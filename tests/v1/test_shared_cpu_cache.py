# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest
import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedChunkHandle,
    SharedCPUCacheError,
    SharedCPUCacheValidationError,
    SharedHandleEnvelope,
    SharedSlabMapping,
)


def _make_key(kv_group: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=1234,
        dtype=torch.float16,
        kv_group=kv_group,
    )


def _make_memory_obj(
    backing: torch.Tensor,
    *,
    offset: int = 128,
    logical_size: int = 16,
    physical_size: int = 64,
    kv_group: int = 0,
) -> TensorMemoryObj:
    raw = backing[offset : offset + logical_size]
    metadata = MemoryObjMetadata(
        shape=torch.Size([8]),
        dtype=torch.float16,
        address=offset,
        phy_size=physical_size,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT
        if kv_group == 0
        else MemoryFormat.KV_DSA_INDEX_FMT,
        cached_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        shapes=[torch.Size([8])],
        dtypes=[torch.float16],
    )
    return TensorMemoryObj(
        raw_data=raw,
        metadata=metadata,
        parent_allocator=None,
    )


def _make_engine_for_contract(*, use_layerwise: bool, sparse: bool, shared: bool):
    engine = object.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(use_mla=True, world_size=2)
    engine.save_only_first_rank = True
    engine.enable_shared_cpu_cache = shared
    engine.dsa_two_groups = True
    engine.shared_cpu_cache_strict = True
    engine.config = SimpleNamespace(
        use_layerwise=use_layerwise,
        enable_sparse_attention=sparse,
        local_cpu=True,
        max_local_cpu_size=1,
        get_extra_config_value=lambda key, default=None: default,
    )
    return engine


class _FakeSharedShapeConnector:
    def get_shape(self, num_tokens: int, kv_group: int = 0) -> torch.Size:
        hidden = 1024 if kv_group == 0 else 128
        return torch.Size([num_tokens, hidden])


def _make_engine_for_sparse_capacity(*, max_local_cpu_size: float):
    engine = object.__new__(LMCacheEngine)
    extra_config = {
        "vllm_max_model_len": 1024,
        "vllm_max_num_seqs": 32,
    }
    engine.config = SimpleNamespace(
        enable_sparse_attention=True,
        chunk_size=256,
        max_local_cpu_size=max_local_cpu_size,
        extra_config=extra_config,
        get_extra_config_value=lambda key, default=None: extra_config.get(
            key,
            default,
        ),
    )
    engine.metadata = SimpleNamespace(
        world_size=8,
        is_first_rank=lambda: True,
        max_model_len=1024,
        kv_dtype=torch.float16,
        get_dtypes=lambda: [torch.float16],
        get_shapes=lambda num_tokens: [torch.Size([num_tokens, 1024])],
    )
    engine.num_layers = 4
    engine.save_only_first_rank = True
    engine.enable_shared_cpu_cache = True
    engine.dsa_two_groups = True
    engine.shared_cpu_cache_strict = True
    engine.gpu_connector = _FakeSharedShapeConnector()
    engine._shared_cpu_active_sparse_requests = {}
    return engine


@pytest.mark.no_shared_allocator
def test_shared_cpu_size_override_wins_over_first_rank_size(monkeypatch):
    captured = {}

    class DummyMixedMemoryAllocator:
        def __init__(self, size, **kwargs):
            captured["size"] = size
            captured["kwargs"] = kwargs

    import lmcache.v1.cache_engine as cache_engine_module

    monkeypatch.setattr(
        cache_engine_module,
        "MixedMemoryAllocator",
        DummyMixedMemoryAllocator,
    )
    config = SimpleNamespace(
        extra_config={
            "save_only_first_rank": True,
            "enable_shared_cpu_cache": True,
            "shared_cpu_cache_size_gb": 3,
            "first_rank_max_local_cpu_size": 9,
        },
        gds_path=None,
        cufile_buffer_size=None,
        max_local_cpu_size=5,
        get_extra_config_value=lambda key, default=None: config.extra_config.get(
            key,
            default,
        ),
    )
    metadata = SimpleNamespace(use_mla=True, is_first_rank=lambda: True)

    allocator = LMCacheEngineBuilder._Create_memory_allocator(
        config,
        metadata,
        None,
    )

    assert isinstance(allocator, DummyMixedMemoryAllocator)
    assert captured["size"] == 3 * 1024**3


def test_shared_cpu_shm_capacity_preflight_reports_sigbus_risk(monkeypatch):
    engine = object.__new__(LMCacheEngine)
    engine.enable_shared_cpu_cache = True
    engine.shared_cpu_cache_name = "/lmcache-too-large"
    engine.metadata = SimpleNamespace(is_first_rank=lambda: True)
    engine.config = SimpleNamespace(
        max_local_cpu_size=2,
        get_extra_config_value=lambda key, default=None: default,
    )

    monkeypatch.setattr("os.path.isdir", lambda path: path == "/dev/shm")
    monkeypatch.setattr(
        "os.statvfs",
        lambda _path: SimpleNamespace(f_bavail=1, f_frsize=1024**3),
    )

    with pytest.raises(ValueError, match="SIGBUS"):
        engine._preflight_shared_cpu_shm_capacity()


class _FakeAddressManager:
    def __init__(self, free_bytes: int):
        self._free_bytes = free_bytes
        self.total_allocated_size = 0

    def get_free_size(self) -> int:
        return self._free_bytes


class _FakeLocalCPUBackend:
    def __init__(self, *, free_bytes: int, hot_cache: dict):
        self.hot_cache = hot_cache
        self.cpu_lock = nullcontext()
        self.memory_allocator = SimpleNamespace(
            buffer=torch.empty(1024, dtype=torch.uint8),
            address_manager=_FakeAddressManager(free_bytes),
            align_bytes=64,
        )


class _FakeResolvableMemoryObj:
    def __init__(self):
        self.is_pinned = False
        self.ref_count_down_count = 0

    def pin(self):
        self.is_pinned = True

    def unpin(self):
        self.is_pinned = False

    def ref_count_down(self):
        self.ref_count_down_count += 1


class _FakeGetBlockingLocalCPUBackend:
    def __init__(self, hot_obj):
        self.hot_obj = hot_obj

    def get_blocking(self, key):
        return self.hot_obj


def test_engine_contract_requires_shared_cache_for_dense_layerwise_tp():
    engine = _make_engine_for_contract(
        use_layerwise=True,
        sparse=False,
        shared=False,
    )

    with pytest.raises(ValueError, match="use_layerwise=true") as exc_info:
        engine._validate_shared_cpu_cache_contract()
    message = str(exc_info.value)
    assert "enable_shared_cpu_cache" in message
    assert "save_only_first_rank" in message
    assert "TP/world_size=2" in message
    assert "shared_cpu_cache_size_gb" in message


def test_engine_contract_requires_broadcast_object_fn_for_shared_tp():
    engine = _make_engine_for_contract(
        use_layerwise=True,
        sparse=False,
        shared=True,
    )
    engine.broadcast_object_fn = None

    with pytest.raises(ValueError, match="broadcast_object_fn"):
        engine._validate_shared_cpu_cache_contract()


def test_rank0_post_init_broadcasts_startup_error_on_storage_failure(
    monkeypatch,
):
    engine = object.__new__(LMCacheEngine)
    engine.post_inited = False
    engine.enable_shared_cpu_cache = True
    engine.use_layerwise = False
    engine.save_only_first_rank = True
    engine.lmcache_worker = None
    engine.event_manager = object()
    engine.storage_manager = None
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_slab_size = None
    engine.shared_cpu_cache_generation = 0
    engine.metadata = SimpleNamespace(
        use_mla=True,
        world_size=2,
        worker_id=0,
        first_rank=0,
        is_first_rank=lambda: True,
    )
    engine.config = SimpleNamespace(
        get_lookup_server_worker_ids=lambda use_mla, world_size: [],
    )
    broadcasts = []
    engine.broadcast_object_fn = lambda payload, src: broadcasts.append(
        (payload, src)
    )

    def fail_storage_manager(*args, **kwargs):
        raise RuntimeError("stale shm segment")

    monkeypatch.setattr(
        "lmcache.v1.cache_engine.StorageManager",
        fail_storage_manager,
    )

    with pytest.raises(RuntimeError, match="stale shm segment"):
        engine.post_init()

    assert len(broadcasts) == 1
    envelope, src = broadcasts[0]
    assert src == 0
    assert envelope["status"] == "error"
    assert envelope["shm_name"] == "/lmcache-test"
    assert "StorageManager" in envelope["message"]
    assert "stale shm segment" in envelope["message"]


def test_sparse_capacity_preflight_fails_when_one_max_request_cannot_fit():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=0.001)

    with pytest.raises(ValueError, match="one maximum request cannot fit"):
        engine._report_shared_cpu_sparse_capacity_sanity()


def test_sparse_capacity_preflight_records_startup_estimate():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)

    engine._report_shared_cpu_sparse_capacity_sanity()

    estimate = engine.config.extra_config[
        "shared_cpu_sparse_startup_capacity_estimate"
    ]
    assert estimate["max_model_len"] == 1024
    assert estimate["max_num_seqs"] == 32
    assert estimate["kv_groups"] == [0, 1]
    assert estimate["one_max_request_bytes"] > 0
    assert estimate["configured_worst_case_bytes"] == (
        estimate["one_max_request_bytes"] * 32
    )


def test_sparse_capacity_shape_helper_keeps_two_dim_token_shape():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    engine.num_layers = 256

    assert engine._shape_numel_without_layer_dim(torch.Size([256, 1024])) == (
        256 * 1024
    )


def test_runtime_capacity_details_exclude_required_hot_chunks_from_evictable():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    hot_key = _make_key()
    miss_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=5678,
        dtype=torch.float16,
        kv_group=0,
    )
    other_key = CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=9999,
        dtype=torch.float16,
        kv_group=0,
    )
    hot_obj = _make_memory_obj(
        torch.empty(1024, dtype=torch.uint8),
        physical_size=64,
    )
    other_obj = _make_memory_obj(
        torch.empty(1024, dtype=torch.uint8),
        offset=256,
        physical_size=64,
    )
    backend = _FakeLocalCPUBackend(
        free_bytes=9000,
        hot_cache={hot_key: hot_obj, other_key: other_obj},
    )
    engine._shared_local_cpu_backend = lambda: backend
    engine._is_rank0_shared_mem_obj = lambda mem_obj: mem_obj in (
        hot_obj,
        other_obj,
    )
    engine.config.chunk_size = 4
    engine._shared_cpu_active_sparse_requests = {"req-old": {}}

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[hot_key, miss_key]],
        chunk_locations_layer_major=[["LocalCPUBackend", "MooncakeStore"]],
        token_count=8,
        chunk_token_lengths=[1, 1],
    )

    expected_missing_bytes = engine._shared_cpu_estimated_physical_chunk_bytes(
        0,
        num_tokens=1,
    )
    assert details["required_bytes"] == expected_missing_bytes
    assert details["available_after_eviction"] == 9064
    assert details["protected_hot_bytes"] == 64
    assert details["hot_chunk_count"] == 1
    assert details["non_shm_hot_chunk_count"] == 0
    assert details["active_sparse_requests"] == 2
    assert details["fits"] is True


def test_runtime_capacity_counts_non_shm_hot_hits_as_required_bytes():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    hot_key = _make_key()
    hot_obj = _make_memory_obj(
        torch.empty(1024, dtype=torch.uint8),
        physical_size=64,
    )
    backend = _FakeLocalCPUBackend(
        free_bytes=0,
        hot_cache={hot_key: hot_obj},
    )
    engine._shared_local_cpu_backend = lambda: backend
    engine._is_rank0_shared_mem_obj = lambda _mem_obj: False
    engine.config.chunk_size = 4

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[hot_key]],
        chunk_locations_layer_major=[["LocalCPUBackend"]],
        token_count=1,
        chunk_token_lengths=[1],
    )

    expected_bytes = engine._shared_cpu_estimated_physical_chunk_bytes(
        0,
        num_tokens=1,
    )
    assert details["required_bytes"] == expected_bytes
    assert details["available_after_eviction"] == 0
    assert details["protected_hot_bytes"] == 0
    assert details["hot_chunk_count"] == 0
    assert details["non_shm_hot_chunk_count"] == 1
    assert details["fits"] is False


def test_runtime_capacity_details_report_failure_before_materialization():
    engine = _make_engine_for_sparse_capacity(max_local_cpu_size=1)
    miss_key = _make_key()
    backend = _FakeLocalCPUBackend(free_bytes=0, hot_cache={})
    engine._shared_local_cpu_backend = lambda: backend
    engine.config.chunk_size = 4

    details = engine._shared_cpu_runtime_capacity_details(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        kv_group=0,
        keys_layer_major=[[miss_key]],
        chunk_locations_layer_major=[["MooncakeStore"]],
        token_count=4,
        chunk_token_lengths=[1],
    )

    assert details["required_bytes"] == engine._shared_cpu_estimated_physical_chunk_bytes(
        0,
        num_tokens=1,
    )
    assert details["available_after_eviction"] == 0
    assert details["fits"] is False


def test_rank0_resolver_rematerializes_non_shm_hot_cache_hit():
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = object()
    hot_obj = _FakeResolvableMemoryObj()
    materialized_obj = _FakeResolvableMemoryObj()
    backend = _FakeGetBlockingLocalCPUBackend(hot_obj)
    key = _make_key()
    materialized_from = []

    engine._shared_local_cpu_backend = lambda: backend
    engine._is_rank0_shared_mem_obj = lambda _obj: False
    engine._validate_rank0_shared_mem_obj = lambda *args, **kwargs: None

    def materialize_shared_copy(**kwargs):
        materialized_from.append(kwargs["src_obj"])
        return materialized_obj

    engine._materialize_shared_rank0_copy = materialize_shared_copy

    resolved = engine._resolve_shared_rank0_layer_mem_objs(
        req_id="req-1",
        phase="sparse_decode_bootstrap",
        layer_id=0,
        kv_group=0,
        keys_layer=[key],
        chunk_locations=["LocalCPUBackend"],
    )

    assert resolved == [materialized_obj]
    assert materialized_from == [hot_obj]
    assert hot_obj.ref_count_down_count == 1
    assert materialized_obj.is_pinned


def test_rank0_handle_builder_rejects_partial_publication():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_generation = 3
    engine.metadata = SimpleNamespace(worker_id=0)
    backing = torch.arange(1024, dtype=torch.uint8)

    with pytest.raises(ValueError, match="partial layer handles"):
        engine._make_shared_handles_for_layer(
            req_id="req-1",
            phase="dense_prefix",
            keys_layer=[_make_key(), _make_key(kv_group=1)],
            mem_objs_layer=[_make_memory_obj(backing)],
            layer_id=0,
            kv_group=0,
        )


def test_rank0_handle_builder_validates_objects_before_publication():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_generation = 3
    engine.metadata = SimpleNamespace(worker_id=0)
    backing = torch.arange(1024, dtype=torch.uint8)

    def reject_publication(*_args, **_kwargs):
        raise ValueError("object is not shm-backed")

    engine._validate_rank0_shared_mem_obj = reject_publication

    with pytest.raises(ValueError, match="not shm-backed"):
        engine._make_shared_handles_for_layer(
            req_id="req-1",
            phase="dense_prefix",
            keys_layer=[_make_key()],
            mem_objs_layer=[_make_memory_obj(backing)],
            layer_id=0,
            kv_group=0,
        )


def test_shared_chunk_handle_preserves_key_and_cached_positions():
    backing = torch.arange(1024, dtype=torch.uint8)
    key = _make_key()
    memory_obj = _make_memory_obj(backing)

    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=key,
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=memory_obj,
        generation=7,
        producer_rank=0,
    )

    encoded = handle.to_dict()
    assert set(encoded) == {
        "request_id",
        "phase",
        "key",
        "layer_id",
        "kv_group",
        "chunk_index",
        "shm_name",
        "offset",
        "physical_size",
        "logical_size",
        "shape",
        "dtype",
        "shapes",
        "dtypes",
        "fmt",
        "cached_positions",
        "generation",
        "producer_rank",
        "status",
    }
    assert encoded["key"] == key
    assert encoded["cached_positions"] == [0, 1, 2, 3]
    forbidden_fragments = (
        "ptr",
        "pointer",
        "data_ptr",
        "host",
        "device",
        "allocator",
        "parent",
        "object",
    )
    assert not any(
        fragment in field
        for field in encoded
        for fragment in forbidden_fragments
    )

    decoded = SharedChunkHandle.from_dict(encoded)
    assert decoded.key == key
    assert decoded.cached_positions == [0, 1, 2, 3]
    assert decoded.offset == 128
    assert decoded.logical_size == 16
    assert decoded.physical_size == 64


def test_shared_chunk_handle_rejects_missing_required_field():
    backing = torch.arange(1024, dtype=torch.uint8)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(backing),
        generation=7,
        producer_rank=0,
    )
    encoded = handle.to_dict()
    encoded.pop("cached_positions")

    with pytest.raises(SharedCPUCacheValidationError, match="cached_positions"):
        SharedChunkHandle.from_dict(encoded)


def test_shared_chunk_handle_rejects_pointer_private_fields():
    backing = torch.arange(1024, dtype=torch.uint8)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(backing),
        generation=7,
        producer_rank=0,
    )
    encoded = handle.to_dict()
    encoded["host_ptr"] = 123456

    with pytest.raises(SharedCPUCacheValidationError, match="forbidden"):
        SharedChunkHandle.from_dict(encoded)


def test_shared_chunk_handle_reports_bad_payload_type_and_dtype():
    with pytest.raises(SharedCPUCacheValidationError, match="expected dict"):
        SharedChunkHandle.from_dict("not-a-dict")  # type: ignore[arg-type]

    backing = torch.arange(1024, dtype=torch.uint8)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=2,
        kv_group=0,
        chunk_index=3,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(backing),
        generation=7,
        producer_rank=0,
    )
    encoded = handle.to_dict()
    encoded["dtype"] = "torch.not_a_dtype"

    with pytest.raises(SharedCPUCacheValidationError, match="Unknown.*dtype"):
        SharedChunkHandle.from_dict(encoded)


def test_passive_allocator_creates_view_and_free_only_invalidates():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=11,
        producer_rank=0,
    )

    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=11,
    )
    view = allocator.create_view(
        handle,
        expected_request_id="req-1",
        expected_phase="sparse_decode_bootstrap",
        expected_layer_id=0,
        expected_kv_group=0,
        expected_chunk_index=0,
    )

    assert view.parent() is allocator
    assert view.metadata.address == handle.offset
    assert view.metadata.phy_size == handle.physical_size
    assert view.metadata.cached_positions.tolist() == [0, 1, 2, 3]
    assert torch.equal(view.raw_tensor, slab[128:144])

    view.ref_count_down()
    assert not view.is_valid()
    allocator.free(view)
    assert not view.is_valid()


def test_passive_allocator_rejects_bounds_generation_and_order_mismatch():
    slab = torch.arange(256, dtype=torch.uint8)
    source_obj = _make_memory_obj(
        slab,
        offset=128,
        logical_size=16,
        physical_size=256,
    )
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=1,
        kv_group=0,
        chunk_index=4,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=3,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="generation=2"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=1,
            expected_kv_group=0,
            expected_chunk_index=4,
        )

    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )
    with pytest.raises(SharedCPUCacheValidationError, match="bounds"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=1,
            expected_kv_group=0,
            expected_chunk_index=5,
        )


def test_passive_allocator_rejects_inconsistent_shape_size():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    bad_handle = replace(handle, shape=torch.Size([4]), shapes=[torch.Size([4])])
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="shape/dtype bytes"):
        allocator.create_view(
            bad_handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
        )


def test_passive_allocator_rejects_key_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    key = _make_key()
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=key,
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="expected="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_key=_make_key(kv_group=1),
        )


def test_passive_allocator_rejects_producer_rank_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=3,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="producer_rank=3"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_producer_rank=0,
        )


def test_passive_allocator_rejects_cached_positions_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="cached_positions"):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_cached_positions=[4, 5, 6, 7],
        )


def test_passive_allocator_requires_cached_positions_when_expected():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="cached_positions is None"):
        allocator.create_view(
            replace(handle, cached_positions=None),
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_cached_positions=[0, 1, 2, 3],
        )


def test_passive_allocator_rejects_expected_metadata_mismatch():
    slab = torch.arange(1024, dtype=torch.uint8)
    source_obj = _make_memory_obj(slab)
    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=source_obj,
        generation=2,
        producer_rank=0,
    )
    allocator = PassiveSharedViewAllocator(
        slab_tensor=slab,
        shm_name="/lmcache-test",
        generation=2,
    )

    with pytest.raises(SharedCPUCacheValidationError, match="shape="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_shape=torch.Size([4]),
        )
    with pytest.raises(SharedCPUCacheValidationError, match="dtype="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_dtype=torch.float32,
        )
    with pytest.raises(SharedCPUCacheValidationError, match="fmt="):
        allocator.create_view(
            handle,
            expected_request_id="req-1",
            expected_phase="dense_prefix",
            expected_layer_id=0,
            expected_kv_group=0,
            expected_chunk_index=0,
            expected_fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        )


def test_passive_allocator_never_allocates():
    allocator = PassiveSharedViewAllocator(
        slab_tensor=torch.empty(16, dtype=torch.uint8),
        shm_name="/lmcache-test",
        generation=1,
    )
    with pytest.raises(SharedCPUCacheError):
        allocator.allocate(torch.Size([1]), torch.uint8)
    with pytest.raises(SharedCPUCacheError):
        allocator.batched_allocate(torch.Size([1]), torch.uint8, 2)


def test_shared_slab_owner_close_unlinks_without_detach(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(
            unlink_shm=lambda shm_name: calls.append(("unlink", shm_name)),
            detach_shm_pinned_ptr=lambda ptr, size: calls.append(
                ("detach", ptr, size)
            ),
        ),
    )
    mapping = SharedSlabMapping(
        shm_name="/lmcache-owner-close",
        size=16,
        ptr=1234,
        tensor=torch.empty(16, dtype=torch.uint8),
        generation=7,
        owner=True,
    )

    mapping.close()
    mapping.close()

    assert calls == [("unlink", "/lmcache-owner-close")]


def test_shared_slab_preflight_reports_null_device_ptr(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(get_device_ptr=lambda ptr: None),
    )
    mapping = SharedSlabMapping(
        shm_name="/lmcache-preflight",
        size=16,
        ptr=1234,
        tensor=torch.empty(16, dtype=torch.uint8),
        generation=7,
        owner=False,
    )

    with pytest.raises(SharedCPUCacheError, match="get_device_ptr returned None"):
        mapping.preflight_device_ptr()


def test_shared_slab_attach_falls_back_to_non_cuda_equivalents(monkeypatch):
    import lmcache

    fallback_ops = SimpleNamespace(
        attach_shm_pinned_ptr=lambda size, name, writable: 4321,
    )
    monkeypatch.delitem(sys.modules, "lmcache.c_ops", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "lmcache.non_cuda_equivalents",
        fallback_ops,
    )
    monkeypatch.setattr(
        lmcache,
        "non_cuda_equivalents",
        fallback_ops,
        raising=False,
    )
    monkeypatch.setattr(
        SharedSlabMapping,
        "_tensor_from_ptr",
        staticmethod(
            lambda _ptr, size: (torch.empty(size, dtype=torch.uint8), object())
        ),
    )

    mapping = SharedSlabMapping.attach(
        shm_name="/lmcache-no-cuda",
        size=16,
        generation=3,
        writable=False,
    )

    assert mapping.ptr == 4321
    assert mapping.owner is False


def test_shared_slab_attach_reports_null_host_ptr(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(attach_shm_pinned_ptr=lambda size, name, writable: 0),
    )

    with pytest.raises(SharedCPUCacheError, match="returned 0"):
        SharedSlabMapping.attach(
            shm_name="/lmcache-attach-null",
            size=16,
            generation=3,
            writable=False,
        )


def test_shared_slab_attach_detaches_if_tensor_view_creation_fails(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "lmcache.c_ops",
        SimpleNamespace(
            attach_shm_pinned_ptr=lambda size, name, writable: 1234,
            detach_shm_pinned_ptr=lambda ptr, size: calls.append((ptr, size)),
        ),
    )

    def fail_tensor_from_ptr(_ptr, _size):
        raise RuntimeError("tensor view failed")

    monkeypatch.setattr(
        SharedSlabMapping,
        "_tensor_from_ptr",
        staticmethod(fail_tensor_from_ptr),
    )

    with pytest.raises(RuntimeError, match="tensor view failed"):
        SharedSlabMapping.attach(
            shm_name="/lmcache-attach-cleanup",
            size=16,
            generation=3,
            writable=False,
        )

    assert calls == [(1234, 16)]


def test_rank0_slab_rejects_empty_allocator_buffer():
    with pytest.raises(SharedCPUCacheError, match="invalid buffer size"):
        SharedSlabMapping.from_rank0_allocator(
            shm_name="/lmcache-rank0-empty",
            allocator_tensor=torch.empty(0, dtype=torch.uint8),
            generation=9,
        )


def test_rank0_startup_preflight_broadcasts_error_before_raising():
    engine = object.__new__(LMCacheEngine)
    broadcasts = []
    engine.enable_shared_cpu_cache = True
    engine.storage_manager = None
    engine.shared_cpu_cache_name = "/lmcache-test"
    engine.shared_cpu_cache_slab_size = None
    engine.shared_cpu_cache_generation = 0
    engine.metadata = SimpleNamespace(
        world_size=2,
        first_rank=0,
        worker_id=0,
        is_first_rank=lambda: True,
    )
    engine.broadcast_object_fn = lambda obj, rank: broadcasts.append((obj, rank))

    with pytest.raises(ValueError, match="requires StorageManager"):
        engine._post_init_shared_cpu_cache()

    assert broadcasts
    envelope, rank = broadcasts[-1]
    assert rank == 0
    assert envelope["status"] == "error"
    assert "requires StorageManager" in envelope["message"]
    assert envelope["shm_name"] == "/lmcache-test"


def test_rank0_startup_preflight_failure_closes_mapping_before_error_broadcast(
    monkeypatch,
):
    engine = object.__new__(LMCacheEngine)
    broadcasts = []
    closed = []

    class FakeMapping:
        def preflight_device_ptr(self):
            raise SharedCPUCacheError("preflight boom")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        "lmcache.v1.cache_engine.SharedSlabMapping.from_rank0_allocator",
        lambda **_kwargs: FakeMapping(),
    )
    engine.enable_shared_cpu_cache = True
    engine.shared_cpu_cache_strict = True
    engine.shared_cpu_cache_mapping = None
    engine.storage_manager = SimpleNamespace(
        local_cpu_backend=SimpleNamespace(
            memory_allocator=SimpleNamespace(
                buffer=torch.empty(16, dtype=torch.uint8),
                shm_name="/lmcache-startup-cleanup",
            )
        )
    )
    engine.shared_cpu_cache_name = None
    engine.shared_cpu_cache_slab_size = None
    engine.shared_cpu_cache_generation = 0
    engine.metadata = SimpleNamespace(
        world_size=2,
        first_rank=0,
        worker_id=0,
        is_first_rank=lambda: True,
    )
    engine.broadcast_object_fn = lambda obj, rank: broadcasts.append((obj, rank))

    with pytest.raises(SharedCPUCacheError, match="preflight boom"):
        engine._post_init_shared_cpu_cache()

    assert closed == [True]
    assert engine.shared_cpu_cache_mapping is None
    envelope, rank = broadcasts[-1]
    assert rank == 0
    assert envelope["status"] == "error"
    assert "preflight boom" in envelope["message"]


def test_receive_shared_envelope_reports_corrupt_payload():
    engine = object.__new__(LMCacheEngine)
    engine.metadata = SimpleNamespace(first_rank=0)
    engine.broadcast_object_fn = lambda obj, rank: {"status": "ok"}

    with pytest.raises(ValueError, match="corrupt envelope"):
        engine._receive_shared_envelope()


def test_skipped_index_envelope_round_trips_without_handles():
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )

    encoded = envelope.to_dict()
    assert set(encoded) == {
        "request_id",
        "phase",
        "request_ordinal",
        "layer_id",
        "kv_group",
        "status",
        "generation",
        "handles",
        "message",
        "error_details",
    }
    decoded = SharedHandleEnvelope.from_dict(encoded)
    assert decoded.status == "skipped"
    assert decoded.kv_group == 1
    assert decoded.handles == []
    assert decoded.message is not None


def test_shared_envelope_rejects_missing_required_field():
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )
    encoded = envelope.to_dict()
    encoded.pop("error_details")

    with pytest.raises(SharedCPUCacheValidationError, match="error_details"):
        SharedHandleEnvelope.from_dict(encoded)


def test_shared_envelope_rejects_pointer_private_fields():
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )
    encoded = envelope.to_dict()
    encoded["device_ptr"] = 123456

    with pytest.raises(SharedCPUCacheValidationError, match="forbidden"):
        SharedHandleEnvelope.from_dict(encoded)


def test_shared_envelope_reports_bad_payload_type_status_and_handles():
    with pytest.raises(SharedCPUCacheValidationError, match="expected dict"):
        SharedHandleEnvelope.from_dict("not-a-dict")  # type: ignore[arg-type]

    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="sparse_decode_bootstrap",
        request_ordinal=0,
        layer_id=5,
        kv_group=1,
        status="skipped",
        generation=9,
        handles=[],
        message="index already resident by non-strict debug path",
    )
    encoded = envelope.to_dict()
    encoded["status"] = "surprise"
    with pytest.raises(SharedCPUCacheValidationError, match="unsupported status"):
        SharedHandleEnvelope.from_dict(encoded)

    encoded = envelope.to_dict()
    encoded["handles"] = {"not": "a list"}
    with pytest.raises(SharedCPUCacheValidationError, match="handles must be a list"):
        SharedHandleEnvelope.from_dict(encoded)


def test_dense_prefix_zero_hit_broadcasts_skipped_not_miss():
    engine = object.__new__(LMCacheEngine)
    engine.storage_manager = SimpleNamespace()
    engine.gpu_connector = SimpleNamespace()
    engine.num_layers = 2
    engine.shared_cpu_cache_generation = 9
    engine.metadata = SimpleNamespace(first_rank=0, worker_id=0)
    broadcasts = []
    engine.broadcast_object_fn = lambda obj, rank: broadcasts.append(obj)
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_finished=lambda monitor_req_id, tokens: broadcasts.append(
            {"stats": int(tokens)}
        )
    )
    ret_mask = torch.zeros(8, dtype=torch.bool)

    yielded = list(
        engine._retrieve_layer_shared_rank0(
            starts=[],
            ends=[],
            keys_layer_major=[],
            chunk_locations_layer_major=[],
            location=None,
            ret_mask=ret_mask,
            monitor_req_id=123,
            req_id="req-1",
            kv_group=0,
            kwargs={"shared_cpu_phase": "dense_prefix"},
        )
    )

    envelopes = [item for item in broadcasts if "status" in item]
    assert [item["status"] for item in envelopes] == ["skipped", "skipped"]
    assert all(item["handles"] == [] for item in envelopes)
    assert torch.equal(yielded[-1], ret_mask)
    assert broadcasts[-1] == {"stats": 0}


def test_strict_shared_envelope_rejects_miss_before_view_creation():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_strict = True
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="miss",
        generation=9,
        handles=[],
        message="missing required dense prefix chunk",
    )

    with pytest.raises(ValueError, match="strict mode received miss envelope"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
        )


def test_shared_envelope_rejects_request_ordinal_mismatch():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_strict = True
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=2,
        layer_id=0,
        kv_group=0,
        status="skipped",
        generation=9,
        handles=[],
    )

    with pytest.raises(ValueError, match="request_ordinal=2"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=1,
            layer_id=0,
            kv_group=0,
        )


def test_shared_envelope_rejects_status_handle_mismatch():
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_generation = 9
    engine.shared_cpu_cache_strict = True
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="ok",
        generation=9,
        handles=[],
    )

    with pytest.raises(ValueError, match="ok envelope"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
        )

    handle = SharedChunkHandle.from_memory_obj(
        request_id="req-1",
        phase="dense_prefix",
        key=_make_key(),
        layer_id=0,
        kv_group=0,
        chunk_index=0,
        shm_name="/lmcache-test",
        memory_obj=_make_memory_obj(torch.arange(1024, dtype=torch.uint8)),
        generation=9,
        producer_rank=0,
    )
    envelope = SharedHandleEnvelope(
        request_id="req-1",
        phase="dense_prefix",
        request_ordinal=0,
        layer_id=0,
        kv_group=0,
        status="skipped",
        generation=9,
        handles=[handle],
    )

    with pytest.raises(ValueError, match="must not carry handles"):
        engine._validate_shared_layerwise_envelope(
            envelope,
            req_id="req-1",
            phase="dense_prefix",
            request_ordinal=0,
            layer_id=0,
            kv_group=0,
        )
