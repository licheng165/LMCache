# SPDX-License-Identifier: Apache-2.0
"""P0: sparse decode warm/cold path transitions at connector and cache-engine level."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LoadSpec,
    ReqMeta,
    WorkerRetrieveState,
    _sparse_slot_mapping_len,
)
from tests.v1.connector_test_utils import make_worker_impl

pytest.importorskip("lmcache_ascend", reason="Ascend package required for engine tests")
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


def _make_sparse_request(*, resumed: bool = False) -> ReqMeta:
    return ReqMeta(
        req_id="req-1",
        token_ids=[0] * 256,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=256,
            can_load=True,
        ),
        is_sparse_decode=True,
        resumed_from_preemption=resumed,
    )


def _publish_sparse_state(
    impl, request: ReqMeta, *, with_data: bool = True
) -> WorkerRetrieveState:
    state = WorkerRetrieveState(
        cached_keys=[["layer-key"]],
        cached_starts=[0],
        cached_ends=[256],
    )
    if with_data:
        state.cached_tensors = [[torch.zeros(256)]]
        state.cached_chunk_ptrs_npu = [
            torch.tensor([123], dtype=torch.int64)
        ]
    impl._publish_worker_retrieve_state(
        state,
        request,
        location="LocalCPUBackend",
        metadata_warm=True,
        token_count=256,
    )
    return state


class TestConnectorWarmColdInvalidate:
    def test_complete_layer_cache_is_resolved_without_legacy_binding(self) -> None:
        impl = make_worker_impl()
        impl._latent_kvcaches = [torch.zeros(1)]
        request = _make_sparse_request()
        state = _publish_sparse_state(impl, request)

        fresh = _make_sparse_request()
        bound = impl._worker_retrieve_state_for_request(fresh)

        assert bound is state
        assert state.prepared_sparse_sources[0].total_tokens == 256
        assert not hasattr(fresh, "cached_keys")

        impl._drop_worker_retrieve_state(request.req_id)
        assert state.prepared_sparse_sources == {}

    def test_shared_store_seed_waits_for_collective_bootstrap(self) -> None:
        impl = make_worker_impl()
        impl._latent_kvcaches = [torch.zeros(1)]
        request = _make_sparse_request()
        state = _publish_sparse_state(impl, request)
        impl.lmcache_engine = SimpleNamespace(enable_shared_cpu_cache=True)

        assert impl._prepared_sparse_source(state, 0, 256) is None
        state.shared_request_active = True
        assert impl._prepared_sparse_source(state, 0, 256) is not None

    def test_prepared_shared_state_skips_scope_string_rebuild(self) -> None:
        impl = make_worker_impl()
        impl._latent_kvcaches = [torch.zeros(1)]
        request = _make_sparse_request()
        state = _publish_sparse_state(impl, request)
        state.shared_request_active = True
        state.shared_generation = 7
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
        )
        impl._shared_request_scope_token = MagicMock(
            side_effect=AssertionError("prepared path rebuilt the scope string")
        )

        assert not impl._should_invalidate_worker_retrieve_state(request, 256)
        impl._shared_request_scope_token.assert_not_called()

    def test_prepared_shared_state_skips_cold_coverage_validation(self) -> None:
        impl = make_worker_impl()
        impl._latent_kvcaches = [torch.zeros(1)]
        request = _make_sparse_request()
        state = _publish_sparse_state(impl, request)
        state.shared_request_active = True
        state.shared_latent_status = "present"
        state.shared_generation = 7
        state.pointer_cache_generation = 7
        impl.lmcache_engine = SimpleNamespace(
            enable_shared_cpu_cache=True,
            shared_cpu_cache_generation=7,
        )
        impl._cached_ranges_cover_prefix = MagicMock(
            side_effect=AssertionError("prepared path repeated coverage validation")
        )

        assert impl._worker_retrieve_state_for_request(request) is state
        impl._cached_ranges_cover_prefix.assert_not_called()

    def test_building_store_kwargs_does_not_mutate_retrieve_state(self) -> None:
        impl = make_worker_impl()
        request = _make_sparse_request()
        state = _publish_sparse_state(impl, request, with_data=False)

        fresh = _make_sparse_request()
        bound = impl._worker_retrieve_state_for_request(fresh)
        kwargs = impl._layerwise_store_kwargs(fresh, 0)
        warm = impl._sparse_decode_bootstrap_reuse_kwargs(256, bound)

        assert bound is state
        assert not any(name.startswith("cached_") for name in kwargs)
        assert not hasattr(fresh, "cached_keys")
        assert warm["_retrieve_metadata_warm"] is True
        assert warm["cached_retrieve_location"] == "LocalCPUBackend"

    def test_store_kwargs_identify_requested_group_without_cache_state(
        self,
    ) -> None:
        impl = make_worker_impl()
        impl.config = SimpleNamespace(dsa_two_groups=True)
        impl.enable_sparse_attention = True
        request = _make_sparse_request()

        kwargs = impl._layerwise_store_kwargs(request, 1)

        assert kwargs["kv_group"] == 1
        assert not any(name.startswith("cached_") for name in kwargs)
        assert not hasattr(request, "cached_keys")

    def test_preemption_invalidates_then_cold_reload(self) -> None:
        impl = make_worker_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            location="LocalCPUBackend",
            token_count=256,
        )

        preempted = _make_sparse_request(resumed=True)
        assert impl._should_invalidate_worker_retrieve_state(preempted, 256)

        impl._drop_worker_retrieve_state("req-1")
        assert "req-1" not in impl._worker_retrieve_state

        rebound = impl._worker_retrieve_state_for_request(_make_sparse_request())
        assert rebound is None

    def test_token_rollback_invalidates_worker_state(self) -> None:
        impl = make_worker_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )
        assert impl._should_invalidate_worker_retrieve_state(
            _make_sparse_request(), 128
        )

    def test_extended_prefix_disables_metadata_warm_kwargs(self) -> None:
        impl = make_worker_impl()
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            location="LocalCPUBackend",
            token_count=256,
        )
        warm = impl._sparse_decode_bootstrap_reuse_kwargs(
            512, state
        )
        assert "_retrieve_metadata_warm" not in warm
        assert warm["cached_retrieve_location"] == "LocalCPUBackend"


class TestAscendEngineWarmColdMetadata:
    def test_has_retrieve_data_cache_cold_vs_warm(self) -> None:
        assert not AscendLMCacheEngine._has_retrieve_data_cache(None, None, 2)
        # Empty per-layer lists describe shape only; they do not prove data is
        # ready for warm reuse.
        assert not AscendLMCacheEngine._has_retrieve_data_cache([[], []], None, 2)

        cached_tensors = [torch.zeros(1), torch.zeros(1)]
        assert AscendLMCacheEngine._has_retrieve_data_cache(cached_tensors, None, 2)

        cached_memory_objs = [[MagicMock()], [MagicMock()]]
        assert AscendLMCacheEngine._has_retrieve_data_cache(
            None, cached_memory_objs, 2
        )

    def test_metadata_refresh_when_prefix_grows(self) -> None:
        assert AscendLMCacheEngine._needs_retrieve_metadata_refresh(
            [["k"]], [0], [256], [0] * 512
        )
        assert not AscendLMCacheEngine._needs_retrieve_metadata_refresh(
            [["k"]], [0], [256], [0] * 256
        )

    def test_warm_metadata_skips_contains_when_tensor_cache_ready(self) -> None:
        engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
        engine.storage_manager = MagicMock()
        engine.storage_manager.storage_backends = {"LocalCPUBackend": MagicMock()}
        engine.retrieve_locations = None
        engine.num_layers = 2

        cached_keys: list[list] = [[MagicMock()], [MagicMock()]]
        cached_starts = [0]
        cached_ends = [256]
        ret_mask = torch.zeros(256, dtype=torch.bool)
        retrieve_kwargs = {
            "_retrieve_metadata_warm": True,
            "_use_cached_retrieve": True,
            "cached_retrieve_location": "RemoteBackend",
        }

        location, _, _, keys = engine._ensure_retrieve_chunk_metadata(
            tokens=[0] * 256,
            mask=None,
            request_configs=None,
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs=retrieve_kwargs,
        )

        engine.storage_manager.contains.assert_not_called()
        assert location == "LocalCPUBackend"
        assert keys == cached_keys

    def test_stale_location_rechecked_when_not_using_tensor_cache(self) -> None:
        engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
        engine.storage_manager = MagicMock()
        engine.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
        engine.num_layers = 2

        cached_keys: list[list] = [[MagicMock()], [MagicMock()]]
        engine.storage_manager.batched_contains.return_value = (
            len(cached_keys),
            {"LocalCPUBackend": [layer[0] for layer in cached_keys]},
        )
        cached_starts = [0]
        cached_ends = [256]
        ret_mask = torch.zeros(256, dtype=torch.bool)
        retrieve_kwargs = {
            "_retrieve_metadata_warm": True,
            "cached_retrieve_location": "RemoteBackend",
        }

        location, _, _, _ = engine._ensure_retrieve_chunk_metadata(
            tokens=[0] * 256,
            mask=None,
            request_configs=None,
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs=retrieve_kwargs,
        )

        engine.storage_manager.batched_contains.assert_called_once()
        engine.storage_manager.contains.assert_not_called()
        assert location == "LocalCPUBackend"
        assert retrieve_kwargs["cached_retrieve_location"] == "LocalCPUBackend"

    def test_metadata_refresh_requires_all_layer_keys(self) -> None:
        from lmcache.utils import CacheEngineKey

        engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
        engine.storage_manager = MagicMock()
        engine.retrieve_locations = ["LocalCPUBackend"]
        engine.num_layers = 2
        engine.token_database = MagicMock()
        base_key = CacheEngineKey("model", 1, 0, 42, torch.bfloat16)
        engine.token_database.process_tokens.return_value = [(0, 256, base_key)]

        def contains_side_effect(key, search_range=None):
            if getattr(key, "layer_id", None) == 0:
                return "LocalCPUBackend"
            return None

        engine.storage_manager.contains.side_effect = contains_side_effect

        cached_keys: list[list] = [[], []]
        cached_starts: list[int] = []
        cached_ends: list[int] = []
        ret_mask = torch.zeros(256, dtype=torch.bool)

        location, starts, ends, keys = engine._ensure_retrieve_chunk_metadata(
            tokens=[0] * 256,
            mask=None,
            request_configs=None,
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs={},
        )

        assert engine.storage_manager.contains.call_count == 2
        assert location is None
        assert starts == []
        assert ends == []
        assert keys == []


class TestSparseDecodeTokenMask:
    def test_decode_token_mask_applies_vllm_prefix_mask(self) -> None:
        request = ReqMeta(
            req_id="req-1",
            token_ids=[0] * 512,
            load_spec=LoadSpec(
                vllm_cached_tokens=384,
                lmcache_cached_tokens=256,
                can_load=True,
            ),
            is_sparse_decode=True,
            decode_token_mask=torch.ones(512, dtype=torch.bool),
        )

        token_mask = make_worker_impl()._load_token_mask_for_retrieve(
            request, 512, 256
        )

        assert token_mask[:256].eq(False).all()
        assert token_mask[256:].eq(True).all()
        assert request.decode_token_mask is not None
        assert request.decode_token_mask[:256].eq(False).all()

    def test_decode_token_mask_uses_lmcache_prefix_on_run2(self) -> None:
        impl = make_worker_impl()
        window = _sparse_slot_mapping_len(16384)
        request = ReqMeta(
            req_id="req-2",
            token_ids=[0] * 18879,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=16384,
                can_load=True,
            ),
            is_sparse_decode=True,
            decode_token_mask=torch.ones(window, dtype=torch.bool),
        )

        retrieve_tokens = impl._load_tokens_for_retrieve(
            request.token_ids,
            request.load_spec.lmcache_cached_tokens,
            is_sparse_decode=True,
        )
        token_mask = impl._load_token_mask_for_retrieve(
            request, len(retrieve_tokens), 256
        )

        assert len(retrieve_tokens) == request.load_spec.lmcache_cached_tokens
        assert token_mask is None
        assert request.decode_token_mask is None

    def test_sparse_retrieve_tokens_cover_lmcache_prefix(self) -> None:
        impl = make_worker_impl()
        tokens = list(range(18879))
        retrieve_tokens = impl._load_tokens_for_retrieve(
            tokens,
            lmcache_cached_tokens=16384,
            is_sparse_decode=True,
        )
        assert len(retrieve_tokens) == 16384
