# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    SaveSpec,
)


class _FakeParent:
    def __init__(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeEngine:
    def __init__(self):
        self.unpinned: list[str] = []
        self.store_steps: dict[str, int] = {}
        self.store_calls: list[str] = []
        self.store_kwargs: list[dict] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store_layer(self, token_ids, **kwargs):
        req_id = kwargs["req_id"]
        self.store_calls.append(req_id)
        self.store_kwargs.append(dict(kwargs))
        self.store_steps.setdefault(req_id, 0)

        def _storer():
            while True:
                self.store_steps[req_id] += 1
                yield None

        return _storer()


class _FakeManager:
    def __init__(self, engine: _FakeEngine):
        self.lmcache_engine = engine


def _make_req(
    req_id: str,
    can_save: bool = True,
    request_configs: dict | None = None,
):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        slot_mapping=[torch.arange(4, dtype=torch.long)],
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
        is_sparse_decode=False,
        load_spec=None,
        cached_keys=[],
        cached_starts=[],
        cached_ends=[],
        cached_memory_objs=[],
        cached_tensors=[],
        request_configs=request_configs,
    )


def _make_connector(requests):
    metadata = LMCacheConnectorMetadata(requests=requests)
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = _FakeManager(engine)
    connector.kv_role = "kv_producer"
    connector.use_layerwise = True
    connector.device = "cpu"
    connector.config = SimpleNamespace(dsa_two_groups=False)
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._kvcaches_list = []
    connector._latent_layer_names = []
    connector._indexer_layer_names = []
    connector._latent_kvcaches = []
    connector._indexer_kvcaches = []
    connector._layerwise_save_storers = {}
    connector._deferred_latent_pending = set()
    # lmcache_ascend patches LMCacheConnectorV1Impl at import time; __new__ skips
    # LMCacheAscendConnectorV1Impl.__init__ which normally sets these.
    connector.store_async = False
    connector._wait_for_save_done = True
    connector._finished_req_ids_waiting_for_save = set()
    connector._late_finished_sending = set()
    return connector, metadata, engine


def _init_indexer_cache_fields(request) -> None:
    request.cached_keys_indexer = []
    request.cached_starts_indexer = []
    request.cached_ends_indexer = []
    request.cached_memory_objs_indexer = []
    request.cached_tensors_indexer = []
    request.cached_chunk_dev_ptrs_indexer = []
    request.cached_chunk_ptrs_npu_indexer = []
    request.cached_shared_handles_indexer = []


def test_layerwise_storer_is_request_scoped_across_interleaved_finalize() -> None:
    connector, metadata, engine = _make_connector(
        [_make_req("req-1"), _make_req("req-2")]
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1", "req-2"]
    assert engine.store_steps["req-1"] == 1
    assert engine.store_steps["req-2"] == 1

    metadata.requests = [_make_req("req-1")]
    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert engine.store_steps["req-2"] == 1
    assert engine.unpinned == ["req-1"]
    assert set(connector._layerwise_save_storers.keys()) == {"req-2"}

    metadata.requests = [_make_req("req-2")]
    connector.wait_for_save()
    assert engine.store_steps["req-2"] == 2
    assert engine.unpinned == ["req-1", "req-2"]
    assert connector._layerwise_save_storers == {}


def test_wait_for_save_repeated_call_does_not_readvance_finalized_storer() -> None:
    connector, metadata, engine = _make_connector([_make_req("req-1")])
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_steps["req-1"] == 1

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2


def test_multi_layer_save_and_finalize() -> None:
    connector, _, engine = _make_connector([_make_req("req-1"), _make_req("req-2")])
    num_layers = 4

    for _ in range(num_layers):
        connector.save_kv_layer("layer_x", torch.zeros(1), None)

    assert engine.store_steps["req-1"] == num_layers
    assert engine.store_steps["req-2"] == num_layers

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == num_layers + 1
    assert engine.store_steps["req-2"] == num_layers + 1
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_skips_requests_that_cannot_save() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])
    connector.kv_role = "kv_both"
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == []
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_kv_producer_ignores_can_save_flag() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1"]
    assert engine.store_steps["req-1"] == 1
    assert set(connector._layerwise_save_storers.keys()) == {"req-1"}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_passes_request_configs() -> None:
    request_configs = {"lmcache.tag.schema": "dsa-index-save-v2"}
    connector, _, engine = _make_connector(
        [_make_req("req-1", request_configs=request_configs)]
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)

    assert engine.store_calls == ["req-1"]
    assert engine.store_kwargs[0]["request_configs"] == request_configs


def test_indexer_save_uses_layer_metadata_slots_not_request_slots() -> None:
    request = _make_req("req-1")
    request.save_spec = SaveSpec(
        skip_leading_tokens=0,
        can_save=True,
        can_save_latent=True,
        can_save_indexer=True,
    )
    request.indexer_slot_mapping = [torch.arange(100, 104, dtype=torch.long)]
    _init_indexer_cache_fields(request)

    connector, _, engine = _make_connector([request])
    connector.config = SimpleNamespace(dsa_two_groups=True)
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    connector.kv_caches = {
        "model.layers.0.self_attn.attn": torch.zeros(1),
        indexer_layer_name: torch.zeros(1),
    }
    metadata_slots = torch.arange(200, 204, dtype=torch.long)
    attn_metadata = {
        indexer_layer_name: SimpleNamespace(slot_mapping=metadata_slots),
    }

    connector.save_kv_layer(indexer_layer_name, torch.zeros(1), attn_metadata)

    assert engine.store_calls == ["req-1"]
    assert engine.store_kwargs[0]["kv_group"] == 1
    assert torch.equal(engine.store_kwargs[0]["slot_mapping"], metadata_slots)
    assert not torch.equal(
        engine.store_kwargs[0]["slot_mapping"],
        request.indexer_slot_mapping[0],
    )


def test_chunked_indexer_save_pads_layer_metadata_slots() -> None:
    request = _make_req("req-1")
    request.token_ids = list(range(16))
    request.slot_mapping = [torch.arange(16, dtype=torch.long)]
    request.save_spec = SaveSpec(
        skip_leading_tokens=8,
        can_save=True,
        can_save_latent=True,
        can_save_indexer=True,
    )
    request.indexer_slot_mapping = [torch.arange(100, 116, dtype=torch.long)]
    _init_indexer_cache_fields(request)

    connector, _, engine = _make_connector([request])
    connector.kv_role = "kv_both"
    connector.config = SimpleNamespace(dsa_two_groups=True)
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    connector.kv_caches = {
        "model.layers.0.self_attn.attn": torch.zeros(1),
        indexer_layer_name: torch.zeros(1),
    }
    metadata_slots = torch.arange(200, 208, dtype=torch.long)
    attn_metadata = {
        indexer_layer_name: SimpleNamespace(slot_mapping=metadata_slots),
    }

    connector.save_kv_layer(indexer_layer_name, torch.zeros(1), attn_metadata)

    assert engine.store_calls == ["req-1"]
    assert engine.store_kwargs[0]["kv_group"] == 1
    assert engine.store_kwargs[0]["offset"] == 8
    assert torch.equal(
        engine.store_kwargs[0]["slot_mapping"],
        torch.cat((torch.zeros(8, dtype=torch.long), metadata_slots)),
    )
    assert torch.equal(
        engine.store_kwargs[0]["mask"],
        torch.tensor(
            [False] * 8 + [True] * 8,
            dtype=torch.bool,
        ),
    )
