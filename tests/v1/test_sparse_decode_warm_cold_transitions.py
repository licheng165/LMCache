# SPDX-License-Identifier: Apache-2.0
"""P0: sparse decode warm/cold path transitions at connector and cache-engine level."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LoadSpec,
    ReqMeta,
    WorkerRetrieveState,
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


class TestConnectorWarmColdInvalidate:
    def test_cold_bind_after_save_builds_warm_state(self) -> None:
        impl = make_worker_impl()
        request = _make_sparse_request()
        request.cached_keys = [["layer-key"]]
        request.cached_starts = [0]
        request.cached_ends = [256]

        impl._save_worker_retrieve_state_from_request(
            request,
            location="LocalCPUBackend",
            metadata_warm=True,
            token_count=256,
        )

        fresh = _make_sparse_request()
        bound = impl._bind_worker_retrieve_state_to_request(fresh)
        warm = impl._sparse_decode_retrieve_warm_kwargs(fresh, 256, bound)

        assert warm["_retrieve_metadata_warm"] is True
        assert warm["cached_retrieve_location"] == "LocalCPUBackend"

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

        rebound = impl._bind_worker_retrieve_state_to_request(_make_sparse_request())
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
        warm = impl._sparse_decode_retrieve_warm_kwargs(
            _make_sparse_request(), 512, state
        )
        assert "_retrieve_metadata_warm" not in warm
        assert warm["cached_retrieve_location"] == "LocalCPUBackend"


class TestAscendEngineWarmColdMetadata:
    def test_has_retrieve_data_cache_cold_vs_warm(self) -> None:
        assert not AscendLMCacheEngine._has_retrieve_data_cache(None, None, 2)
        # Empty per-layer lists still mean the tensor cache structure exists.
        assert AscendLMCacheEngine._has_retrieve_data_cache([[], []], None, 2)

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
        engine.storage_manager.contains.return_value = "LocalCPUBackend"
        engine.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
        engine.num_layers = 2

        cached_keys: list[list] = [[MagicMock()], [MagicMock()]]
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

        engine.storage_manager.contains.assert_called_once()
        assert location == "LocalCPUBackend"
        assert retrieve_kwargs["cached_retrieve_location"] == "LocalCPUBackend"
