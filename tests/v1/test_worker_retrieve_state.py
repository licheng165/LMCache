# SPDX-License-Identifier: Apache-2.0
"""Tests for worker-local sparse decode retrieve state cache."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    WorkerRetrieveState,
)


def _make_impl() -> LMCacheConnectorV1Impl:
    impl = object.__new__(LMCacheConnectorV1Impl)
    impl._worker_retrieve_state = {}
    return impl


def _make_request(*, resumed: bool = False) -> ReqMeta:
    return ReqMeta(
        req_id="req-1",
        token_ids=[1, 2, 3],
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=3,
            can_load=True,
        ),
        is_sparse_decode=True,
        resumed_from_preemption=resumed,
    )


class TestWorkerRetrieveState:
    def test_bind_rehydrates_scheduler_empty_metadata(self):
        impl = _make_impl()
        request = _make_request()
        assert request.cached_keys == []

        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["layer0-key"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            location="local",
            token_count=256,
        )

        bound = impl._bind_worker_retrieve_state_to_request(request)
        assert bound is not None
        assert request.cached_keys == [["layer0-key"]]
        assert request.cached_starts == [0]
        assert request.cached_ends == [256]

    def test_save_then_bind_round_trip(self):
        impl = _make_impl()
        request = _make_request()
        request.cached_keys = [["k"]]
        request.cached_starts = [0]
        request.cached_ends = [256]

        impl._save_worker_retrieve_state_from_request(
            request,
            location="local",
            metadata_warm=True,
            token_count=256,
        )

        fresh = _make_request()
        assert fresh.cached_keys == []
        impl._bind_worker_retrieve_state_to_request(fresh)
        assert fresh.cached_keys == [["k"]]

    def test_warm_kwargs_only_when_prefix_unchanged(self):
        impl = _make_impl()
        request = _make_request()
        state = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            location="local",
            token_count=256,
        )

        warm = impl._sparse_decode_retrieve_warm_kwargs(request, 256, state)
        assert warm["_retrieve_metadata_warm"] is True
        assert warm["cached_retrieve_location"] == "local"

        extended = impl._sparse_decode_retrieve_warm_kwargs(request, 512, state)
        assert "_retrieve_metadata_warm" not in extended
        assert extended["cached_retrieve_location"] == "local"

    def test_invalidate_on_preemption_and_token_rollback(self):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[256],
            metadata_warm=True,
            token_count=256,
        )

        assert impl._should_invalidate_worker_retrieve_state(
            _make_request(resumed=True), 256
        )
        assert impl._should_invalidate_worker_retrieve_state(_make_request(), 128)

    def test_prune_drops_finished_requests(self):
        impl = _make_impl()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True),
            "req-2": WorkerRetrieveState(metadata_warm=True),
        }
        impl._prune_worker_retrieve_state({"req-1"})
        assert set(impl._worker_retrieve_state) == {"req-1"}
