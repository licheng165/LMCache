# SPDX-License-Identifier: Apache-2.0
"""Tests for worker-local sparse decode retrieve state cache."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

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

    def test_sparse_decode_window_does_not_invalidate_full_prompt_cache(self):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[18879],
            metadata_warm=True,
            token_count=18879,
        )
        req = _make_request()
        req.token_ids = [0] * 18879
        assert not impl._should_invalidate_worker_retrieve_state(req, 2048)

    def test_sparse_decode_prompt_shrink_invalidates(self):
        impl = _make_impl()
        impl._worker_retrieve_state["req-1"] = WorkerRetrieveState(
            cached_keys=[["k"]],
            cached_starts=[0],
            cached_ends=[18879],
            metadata_warm=True,
            token_count=18879,
        )
        req = _make_request()
        req.token_ids = [0] * 4096
        assert impl._should_invalidate_worker_retrieve_state(req, 2048)

    def test_prune_keeps_metadata_warm_states_until_request_finished(self):
        impl = _make_impl()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(
                metadata_warm=True, cached_keys=[["k"]]
            ),
            "req-2": WorkerRetrieveState(
                metadata_warm=True, cached_keys=[["k2"]]
            ),
        }
        impl._prune_worker_retrieve_state({"req-1"})
        assert set(impl._worker_retrieve_state) == {"req-1", "req-2"}

    def test_prune_drops_non_warm_finished_requests(self):
        impl = _make_impl()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "req-2": WorkerRetrieveState(),
        }
        impl._prune_worker_retrieve_state({"req-1"})
        assert set(impl._worker_retrieve_state) == {"req-1"}

    def test_drop_and_prune_release_lookup_pins(self):
        engine = MagicMock()
        impl = _make_impl()
        impl._manager = SimpleNamespace(lmcache_engine=engine)

        impl._drop_worker_retrieve_state("req-1")
        engine.lookup_unpin.assert_called_once_with("req-1")

        engine.lookup_unpin.reset_mock()
        impl._worker_retrieve_state = {
            "req-1": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "req-2": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k2"]]),
        }
        impl._prune_worker_retrieve_state({"req-1"})
        engine.lookup_unpin.assert_not_called()

    def test_defer_lookup_unpin_for_active_sparse_decode(self):
        impl = _make_impl()
        request = _make_request()
        assert impl._should_defer_lookup_unpin_for_sparse_decode(request)

        finished = _make_request()
        finished.load_spec.can_load = False
        assert not impl._should_defer_lookup_unpin_for_sparse_decode(finished)

    def test_maybe_lookup_unpin_skips_active_sparse_decode(self):
        engine = MagicMock()
        impl = _make_impl()
        impl._manager = SimpleNamespace(lmcache_engine=engine)

        impl._maybe_lookup_unpin_for_request(_make_request())
        engine.lookup_unpin.assert_not_called()

        non_sparse = _make_request()
        non_sparse.is_sparse_decode = False
        impl._maybe_lookup_unpin_for_request(non_sparse)
        engine.lookup_unpin.assert_called_once_with("req-1")
