# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LMCache DSA offload state machine and typed-permit
arbitration (design "GLM5.1 DSA按上下文长度动态切换详细设计" sections 10-12).

These tests pin the core scheduler-side invariants:

* the per-request route state machine is one-way
  (RESIDENT -> PROMOTING -> OFFLOADED);
* promotion only begins when the threshold is crossed AND a releasable frontier
  beyond the scratch capacity exists (``F > S``);
* raw ``DSACommitEvidence`` becomes a release only after generation/frontier/
  rank arbitration, and stale-generation or invalidated requests never release;
* threshold ``0`` preserves the legacy "every decode request is eligible"
  behaviour.
"""

# Standard
# (no standard imports needed)

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    DSA_OFFLOAD_STATE_OFFLOADED,
    DSA_OFFLOAD_STATE_PROMOTING,
    DSA_OFFLOAD_STATE_RESIDENT,
    DSAOffloadRouteMeta,
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    ReqMeta,
    RequestTracker,
)
from lmcache.v1.config import LMCacheEngineConfig

# Local
from vllm.v1.outputs import DSACommitEvidence, DSAReleasePermit


def _make_impl(scratch_capacity=4096, threshold=8192, chunk=256, auth_rank=0):
    """Build a minimal LMCacheConnectorV1Impl harness for unit testing."""
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._dsa_scratch_capacity = scratch_capacity
    impl._dsa_offload_threshold = threshold
    impl._lmcache_chunk_size = chunk
    impl._decode_window_save_window_size = 0
    impl._dsa_promotion_timeout_seconds = 30.0
    impl._dsa_promotion_max_retries = 3
    impl._dsa_authoritative_store_rank = auth_rank
    impl.enable_sparse_attention = True
    impl.kv_role = "kv_both"
    impl._request_trackers = {}
    return impl


def _make_tracker(req_id="r", prompt_len=100, tokens=100, blocks=64):
    return RequestTracker(
        req_id=req_id,
        prompt_len=prompt_len,
        token_ids=list(range(tokens)),
        allocated_block_ids=list(range(blocks)),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_dsa_offload_threshold_default_and_env():
    assert LMCacheEngineConfig.from_defaults().dsa_offload_token_threshold == 0
    cfg = LMCacheEngineConfig.from_defaults(dsa_offload_token_threshold=8192)
    assert cfg.dsa_offload_token_threshold == 8192


def test_dsa_offload_threshold_rejects_negative():
    with pytest.raises(ValueError, match="dsa_offload_token_threshold"):
        LMCacheEngineConfig.from_defaults(
            dsa_offload_token_threshold=-1
        ).validate()


# ---------------------------------------------------------------------------
# Route state machine
# ---------------------------------------------------------------------------


def test_threshold_zero_legacy_marks_offloaded_in_decode():
    impl = _make_impl(threshold=0)
    tracker = _make_tracker()
    impl._request_trackers[tracker.req_id] = tracker
    state, is_sparse, committed = impl._step_dsa_offload_route(tracker, True)
    assert state == DSA_OFFLOAD_STATE_OFFLOADED
    assert is_sparse is True
    assert tracker.dsa_offload_state == DSA_OFFLOAD_STATE_OFFLOADED


def test_threshold_zero_keeps_resident_in_prefill():
    impl = _make_impl(threshold=0)
    tracker = _make_tracker()
    impl._request_trackers[tracker.req_id] = tracker
    state, is_sparse, _ = impl._step_dsa_offload_route(tracker, False)
    assert state == DSA_OFFLOAD_STATE_RESIDENT
    assert is_sparse is False


def test_below_threshold_stays_resident():
    impl = _make_impl(threshold=8192, scratch_capacity=256)
    tracker = _make_tracker(tokens=1000)
    impl._request_trackers[tracker.req_id] = tracker
    state, is_sparse, _ = impl._step_dsa_offload_route(tracker, True)
    assert state == DSA_OFFLOAD_STATE_RESIDENT
    assert is_sparse is False


def test_crossed_threshold_but_frontier_within_scratch_stays_resident():
    # finalized >= threshold but the only aligned frontier (<= tokens) is not
    # beyond scratch capacity -> must remain resident (design 10.4).
    impl = _make_impl(threshold=512, scratch_capacity=4096, chunk=256)
    tracker = _make_tracker(tokens=600)  # frontier floor=512 <= scratch
    impl._request_trackers[tracker.req_id] = tracker
    state, is_sparse, _ = impl._step_dsa_offload_route(tracker, True)
    assert state == DSA_OFFLOAD_STATE_RESIDENT
    assert is_sparse is False


def test_crossed_threshold_with_frontier_beyond_scratch_enters_promoting():
    impl = _make_impl(threshold=4096, scratch_capacity=256, chunk=256)
    tracker = _make_tracker(tokens=4608)  # floor(4608/256)*256 = 4608 > 256
    impl._request_trackers[tracker.req_id] = tracker
    state, is_sparse, _ = impl._step_dsa_offload_route(tracker, True)
    assert state == DSA_OFFLOAD_STATE_PROMOTING
    assert is_sparse is False  # promoting keeps a resident forward
    assert tracker.dsa_route_generation == 1
    assert tracker.dsa_promotion_inflight_end == 4608


def test_offloaded_is_monotonic():
    impl = _make_impl(threshold=8192, scratch_capacity=256)
    tracker = _make_tracker(tokens=10000)
    tracker.dsa_offload_state = DSA_OFFLOAD_STATE_OFFLOADED
    tracker.decode_window_save_committed_end = 8192
    impl._request_trackers[tracker.req_id] = tracker
    state, is_sparse, committed = impl._step_dsa_offload_route(tracker, True)
    assert state == DSA_OFFLOAD_STATE_OFFLOADED
    assert is_sparse is True
    assert committed == 8192


# ---------------------------------------------------------------------------
# Promotion candidate + save meta
# ---------------------------------------------------------------------------


def test_promotion_candidate_none_when_frontier_le_scratch():
    impl = _make_impl(threshold=2048, scratch_capacity=4096, chunk=256)
    tracker = _make_tracker(tokens=4096)
    assert impl._dsa_promotion_candidate(tracker, 4096) is None


def test_promotion_candidate_aligned_to_chunk():
    impl = _make_impl(threshold=2048, scratch_capacity=4096, chunk=256)
    tracker = _make_tracker(tokens=4609)  # floor -> 4608
    assert impl._dsa_promotion_candidate(tracker, 4609) == 4608


def test_promotion_save_meta_tagged_correctly():
    tracker = _make_tracker(tokens=8192, blocks=512)
    meta = ReqMeta.from_dsa_promotion_save(
        tracker, block_size=16, promotion_frontier=4096
    )
    assert meta is not None
    assert meta.is_dsa_promotion_save is True
    assert meta.promotion_frontier == 4096
    assert meta.is_decode_window_save is False
    assert meta.is_sparse_decode is False


# ---------------------------------------------------------------------------
# Arbitration: raw evidence -> validated permits
# ---------------------------------------------------------------------------


def _seed_promoting(impl, tracker, generation=1):
    tracker.dsa_offload_state = DSA_OFFLOAD_STATE_PROMOTING
    tracker.dsa_route_generation = generation
    tracker.dsa_promotion_inflight_end = 4608


def test_arbitration_ignores_stale_generation():
    impl = _make_impl(threshold=4096, scratch_capacity=256, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=2)
    ev = DSACommitEvidence(tracker.req_id, "promotion_store", 4608, 1, 0)
    permits = impl.arbitrate_dsa_release([ev], set())
    assert permits == {}
    assert tracker.dsa_offload_state == DSA_OFFLOAD_STATE_PROMOTING


def test_arbitration_drops_invalidated_request():
    impl = _make_impl(threshold=4096, scratch_capacity=256, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    ev = DSACommitEvidence(tracker.req_id, "promotion_store", 4608, 1, 0)
    permits = impl.arbitrate_dsa_release([ev], {tracker.req_id})
    assert permits == {}
    assert tracker.dsa_offload_state == DSA_OFFLOAD_STATE_PROMOTING


def test_arbitration_no_permit_when_frontier_le_scratch():
    impl = _make_impl(scratch_capacity=8192, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    ev = DSACommitEvidence(tracker.req_id, "promotion_store", 4096, 1, 0)
    permits = impl.arbitrate_dsa_release([ev], set())
    assert permits == {}


def test_arbitration_promotion_store_yields_permit_and_offloaded():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    ev = DSACommitEvidence(tracker.req_id, "promotion_store", 4608, 1, 0)
    permits = impl.arbitrate_dsa_release([ev], set())
    assert tracker.req_id in permits
    permit = permits[tracker.req_id]
    assert isinstance(permit, DSAReleasePermit)
    assert permit.frontier == 4608
    assert permit.generation == 1
    assert tracker.dsa_offload_state == DSA_OFFLOAD_STATE_OFFLOADED
    assert tracker.decode_window_save_committed_end == 4608
    assert tracker.dsa_promotion_inflight_end is None


def test_arbitration_store_requires_authoritative_rank():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256, auth_rank=0)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    # Reported only by rank 1 while authoritative rank is 0 -> rejected.
    ev = DSACommitEvidence(tracker.req_id, "promotion_store", 4608, 1, 1)
    assert impl.arbitrate_dsa_release([ev], set()) == {}
    # Once rank 0 also reports, the permit is issued.
    ev0 = DSACommitEvidence(tracker.req_id, "promotion_store", 4608, 1, 0)
    permits = impl.arbitrate_dsa_release([ev, ev0], set())
    assert tracker.req_id in permits


def test_arbitration_decode_window_store_permit_for_offloaded():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    tracker.dsa_offload_state = DSA_OFFLOAD_STATE_OFFLOADED
    tracker.dsa_route_generation = 5
    tracker.decode_window_save_committed_end = 4608
    impl._request_trackers[tracker.req_id] = tracker
    ev = DSACommitEvidence(tracker.req_id, "decode_window_store", 4864, 5, 0)
    permits = impl.arbitrate_dsa_release([ev], set())
    assert permits[tracker.req_id].frontier == 4864
    assert tracker.decode_window_save_committed_end == 4864


def test_arbitration_empty_evidence_no_permit():
    impl = _make_impl()
    assert impl.arbitrate_dsa_release([], set()) == {}


def test_arbitration_failed_evidence_ignored():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    ev = DSACommitEvidence(
        tracker.req_id, "promotion_store", 4608, 1, 0, status="failed"
    )
    assert impl.arbitrate_dsa_release([ev], set()) == {}
    assert tracker.dsa_offload_state == DSA_OFFLOAD_STATE_PROMOTING


# ---------------------------------------------------------------------------
# Step 3: backend-fence deferral, failure and retry paths
# ---------------------------------------------------------------------------


def test_failed_promotion_evidence_triggers_retry():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    # Save already dispatched for generation 1.
    _seed_promoting(impl, tracker, generation=1)
    tracker.dsa_promotion_emitted = True
    tracker.dsa_promotion_inflight_end = 4608
    ev = DSACommitEvidence(
        tracker.req_id, "promotion_store", 4608, 1, 0, status="failed"
    )
    permits = impl.arbitrate_dsa_release([ev], set())
    assert permits == {}
    # Retry bumps generation and re-arms dispatch.
    assert tracker.dsa_route_generation == 2
    assert tracker.dsa_promotion_retry_count == 1
    assert tracker.dsa_promotion_emitted is False
    assert tracker.dsa_promotion_inflight_end == 4608
    assert tracker.dsa_offload_state == DSA_OFFLOAD_STATE_PROMOTING


def test_failed_promotion_ignores_stale_generation():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=5)
    tracker.dsa_promotion_emitted = True
    # Failure reported for an old generation -> no retry side effect.
    ev = DSACommitEvidence(
        tracker.req_id, "promotion_store", 4608, 1, 0, status="failed"
    )
    impl.arbitrate_dsa_release([ev], set())
    assert tracker.dsa_route_generation == 5
    assert tracker.dsa_promotion_retry_count == 0


def test_promotion_exhausts_retries_and_keeps_resident():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    tracker.dsa_promotion_retry_count = impl._dsa_promotion_max_retries
    tracker.dsa_promotion_inflight_end = 4608
    ev = DSACommitEvidence(
        tracker.req_id, "promotion_store", 4608, 1, 0, status="failed"
    )
    impl.arbitrate_dsa_release([ev], set())
    # Exhausted: inflight cleared, no release, request stays safe/resident.
    assert tracker.dsa_promotion_inflight_end is None
    assert impl.arbitrate_dsa_release(
        [
            DSACommitEvidence(
                tracker.req_id, "promotion_store", 4608, 1, 0
            )
        ],
        set(),
    ) == {}


def test_promotion_deadline_retries():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    tracker.dsa_promotion_emitted = True
    tracker.dsa_promotion_inflight_end = 4608
    tracker.dsa_promotion_deadline = 0.0  # already expired
    state, _, _ = impl._step_dsa_offload_route(tracker, True)
    assert state == DSA_OFFLOAD_STATE_PROMOTING
    assert tracker.dsa_route_generation == 2
    assert tracker.dsa_promotion_emitted is False


def test_promotion_success_clears_inflight_and_emitted():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    tracker = _make_tracker(tokens=5000)
    impl._request_trackers[tracker.req_id] = tracker
    _seed_promoting(impl, tracker, generation=1)
    tracker.dsa_promotion_emitted = True
    ev = DSACommitEvidence(tracker.req_id, "promotion_store", 4608, 1, 0)
    permits = impl.arbitrate_dsa_release([ev], set())
    assert tracker.req_id in permits
    assert tracker.dsa_offload_state == DSA_OFFLOAD_STATE_OFFLOADED
    assert tracker.dsa_promotion_inflight_end is None


def test_fail_staged_promotion_emits_failure_evidence():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    # Worker-side evidence list must exist for staging failure.
    impl._dsa_commit_evidence = []
    impl._parent = type(
        "P", (), {"_get_connector_metadata": lambda self: None}
    )()
    req = type(
        "R",
        (),
        {
            "req_id": "r",
            "is_dsa_promotion_save": True,
            "promotion_frontier": 4608,
        },
    )()
    save_context = {"dsa_promotion_saves": [req]}
    impl._fail_staged_dsa_promotion_saves(save_context)
    evidence = impl._dsa_commit_evidence
    assert len(evidence) == 1
    assert evidence[0].status == "failed"
    assert evidence[0].kind == "promotion_store"
    assert evidence[0].frontier == 4608
    # Staged candidates are cleared from the save context.
    assert "dsa_promotion_saves" not in save_context


def test_mark_promotion_save_completed_emits_success_evidence():
    impl = _make_impl(scratch_capacity=256, threshold=4096, chunk=256)
    impl._dsa_commit_evidence = []
    impl._parent = type(
        "P", (), {"_get_connector_metadata": lambda self: None}
    )()
    req = type(
        "R",
        (),
        {
            "req_id": "r",
            "is_dsa_promotion_save": True,
            "promotion_frontier": 4608,
            "token_ids": list(range(5000)),
        },
    )()
    impl._mark_dsa_promotion_save_completed(req)
    evidence = impl._dsa_commit_evidence
    assert len(evidence) == 1
    assert evidence[0].status == "succeeded"
    assert evidence[0].kind == "promotion_store"
    assert evidence[0].frontier == 4608


# ---------------------------------------------------------------------------
# Connector metadata route table
# ---------------------------------------------------------------------------


def test_connector_metadata_carries_route_table():
    meta = LMCacheConnectorMetadata()
    meta.dsa_offload_routes["r"] = DSAOffloadRouteMeta(
        state=DSA_OFFLOAD_STATE_OFFLOADED,
        committed_end=8192,
        generation=1,
        window_anchor=256,
    )
    assert meta.dsa_offload_routes["r"].state == DSA_OFFLOAD_STATE_OFFLOADED
