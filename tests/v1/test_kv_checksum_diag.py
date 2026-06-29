# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KV checksum diagnostics (experiments B and C)."""

from __future__ import annotations

# Standard
from typing import List, Tuple

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_checksum_diag import (
    KV_FORMAT_MLA,
    collect_kv_checksum_samples,
    compare_phases,
    compare_runs_same_phase,
    on_compute_before_decode_scatter,
    on_decode_scatter_complete,
    on_prefill_retrieve_complete,
    record_phase_samples,
    reset_kv_checksum_diag_state,
    sample_layer_ids,
    sample_token_indices,
    tensor_fingerprint,
)


def _make_mla_kvcaches(
    num_layers: int,
    num_slots: int,
    hidden_k: int = 64,
    hidden_v: int = 32,
    *,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer_id in range(num_layers):
        k = torch.randn(num_slots, hidden_k, generator=gen) + layer_id
        v = torch.randn(num_slots, hidden_v, generator=gen) + layer_id * 0.1
        caches.append((k, v))
    return caches


def _clone_kvcaches(
    kvcaches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(k.clone(), v.clone()) for k, v in kvcaches]


def _corrupt_sampled_k_slot(
    kvcaches: List[Tuple[torch.Tensor, torch.Tensor]],
    slot_mapping: torch.Tensor,
    prompt_len: int,
    *,
    layer_id: int = 0,
    delta: float = 1.0,
) -> int:
    """Corrupt K at a token index that sample_token_indices() always checks."""
    token_idx = sample_token_indices(prompt_len)[0]
    slot = int(slot_mapping[token_idx].item())
    kvcaches[layer_id][0][slot] += delta
    return token_idx


@pytest.fixture(autouse=True)
def _enable_kv_checksum_diag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMCACHE_DIAG_KV_CHECKSUM", "1")
    reset_kv_checksum_diag_state()
    yield
    reset_kv_checksum_diag_state()


class TestKvChecksumSampling:
    def test_sample_token_indices_tail_window(self) -> None:
        indices = sample_token_indices(18879, sparse_window=2048, num_samples=8)
        assert 0 in indices
        assert 18878 in indices
        assert min(indices) >= 0
        assert max(indices) < 18879
        assert indices == sorted(indices)

    def test_sample_layer_ids(self) -> None:
        assert sample_layer_ids(1) == [0]
        assert sample_layer_ids(4) == [0, 2, 3]

    def test_tensor_fingerprint_stable(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0])
        assert tensor_fingerprint(t) == tensor_fingerprint(t.clone())

    def test_collect_kv_checksum_samples(self) -> None:
        kvcaches = _make_mla_kvcaches(num_layers=2, num_slots=128)
        slot_mapping = torch.arange(64, dtype=torch.long)
        samples = collect_kv_checksum_samples(
            kvcaches,
            slot_mapping,
            token_indices=[0, 63],
            layer_ids=[0, 1],
            kv_format=KV_FORMAT_MLA,
        )
        assert len(samples) == 4
        assert all(s.k_fp and s.v_fp for s in samples)


class TestExperimentBStoreRetrieveRoundtrip:
    """Run1 prefill compute KV vs prefill retrieve scatter checksum."""

    def test_prefill_retrieve_matches_compute_baseline(self) -> None:
        prompt_len = 512
        num_layers = 4
        num_slots = 1024
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        compute_kv = _make_mla_kvcaches(num_layers, num_slots, seed=42)

        on_compute_before_decode_scatter(
            req_id="run1-req",
            token_ids=token_ids,
            kvcaches=compute_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )

        retrieve_kv = _clone_kvcaches(compute_kv)
        on_prefill_retrieve_complete(
            req_id="run1-req-retrieve",
            token_ids=token_ids,
            kvcaches=retrieve_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )

        from lmcache.v1 import kv_checksum_diag as mod
        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        baseline = mod._COMPUTE_BASELINE_BY_FP[fp]
        scatter = mod._KV_SAMPLES[fp][2]["prefill_retrieve_scatter"]
        compared, n_mismatch, _ = mod._count_mismatches(baseline, scatter)
        assert n_mismatch == 0 and compared > 0

    def test_prefill_retrieve_detects_scatter_mismatch(self) -> None:
        prompt_len = 256
        num_layers = 2
        num_slots = 512
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        compute_kv = _make_mla_kvcaches(num_layers, num_slots, seed=7)

        on_compute_before_decode_scatter(
            req_id="run1",
            token_ids=token_ids,
            kvcaches=compute_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )

        bad_retrieve_kv = _clone_kvcaches(compute_kv)
        _corrupt_sampled_k_slot(bad_retrieve_kv, slot_mapping, prompt_len)

        on_prefill_retrieve_complete(
            req_id="run1-bad-retrieve",
            token_ids=token_ids,
            kvcaches=bad_retrieve_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )

        from lmcache.v1 import kv_checksum_diag as mod
        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        baseline = mod._COMPUTE_BASELINE_BY_FP[fp]
        scatter = mod._KV_SAMPLES[fp][2]["prefill_retrieve_scatter"]
        compared, n_mismatch, _ = mod._count_mismatches(baseline, scatter)
        assert n_mismatch > 0 and compared > 0


class TestExperimentBDecodeScatterVsCompute:
    """Run1: compute KV before decode scatter vs decode step 1 scatter."""

    def test_decode_scatter_matches_compute_when_identical(self) -> None:
        prompt_len = 512
        num_layers = 4
        num_slots = 1024
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        kv = _make_mla_kvcaches(num_layers, num_slots, seed=99)

        on_compute_before_decode_scatter(
            req_id="run1-decode",
            token_ids=token_ids,
            kvcaches=kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )
        on_decode_scatter_complete(
            req_id="run1-decode",
            token_ids=token_ids,
            kvcaches=_clone_kvcaches(kv),
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=0,
        )

        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        assert compare_phases(
            prompt_fp=fp,
            run_number=1,
            phase_left="compute_before_decode_scatter",
            phase_right="decode_step0_scatter",
            experiment="B_compute_vs_decode_scatter",
            req_id="run1-decode",
        )

    def test_decode_scatter_mismatch_flags_batched_from_gpu(self) -> None:
        prompt_len = 256
        num_layers = 2
        num_slots = 512
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        compute_kv = _make_mla_kvcaches(num_layers, num_slots, seed=11)

        on_compute_before_decode_scatter(
            req_id="run1",
            token_ids=token_ids,
            kvcaches=compute_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )

        scattered_kv = _clone_kvcaches(compute_kv)
        _corrupt_sampled_k_slot(scattered_kv, slot_mapping, prompt_len, layer_id=1)

        on_decode_scatter_complete(
            req_id="run1",
            token_ids=token_ids,
            kvcaches=scattered_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=0,
        )

        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        assert not compare_phases(
            prompt_fp=fp,
            run_number=1,
            phase_left="compute_before_decode_scatter",
            phase_right="decode_step0_scatter",
            experiment="B_compute_vs_decode_scatter",
            req_id="run1",
        )


class TestExperimentCRun1VsRun2DecodeScatter:
    """Run2 without skip: decode step 1 scatter KV vs Run1."""

    def test_run2_decode_scatter_matches_run1(self) -> None:
        prompt_len = 512
        num_layers = 4
        num_slots = 1024
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        kv = _make_mla_kvcaches(num_layers, num_slots, seed=55)

        on_compute_before_decode_scatter(
            req_id="run1-req",
            token_ids=token_ids,
            kvcaches=kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )
        on_decode_scatter_complete(
            req_id="run1-req",
            token_ids=token_ids,
            kvcaches=_clone_kvcaches(kv),
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=0,
        )

        on_decode_scatter_complete(
            req_id="run2-req",
            token_ids=token_ids,
            kvcaches=_clone_kvcaches(kv),
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=0,
        )

        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        assert compare_runs_same_phase(
            prompt_fp=fp,
            run_left=1,
            run_right=2,
            phase="decode_step0_scatter",
            experiment="C_run1_vs_runN_decode_scatter",
            req_id="run2-req",
        )

    def test_run2_mismatch_run1_ok_implies_compute_fallback(self) -> None:
        """Run2 scatter diverges while Run1 compute baseline stays valid."""
        prompt_len = 256
        num_layers = 2
        num_slots = 512
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        run1_kv = _make_mla_kvcaches(num_layers, num_slots, seed=33)

        on_compute_before_decode_scatter(
            req_id="run1",
            token_ids=token_ids,
            kvcaches=run1_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
        )
        on_decode_scatter_complete(
            req_id="run1",
            token_ids=token_ids,
            kvcaches=_clone_kvcaches(run1_kv),
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=0,
        )

        run2_kv = _clone_kvcaches(run1_kv)
        _corrupt_sampled_k_slot(run2_kv, slot_mapping, prompt_len)
        on_decode_scatter_complete(
            req_id="run2",
            token_ids=token_ids,
            kvcaches=run2_kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=0,
        )

        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        assert compare_phases(
            prompt_fp=fp,
            run_number=1,
            phase_left="compute_before_decode_scatter",
            phase_right="decode_step0_scatter",
            experiment="B_compute_vs_decode_scatter",
            req_id="run1",
        )
        assert not compare_runs_same_phase(
            prompt_fp=fp,
            run_left=1,
            run_right=2,
            phase="decode_step0_scatter",
            experiment="C_run1_vs_runN_decode_scatter",
            req_id="run2",
        )

    def test_decode_step_gt_zero_records_per_step(self) -> None:
        """decode_step>0 now records a per-step phase (decode_step{k}_scatter)
        and compares Run1 vs RunN at the same step (no longer skipped)."""
        prompt_len = 64
        num_layers = 1
        num_slots = 128
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        kv = _make_mla_kvcaches(num_layers, num_slots)

        # Run1 step 1
        on_decode_scatter_complete(
            req_id="run1",
            token_ids=token_ids,
            kvcaches=_clone_kvcaches(kv),
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=1,
        )
        # Run2 step 1 (identical KV -> should match Run1 step 1)
        on_decode_scatter_complete(
            req_id="run2",
            token_ids=token_ids,
            kvcaches=_clone_kvcaches(kv),
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
            worker_id=0,
            decode_step=1,
        )

        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        # Per-step phase recorded for both runs
        from lmcache.v1 import kv_checksum_diag as mod

        run1_samples = mod._KV_SAMPLES.get(fp, {}).get(1, {}).get(
            "decode_step1_scatter"
        )
        run2_samples = mod._KV_SAMPLES.get(fp, {}).get(2, {}).get(
            "decode_step1_scatter"
        )
        assert run1_samples is not None
        assert run2_samples is not None
        # Same KV -> Run1 vs Run2 at step 1 should match
        assert compare_runs_same_phase(
            prompt_fp=fp,
            run_left=1,
            run_right=2,
            phase="decode_step1_scatter",
            experiment="C_run1_vs_runN_step1",
            req_id="run2",
        )


class TestKvChecksumDisabled:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LMCACHE_DIAG_KV_CHECKSUM", raising=False)
        reset_kv_checksum_diag_state()
        kv = _make_mla_kvcaches(1, 64)
        slot_mapping = torch.arange(32, dtype=torch.long)
        on_compute_before_decode_scatter(
            req_id="r",
            token_ids=list(range(32)),
            kvcaches=kv,
            slot_mapping=slot_mapping,
            num_layers=1,
            prompt_len=32,
            worker_id=0,
        )
        from lmcache.v1 import kv_checksum_diag as mod

        assert not mod._KV_SAMPLES


class TestLayerwiseStoreGeneratorStructure:
    """Verify the layerwise store generator yield pattern.

    store_layer yields num_layers+1 times (num_layers in the per-layer loop +
    one final yield). save_kv_layer calls next() num_layers times, so the
    "Stored X" log and final per-layer put complete INSIDE save_kv_layer. The
    generator is then paused at its final yield; wait_for_save's next() raises
    StopIteration. Therefore any post-store diagnostic hook MUST run BEFORE
    that next() (which is why the prefill_compute baseline hook is placed
    before the next() in wait_for_save).
    """

    def test_store_generator_yields_num_layers_plus_one(self) -> None:
        num_layers = 4

        def mock_store_layer():
            for _ in range(num_layers):
                yield  # per-layer yield (store_layer line 985)
            # "Stored X" log fires here (store_layer line 994)
            yield  # final yield (store_layer line 1012)

        storer = mock_store_layer()

        # save_kv_layer calls next() num_layers times -- none raise.
        for i in range(num_layers):
            next(storer)  # consumes per-layer yields

        # The store has completed (log fired during the num_layers-th next()).
        # The generator is now paused at the final yield.
        hook_ran = True  # a hook here (before the final next()) WOULD run
        assert hook_ran

        # wait_for_save's next() consumes the final yield and raises
        # StopIteration -- any code AFTER this next() would NOT run.
        with pytest.raises(StopIteration):
            next(storer)

    def test_hook_after_next_does_not_run_on_stopiteration(self) -> None:
        """Confirm a hook placed AFTER the final next() is unreachable."""
        num_layers = 2

        def gen():
            for _ in range(num_layers):
                yield
            yield

        storer = gen()
        for _ in range(num_layers):
            next(storer)

        hook_after_ran = False
        try:
            next(storer)
            hook_after_ran = True  # unreachable
        except StopIteration:
            pass
        assert not hook_after_ran

    def test_hook_before_next_runs(self) -> None:
        """Confirm a hook placed BEFORE the final next() runs (the fix)."""
        num_layers = 2

        def gen():
            for _ in range(num_layers):
                yield
            yield

        storer = gen()
        for _ in range(num_layers):
            next(storer)

        hook_before_ran = False
        # Hook BEFORE the final next() -- this is where the prefill_compute
        # baseline is recorded.
        hook_before_ran = True
        try:
            next(storer)
        except StopIteration:
            pass
        assert hook_before_ran
