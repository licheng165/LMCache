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
        bad_retrieve_kv[0][0][slot_mapping[10].item()] += 1.0

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
            phase_right="decode_step1_scatter",
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
        scattered_kv[1][1][slot_mapping[50].item()] += 0.5

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
            phase_right="decode_step1_scatter",
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
            phase="decode_step1_scatter",
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
        run2_kv[0][0][slot_mapping[20].item()] += 2.0
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
            phase_right="decode_step1_scatter",
            experiment="B_compute_vs_decode_scatter",
            req_id="run1",
        )
        assert not compare_runs_same_phase(
            prompt_fp=fp,
            run_left=1,
            run_right=2,
            phase="decode_step1_scatter",
            experiment="C_run1_vs_runN_decode_scatter",
            req_id="run2",
        )

    def test_decode_step_gt_zero_skips_experiment_c(self) -> None:
        prompt_len = 64
        num_layers = 1
        num_slots = 128
        token_ids = list(range(prompt_len))
        slot_mapping = torch.arange(prompt_len, dtype=torch.long)
        kv = _make_mla_kvcaches(num_layers, num_slots)

        record_phase_samples(
            req_id="run1",
            token_ids=token_ids,
            phase="decode_step1_scatter",
            kvcaches=kv,
            slot_mapping=slot_mapping,
            num_layers=num_layers,
            prompt_len=prompt_len,
        )

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

        from lmcache.v1 import kv_checksum_diag as mod
        from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

        fp = prompt_fingerprint(token_ids)
        assert 2 not in mod._KV_SAMPLES.get(fp, {})


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
