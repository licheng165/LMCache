# SPDX-License-Identifier: Apache-2.0
"""Opt-in KV checksum diagnostics for store/retrieve/scatter experiments."""

from __future__ import annotations

# Standard
from dataclasses import dataclass
import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.ext_prefix_hit_diag import prompt_fingerprint

logger = init_logger(__name__)

KV_FORMAT_MLA = 3
KV_FORMAT_DSA = 4

# prompt_fp -> run_number -> phase -> samples
_KV_SAMPLES: Dict[str, Dict[int, Dict[str, List["KvChecksumSample"]]]] = {}
# prompt_fp -> compute baseline from run 1 decode step 0 (before scatter)
_COMPUTE_BASELINE_BY_FP: Dict[str, List["KvChecksumSample"]] = {}
# prompt_fp -> compute baseline from run 1 prefill (full prompt positions)
_PREFILL_COMPUTE_BASELINE_BY_FP: Dict[str, List["KvChecksumSample"]] = {}
_REQ_TO_RUN: Dict[str, Tuple[str, int]] = {}
_FP_RUN_COUNT: Dict[str, int] = {}


def kv_checksum_diag_enabled() -> bool:
    """Enable with LMCACHE_DIAG_KV_CHECKSUM=1."""
    return os.environ.get("LMCACHE_DIAG_KV_CHECKSUM", "").lower() in (
        "1",
        "true",
        "yes",
    )


def log_kv_checksum_diag(fmt: str, *args: Any) -> None:
    if kv_checksum_diag_enabled():
        logger.info("[LMCache-Diag-KVChecksum] " + fmt, *args)


def sample_token_indices(
    prompt_len: int,
    *,
    sparse_window: int = 2048,
    num_samples: int = 8,
) -> List[int]:
    """Sample prompt token indices: head, tail, and evenly across sparse window.

    If LMCACHE_DIAG_KV_FULL_SCAN=1, sample every 64th token across the FULL
    prompt (not just the sparse window) to catch V/K corruption at any position.
    """
    if prompt_len <= 0:
        return []
    if os.environ.get("LMCACHE_DIAG_KV_FULL_SCAN", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        step = max(1, prompt_len // 256)  # ~256 samples across full prompt
        return list(range(0, prompt_len, step))
    indices: set[int] = {0, prompt_len - 1}
    start = max(0, prompt_len - sparse_window)
    if num_samples <= 1:
        return sorted(indices)
    span = max(1, prompt_len - 1 - start)
    for i in range(num_samples):
        idx = start + (i * span) // max(1, num_samples - 1)
        indices.add(min(idx, prompt_len - 1))
    return sorted(indices)


def sample_layer_ids(num_layers: int) -> List[int]:
    if num_layers <= 0:
        return []
    if num_layers == 1:
        return [0]
    mid = num_layers // 2
    if num_layers - 1 == mid:
        return [0, mid]
    return [0, mid, num_layers - 1]


def tensor_fingerprint(t: torch.Tensor) -> str:
    t_cpu = t.detach().contiguous().cpu()
    if t_cpu.dtype == torch.bfloat16:
        t_cpu = t_cpu.float()
    return hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()[:16]


def _layer_kv_tensors(layer_cache: Any) -> Tuple[torch.Tensor, ...]:
    if isinstance(layer_cache, (tuple, list)):
        return tuple(layer_cache)
    return (layer_cache,)


def slot_kv_fingerprints(
    layer_cache: Any,
    slot: int,
    *,
    kv_format: int = KV_FORMAT_MLA,
) -> Dict[str, str]:
    """Fingerprint K/V (and DSA if present) at a flat paged slot index."""
    tensors = _layer_kv_tensors(layer_cache)
    out: Dict[str, str] = {}
    if kv_format in (KV_FORMAT_MLA, KV_FORMAT_DSA):
        k = tensors[0].reshape(-1, tensors[0].shape[-1])
        v = tensors[1].reshape(-1, tensors[1].shape[-1])
        slot_i = int(slot)
        out["k"] = tensor_fingerprint(k[slot_i])
        out["v"] = tensor_fingerprint(v[slot_i])
        if kv_format == KV_FORMAT_DSA and len(tensors) > 2:
            dsa = tensors[2].reshape(-1, tensors[2].shape[-1])
            out["dsa"] = tensor_fingerprint(dsa[slot_i])
        return out
    # MERGED / SEPARATE fallback: flatten last two dims per plane
    k = tensors[0].reshape(-1, *tensors[0].shape[-2:])
    v = tensors[1].reshape(-1, *tensors[1].shape[-2:])
    slot_i = int(slot)
    out["k"] = tensor_fingerprint(k[slot_i])
    out["v"] = tensor_fingerprint(v[slot_i])
    return out


@dataclass(frozen=True)
class KvChecksumSample:
    layer_id: int
    token_idx: int
    slot: int
    k_fp: str
    v_fp: str
    dsa_fp: Optional[str] = None


def collect_kv_checksum_samples(
    kvcaches: Sequence[Any],
    slot_mapping: Union[torch.Tensor, Sequence[int]],
    token_indices: Sequence[int],
    layer_ids: Sequence[int],
    *,
    kv_format: int = KV_FORMAT_MLA,
) -> List[KvChecksumSample]:
    if isinstance(slot_mapping, torch.Tensor):
        slots = slot_mapping.detach().cpu().tolist()
    else:
        slots = list(slot_mapping)

    samples: List[KvChecksumSample] = []
    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= len(kvcaches):
            continue
        layer_cache = kvcaches[layer_id]
        for token_idx in token_indices:
            if token_idx < 0 or token_idx >= len(slots):
                continue
            slot = int(slots[token_idx])
            fps = slot_kv_fingerprints(layer_cache, slot, kv_format=kv_format)
            samples.append(
                KvChecksumSample(
                    layer_id=layer_id,
                    token_idx=int(token_idx),
                    slot=slot,
                    k_fp=fps["k"],
                    v_fp=fps["v"],
                    dsa_fp=fps.get("dsa"),
                )
            )
    return samples


def register_worker_run(
    req_id: str,
    token_ids: Optional[Union[torch.Tensor, List[int]]],
) -> Tuple[str, int]:
    if req_id in _REQ_TO_RUN:
        return _REQ_TO_RUN[req_id]
    fp = prompt_fingerprint(token_ids)
    run_number = _FP_RUN_COUNT.get(fp, 0) + 1
    _FP_RUN_COUNT[fp] = run_number
    _REQ_TO_RUN[req_id] = (fp, run_number)
    return fp, run_number


def _aggregate_fingerprint(samples: Sequence[KvChecksumSample]) -> str:
    parts = [
        f"L{s.layer_id}T{s.token_idx}:k={s.k_fp},v={s.v_fp}"
        + (f",dsa={s.dsa_fp}" if s.dsa_fp else "")
        for s in samples
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:16]


def _store_samples(
    prompt_fp: str,
    run_number: int,
    phase: str,
    samples: Sequence[KvChecksumSample],
) -> None:
    _KV_SAMPLES.setdefault(prompt_fp, {}).setdefault(run_number, {})[phase] = list(
        samples
    )


def _lookup_sample(
    samples: Sequence[KvChecksumSample],
    layer_id: int,
    token_idx: int,
) -> Optional[KvChecksumSample]:
    for sample in samples:
        if sample.layer_id == layer_id and sample.token_idx == token_idx:
            return sample
    return None


def _count_mismatches(
    left: Sequence[KvChecksumSample],
    right: Sequence[KvChecksumSample],
) -> Tuple[int, int, List[str]]:
    mismatches: List[str] = []
    compared = 0
    for sample in left:
        other = _lookup_sample(right, sample.layer_id, sample.token_idx)
        if other is None:
            continue
        compared += 1
        if sample.k_fp != other.k_fp or sample.v_fp != other.v_fp:
            mismatches.append(
                f"L{sample.layer_id}T{sample.token_idx}: "
                f"k {sample.k_fp}!={other.k_fp} v {sample.v_fp}!={other.v_fp}"
            )
        if sample.dsa_fp and other.dsa_fp and sample.dsa_fp != other.dsa_fp:
            mismatches.append(
                f"L{sample.layer_id}T{sample.token_idx}: "
                f"dsa {sample.dsa_fp}!={other.dsa_fp}"
            )
    return compared, len(mismatches), mismatches


def record_phase_samples(
    *,
    req_id: str,
    token_ids: Optional[Union[torch.Tensor, List[int]]],
    phase: str,
    kvcaches: Sequence[Any],
    slot_mapping: Union[torch.Tensor, Sequence[int]],
    num_layers: int,
    prompt_len: int,
    worker_id: int = 0,
    kv_format: int = KV_FORMAT_MLA,
) -> Tuple[str, int, str]:
    """Collect and store checksum samples; returns (prompt_fp, run_number, agg_fp)."""
    prompt_fp, run_number = register_worker_run(req_id, token_ids)
    # Sample positions within the slot_mapping range, NOT prompt_len. For
    # prefill retrieve slot_mapping covers the full prompt (18879); for sparse
    # decode slot_mapping is the packed 2048-token window. Sampling prompt
    # positions against a 2048-length decode slot_mapping would skip almost
    # all tokens (the original n_samples=3 bug).
    if isinstance(slot_mapping, torch.Tensor):
        slot_count = int(slot_mapping.numel())
    else:
        slot_count = len(slot_mapping)
    token_indices = sample_token_indices(slot_count)
    layer_ids = sample_layer_ids(num_layers)
    samples = collect_kv_checksum_samples(
        kvcaches,
        slot_mapping,
        token_indices,
        layer_ids,
        kv_format=kv_format,
    )
    _store_samples(prompt_fp, run_number, phase, samples)
    agg_fp = _aggregate_fingerprint(samples)
    log_kv_checksum_diag(
        "record phase=%s req=%s fp=%s run=%d worker_id=%d layers=%s tokens=%s "
        "agg_fp=%s n_samples=%d",
        phase,
        req_id,
        prompt_fp,
        run_number,
        worker_id,
        list(layer_ids),
        token_indices,
        agg_fp,
        len(samples),
    )
    return prompt_fp, run_number, agg_fp


def compare_phases(
    *,
    prompt_fp: str,
    run_number: int,
    phase_left: str,
    phase_right: str,
    experiment: str,
    req_id: str,
    worker_id: int = 0,
    max_log_mismatches: int = 5,
) -> bool:
    """Return True if all compared slots match."""
    run_samples = _KV_SAMPLES.get(prompt_fp, {}).get(run_number, {})
    left = run_samples.get(phase_left)
    right = run_samples.get(phase_right)
    if not left or not right:
        log_kv_checksum_diag(
            "compare skip experiment=%s req=%s fp=%s run=%d missing phase "
            "left=%s right=%s",
            experiment,
            req_id,
            prompt_fp,
            run_number,
            phase_left,
            phase_right,
        )
        return False

    compared, n_mismatch, mismatches = _count_mismatches(left, right)
    ok = n_mismatch == 0 and compared > 0
    log_kv_checksum_diag(
        "compare experiment=%s req=%s fp=%s run=%d worker_id=%d "
        "left=%s right=%s compared=%d mismatches=%d match=%s",
        experiment,
        req_id,
        prompt_fp,
        run_number,
        worker_id,
        phase_left,
        phase_right,
        compared,
        n_mismatch,
        ok,
    )
    for line in mismatches[:max_log_mismatches]:
        log_kv_checksum_diag("  mismatch %s", line)
    if len(mismatches) > max_log_mismatches:
        log_kv_checksum_diag(
            "  ... %d more mismatches omitted",
            len(mismatches) - max_log_mismatches,
        )
    return ok


def compare_runs_same_phase(
    *,
    prompt_fp: str,
    run_left: int,
    run_right: int,
    phase: str,
    experiment: str,
    req_id: str,
    worker_id: int = 0,
) -> bool:
    left = _KV_SAMPLES.get(prompt_fp, {}).get(run_left, {}).get(phase)
    right = _KV_SAMPLES.get(prompt_fp, {}).get(run_right, {}).get(phase)
    if not left or not right:
        log_kv_checksum_diag(
            "compare_runs skip experiment=%s req=%s fp=%s run=%d vs %d phase=%s",
            experiment,
            req_id,
            prompt_fp,
            run_left,
            run_right,
            phase,
        )
        return False
    compared, n_mismatch, mismatches = _count_mismatches(left, right)
    ok = n_mismatch == 0 and compared > 0
    log_kv_checksum_diag(
        "compare_runs experiment=%s req=%s fp=%s run=%d vs run=%d phase=%s "
        "worker_id=%d compared=%d mismatches=%d match=%s",
        experiment,
        req_id,
        prompt_fp,
        run_left,
        run_right,
        phase,
        worker_id,
        compared,
        n_mismatch,
        ok,
    )
    for line in mismatches[:5]:
        log_kv_checksum_diag("  mismatch %s", line)
    return ok


def on_compute_before_decode_scatter(
    *,
    req_id: str,
    token_ids: Optional[Union[torch.Tensor, List[int]]],
    kvcaches: Sequence[Any],
    slot_mapping: Union[torch.Tensor, Sequence[int]],
    num_layers: int,
    prompt_len: int,
    worker_id: int,
    kv_format: int = KV_FORMAT_MLA,
) -> None:
    """Experiment B baseline: GPU compute KV before first decode sparse scatter."""
    if not kv_checksum_diag_enabled():
        return
    prompt_fp, run_number, _ = record_phase_samples(
        req_id=req_id,
        token_ids=token_ids,
        phase="compute_before_decode_scatter",
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        num_layers=num_layers,
        prompt_len=prompt_len,
        worker_id=worker_id,
        kv_format=kv_format,
    )
    if run_number == 1:
        samples = _KV_SAMPLES[prompt_fp][run_number]["compute_before_decode_scatter"]
        _COMPUTE_BASELINE_BY_FP[prompt_fp] = list(samples)
        log_kv_checksum_diag(
            "experiment B baseline stored fp=%s run=1 req=%s (prefill compute KV)",
            prompt_fp,
            req_id,
        )


def on_decode_scatter_complete(
    *,
    req_id: str,
    token_ids: Optional[Union[torch.Tensor, List[int]]],
    kvcaches: Sequence[Any],
    slot_mapping: Union[torch.Tensor, Sequence[int]],
    num_layers: int,
    prompt_len: int,
    worker_id: int,
    decode_step: int,
    kv_format: int = KV_FORMAT_MLA,
) -> None:
    """After decode step scatter: per-step Run1 vs RunN comparison (experiment C).

    Records EVERY decode step (not just step 0) so we can see at which step
    Run2's scattered KV starts to diverge from Run1's. At a given decode step
    both runs have the same sequence length (prompt_len + step), so the 2048
    window covers the same prompt tokens -- a real mismatch means the scatter
    wrote different KV for the same token, not a window-shift artifact.
    """
    if not kv_checksum_diag_enabled():
        return

    phase = f"decode_step{decode_step}_scatter"

    prompt_fp, run_number, _ = record_phase_samples(
        req_id=req_id,
        token_ids=token_ids,
        phase=phase,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        num_layers=num_layers,
        prompt_len=prompt_len,
        worker_id=worker_id,
        kv_format=kv_format,
    )

    # Experiment B: vs compute baseline (only meaningful for step 0; later
    # steps' "compute_before" was the uninit decode window, so skip).
    if decode_step == 0:
        compare_phases(
            prompt_fp=prompt_fp,
            run_number=run_number,
            phase_left="compute_before_decode_scatter",
            phase_right=phase,
            experiment="B_compute_vs_decode_scatter",
            req_id=req_id,
            worker_id=worker_id,
        )

    # Experiment C: Run1 vs RunN at the SAME decode step. This is the key
    # per-step divergence check. Run1 is the reference (correct output).
    if run_number > 1:
        compare_runs_same_phase(
            prompt_fp=prompt_fp,
            run_left=1,
            run_right=run_number,
            phase=phase,
            experiment=f"C_run1_vs_runN_step{decode_step}",
            req_id=req_id,
            worker_id=worker_id,
        )


def on_prefill_compute_complete(
    *,
    req_id: str,
    token_ids: Optional[Union[torch.Tensor, List[int]]],
    kvcaches: Sequence[Any],
    slot_mapping: Union[torch.Tensor, Sequence[int]],
    num_layers: int,
    prompt_len: int,
    worker_id: int,
    kv_format: int = KV_FORMAT_MLA,
) -> None:
    """Record Run1 prefill compute KV at full prompt positions.

    Unlike ``on_compute_before_decode_scatter`` (which samples the 2048-token
    decode window and is misaligned with prefill-retrieve prompt positions),
    this baseline samples the FULL prompt slot_mapping so that
    ``on_prefill_retrieve_complete`` can compare at all sampled prompt tokens
    (0, 16831, ..., 18878), not just the overlapping token 0.
    """
    if not kv_checksum_diag_enabled():
        return
    prompt_fp, run_number, _ = record_phase_samples(
        req_id=req_id,
        token_ids=token_ids,
        phase="prefill_compute",
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        num_layers=num_layers,
        prompt_len=prompt_len,
        worker_id=worker_id,
        kv_format=kv_format,
    )
    if run_number == 1:
        samples = _KV_SAMPLES[prompt_fp][run_number]["prefill_compute"]
        _PREFILL_COMPUTE_BASELINE_BY_FP[prompt_fp] = list(samples)
        log_kv_checksum_diag(
            "experiment B prefill_compute baseline stored fp=%s run=1 "
            "req=%s worker_id=%d n_samples=%d (full prompt positions)",
            prompt_fp,
            req_id,
            worker_id,
            len(samples),
        )


def on_prefill_retrieve_complete(
    *,
    req_id: str,
    token_ids: Optional[Union[torch.Tensor, List[int]]],
    kvcaches: Sequence[Any],
    slot_mapping: Union[torch.Tensor, Sequence[int]],
    num_layers: int,
    prompt_len: int,
    worker_id: int,
    kv_format: int = KV_FORMAT_MLA,
) -> None:
    """Experiment B: prefill retrieve scatter vs run1 prefill compute baseline."""
    if not kv_checksum_diag_enabled():
        return
    prompt_fp, run_number, _ = record_phase_samples(
        req_id=req_id,
        token_ids=token_ids,
        phase="prefill_retrieve_scatter",
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        num_layers=num_layers,
        prompt_len=prompt_len,
        worker_id=worker_id,
        kv_format=kv_format,
    )
    # Prefer the prefill_compute baseline (full prompt positions) so the
    # comparison covers ALL sampled tokens, not just the token-0 overlap with
    # the decode-window baseline.
    baseline = _PREFILL_COMPUTE_BASELINE_BY_FP.get(prompt_fp)
    baseline_label = "prefill_compute"
    if baseline is None:
        baseline = _COMPUTE_BASELINE_BY_FP.get(prompt_fp)
        baseline_label = "compute_before_decode_scatter"
    if baseline is None:
        log_kv_checksum_diag(
            "prefill_retrieve complete fp=%s run=%d req=%s (no run1 compute baseline yet)",
            prompt_fp,
            run_number,
            req_id,
        )
        return
    scatter = _KV_SAMPLES[prompt_fp][run_number]["prefill_retrieve_scatter"]
    compared, n_mismatch, mismatches = _count_mismatches(baseline, scatter)
    log_kv_checksum_diag(
        "experiment B prefill_retrieve vs run1 %s fp=%s run=%d req=%s "
        "worker_id=%d compared=%d mismatches=%d",
        baseline_label,
        prompt_fp,
        run_number,
        req_id,
        worker_id,
        compared,
        n_mismatch,
    )
    for line in mismatches[:5]:
        log_kv_checksum_diag("  mismatch %s", line)


def log_sparse_scatter_entry(
    *,
    req_id: Optional[str],
    layer_id: int,
    worker_id: int,
    num_sparse: int,
    total_tokens: int,
    chunk_size: int,
    chunk_ptrs: int,
    use_fast_path: bool,
) -> None:
    if not kv_checksum_diag_enabled():
        return
    log_kv_checksum_diag(
        "sparse_scatter_entry req=%s layer=%d worker_id=%d num_sparse=%d "
        "total_tokens=%d chunk_size=%d chunk_ptrs=%d fast=%s "
        "(sparse_mla_dsa_batched_direct_kv_transfer)",
        req_id,
        layer_id,
        worker_id,
        num_sparse,
        total_tokens,
        chunk_size,
        chunk_ptrs,
        use_fast_path,
    )


def reset_kv_checksum_diag_state() -> None:
    """Clear module state (for unit tests)."""
    _KV_SAMPLES.clear()
    _COMPUTE_BASELINE_BY_FP.clear()
    _PREFILL_COMPUTE_BASELINE_BY_FP.clear()
    _REQ_TO_RUN.clear()
    _FP_RUN_COUNT.clear()
