# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostics for vLLM external prefix cache hit rate."""

# Standard
from dataclasses import dataclass, field
import hashlib
import os
from typing import Dict, List, Optional, Tuple, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def ext_prefix_hit_diag_enabled() -> bool:
    """Enable with environment variable LMCACHE_DIAG_EXT_PREFIX_HIT=1."""
    return os.environ.get("LMCACHE_DIAG_EXT_PREFIX_HIT", "").lower() in (
        "1",
        "true",
        "yes",
    )


def log_ext_prefix_hit_diag(fmt: str, *args) -> None:
    if ext_prefix_hit_diag_enabled():
        logger.info("[LMCache-Diag-ExtPrefix] " + fmt, *args)


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator * 100.0 / denominator


def prompt_fingerprint(
    token_ids: Optional[Union[torch.Tensor, List[int]]],
) -> str:
    """Stable prompt id across processes (does not use Python hash())."""
    if token_ids is None:
        return "empty"
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if not token_ids:
        return "empty"
    digest = hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()
    return f"{digest[:12]}_n{len(token_ids)}"


@dataclass
class PromptRunMetrics:
    req_id: str
    prompt_len: int = 0
    lookup_hit: int = 0
    lookup_hit_pct: float = 0.0
    returns_to_vllm: int = 0
    external_allocated: int = 0
    worker_load_pct_by_rank: Dict[int, float] = field(default_factory=dict)
    worker_retrieved_by_rank: Dict[int, int] = field(default_factory=dict)
    store_new_chunks_by_rank: Dict[int, int] = field(default_factory=dict)
    store_tokens_by_rank: Dict[int, int] = field(default_factory=dict)
    store_skipped_by_rank: Dict[int, int] = field(default_factory=dict)


class ExtPrefixHitRunTracker:
    """Track repeated runs of the same prompt to explain rising hit rates."""

    def __init__(self) -> None:
        self._runs_by_fp: Dict[str, List[PromptRunMetrics]] = {}
        self._req_to_fp_run: Dict[str, Tuple[str, int]] = {}

    def register_request(
        self,
        req_id: str,
        token_ids: Optional[Union[torch.Tensor, List[int]]],
    ) -> Tuple[str, int]:
        if not ext_prefix_hit_diag_enabled():
            return prompt_fingerprint(token_ids), 0

        if req_id in self._req_to_fp_run:
            return self._req_to_fp_run[req_id]

        fp = prompt_fingerprint(token_ids)
        prompt_len = len(token_ids) if token_ids is not None else 0
        if isinstance(token_ids, torch.Tensor):
            prompt_len = int(token_ids.numel())

        runs = self._runs_by_fp.setdefault(fp, [])
        run_number = len(runs) + 1
        runs.append(PromptRunMetrics(req_id=req_id, prompt_len=prompt_len))
        self._req_to_fp_run[req_id] = (fp, run_number)

        log_ext_prefix_hit_diag(
            "prompt_run_begin fp=%s run=%d req=%s prompt_len=%d "
            "(repeated runs of same fp: hit rate often rises as storage fills)",
            fp,
            run_number,
            req_id,
            prompt_len,
        )
        if run_number >= 2:
            self._log_cross_run_hint(fp, run_number)
        return fp, run_number

    def _get_metrics(self, req_id: str) -> Optional[Tuple[str, int, PromptRunMetrics]]:
        if req_id not in self._req_to_fp_run:
            return None
        fp, run_number = self._req_to_fp_run[req_id]
        return fp, run_number, self._runs_by_fp[fp][run_number - 1]

    def _prev_metrics(self, fp: str, run_number: int) -> Optional[PromptRunMetrics]:
        if run_number <= 1:
            return None
        runs = self._runs_by_fp.get(fp, [])
        if len(runs) < run_number - 1:
            return None
        return runs[run_number - 2]

    def _log_cross_run_hint(self, fp: str, run_number: int) -> None:
        prev = self._prev_metrics(fp, run_number)
        if prev is None:
            return
        log_ext_prefix_hit_diag(
            "prompt_run_context fp=%s entering run=%d (prev run=%d req=%s) "
            "prev_lookup_hit=%d (%.1f%%) prev_store_chunks=%s",
            fp,
            run_number,
            run_number - 1,
            prev.req_id,
            prev.lookup_hit,
            prev.lookup_hit_pct,
            prev.store_new_chunks_by_rank or "none",
        )

    def record_scheduler_lookup(
        self,
        req_id: str,
        token_ids: Optional[Union[torch.Tensor, List[int]]],
        *,
        prompt_len: int,
        lookup_hit: int,
        returns_to_vllm: int,
        vllm_computed: int,
    ) -> None:
        if not ext_prefix_hit_diag_enabled():
            return

        fp, run_number = self.register_request(req_id, token_ids)
        metrics = self._runs_by_fp[fp][run_number - 1]
        metrics.prompt_len = prompt_len
        metrics.lookup_hit = lookup_hit
        metrics.lookup_hit_pct = pct(lookup_hit, prompt_len)
        metrics.returns_to_vllm = returns_to_vllm

        prev = self._prev_metrics(fp, run_number)
        if prev is None:
            return

        delta_lookup = lookup_hit - prev.lookup_hit
        log_ext_prefix_hit_diag(
            "prompt_run_scheduler fp=%s run=%d vs run=%d "
            "lookup_hit %d (%.1f%%) vs %d (%.1f%%) delta=%+d "
            "returns_to_vllm %d vs %d vllm_computed=%d",
            fp,
            run_number,
            run_number - 1,
            lookup_hit,
            metrics.lookup_hit_pct,
            prev.lookup_hit,
            prev.lookup_hit_pct,
            delta_lookup,
            returns_to_vllm,
            prev.returns_to_vllm,
            vllm_computed,
        )
        if run_number >= 3 and delta_lookup > 0:
            runs = self._runs_by_fp[fp]
            progression = " -> ".join(
                f"run{i + 1}:{runs[i].lookup_hit}"
                for i in range(min(run_number, len(runs)))
            )
            log_ext_prefix_hit_diag(
                "prompt_run_history fp=%s lookup_hit progression [%s] "
                "(rising across repeats is normal when each run stores more)",
                fp,
                progression,
            )
            log_ext_prefix_hit_diag(
                "prompt_run_why_run3_higher fp=%s run=%d scheduler: "
                "lookup_hit rose vs run=%d — storage accumulated from "
                "run1 store + run%d store/load (check store_new_chunks per rank)",
                fp,
                run_number,
                run_number - 1,
                run_number - 1,
            )
        elif run_number >= 2 and delta_lookup > 0:
            log_ext_prefix_hit_diag(
                "prompt_run_why_higher fp=%s run=%d scheduler: "
                "lookup_hit rose vs run=%d — likely run%d store_layer wrote "
                "more worker_id keys (see store logs)",
                fp,
                run_number,
                run_number - 1,
                run_number - 1,
            )

    def record_scheduler_alloc(
        self,
        req_id: str,
        *,
        external_allocated: int,
    ) -> None:
        if not ext_prefix_hit_diag_enabled():
            return

        ctx = self._get_metrics(req_id)
        if ctx is None:
            return
        fp, run_number, metrics = ctx
        metrics.external_allocated = external_allocated
        prev = self._prev_metrics(fp, run_number)
        if prev is None:
            return
        log_ext_prefix_hit_diag(
            "prompt_run_alloc fp=%s run=%d vs run=%d "
            "external_allocated %d vs %d delta=%+d",
            fp,
            run_number,
            run_number - 1,
            external_allocated,
            prev.external_allocated,
            external_allocated - prev.external_allocated,
        )

    def record_worker_load(
        self,
        req_id: str,
        token_ids: Optional[Union[torch.Tensor, List[int]]],
        *,
        worker_id: int,
        prompt_len: int,
        lookup_hit: int,
        retrieved: int,
        expected: int,
        total_loaded: int,
    ) -> None:
        if not ext_prefix_hit_diag_enabled():
            return

        fp, run_number = self.register_request(req_id, token_ids)
        metrics = self._runs_by_fp[fp][run_number - 1]
        load_pct = pct(total_loaded, prompt_len)
        metrics.worker_load_pct_by_rank[worker_id] = load_pct
        metrics.worker_retrieved_by_rank[worker_id] = retrieved

        prev = self._prev_metrics(fp, run_number)
        if prev is None:
            log_ext_prefix_hit_diag(
                "prompt_run_worker fp=%s run=%d rank=%d first_run "
                "load_pct=%.1f%% retrieved=%d expected=%d",
                fp,
                run_number,
                worker_id,
                load_pct,
                retrieved,
                expected,
            )
            return

        prev_pct = prev.worker_load_pct_by_rank.get(worker_id, 0.0)
        prev_retrieved = prev.worker_retrieved_by_rank.get(worker_id, 0)
        delta_pct = load_pct - prev_pct
        log_ext_prefix_hit_diag(
            "prompt_run_worker fp=%s run=%d vs run=%d rank=%d "
            "load_pct %.1f%% vs %.1f%% delta=%+.1f%% "
            "retrieved %d vs %d (expected=%d)",
            fp,
            run_number,
            run_number - 1,
            worker_id,
            load_pct,
            prev_pct,
            delta_pct,
            retrieved,
            prev_retrieved,
            expected,
        )
        if delta_pct > 0.01:
            prev_store = prev.store_new_chunks_by_rank.get(worker_id, 0)
            avg_prev = (
                sum(prev.worker_load_pct_by_rank.values())
                / len(prev.worker_load_pct_by_rank)
                if prev.worker_load_pct_by_rank
                else prev_pct
            )
            avg_cur = (
                sum(metrics.worker_load_pct_by_rank.values())
                / len(metrics.worker_load_pct_by_rank)
                if metrics.worker_load_pct_by_rank
                else load_pct
            )
            log_ext_prefix_hit_diag(
                "prompt_run_why_higher fp=%s run=%d rank=%d load rose vs run=%d: "
                "run=%d store wrote %d new_chunks on this rank; "
                "vLLM external hit ~ avg TP load_pct (prev_avg=%.1f%% cur_partial_avg=%.1f%%)",
                fp,
                run_number,
                worker_id,
                run_number - 1,
                run_number - 1,
                prev_store,
                avg_prev,
                avg_cur,
            )
        if run_number >= 3 and metrics.worker_load_pct_by_rank:
            runs = self._runs_by_fp[fp]
            all_ranks: set[int] = set()
            for past in runs[:run_number]:
                all_ranks.update(past.worker_load_pct_by_rank.keys())
            rank_progress = []
            for ridx in sorted(all_ranks):
                seq = " -> ".join(
                    f"{runs[i].worker_load_pct_by_rank.get(ridx, 0):.0f}%"
                    for i in range(run_number)
                )
                rank_progress.append(f"rank{ridx}:[{seq}]")
            if rank_progress:
                log_ext_prefix_hit_diag(
                    "prompt_run_worker_history fp=%s run=%d load_pct by rank %s",
                    fp,
                    run_number,
                    " ".join(rank_progress),
                )

    def record_worker_store(
        self,
        req_id: str,
        token_ids: Optional[Union[torch.Tensor, List[int]]],
        *,
        worker_id: int,
        new_chunks: int,
        stored_tokens: int,
        skipped_existing: int,
        total_tokens: int,
    ) -> None:
        if not ext_prefix_hit_diag_enabled():
            return

        fp, run_number = self.register_request(req_id, token_ids)
        metrics = self._runs_by_fp[fp][run_number - 1]
        metrics.store_new_chunks_by_rank[worker_id] = new_chunks
        metrics.store_tokens_by_rank[worker_id] = stored_tokens
        metrics.store_skipped_by_rank[worker_id] = skipped_existing

        prev = self._prev_metrics(fp, run_number)
        if prev is None:
            log_ext_prefix_hit_diag(
                "prompt_run_store fp=%s run=%d rank=%d first_run "
                "new_chunks=%d stored_tokens=%d skipped_existing=%d total=%d "
                "(run2+ lookup_hit depends on this)",
                fp,
                run_number,
                worker_id,
                new_chunks,
                stored_tokens,
                skipped_existing,
                total_tokens,
            )
            return

        prev_chunks = prev.store_new_chunks_by_rank.get(worker_id, 0)
        prev_skipped = prev.store_skipped_by_rank.get(worker_id, 0)
        log_ext_prefix_hit_diag(
            "prompt_run_store fp=%s run=%d vs run=%d rank=%d "
            "new_chunks %d vs %d skipped_existing %d vs %d stored_tokens %d "
            "(skipped_existing>0 means keys already present for next run lookup)",
            fp,
            run_number,
            run_number - 1,
            worker_id,
            new_chunks,
            prev_chunks,
            skipped_existing,
            prev_skipped,
            stored_tokens,
        )
        if run_number >= 2 and skipped_existing > prev_skipped:
            log_ext_prefix_hit_diag(
                "prompt_run_why_next_run_higher fp=%s after run=%d rank=%d: "
                "skipped_existing=%d (storage warmer) — run=%d lookup/load "
                "should improve if hash/worker_id keys align",
                fp,
                run_number,
                worker_id,
                skipped_existing,
                run_number + 1,
            )


_RUN_TRACKER: Optional[ExtPrefixHitRunTracker] = None


def get_run_tracker() -> ExtPrefixHitRunTracker:
    global _RUN_TRACKER
    if _RUN_TRACKER is None:
        _RUN_TRACKER = ExtPrefixHitRunTracker()
    return _RUN_TRACKER


def record_store_layer_if_enabled(
    *,
    req_id: Optional[str],
    token_ids: Optional[Union[torch.Tensor, List[int]]],
    worker_id: int,
    new_chunks: int,
    stored_tokens: int,
    skipped_existing: int,
    total_tokens: int,
) -> None:
    if not ext_prefix_hit_diag_enabled() or req_id is None:
        return
    get_run_tracker().record_worker_store(
        req_id,
        token_ids,
        worker_id=worker_id,
        new_chunks=new_chunks,
        stored_tokens=stored_tokens,
        skipped_existing=skipped_existing,
        total_tokens=total_tokens,
    )
