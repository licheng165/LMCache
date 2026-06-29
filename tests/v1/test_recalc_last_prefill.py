# SPDX-License-Identifier: Apache-2.0
"""Prefill full-hit recalc_last: trim tokens/slots (not suffix mask)."""

# Third Party
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LoadSpec,
    LMCacheConnectorV1Impl,
    ReqMeta,
)


def _make_prefill_req(*, prompt_len: int = 18879) -> ReqMeta:
    return ReqMeta(
        req_id="req-1",
        token_ids=list(range(prompt_len)),
        slot_mapping=[torch.arange(prompt_len, dtype=torch.long)],
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=prompt_len,
            can_load=True,
        ),
        is_sparse_decode=False,
    )


class TestFullHitRecalcLast:
    def test_detects_full_hit_prefill(self) -> None:
        spec = LoadSpec(0, 18879, True)
        assert LMCacheConnectorV1Impl._full_hit_recalc_last_token(
            spec, 18879, is_sparse_decode=False
        )
        assert not LMCacheConnectorV1Impl._full_hit_recalc_last_token(
            spec, 18879, is_sparse_decode=True
        )

    def test_trim_tokens_and_slots(self) -> None:
        req = _make_prefill_req()
        tokens = list(range(18879))
        slots = torch.arange(18879, dtype=torch.long)
        out_tokens, out_slots = LMCacheConnectorV1Impl._trim_prefill_for_recalc_last(
            req, tokens, slots
        )
        assert len(out_tokens) == 18878
        assert out_slots.numel() == 18878
        assert out_tokens[-1] == 18877
        assert out_slots[-1].item() == 18877

    def test_sparse_decode_untouched(self) -> None:
        req = _make_prefill_req()
        req.is_sparse_decode = True
        tokens = list(range(18879))
        slots = torch.arange(18879, dtype=torch.long)
        out_tokens, out_slots = LMCacheConnectorV1Impl._trim_prefill_for_recalc_last(
            req, tokens, slots
        )
        assert out_tokens is tokens
        assert out_slots is slots
