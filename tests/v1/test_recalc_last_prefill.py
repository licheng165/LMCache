# SPDX-License-Identifier: Apache-2.0
"""Prefill full-hit recalc_last token mask / slot_mapping."""

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

    def test_no_recalc_on_partial_hit(self) -> None:
        spec = LoadSpec(0, 4096, True)
        assert not LMCacheConnectorV1Impl._full_hit_recalc_last_token(
            spec, 18879, is_sparse_decode=False
        )

    def test_apply_masks_last_token_and_trims_slots(self) -> None:
        req = _make_prefill_req()
        mask = torch.ones(18879, dtype=torch.bool)
        slots = torch.arange(18879, dtype=torch.long)
        trimmed = LMCacheConnectorV1Impl._apply_full_hit_recalc_last_prefill(
            req, mask, slots
        )
        assert int(mask.sum().item()) == 18878
        assert mask[-1].item() is False
        assert trimmed.numel() == 18878
        assert trimmed[-1].item() == 18877

    def test_sparse_decode_untouched(self) -> None:
        req = _make_prefill_req()
        req.is_sparse_decode = True
        mask = torch.ones(18879, dtype=torch.bool)
        slots = torch.arange(18879, dtype=torch.long)
        out = LMCacheConnectorV1Impl._apply_full_hit_recalc_last_prefill(
            req, mask, slots
        )
        assert out is slots
        assert int(mask.sum().item()) == 18879
