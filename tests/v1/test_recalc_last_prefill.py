# SPDX-License-Identifier: Apache-2.0
"""Prefill full-hit recalc_last: keep tokens/slots for partial-chunk key match."""

# Standard
from types import SimpleNamespace

# Third Party
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    DENSE_PREFIX_DIAG_ENV,
    LoadSpec,
    LMCacheConnectorV1Impl,
    ReqMeta,
    RequestTracker,
    _dense_prefix_diag_enabled,
    _dense_prefix_diag_message,
    _mark_dense_prefix_loaded,
)


def _block_ids(num_tokens: int, block_size: int) -> list[int]:
    return list(range((num_tokens + block_size - 1) // block_size))


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

    def test_preserves_tokens_and_slots_for_partial_chunk(self) -> None:
        req = _make_prefill_req()
        tokens = list(range(18879))
        slots = torch.arange(18879, dtype=torch.long)
        out_tokens, out_slots = LMCacheConnectorV1Impl._trim_prefill_for_recalc_last(
            req, tokens, slots
        )
        assert out_tokens is tokens
        assert out_slots is slots
        assert len(out_tokens) == 18879
        assert out_slots.numel() == 18879

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

    def test_full_hit_new_request_keeps_prompt_tokens_for_restore(self) -> None:
        prompt_len = 18879
        block_size = 16
        prompt_tokens = list(range(prompt_len))
        new_request = SimpleNamespace(
            req_id="req-full-hit",
            prompt_token_ids=prompt_tokens,
            block_ids=(
                _block_ids(prompt_len, block_size),
                _block_ids(prompt_len, block_size),
            ),
            sampling_params=SimpleNamespace(extra_args=None),
        )

        tracker = RequestTracker.from_new_request(
            lmcache_config=None,
            new_request=new_request,
            num_tokens_to_compute=1,
            lmcache_cached_tokens=prompt_len,
            skip_save=False,
        )

        assert tracker.token_ids == prompt_tokens

    def test_dense_full_hit_req_meta_uses_lmcache_hit_length(self) -> None:
        prompt_len = 18879
        block_size = 16
        tracker = RequestTracker(
            req_id="req-full-hit",
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=_block_ids(prompt_len, block_size),
            allocated_block_ids_indexer=_block_ids(prompt_len, block_size),
            num_saved_tokens=prompt_len,
        )

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=block_size,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=prompt_len,
                can_load=True,
            ),
            dsa_two_groups=True,
        )

        assert req_meta is not None
        assert len(req_meta.token_ids) == prompt_len
        assert req_meta.slot_mapping[0].numel() == prompt_len
        assert req_meta.indexer_slot_mapping[0].numel() == prompt_len
        assert req_meta.save_spec.can_save is False

    def test_dense_prefix_diag_env_flag(self, monkeypatch) -> None:
        monkeypatch.delenv(DENSE_PREFIX_DIAG_ENV, raising=False)
        assert not _dense_prefix_diag_enabled()

        monkeypatch.setenv(DENSE_PREFIX_DIAG_ENV, "1")
        assert _dense_prefix_diag_enabled()

    def test_mark_dense_prefix_loaded_sets_forward_context_flag(self) -> None:
        forward_context = SimpleNamespace()

        _mark_dense_prefix_loaded(
            forward_context,
            req_id="req-diag",
            token_count=18879,
            kv_group=0,
        )

        assert forward_context.lmcache_dense_prefix_loaded is True
        assert forward_context.lmcache_dense_prefix_loaded_reqs == [
            {
                "req_id": "req-diag",
                "token_count": 18879,
                "kv_group": 0,
            }
        ]

    def test_dense_prefix_diag_message_identifies_dsa_group(self) -> None:
        message = _dense_prefix_diag_message(
            phase="adapter_dense_prefix_retrieve",
            req_id="req-diag",
            kv_group=1,
            tokens=list(range(8)),
            token_mask=torch.ones(8, dtype=torch.bool),
            slot_mapping=torch.arange(20, 28, dtype=torch.long),
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=8,
                can_load=True,
            ),
            recalc_last_applied=True,
        )

        assert "phase=adapter_dense_prefix_retrieve" in message
        assert "req_id=req-diag" in message
        assert "kv_group=1" in message
        assert "group=dsa_index" in message
        assert "lmcache_cached_tokens=8" in message
        assert "recalc_last=True" in message
        assert "tokens.len=8" in message
        assert "token_mask_true=8" in message
        assert "slot_mapping.len=8" in message
        assert "slot_mapping.min=20" in message
        assert "slot_mapping.max=27" in message
