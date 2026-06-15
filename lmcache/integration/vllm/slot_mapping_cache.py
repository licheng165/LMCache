# SPDX-License-Identifier: Apache-2.0
"""Incremental CPU and device slot_mapping caches for vLLM KV connector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from lmcache import utils


class SlotMappingBuilder:
    """Build vLLM paged-KV slot indices from physical block ids."""

    @staticmethod
    def slots_for_blocks(block_ids: list[int], block_size: int) -> torch.Tensor:
        num_blocks = len(block_ids)
        if num_blocks == 0:
            return torch.empty(0, dtype=torch.long)
        block_ids_t = torch.tensor(block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        return (
            block_offsets.reshape((1, block_size))
            + block_ids_t.reshape((num_blocks, 1)) * block_size
        ).flatten()

    @staticmethod
    def block_fingerprint(
        block_ids: tuple[int, ...], block_size: int, num_tokens: int
    ) -> tuple[int, ...]:
        """Block ids that cover token indices ``[0, num_tokens)``."""
        return block_ids[: utils.cdiv(num_tokens, block_size)]

    @staticmethod
    def is_prefix_extension(
        cached_fp: tuple[int, ...], new_fp: tuple[int, ...]
    ) -> bool:
        return len(new_fp) > len(cached_fp) and new_fp[: len(cached_fp)] == cached_fp


@dataclass
class CpuSlotMappingCache:
    """Scheduler-side CPU cache; extends incrementally when blocks are appended."""

    _tensor: Optional[torch.Tensor] = field(default=None, repr=False)
    _num_blocks: int = field(default=0, repr=False)

    def invalidate(self) -> None:
        self._tensor = None
        self._num_blocks = 0

    def get(self, block_ids: list[int], block_size: int, num_tokens: int) -> torch.Tensor:
        num_blocks = len(block_ids)

        if self._tensor is not None and self._num_blocks > num_blocks:
            self.invalidate()

        if (
            self._tensor is not None
            and self._num_blocks == num_blocks
            and self._tensor.numel() >= num_tokens
        ):
            return self._tensor[:num_tokens]

        if self._tensor is not None and self._num_blocks < num_blocks:
            new_slots = SlotMappingBuilder.slots_for_blocks(
                block_ids[self._num_blocks :], block_size
            )
            self._tensor = torch.cat([self._tensor, new_slots])
            self._num_blocks = num_blocks
            return self._tensor[:num_tokens]

        self._tensor = SlotMappingBuilder.slots_for_blocks(block_ids, block_size)
        self._num_blocks = num_blocks
        return self._tensor[:num_tokens]


@dataclass
class DeviceSlotMappingCache:
    """Worker-side device cache keyed by request id and block fingerprint."""

    device: torch.device | str
    _entries: dict[str, tuple[tuple[int, ...], torch.Tensor]] = field(
        default_factory=dict, repr=False
    )

    def clear(self, req_id: str) -> None:
        self._entries.pop(req_id, None)

    def get(
        self,
        req_id: str,
        cpu_mapping: torch.Tensor,
        block_fingerprint: tuple[int, ...],
        block_size: int,
        num_tokens: int,
    ) -> torch.Tensor:
        fp = SlotMappingBuilder.block_fingerprint(
            block_fingerprint, block_size, num_tokens
        )
        entry = self._entries.get(req_id)

        if entry is not None:
            cached_fp, cached_tensor = entry
            if cached_fp == fp and cached_tensor.numel() >= num_tokens:
                return cached_tensor[:num_tokens]

            if (
                SlotMappingBuilder.is_prefix_extension(cached_fp, fp)
                and cpu_mapping.numel() >= num_tokens
            ):
                cached_tensor = self._extend_cached(
                    cached_tensor, cpu_mapping, num_tokens
                )
                self._entries[req_id] = (fp, cached_tensor)
                return cached_tensor[:num_tokens]

            if cached_fp != fp:
                self._entries.pop(req_id, None)

        device_mapping = cpu_mapping[:num_tokens].to(
            device=self.device, dtype=torch.long
        )
        self._entries[req_id] = (fp, device_mapping)
        return device_mapping

    def _extend_cached(
        self,
        cached_tensor: torch.Tensor,
        cpu_mapping: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        extend_start = cached_tensor.numel()
        if num_tokens <= extend_start:
            return cached_tensor[:num_tokens]
        new_slice = cpu_mapping[extend_start:num_tokens].to(
            device=self.device, dtype=torch.long
        )
        return torch.cat([cached_tensor, new_slice])
