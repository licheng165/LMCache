# SPDX-License-Identifier: Apache-2.0
"""Incremental CPU and device slot_mapping caches for vLLM KV connector."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch

from lmcache import utils
from lmcache.logging import init_logger

logger = init_logger(__name__)

_PERF_LOG_DISABLED = frozenset({"0", "false", "no", "off", ""})


def connector_perf_log_enabled() -> bool:
    value = os.environ.get("LMCACHE_CONNECTOR_PERF_LOG")
    if value is None:
        return False
    return value.strip().lower() not in _PERF_LOG_DISABLED


def _perf_log(msg: str, *args) -> None:
    """Emit connector perf diagnostics at INFO (visible under default log level)."""
    if connector_perf_log_enabled():
        logger.info(msg, *args)


@dataclass
class ConnectorPerfTimer:
    """Accumulate section timings for connector perf logging."""

    enabled: bool = field(default=False, repr=False)
    _start: float = field(default=0.0, repr=False)
    _parts: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.enabled:
            self._start = time.perf_counter()

    @classmethod
    def create(cls) -> "ConnectorPerfTimer":
        return cls(enabled=connector_perf_log_enabled())

    def record(self, name: str, elapsed_ms: float) -> None:
        if self.enabled:
            self._parts[name] = self._parts.get(name, 0.0) + elapsed_ms

    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - t0) * 1000)

    def log(self, context: str, **kwargs: object) -> None:
        if not self.enabled:
            return
        total_ms = (time.perf_counter() - self._start) * 1000
        parts = " ".join(
            f"{name}={elapsed_ms:.3f}ms"
            for name, elapsed_ms in sorted(self._parts.items())
        )
        extras = " ".join(f"{name}={value}" for name, value in kwargs.items())
        detail = " ".join(part for part in (parts, extras) if part)
        if detail:
            _perf_log("connector_perf %s total=%.3fms %s", context, total_ms, detail)
        else:
            _perf_log("connector_perf %s total=%.3fms", context, total_ms)


@dataclass
class SlotMappingCacheCounters:
    """Aggregate hit/miss counters for connector perf diagnostics."""

    hit: int = 0
    extend: int = 0
    rebuild: int = 0
    fp_mismatch: int = 0

    def reset(self) -> None:
        self.hit = 0
        self.extend = 0
        self.rebuild = 0
        self.fp_mismatch = 0


CPU_SLOT_MAPPING_COUNTERS = SlotMappingCacheCounters()
DEVICE_SLOT_MAPPING_COUNTERS = SlotMappingCacheCounters()


def reset_slot_mapping_cache_counters() -> None:
    CPU_SLOT_MAPPING_COUNTERS.reset()
    DEVICE_SLOT_MAPPING_COUNTERS.reset()


def log_slot_mapping_cache_summary(context: str) -> None:
    if not connector_perf_log_enabled():
        return
    _perf_log(
        "slot_mapping_cache %s cpu(hit=%d extend=%d rebuild=%d fp_mismatch=%d) "
        "device(hit=%d extend=%d rebuild=%d fp_mismatch=%d)",
        context,
        CPU_SLOT_MAPPING_COUNTERS.hit,
        CPU_SLOT_MAPPING_COUNTERS.extend,
        CPU_SLOT_MAPPING_COUNTERS.rebuild,
        CPU_SLOT_MAPPING_COUNTERS.fp_mismatch,
        DEVICE_SLOT_MAPPING_COUNTERS.hit,
        DEVICE_SLOT_MAPPING_COUNTERS.extend,
        DEVICE_SLOT_MAPPING_COUNTERS.rebuild,
        DEVICE_SLOT_MAPPING_COUNTERS.fp_mismatch,
    )
    reset_slot_mapping_cache_counters()


def _mapping_fp_head(
    block_ids: tuple[int, ...], block_size: int, num_tokens: int, n: int = 4
) -> tuple[int, ...]:
    return SlotMappingBuilder.block_fingerprint(block_ids, block_size, num_tokens)[:n]


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
    def slots_for_token_range(
        block_ids: list[int], block_size: int, start_token: int, end_token: int
    ) -> torch.Tensor:
        """Slot indices for token indices ``[start_token, end_token)``."""
        if start_token >= end_token or not block_ids:
            return torch.empty(0, dtype=torch.long)
        tokens = torch.arange(start_token, end_token, dtype=torch.long)
        block_ids_t = torch.tensor(block_ids, dtype=torch.long)
        block_indices = tokens // block_size
        return block_ids_t[block_indices] * block_size + (tokens % block_size)

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

    @staticmethod
    def fingerprint_compatible(
        cached_fp: tuple[int, ...], new_fp: tuple[int, ...]
    ) -> bool:
        return cached_fp == new_fp or SlotMappingBuilder.is_prefix_extension(
            cached_fp, new_fp
        )


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
            CPU_SLOT_MAPPING_COUNTERS.hit += 1
            _perf_log(
                "cpu slot_mapping hit num_tokens=%d num_blocks=%d fp_head=%s",
                num_tokens,
                num_blocks,
                _mapping_fp_head(tuple(block_ids), block_size, num_tokens),
            )
            return self._tensor[:num_tokens]

        if self._tensor is not None and self._num_blocks == num_blocks:
            extend_start = self._tensor.numel()
            new_slots = SlotMappingBuilder.slots_for_token_range(
                block_ids, block_size, extend_start, num_tokens
            )
            self._tensor = torch.cat([self._tensor, new_slots])
            CPU_SLOT_MAPPING_COUNTERS.extend += 1
            _perf_log(
                "cpu slot_mapping extend num_tokens=%d extend_start=%d "
                "num_blocks=%d fp_head=%s",
                num_tokens,
                extend_start,
                num_blocks,
                _mapping_fp_head(tuple(block_ids), block_size, num_tokens),
            )
            return self._tensor[:num_tokens]

        if self._tensor is not None and self._num_blocks < num_blocks:
            prev_blocks = self._num_blocks
            new_slots = SlotMappingBuilder.slots_for_blocks(
                block_ids[self._num_blocks :], block_size
            )
            self._tensor = torch.cat([self._tensor, new_slots])
            self._num_blocks = num_blocks
            CPU_SLOT_MAPPING_COUNTERS.extend += 1
            _perf_log(
                "cpu slot_mapping extend_blocks num_tokens=%d num_blocks=%d "
                "prev_blocks=%d fp_head=%s",
                num_tokens,
                num_blocks,
                prev_blocks,
                _mapping_fp_head(tuple(block_ids), block_size, num_tokens),
            )
            return self._tensor[:num_tokens]

        self._tensor = SlotMappingBuilder.slots_for_blocks(block_ids, block_size)
        self._num_blocks = num_blocks
        CPU_SLOT_MAPPING_COUNTERS.rebuild += 1
        _perf_log(
            "cpu slot_mapping rebuild num_tokens=%d num_blocks=%d fp_head=%s",
            num_tokens,
            num_blocks,
            _mapping_fp_head(tuple(block_ids), block_size, num_tokens),
        )
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

    @staticmethod
    def _resolve_cpu_mapping(
        cpu_mapping: torch.Tensor,
        block_fingerprint: tuple[int, ...],
        block_size: int,
        num_tokens: int,
    ) -> torch.Tensor:
        if cpu_mapping.numel() >= num_tokens:
            return cpu_mapping[:num_tokens]
        if not block_fingerprint:
            return cpu_mapping[:num_tokens]
        built = SlotMappingBuilder.slots_for_blocks(
            list(block_fingerprint), block_size
        )
        return built[:num_tokens]

    def get(
        self,
        req_id: str,
        cpu_mapping: torch.Tensor,
        block_fingerprint: tuple[int, ...],
        block_size: int,
        num_tokens: int,
    ) -> torch.Tensor:
        cpu_mapping = self._resolve_cpu_mapping(
            cpu_mapping, block_fingerprint, block_size, num_tokens
        )
        fp = SlotMappingBuilder.block_fingerprint(
            block_fingerprint, block_size, num_tokens
        )
        entry = self._entries.get(req_id)

        if entry is not None:
            cached_fp, cached_tensor = entry
            if cached_fp == fp and cached_tensor.numel() >= num_tokens:
                DEVICE_SLOT_MAPPING_COUNTERS.hit += 1
                _perf_log(
                    "device slot_mapping hit req=%s num_tokens=%d fp_head=%s",
                    req_id,
                    num_tokens,
                    _mapping_fp_head(block_fingerprint, block_size, num_tokens),
                )
                return cached_tensor[:num_tokens]

            if (
                SlotMappingBuilder.fingerprint_compatible(cached_fp, fp)
                and cpu_mapping.numel() >= num_tokens
            ):
                if cached_tensor.numel() < num_tokens:
                    cached_tensor = self._extend_cached(
                        cached_tensor, cpu_mapping, num_tokens
                    )
                self._entries[req_id] = (fp, cached_tensor)
                DEVICE_SLOT_MAPPING_COUNTERS.extend += 1
                _perf_log(
                    "device slot_mapping extend req=%s num_tokens=%d "
                    "cached_fp_head=%s fp_head=%s",
                    req_id,
                    num_tokens,
                    cached_fp[:4],
                    fp[:4],
                )
                return cached_tensor[:num_tokens]

            if not SlotMappingBuilder.fingerprint_compatible(cached_fp, fp):
                self._entries.pop(req_id, None)
                DEVICE_SLOT_MAPPING_COUNTERS.fp_mismatch += 1
                _perf_log(
                    "device slot_mapping fp_mismatch req=%s num_tokens=%d "
                    "cached_fp_head=%s fp_head=%s",
                    req_id,
                    num_tokens,
                    cached_fp[:4],
                    fp[:4],
                )

        device_mapping = cpu_mapping[:num_tokens].to(
            device=self.device, dtype=torch.long
        )
        self._entries[req_id] = (fp, device_mapping)
        DEVICE_SLOT_MAPPING_COUNTERS.rebuild += 1
        _perf_log(
            "device slot_mapping rebuild req=%s num_tokens=%d fp_head=%s",
            req_id,
            num_tokens,
            _mapping_fp_head(block_fingerprint, block_size, num_tokens),
        )
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
