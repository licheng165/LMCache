# SPDX-License-Identifier: Apache-2.0
"""Prepared source state for layerwise sparse cache retrieval."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch


@dataclass(frozen=True, slots=True)
class PreparedSparseSourceLayer:
    """Stable CPU source and pointer table for one sparse cache layer."""

    tensors: tuple[torch.Tensor, ...]
    chunk_ptrs_npu: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreparedSparseSource:
    """Request-owned sparse source resolved once after cache bootstrap."""

    layers: tuple[PreparedSparseSourceLayer, ...]
    total_tokens: int
    chunk_token_counts: tuple[int, ...] = field(default_factory=tuple)
    pointer_device: Optional[torch.device] = None


def build_prepared_sparse_source(
    cached_tensors: Sequence[Sequence[torch.Tensor]],
    cached_chunk_ptrs_npu: Sequence[Optional[torch.Tensor]],
    *,
    num_layers: int,
    total_tokens: int,
    chunk_token_counts: Optional[Sequence[int]] = None,
    expected_pointer_device: Optional[torch.device] = None,
) -> Optional[PreparedSparseSource]:
    """Seal a complete layer cache into immutable hot-path source metadata.

    Args:
        cached_tensors: CPU chunk tensors in layer-major order.
        cached_chunk_ptrs_npu: NPU pointer tables in layer-major order.
        num_layers: Exact layer count required for a complete binding.
        total_tokens: Number of valid source tokens represented by the cache.
        chunk_token_counts: Request-owned token coverage for each CPU chunk.
        expected_pointer_device: Accelerator device that owns pointer tables.

    Returns:
        A prepared source, or ``None`` while bootstrap data is incomplete.

    Raises:
        TypeError: A completed cache contains an object of the wrong type.
        ValueError: Pointer metadata is malformed or has partial coverage.

    Incomplete caches are expected while the bootstrap generator is running and
    return ``None``. Once a layer has a pointer tensor, malformed pointer
    coverage is an invariant violation and is reported immediately.
    """
    if num_layers <= 0 or total_tokens <= 0:
        return None
    if len(cached_tensors) != num_layers:
        return None
    if len(cached_chunk_ptrs_npu) != num_layers:
        return None

    normalized_chunk_counts: tuple[int, ...] = ()
    if chunk_token_counts is not None:
        normalized_chunk_counts = tuple(int(count) for count in chunk_token_counts)
        if any(count <= 0 for count in normalized_chunk_counts):
            raise ValueError("Prepared sparse chunk token counts must be positive.")
        covered_tokens = sum(normalized_chunk_counts)
        if covered_tokens < total_tokens:
            return None

    layers: list[PreparedSparseSourceLayer] = []
    pointer_device: Optional[torch.device] = None
    for layer_id in range(num_layers):
        layer_tensors = cached_tensors[layer_id]
        if isinstance(layer_tensors, torch.Tensor):
            raise TypeError(
                "Prepared sparse source layers must contain tensor sequences: "
                f"layer_id={layer_id}"
            )
        tensors = tuple(layer_tensors)
        chunk_ptrs_npu = cached_chunk_ptrs_npu[layer_id]
        if not tensors or chunk_ptrs_npu is None:
            return None
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError(
                "Prepared sparse source contains a non-tensor entry: "
                f"layer_id={layer_id}"
            )
        if not isinstance(chunk_ptrs_npu, torch.Tensor):
            raise TypeError(
                "Prepared sparse pointer cache must contain tensors: "
                f"layer_id={layer_id}, type={type(chunk_ptrs_npu).__name__}"
            )
        if chunk_ptrs_npu.ndim != 1 or chunk_ptrs_npu.dtype != torch.int64:
            raise ValueError(
                "Prepared sparse pointer cache must be a 1D int64 tensor: "
                f"layer_id={layer_id}, shape={tuple(chunk_ptrs_npu.shape)}, "
                f"dtype={chunk_ptrs_npu.dtype}"
            )
        if not chunk_ptrs_npu.is_contiguous():
            raise ValueError(
                "Prepared sparse pointer cache must be contiguous: "
                f"layer_id={layer_id}, stride={chunk_ptrs_npu.stride()}"
            )
        if int(chunk_ptrs_npu.numel()) != len(tensors):
            raise ValueError(
                "Prepared sparse pointer coverage does not match CPU chunks: "
                f"layer_id={layer_id}, pointers={chunk_ptrs_npu.numel()}, "
                f"chunks={len(tensors)}"
            )
        if normalized_chunk_counts and len(normalized_chunk_counts) != len(tensors):
            raise ValueError(
                "Prepared sparse chunk coverage does not match CPU chunks: "
                f"layer_id={layer_id}, coverage={len(normalized_chunk_counts)}, "
                f"chunks={len(tensors)}"
            )
        if pointer_device is None:
            pointer_device = chunk_ptrs_npu.device
        elif chunk_ptrs_npu.device != pointer_device:
            raise ValueError(
                "Prepared sparse pointer tables must share one device: "
                f"layer_id={layer_id}, device={chunk_ptrs_npu.device}, "
                f"expected={pointer_device}"
            )
        if (
            expected_pointer_device is not None
            and (
                chunk_ptrs_npu.device.type != expected_pointer_device.type
                or (
                    expected_pointer_device.index is not None
                    and chunk_ptrs_npu.device.index != expected_pointer_device.index
                )
            )
        ):
            raise ValueError(
                "Prepared sparse pointer table is on the wrong device: "
                f"layer_id={layer_id}, device={chunk_ptrs_npu.device}, "
                f"expected={expected_pointer_device}"
            )

        layers.append(
            PreparedSparseSourceLayer(
                tensors=tensors,
                chunk_ptrs_npu=chunk_ptrs_npu,
            )
        )

    layer_tuple = tuple(layers)
    return PreparedSparseSource(
        layers=layer_tuple,
        total_tokens=int(total_tokens),
        chunk_token_counts=normalized_chunk_counts,
        pointer_device=pointer_device,
    )
