# SPDX-License-Identifier: Apache-2.0
"""Prepared source state for layerwise sparse cache retrieval."""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

# Third Party
import torch


def _tensor_layout_signature(tensor: torch.Tensor) -> tuple:
    return (
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
        str(tensor.device),
        int(tensor.element_size()),
    )


@dataclass(frozen=True, slots=True)
class PreparedSparseSourceLayer:
    """Stable CPU source and pointer table for one sparse cache layer."""

    tensors: tuple[torch.Tensor, ...]
    chunk_ptrs_npu: torch.Tensor
    layout_signature: tuple


class PreparedSparseLayoutKey:
    """Exact layout identity with a cached hash for per-step plan lookup."""

    __slots__ = ("signature", "_hash")

    def __init__(self, signature: tuple) -> None:
        self.signature = signature
        self._hash = hash(signature)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, PreparedSparseLayoutKey):
            return NotImplemented
        return self.signature == other.signature


_PREPARED_SPARSE_LAYOUT_KEYS: dict[tuple, PreparedSparseLayoutKey] = {}


def _prepared_sparse_layout_key(signature: tuple) -> PreparedSparseLayoutKey:
    key = _PREPARED_SPARSE_LAYOUT_KEYS.get(signature)
    if key is not None:
        return key
    return _PREPARED_SPARSE_LAYOUT_KEYS.setdefault(
        signature,
        PreparedSparseLayoutKey(signature),
    )


@dataclass(frozen=True, slots=True)
class PreparedSparseSource:
    """Request-owned sparse source resolved once after cache bootstrap."""

    layers: tuple[PreparedSparseSourceLayer, ...]
    total_tokens: int
    layout_signature: tuple
    layout_key: PreparedSparseLayoutKey = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layout_key",
            _prepared_sparse_layout_key(self.layout_signature),
        )


def build_prepared_sparse_source(
    cached_tensors: Sequence[Sequence[torch.Tensor]],
    cached_chunk_ptrs_npu: Sequence[Optional[torch.Tensor]],
    *,
    num_layers: int,
    total_tokens: int,
) -> Optional[PreparedSparseSource]:
    """Seal a complete layer cache into immutable hot-path source metadata.

    Args:
        cached_tensors: CPU chunk tensors in layer-major order.
        cached_chunk_ptrs_npu: NPU pointer tables in layer-major order.
        num_layers: Exact layer count required for a complete binding.
        total_tokens: Number of valid source tokens represented by the cache.

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

    layers: list[PreparedSparseSourceLayer] = []
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

        layer_signature = (
            tuple(_tensor_layout_signature(tensor) for tensor in tensors),
            chunk_ptrs_npu.dtype,
            str(chunk_ptrs_npu.device),
            int(chunk_ptrs_npu.numel()),
        )
        layers.append(
            PreparedSparseSourceLayer(
                tensors=tensors,
                chunk_ptrs_npu=chunk_ptrs_npu,
                layout_signature=layer_signature,
            )
        )

    layer_tuple = tuple(layers)
    return PreparedSparseSource(
        layers=layer_tuple,
        total_tokens=int(total_tokens),
        layout_signature=tuple(layer.layout_signature for layer in layer_tuple),
    )
