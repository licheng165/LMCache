# SPDX-License-Identifier: Apache-2.0
"""Profile real CPU allocation and hot-cache work in sparse cold bootstrap.

Token hashing, layer-key construction, remote I/O, persistent allocator construction,
cleanup, and correctness checks are intentionally excluded from timed regions.
"""

# Standard
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import gc
import json
import math
import statistics
import threading
import time

# Third Party
import torch

# First Party
from lmcache.utils import LayerCacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryAllocator,
    TensorMemoryObj,
    get_size_bytes,
)
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


ALIGN_BYTES = 4096
TOKEN_DIMS = {0: 576, 1: 128}
VARIANTS = {
    "production": ("production", "cached"),
    "address_backed": ("address_backed", "cached"),
    "staged": ("staged", "production"),
    "single_loop": ("single_loop", "production"),
    "contiguous_split": ("contiguous_split", "production"),
    "cached_metadata": ("staged", "cached"),
    "cached_split": ("contiguous_split", "cached"),
}
ALLOCATOR_STAGES = (
    "allocator_setup_s",
    "address_s",
    "views_s",
    "wrappers_s",
    "combined_build_s",
    "accounting_s",
)
TENSOR_ACCESS_STAGES = (
    "tensor_s",
    "raw_tensor_s",
    "data_ptr_s",
    "cached_build_s",
    "cached_reuse_s",
    "bulk_typed_s",
)


def _verify_result(
    buffer: torch.Tensor,
    keys: list[LayerCacheEngineKey],
    objects: list[TensorMemoryObj],
    metadata: list[tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]],
    allocator: TensorMemoryAllocator,
    backend: SimpleNamespace,
) -> None:
    if len(objects) != len(keys) or len(backend.hot_cache) != len(keys):
        raise AssertionError("allocation or hot-cache object count mismatch")

    shapes, dtypes, fmt, _ = metadata[0]
    ranges = []
    for key, obj in zip(keys, objects, strict=True):
        meta = obj.metadata
        if (
            not obj.is_valid()
            or obj.get_ref_count() != 2
            or obj.parent() is not allocator
            or backend.hot_cache.get(key) is not obj
            or obj.get_shapes() != shapes
            or obj.get_dtypes() != dtypes
            or obj.get_memory_format() != fmt
            or obj.raw_tensor is None
            or obj.raw_tensor.numel() != meta.phy_size
        ):
            raise AssertionError("MemoryObj metadata or ownership mismatch")
        ranges.append((meta.address, meta.address + meta.phy_size))

    ranges.sort()
    if (
        ranges[0][0] < 0
        or ranges[-1][1] > buffer.numel()
        or any(
            left[1] > right[0] for left, right in zip(ranges, ranges[1:], strict=False)
        )
    ):
        raise AssertionError("allocated objects overlap or exceed the backing buffer")


class ProfiledTensorMemoryAllocator(TensorMemoryAllocator):
    """Benchmark-only layouts built from production allocator primitives."""

    def allocate_profiled(
        self,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        batch_size: int,
        fmt: MemoryFormat,
        mode: str,
    ) -> tuple[list[TensorMemoryObj], dict[str, float]]:
        stages = {name: 0.0 for name in ALLOCATOR_STAGES}

        started = time.perf_counter()
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        raw_size = get_size_bytes(shapes, dtypes)
        aligned_size = self.address_manager.compute_aligned_size(raw_size)
        stages["allocator_setup_s"] = time.perf_counter() - started

        started = time.perf_counter()
        allocations = self.address_manager.batched_allocate(
            aligned_size,
            batch_size,
        )
        addresses = (
            None if mode == "single_loop" else [address for address, _ in allocations]
        )
        stages["address_s"] = time.perf_counter() - started

        if mode == "single_loop":
            started = time.perf_counter()
            objects = [
                self._make_object(
                    self._get_buffer_slice(address, aligned_size),
                    address,
                    aligned_size,
                    shapes,
                    dtypes,
                    fmt,
                )
                for address, _ in allocations
            ]
            stages["combined_build_s"] = time.perf_counter() - started
            self._record_accounting(batch_size, stages)
        else:
            assert addresses is not None
            started = time.perf_counter()
            if mode == "contiguous_split":
                if any(
                    right != left + aligned_size
                    for left, right in zip(addresses, addresses[1:], strict=False)
                ):
                    raise RuntimeError("contiguous_split requires contiguous addresses")
                whole = self._get_buffer_slice(
                    addresses[0],
                    aligned_size * batch_size,
                )
                views = list(whole.split(aligned_size))
            else:
                views = [
                    self._get_buffer_slice(address, aligned_size)
                    for address in addresses
                ]
            stages["views_s"] = time.perf_counter() - started

            self._record_accounting(batch_size, stages)
            started = time.perf_counter()
            objects = []
            for view, address in zip(views, addresses, strict=True):
                objects.append(
                    self._make_object(
                        view,
                        address,
                        aligned_size,
                        shapes,
                        dtypes,
                        fmt,
                    )
                )
            stages["wrappers_s"] = time.perf_counter() - started

        return objects, stages

    def _record_accounting(
        self,
        batch_size: int,
        stages: dict[str, float],
    ) -> None:
        started = time.perf_counter()
        self.num_active_allocations += batch_size
        self.stats_monitor.update_local_cache_usage(
            self.address_manager.total_allocated_size
        )
        self.stats_monitor.update_active_memory_objs_count(self.num_active_allocations)
        stages["accounting_s"] = time.perf_counter() - started

    def _make_object(
        self,
        raw_data: torch.Tensor,
        address: int,
        aligned_size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
    ) -> TensorMemoryObj:
        return TensorMemoryObj(
            raw_data=raw_data,
            metadata=MemoryObjMetadata(
                shapes[0],
                dtypes[0],
                address,
                aligned_size,
                1,
                0,
                fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=self,
        )


def make_keys(
    chunks: int,
    layers: int,
    kv_group: int,
) -> list[LayerCacheEngineKey]:
    return [
        LayerCacheEngineKey(
            model_name="allocator-benchmark",
            world_size=8,
            worker_id=0,
            chunk_hash=chunk,
            dtype=torch.bfloat16,
            layer_id=layer,
            kv_group=kv_group,
        )
        for layer in range(layers)
        for chunk in range(chunks)
    ]


def make_connector(chunk_size: int) -> MooncakestoreConnector:
    # Deliberately construct only the state read by the production metadata helper.
    connector = object.__new__(MooncakestoreConnector)
    connector._dsa_raw_token_dims = TOKEN_DIMS  # noqa: SLF001
    connector.meta_shapes = [torch.Size([chunk_size * TOKEN_DIMS[0]])]
    connector.meta_dtypes = [torch.bfloat16]
    connector.meta_fmt = MemoryFormat.KV_MLA_LATENT_FMT
    connector.single_token_size = TOKEN_DIMS[0] * 2
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=chunk_size)
    )
    return connector


def prepare_metadata(
    connector: MooncakestoreConnector,
    keys: list[LayerCacheEngineKey],
    mode: str,
) -> list[tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]]:
    # Private access is intentional: this is the production path under measurement.
    if mode == "cached":
        value = connector._metadata_for_raw_key(keys[0])  # noqa: SLF001
        if not all(key.kv_group == keys[0].kv_group for key in keys):
            raise RuntimeError("cached metadata requires one KV group")
        return [value] * len(keys)

    metadata = [
        connector._metadata_for_raw_key(key)  # noqa: SLF001
        for key in keys
    ]
    first_shapes, first_dtypes, first_fmt, _ = metadata[0]
    if not all(
        shapes == first_shapes and dtypes == first_dtypes and fmt == first_fmt
        for shapes, dtypes, fmt, _ in metadata
    ):
        raise RuntimeError("metadata unexpectedly differs within one KV group")
    return metadata


def allocate_objects(
    allocator: TensorMemoryAllocator,
    metadata: list[tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]],
    mode: str,
) -> tuple[list[TensorMemoryObj], dict[str, float]]:
    shapes, dtypes, fmt, _ = metadata[0]
    stages = {name: 0.0 for name in ALLOCATOR_STAGES}
    if mode in ("production", "address_backed"):
        started = time.perf_counter()
        allocate = (
            allocator.batched_allocate_address_backed
            if mode == "address_backed"
            else allocator.batched_allocate
        )
        objects = allocate(
            shapes,
            dtypes,
            len(metadata),
            fmt,
        )
        if objects is None:
            raise RuntimeError("production batched allocation failed")
        stages["allocator_s"] = time.perf_counter() - started
        return objects, stages

    assert isinstance(allocator, ProfiledTensorMemoryAllocator)
    started = time.perf_counter()
    objects, stages = allocator.allocate_profiled(
        shapes,
        dtypes,
        len(metadata),
        fmt,
        mode,
    )
    stages["allocator_s"] = time.perf_counter() - started
    return objects, stages


def put_hot_cache(
    keys: list[LayerCacheEngineKey],
    objects: list[TensorMemoryObj],
    policy: LRUCachePolicy,
) -> tuple[float, SimpleNamespace]:
    backend = SimpleNamespace(
        use_hot=True,
        cpu_lock=threading.RLock(),
        hot_cache=policy.init_mutable_mapping(),
        cache_policy=policy,
        batched_msg_sender=None,
    )
    started = time.perf_counter()
    LocalCPUBackend.batched_submit_put_task(backend, keys, objects)
    return time.perf_counter() - started, backend


def run_sample(
    buffer: torch.Tensor,
    connector: MooncakestoreConnector,
    keys: list[LayerCacheEngineKey],
    variant: str,
    policy: LRUCachePolicy,
) -> dict[str, float]:
    allocator_mode, metadata_mode = VARIANTS[variant]
    allocator_cls = (
        TensorMemoryAllocator
        if allocator_mode in ("production", "address_backed")
        else ProfiledTensorMemoryAllocator
    )
    allocator = allocator_cls(buffer, align_bytes=ALIGN_BYTES)
    sample = {name: 0.0 for name in ALLOCATOR_STAGES}
    total_started = time.perf_counter()

    started = time.perf_counter()
    metadata = prepare_metadata(connector, keys, metadata_mode)
    sample["metadata_s"] = time.perf_counter() - started
    objects, stages = allocate_objects(allocator, metadata, allocator_mode)
    sample.update(stages)
    sample["hot_cache_s"], backend = put_hot_cache(keys, objects, policy)
    sample["total_s"] = time.perf_counter() - total_started

    _verify_result(buffer, keys, objects, metadata, allocator, backend)
    backend.hot_cache.clear()
    allocator.batched_free(objects)
    return sample


def run_hot_sample(
    buffer: torch.Tensor,
    keys: list[LayerCacheEngineKey],
    shape: torch.Size,
    fmt: MemoryFormat,
    policy: LRUCachePolicy,
    batched_policy: LRUCachePolicy,
) -> dict[str, float]:
    allocator = TensorMemoryAllocator(buffer, align_bytes=ALIGN_BYTES)
    objects = allocator.batched_allocate(
        [shape],
        [torch.bfloat16],
        len(keys),
        fmt,
    )
    if objects is None:
        raise RuntimeError("hot-cache setup allocation failed")

    sample: dict[str, float] = {}
    replacement_allocator = TensorMemoryAllocator(buffer, align_bytes=ALIGN_BYTES)
    started = time.perf_counter()
    for obj in objects:
        obj.parent_allocator = replacement_allocator
    sample["rebind_s"] = time.perf_counter() - started

    started = time.perf_counter()
    for obj in objects:
        obj.ref_count_up()
    sample["refcount_s"] = time.perf_counter() - started

    hot_cache = policy.init_mutable_mapping()
    started = time.perf_counter()
    for key, obj in zip(keys, objects, strict=True):
        hot_cache[key] = obj
    sample["dictionary_s"] = time.perf_counter() - started

    started = time.perf_counter()
    for key in keys:
        policy.update_on_put(key)
    sample["policy_s"] = time.perf_counter() - started

    started = time.perf_counter()
    batched_policy.update_on_put_many(keys)
    sample["batched_lru_policy_s"] = time.perf_counter() - started

    if len(hot_cache) != len(keys) or any(
        hot_cache.get(key) is not obj
        or obj.parent() is not replacement_allocator
        or obj.get_ref_count() != 2
        for key, obj in zip(keys, objects, strict=True)
    ):
        raise AssertionError("isolated hot-cache stage result mismatch")

    hot_cache.clear()
    allocator.batched_free(objects)
    return sample


def _typed_views(objects: list[TensorMemoryObj]) -> list[torch.Tensor]:
    views = []
    for obj in objects:
        view = obj.tensor
        if view is None:
            raise RuntimeError("valid TensorMemoryObj returned no tensor")
        views.append(view)
    return views


def _bulk_typed_views(
    buffer: torch.Tensor,
    objects: list[TensorMemoryObj],
    full_shape: torch.Size,
) -> list[torch.Tensor]:
    first = objects[0].metadata
    dtype = first.dtype
    if dtype is None or len(full_shape) != 1:
        raise RuntimeError("bulk typed-view benchmark requires one flat dtype/shape")
    stride_bytes = first.phy_size
    if (
        stride_bytes % dtype.itemsize
        or full_shape.numel() * dtype.itemsize > stride_bytes
    ):
        raise RuntimeError("physical stride cannot hold the typed view")
    if any(
        obj.metadata.address != first.address + index * stride_bytes
        or obj.metadata.phy_size != stride_bytes
        or obj.metadata.dtype != dtype
        for index, obj in enumerate(objects)
    ):
        raise RuntimeError("bulk typed-view benchmark requires contiguous objects")

    whole = buffer[first.address : first.address + len(objects) * stride_bytes]
    matrix = torch.as_strided(
        whole.view(dtype),
        (len(objects), full_shape.numel()),
        (stride_bytes // dtype.itemsize, 1),
    )
    views = list(matrix.unbind())
    for index, obj in enumerate(objects):
        if obj.get_shape() != full_shape:
            view = obj.tensor
            if view is None:
                raise RuntimeError("partial TensorMemoryObj returned no tensor")
            views[index] = view
    return views


def _verify_tensor_access(
    objects: list[TensorMemoryObj],
    typed_views: list[torch.Tensor],
    raw_views: list[torch.Tensor],
    pointers: list[int],
) -> None:
    if not (
        len(objects) == len(typed_views) == len(raw_views) == len(pointers)
    ):
        raise AssertionError("tensor-access result count mismatch")
    for obj, view, raw, pointer in zip(
        objects, typed_views, raw_views, pointers, strict=True
    ):
        if (
            view.data_ptr() != obj.data_ptr
            or view.dtype != obj.metadata.dtype
            or view.shape != obj.get_shape()
            or raw.data_ptr() != obj.data_ptr
            or raw.dtype != torch.uint8
            or raw.numel() < obj.get_size()
            or pointer != obj.data_ptr
        ):
            raise AssertionError("tensor-access view or pointer mismatch")


def run_tensor_access_sample(
    buffer: torch.Tensor,
    keys: list[LayerCacheEngineKey],
    chunk_size: int,
    num_tokens: int,
    kv_group: int,
) -> dict[str, float]:
    full_shape = torch.Size([chunk_size * TOKEN_DIMS[kv_group]])
    fmt = (
        MemoryFormat.KV_DSA_INDEX_FMT
        if kv_group == 1
        else MemoryFormat.KV_MLA_LATENT_FMT
    )
    allocator = TensorMemoryAllocator(buffer, align_bytes=ALIGN_BYTES)
    objects = allocator.batched_allocate(
        [full_shape],
        [torch.bfloat16],
        len(keys),
        fmt,
    )
    if objects is None:
        raise RuntimeError("tensor-access setup allocation failed")

    tail_tokens = num_tokens % chunk_size
    if tail_tokens:
        tail_chunk = math.ceil(num_tokens / chunk_size) - 1
        tail_bytes = tail_tokens * TOKEN_DIMS[kv_group] * 2
        tail_count = 0
        for key, obj in zip(keys, objects, strict=True):
            if key.chunk_hash == tail_chunk:
                tail_count += 1
                MooncakestoreConnector._reshape_partial_chunk_with_token_size(
                    obj,
                    tail_bytes,
                    TOKEN_DIMS[kv_group] * 2,
                )
        if tail_count * math.ceil(num_tokens / chunk_size) != len(objects):
            raise AssertionError("partial-tail object count mismatch")

    sample = {name: 0.0 for name in TENSOR_ACCESS_STAGES}
    started = time.perf_counter()
    typed_views = _typed_views(objects)
    sample["tensor_s"] = time.perf_counter() - started

    started = time.perf_counter()
    raw_views = []
    for obj in objects:
        raw = obj.raw_tensor
        if raw is None:
            raise RuntimeError("valid TensorMemoryObj returned no raw tensor")
        raw_views.append(raw)
    sample["raw_tensor_s"] = time.perf_counter() - started

    started = time.perf_counter()
    pointers = [obj.data_ptr for obj in objects]
    sample["data_ptr_s"] = time.perf_counter() - started

    started = time.perf_counter()
    cached_views = []
    for obj in objects:
        view = obj.tensor
        if view is None:
            raise RuntimeError("valid TensorMemoryObj returned no tensor")
        obj._benchmark_cached_tensor = view  # type: ignore[attr-defined]
        cached_views.append(view)
    sample["cached_build_s"] = time.perf_counter() - started

    started = time.perf_counter()
    reused_views = [
        obj._benchmark_cached_tensor  # type: ignore[attr-defined]
        for obj in objects
    ]
    sample["cached_reuse_s"] = time.perf_counter() - started

    started = time.perf_counter()
    bulk_views = _bulk_typed_views(buffer, objects, full_shape)
    sample["bulk_typed_s"] = time.perf_counter() - started

    for views in (typed_views, cached_views, reused_views, bulk_views):
        _verify_tensor_access(objects, views, raw_views, pointers)
    for obj in objects:
        del obj._benchmark_cached_tensor  # type: ignore[attr-defined]
    del typed_views, raw_views, pointers, cached_views, reused_views, bulk_views
    allocator.batched_free(objects)
    return sample


def medians(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: statistics.median(sample[name] for sample in samples)
        for name in samples[0]
    }


def run_layer_page_sample(
    buffer: torch.Tensor,
    *,
    chunks: int,
    layers: int,
    shape: torch.Size,
    fmt: MemoryFormat,
) -> dict[str, float]:
    allocator = TensorMemoryAllocator(buffer, align_bytes=ALIGN_BYTES)
    started = time.perf_counter()
    pages = allocator.batched_allocate_layer_pages(
        shape, torch.bfloat16, chunks, layers, fmt
    )
    allocation_s = time.perf_counter() - started
    if pages is None:
        raise RuntimeError("layer-page allocation failed")

    started = time.perf_counter()
    pointers = [
        page.layer_data_ptr(layer) for layer in range(layers) for page in pages
    ]
    pointer_s = time.perf_counter() - started
    started = time.perf_counter()
    views = [page.layer_tensor(layer) for layer in range(layers) for page in pages]
    view_s = time.perf_counter() - started
    if len(pages) != chunks or any(
        tensor.data_ptr() != pointer
        for tensor, pointer in zip(views, pointers, strict=True)
    ):
        raise AssertionError("layer-page pointer or view layout mismatch")
    allocator.batched_free(pages)
    return {
        "allocation_s": allocation_s,
        "pointer_s": pointer_s,
        "view_s": view_s,
        "objects": float(len(pages)),
    }


def ms(seconds: float) -> float:
    return seconds * 1000


def print_results(
    results: list[dict],
    hot_results: dict[int, dict[str, float]],
    tensor_access_results: dict[int, dict[str, float]],
    layer_page_results: dict[int, dict[str, float]],
) -> None:
    print("\nEnd-to-end production-object path (median ms)")
    print(
        f"{'group':>5} {'variant':20} {'total':>9} {'speedup':>9} "
        f"{'metadata':>10} {'allocator':>10} {'hot cache':>10}"
    )
    baseline = {}
    for result in results:
        group = result["kv_group"]
        if group not in baseline or result["variant"] == "production":
            baseline[group] = result["median"]["total_s"]
    for result in results:
        sample = result["median"]
        print(
            f"{result['kv_group']:5d} {result['variant']:20} "
            f"{ms(sample['total_s']):9.3f} "
            f"{baseline[result['kv_group']] / sample['total_s']:9.3f}x "
            f"{ms(sample['metadata_s']):10.3f} "
            f"{ms(sample['allocator_s']):10.3f} "
            f"{ms(sample['hot_cache_s']):10.3f}"
        )

    print("\nAllocator attribution (median ms; production is total-only)")
    print(
        f"{'group':>5} {'variant':20} {'setup':>8} {'address':>9} "
        f"{'views':>8} {'wrappers':>10} {'combined':>10} {'account':>9}"
    )
    for result in results:
        sample = result["median"]
        print(
            f"{result['kv_group']:5d} {result['variant']:20} "
            f"{ms(sample['allocator_setup_s']):8.3f} "
            f"{ms(sample['address_s']):9.3f} "
            f"{ms(sample['views_s']):8.3f} "
            f"{ms(sample['wrappers_s']):10.3f} "
            f"{ms(sample['combined_build_s']):10.3f} "
            f"{ms(sample['accounting_s']):9.3f}"
        )

    print("\nHot-cache isolated stages (median ms)")
    print(
        f"{'group':>5} {'rebind':>9} {'refcount':>10} {'dict':>9} "
        f"{'LRU/key':>11} {'LRU/batch':>12}"
    )
    for group, sample in hot_results.items():
        print(
            f"{group:5d} {ms(sample['rebind_s']):9.3f} "
            f"{ms(sample['refcount_s']):10.3f} "
            f"{ms(sample['dictionary_s']):9.3f} "
            f"{ms(sample['policy_s']):11.3f} "
            f"{ms(sample['batched_lru_policy_s']):12.3f}"
        )

    print("\nTensorMemoryObj access (median ms; correctness checks excluded)")
    print(
        f"{'group':>5} {'tensor':>9} {'raw':>9} {'pointer':>9} "
        f"{'cache build':>12} {'cache reuse':>12} {'bulk typed':>12} "
        f"{'bulk speedup':>13}"
    )
    for group, sample in tensor_access_results.items():
        print(
            f"{group:5d} {ms(sample['tensor_s']):9.3f} "
            f"{ms(sample['raw_tensor_s']):9.3f} "
            f"{ms(sample['data_ptr_s']):9.3f} "
            f"{ms(sample['cached_build_s']):12.3f} "
            f"{ms(sample['cached_reuse_s']):12.3f} "
            f"{ms(sample['bulk_typed_s']):12.3f} "
            f"{sample['tensor_s'] / sample['bulk_typed_s']:13.3f}x"
        )

    print("\nExperimental layer-page path (median ms)")
    print(
        f"{'group':>5} {'objects':>9} {'allocator':>10} "
        f"{'pointers':>10} {'views':>10}"
    )
    for group, sample in layer_page_results.items():
        print(
            f"{group:5d} {int(sample['objects']):9d} "
            f"{ms(sample['allocation_s']):10.3f} "
            f"{ms(sample['pointer_s']):10.3f} "
            f"{ms(sample['view_s']):10.3f}"
        )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=61)
    parser.add_argument("--kv-groups", default="0,1")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    groups = [int(value) for value in args.kv_groups.split(",") if value]
    variants = [value for value in args.variants.split(",") if value]
    if not groups or any(group not in TOKEN_DIMS for group in groups):
        parser.error("--kv-groups must contain 0 and/or 1")
    if len(groups) != len(set(groups)):
        parser.error("--kv-groups must not contain duplicates")
    if not variants or any(variant not in VARIANTS for variant in variants):
        parser.error(f"--variants must use: {','.join(VARIANTS)}")
    if len(variants) != len(set(variants)):
        parser.error("--variants must not contain duplicates")
    if min(args.num_tokens, args.chunk_size, args.num_layers, args.repeats) < 1:
        parser.error("tokens, chunk size, layers, and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")

    chunks = math.ceil(args.num_tokens / args.chunk_size)
    object_count = chunks * args.num_layers
    connector = make_connector(args.chunk_size)
    results: list[dict] = []
    hot_results: dict[int, dict[str, float]] = {}
    tensor_access_results: dict[int, dict[str, float]] = {}
    layer_page_results: dict[int, dict[str, float]] = {}
    print(
        f"tokens={args.num_tokens} chunks={chunks} layers={args.num_layers} "
        f"objects/group={object_count}"
    )

    for group in groups:
        object_bytes = args.chunk_size * TOKEN_DIMS[group] * 2
        aligned_bytes = math.ceil(object_bytes / ALIGN_BYTES) * ALIGN_BYTES
        pool_bytes = aligned_bytes * object_count
        print(
            f"[group {group}] object={object_bytes / 1e3:.3f} KB "
            f"pool={pool_bytes / 1e9:.3f} GB",
            flush=True,
        )
        buffer = torch.empty(pool_bytes, dtype=torch.uint8)
        keys = make_keys(chunks, args.num_layers, group)
        samples = {variant: [] for variant in variants}
        policy = LRUCachePolicy()

        for repeat in range(args.warmup + args.repeats):
            order = variants[:: 1 if repeat % 2 == 0 else -1]
            for variant in order:
                gc.collect()
                policy.chunk_hash_to_init_timestamp.clear()
                sample = run_sample(
                    buffer,
                    connector,
                    keys,
                    variant,
                    policy,
                )
                if repeat >= args.warmup:
                    samples[variant].append(sample)
        results.extend(
            {
                "kv_group": group,
                "variant": variant,
                "median": medians(samples[variant]),
                "samples": samples[variant],
            }
            for variant in variants
        )

        fmt = (
            MemoryFormat.KV_DSA_INDEX_FMT
            if group == 1
            else MemoryFormat.KV_MLA_LATENT_FMT
        )
        hot_samples = []
        hot_policy = LRUCachePolicy()
        batched_policy = LRUCachePolicy()
        for repeat in range(args.warmup + args.repeats):
            gc.collect()
            hot_policy.chunk_hash_to_init_timestamp.clear()
            batched_policy.chunk_hash_to_init_timestamp.clear()
            sample = run_hot_sample(
                buffer,
                keys,
                torch.Size([args.chunk_size * TOKEN_DIMS[group]]),
                fmt,
                hot_policy,
                batched_policy,
            )
            if repeat >= args.warmup:
                hot_samples.append(sample)
        hot_results[group] = medians(hot_samples)

        tensor_access_samples = []
        for repeat in range(args.warmup + args.repeats):
            gc.collect()
            sample = run_tensor_access_sample(
                buffer,
                keys,
                args.chunk_size,
                args.num_tokens,
                group,
            )
            if repeat >= args.warmup:
                tensor_access_samples.append(sample)
        tensor_access_results[group] = medians(tensor_access_samples)

        page_samples = []
        for repeat in range(args.warmup + args.repeats):
            gc.collect()
            sample = run_layer_page_sample(
                buffer,
                chunks=chunks,
                layers=args.num_layers,
                shape=torch.Size([args.chunk_size * TOKEN_DIMS[group]]),
                fmt=fmt,
            )
            if repeat >= args.warmup:
                page_samples.append(sample)
        layer_page_results[group] = medians(page_samples)
        del buffer
        gc.collect()

    print_results(results, hot_results, tensor_access_results, layer_page_results)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "num_tokens": args.num_tokens,
                "chunk_size": args.chunk_size,
                "num_layers": args.num_layers,
                "kv_groups": groups,
                "variants": variants,
                "warmup": args.warmup,
                "repeats": args.repeats,
            },
            "results": results,
            "hot_results": hot_results,
            "tensor_access_results": tensor_access_results,
            "layer_page_results": layer_page_results,
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
