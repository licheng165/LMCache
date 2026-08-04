#!/usr/bin/env python3
"""Reproduce cross-process Mooncake page lookup failures with CPU payloads.

The producer remains alive after publishing layer-merged pages so its registered
CPU buffer remains valid while an independent consumer performs the same lookup
and retrieval used by layerwise LMCache serving. Older Ascend Mooncake builds
still require a visible device while initializing their transfer engine.
"""

# Standard
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty
from typing import Any
import asyncio
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import uuid

# Third Party
import torch

# First Party
from lmcache.utils import LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.memory_management import LayerPageMemoryObj, MixedMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import mooncake_page_key
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.token_database import ChunkedTokenDatabase


TOKEN_DIMS = {0: 576, 1: 128}


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="LMCache YAML config")
    parser.add_argument("--model", default="/workspace/models/GLM-5.1-w4a8")
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--consumer-delay", type=float, default=1.0)
    parser.add_argument("--process-timeout", type=float, default=120.0)
    parser.add_argument(
        "--client-protocol",
        default="tcp",
        help="Mooncake transport for CPU test clients (default: tcp)",
    )
    parser.add_argument(
        "--mooncake-device",
        default="0",
        help=(
            "device exposed and initialized for Mooncake client setup; use "
            "'none' with a Mooncake build that honors MC_FORCE_TCP before "
            "installing AscendDirect (default: 0)"
        ),
    )
    parser.add_argument(
        "--client-global-segment-size",
        type=int,
        default=0,
        help="Mooncake segment created by each test client (default: 0)",
    )
    parser.add_argument(
        "--prefer-local-alloc",
        action="store_true",
        help="Prefer the test client's segment, as in the serving config",
    )
    parser.add_argument(
        "--output-json", type=Path, help="Optional path for the combined result"
    )
    parser.add_argument(
        "--fail-on-visibility-error",
        action="store_true",
        help="Exit nonzero when the consumer cannot see/retrieve producer pages",
    )
    return parser


def _validate_args(args: Namespace) -> None:
    for name in ("num_layers", "world_size", "chunks"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.consumer_delay < 0 or args.process_timeout <= 0:
        raise ValueError("delays must be non-negative and timeout must be positive")
    if args.client_global_segment_size < 0:
        raise ValueError("--client-global-segment-size cannot be negative")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "Run with PYTHONHASHSEED=0 so both processes build same keys"
        )


def _load_config(args: dict[str, Any]) -> LMCacheEngineConfig:
    config = LMCacheEngineConfig.from_file(args["config"])
    extra = dict(config.extra_config or {})
    extra.update(
        {
            "save_only_first_rank": True,
            "save_chunk_meta": False,
            "mooncake_page_first_multi_buffer": True,
            "mooncake_layer_merged_page_objects": True,
            "mooncake_dsa_raw_token_dims": TOKEN_DIMS,
            "global_segment_size": args["client_global_segment_size"],
            "local_buffer_size": 0,
            "mooncake_prefer_local_alloc": args["prefer_local_alloc"],
            "protocol": args["client_protocol"],
        }
    )
    config.extra_config = extra
    config.local_cpu = True
    config.use_layerwise = True
    config.dsa_two_groups = True
    config.enable_shared_cpu_cache = True
    config.experimental_sampled_layerwise_lookup = True
    return config


def _metadata(args: dict[str, Any], chunk_size: int) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name=args["model"],
        world_size=args["world_size"],
        local_world_size=args["world_size"],
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(args["num_layers"], 1, chunk_size, 1, TOKEN_DIMS[0]),
        use_mla=True,
        role="worker",
        chunk_size=chunk_size,
    )


def _pool_bytes(args: dict[str, Any], chunk_size: int) -> int:
    payload = (
        args["chunks"]
        * args["num_layers"]
        * chunk_size
        * sum(TOKEN_DIMS.values())
        * torch.empty((), dtype=torch.bfloat16).element_size()
    )
    return payload + max(payload // 8, 1 << 20)


def _keys(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    args: dict[str, Any],
) -> dict[int, tuple[list[LayerCacheEngineKey], list[LayerCacheEngineKey]]]:
    tokens = _tokens(config.chunk_size, args["chunks"])
    request_config = _request_config(args)
    database = ChunkedTokenDatabase(config, metadata)
    result = {}
    for group in TOKEN_DIMS:
        chunk_keys = [
            key
            for _, _, key in database.process_tokens(
                tokens, request_configs=request_config, kv_group=group
            )
        ]
        layer_keys = [
            key.get_layer(layer)
            for key in chunk_keys
            for layer in range(args["num_layers"])
        ]
        result[group] = (
            [key.get_layer(0) for key in chunk_keys],
            layer_keys,
        )
    return result


def _tokens(chunk_size: int, chunks: int) -> list[int]:
    return [
        (index * 17 + 11) % 32000
        for index in range(chunk_size * chunks)
    ]


def _request_config(args: dict[str, Any]) -> dict[str, str]:
    return {"lmcache.tag.mooncake_cpu_repro": args["run_id"]}


async def _open_client(args: dict[str, Any]):
    # The explicit LMCache config must win over any unrelated shell setting.
    os.environ.pop("MOONCAKE_CONFIG_PATH", None)
    if args["client_protocol"] == "tcp":
        os.environ["MC_FORCE_TCP"] = "1"
    config = _load_config(args)
    metadata = _metadata(args, config.chunk_size)
    allocator = MixedMemoryAllocator(_pool_bytes(args, config.chunk_size))
    backend = LocalCPUBackend(
        config, metadata, dst_device="cpu", memory_allocator=allocator
    )
    connector = None
    try:
        connector = MooncakestoreConnector(
            "", 0, "", asyncio.get_running_loop(), backend, config
        )
        if connector.registered_buffer_ptr is None:
            device = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
            raise RuntimeError(
                "Mooncake failed to register the CPU test buffer "
                f"(ASCEND_RT_VISIBLE_DEVICES={device!r})"
            )
    except Exception:
        if connector is not None:
            await connector.close()
        backend.close()
        raise
    return config, metadata, allocator.buffer, backend, connector


def _prepare_child_environment(args: dict[str, Any]) -> None:
    device = args["mooncake_device"]
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "" if device == "none" else device


def _initialize_mooncake_device(args: dict[str, Any]) -> None:
    _prepare_child_environment(args)
    if args["mooncake_device"] != "none":
        import torch_npu

        torch_npu.npu.set_device(0)

        # The serving plugin installs Ascend's aclrtMallocHost allocator before
        # LMCache creates its CPU slab. This standalone script imports LMCache
        # earlier, so update the already-loaded allocator module explicitly.
        import lmcache_ascend  # noqa: F401
        import lmcache_ascend.c_ops as ascend_c_ops
        import lmcache.v1.memory_management as memory_management

        memory_management.lmc_ops = ascend_c_ops


def _pattern(group: int, chunk: int, layer: int) -> int:
    return (group * 97 + chunk * 29 + layer * 7 + 1) % 251 + 1


def _fill_pages(pages: list[LayerPageMemoryObj], group: int) -> None:
    for chunk, page in enumerate(pages):
        raw = page.raw_data
        for layer in range(page.num_layers):
            begin = page.group_prefix_sum[layer]
            end = page.group_prefix_sum[layer + 1]
            raw[begin:end].fill_(_pattern(group, chunk, layer))


def _verify_pages(
    pages: list[LayerPageMemoryObj], group: int
) -> list[dict[str, int]]:
    mismatches = []
    for chunk, page in enumerate(pages):
        raw = page.raw_data
        for layer in range(page.num_layers):
            begin = page.group_prefix_sum[layer]
            end = page.group_prefix_sum[layer + 1]
            expected = _pattern(group, chunk, layer)
            if not bool(torch.all(raw[begin:end] == expected)):
                mismatches.append(
                    {"group": group, "chunk": chunk, "layer": layer}
                )
    return mismatches


def _timed(call, *args):
    started = time.perf_counter()
    value = call(*args)
    return value, round((time.perf_counter() - started) * 1000, 3)


def _lookup_group(
    connector: MooncakestoreConnector,
    representatives: list[LayerCacheEngineKey],
    layer_keys: list[LayerCacheEngineKey],
    num_layers: int,
) -> dict[str, Any]:
    page_keys = [mooncake_page_key(key, num_layers) for key in representatives]
    raw, raw_ms = _timed(connector.store.batch_is_exist, page_keys)
    page_hits, page_ms = _timed(
        connector.batched_contains_layer_pages, representatives
    )
    layer_hits, layer_ms = _timed(connector.batched_contains, layer_keys)
    return {
        "page_keys": page_keys,
        "raw_exists": [int(value) for value in raw],
        "page_hits": page_hits,
        "expected_pages": len(representatives),
        "layer_hits": layer_hits,
        "expected_layer_keys": len(layer_keys),
        "timing_ms": {
            "raw_exists": raw_ms,
            "page_lookup": page_ms,
            "page_aware_layer_lookup": layer_ms,
        },
    }


def _client_config(connector: MooncakestoreConnector) -> dict[str, Any]:
    config = connector.config
    return {
        "local_hostname": config.local_hostname,
        "metadata_server": config.metadata_server,
        "master_server_address": config.master_server_address,
        "protocol": config.protocol,
        "global_segment_size": config.global_segment_size,
        "prefer_local_alloc": config.prefer_local_alloc,
        "force_tcp": os.environ.get("MC_FORCE_TCP"),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "registered_buffer": connector.registered_buffer_ptr is not None,
    }


def _scheduler_lookup(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    args: dict[str, Any],
    page_keys: list[str],
    master_address: str,
) -> dict[str, Any]:
    client = None
    try:
        client = MooncakeLookupClient(config, metadata, master_address)
        raw, raw_ms = _timed(client.store.batch_is_exist, page_keys)
        hit_tokens, lookup_ms = _timed(
            client.lookup,
            _tokens(config.chunk_size, args["chunks"]),
            None,
            _request_config(args),
        )
        return {
            "raw_exists": [int(value) for value in raw],
            "hit_tokens": hit_tokens,
            "expected_tokens": config.chunk_size * args["chunks"],
            "timing_ms": {"raw_exists": raw_ms, "lookup": lookup_ms},
            "setup": {
                "local_hostname": "localhost",
                "metadata_server": "P2PHANDSHAKE",
                "global_segment_size": 0,
                "local_buffer_size": 16 * 1024 * 1024,
                "protocol": "tcp",
                "master_server_address": master_address,
            },
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if client is not None:
            try:
                client.store.close()
            except Exception:
                pass


async def _producer(args: dict[str, Any], queue, stop: Event) -> None:
    backend = connector = None
    pages_by_group: list[LayerPageMemoryObj] = []
    try:
        config, metadata, _owner, backend, connector = await _open_client(args)
        keys_by_group = _keys(config, metadata, args)
        groups = {}
        for group, (representatives, layer_keys) in keys_by_group.items():
            shapes, dtypes, fmt, _ = connector._metadata_for_raw_key(
                representatives[0]
            )
            pages = backend.batched_allocate_layer_pages(
                shapes,
                dtypes,
                len(representatives),
                args["num_layers"],
                fmt,
                busy_loop=False,
            )
            if pages is None:
                raise RuntimeError(f"Unable to allocate group {group} pages")
            pages_by_group.extend(pages)
            _fill_pages(pages, group)
            repeated = [page for page in pages for _ in range(args["num_layers"])]
            _, put_ms = await _timed_async(connector.batched_put, layer_keys, repeated)
            lookup = _lookup_group(
                connector, representatives, layer_keys, args["num_layers"]
            )
            lookup["put_ms"] = put_ms
            groups[str(group)] = lookup
        queue.put(
            {
                "role": "producer",
                "status": "ready",
                "pid": os.getpid(),
                "client": _client_config(connector),
                "groups": groups,
            }
        )
        await asyncio.to_thread(stop.wait)
    except BaseException as exc:
        queue.put(_error("producer", exc))
    finally:
        try:
            if connector is not None:
                await connector.close()
        finally:
            for page in pages_by_group:
                if page.is_valid():
                    page.ref_count_down()
            if backend is not None:
                backend.close()


async def _timed_async(call, *args):
    started = time.perf_counter()
    value = await call(*args)
    return value, round((time.perf_counter() - started) * 1000, 3)


async def _consumer(args: dict[str, Any], queue) -> None:
    backend = connector = None
    try:
        config, metadata, _owner, backend, connector = await _open_client(args)
        groups = {}
        keys_by_group = _keys(config, metadata, args)
        page_keys = [
            mooncake_page_key(key, args["num_layers"])
            for representatives, _ in keys_by_group.values()
            for key in representatives
        ]
        scheduler = _scheduler_lookup(
            config,
            metadata,
            args,
            page_keys,
            connector.config.master_server_address,
        )
        for group, (representatives, layer_keys) in keys_by_group.items():
            result = _lookup_group(
                connector, representatives, layer_keys, args["num_layers"]
            )
            result["get_attempted"] = result["page_hits"] == len(representatives)
            result["get_ms"] = None
            result["get_error"] = None
            result["mismatches"] = []
            if result["get_attempted"]:
                pages = []
                try:
                    pages, result["get_ms"] = await _timed_async(
                        connector.batched_get_layer_pages, representatives
                    )
                    result["retrieved_pages"] = len(pages)
                    result["mismatches"] = _verify_pages(pages, group)
                except Exception as exc:
                    result["retrieved_pages"] = 0
                    result["get_error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    for page in pages:
                        page.ref_count_down()
            else:
                result["retrieved_pages"] = 0
            groups[str(group)] = result
        queue.put(
            {
                "role": "consumer",
                "status": "done",
                "pid": os.getpid(),
                "client": _client_config(connector),
                "groups": groups,
                "scheduler": scheduler,
            }
        )
    except BaseException as exc:
        queue.put(_error("consumer", exc))
    finally:
        try:
            if connector is not None:
                await connector.close()
        finally:
            if backend is not None:
                backend.close()


def _error(role: str, exc: BaseException) -> dict[str, Any]:
    return {
        "role": role,
        "status": "error",
        "pid": os.getpid(),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def _producer_entry(args: dict[str, Any], queue, stop: Event) -> None:
    try:
        _initialize_mooncake_device(args)
    except BaseException as exc:
        queue.put(_error("producer", exc))
        return
    asyncio.run(_producer(args, queue, stop))


def _consumer_entry(args: dict[str, Any], queue) -> None:
    try:
        _initialize_mooncake_device(args)
    except BaseException as exc:
        queue.put(_error("consumer", exc))
        return
    asyncio.run(_consumer(args, queue))


def classify_result(producer: dict[str, Any], consumer: dict[str, Any]) -> str:
    """Return a stable diagnosis for a completed producer/consumer run."""
    if producer.get("status") != "ready" or consumer.get("status") != "done":
        return "infrastructure_error"
    producer_groups = producer["groups"]
    consumer_groups = consumer["groups"]
    if any(
        group["page_hits"] != group["expected_pages"]
        for group in producer_groups.values()
    ):
        return "producer_put_not_visible"
    if any(
        group["page_hits"] != group["expected_pages"]
        or group["layer_hits"] != group["expected_layer_keys"]
        for group in consumer_groups.values()
    ):
        return "cross_process_visibility_failure"
    scheduler = consumer.get("scheduler")
    if scheduler:
        if scheduler.get("error"):
            return "scheduler_lookup_client_error"
        if scheduler["hit_tokens"] != scheduler["expected_tokens"]:
            return "scheduler_lookup_client_failure"
    if any(
        group["retrieved_pages"] != group["expected_pages"]
        for group in consumer_groups.values()
    ):
        return "lookup_visible_get_failed"
    if any(group["mismatches"] for group in consumer_groups.values()):
        return "payload_mismatch"
    return "ok"


def _receive(queue, role: str, timeout: float) -> dict[str, Any]:
    try:
        result = queue.get(timeout=timeout)
    except Empty:
        return {"role": role, "status": "error", "error": "process timeout"}
    if result.get("role") != role:
        return {
            "role": role,
            "status": "error",
            "error": f"received unexpected {result.get('role')} result",
        }
    return result


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    shared_args = vars(args).copy()
    shared_args.pop("output_json")
    shared_args.pop("fail_on_visibility_error")
    shared_args["run_id"] = uuid.uuid4().hex

    # Spawned children must inherit visibility before importing the Ascend stack.
    _prepare_child_environment(shared_args)
    context = mp.get_context("spawn")
    queue = context.Queue()
    stop = context.Event()
    producer_process = context.Process(
        target=_producer_entry, args=(shared_args, queue, stop), name="producer"
    )
    consumer_process = context.Process(
        target=_consumer_entry, args=(shared_args, queue), name="consumer"
    )
    producer_process.start()
    producer = _receive(queue, "producer", args.process_timeout)
    consumer = {"role": "consumer", "status": "skipped"}
    try:
        if producer.get("status") == "ready":
            time.sleep(args.consumer_delay)
            consumer_process.start()
            consumer = _receive(queue, "consumer", args.process_timeout)
            consumer_process.join(timeout=5)
    finally:
        stop.set()
        producer_process.join(timeout=10)
        for process in (consumer_process, producer_process):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join()

    verdict = classify_result(producer, consumer)
    result = {
        "schema": 1,
        "verdict": verdict,
        "run_id": shared_args["run_id"],
        "config": {
            "path": args.config,
            "model": args.model,
            "world_size": args.world_size,
            "num_layers": args.num_layers,
            "chunks": args.chunks,
            "consumer_delay": args.consumer_delay,
            "client_global_segment_size": args.client_global_segment_size,
            "client_protocol": args.client_protocol,
            "mooncake_device": args.mooncake_device,
            "prefer_local_alloc": args.prefer_local_alloc,
        },
        "producer": producer,
        "consumer": consumer,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if verdict == "infrastructure_error":
        return 1
    if args.fail_on_visibility_error and verdict != "ok":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
