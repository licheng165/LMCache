# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, List, Optional, Sequence, Set
import asyncio
import os
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.connector import CreateConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.multilocation_debug import (
    key_to_debug_string,
    record_key_batch_event,
    record_key_event,
)
from lmcache.v1.storage_backend.naive_serde import CreateSerde

logger = init_logger(__name__)


def _pd_diag_enabled() -> bool:
    return os.environ.get("LMCACHE_PD_DIAG", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _pd_diag_key(key: CacheEngineKey) -> str:
    try:
        return key.to_string()
    except Exception:
        return str(key)


def _put_diag_limit(keys: Sequence[CacheEngineKey]) -> Optional[int]:
    if not keys:
        return 4
    layer_id = getattr(keys[0], "layer_id", None)
    return None if layer_id in (None, 0) else 4


class RemoteBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: Optional[LocalCPUBackend],
        dst_device: str = "cuda",
    ):
        super().__init__(dst_device=dst_device)
        self.put_tasks: Set[CacheEngineKey] = set()
        self.lock = threading.Lock()

        assert config.remote_url is not None

        self.remote_url = config.remote_url

        self.local_cpu_backend = local_cpu_backend

        self.loop = loop
        self.config = config
        self.metadata = metadata

        # Re-establish connection only when the connection
        # has been lost for 10 secs
        self.connection: Optional[RemoteConnector] = None
        self.min_reconnect_interval = 10
        self.failure_time = -1000000.0
        self.init_connection()

        assert config.remote_serde is not None
        self.serializer, self.deserializer = CreateSerde(
            config.remote_serde, metadata, config
        )

        # Precompute MLA mode status.  When save_only_first_rank is disabled,
        # workers store distinct per-rank keys, so remote must not collapse
        # non-zero worker ids to rank 0 or skip their remote writes.
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        remote_worker_id_as0 = config.get_extra_config_value(
            "remote_enable_mla_worker_id_as0", save_only_first_rank
        )
        if (
            remote_worker_id_as0
            and metadata.use_mla
            and metadata.world_size > 1
            and not save_only_first_rank
        ):
            logger.warning(
                "remote_enable_mla_worker_id_as0=True with "
                "save_only_first_rank=False is inconsistent for per-rank "
                "MLA store/lookup; non-zero rank remote writes will be "
                "skipped and decoders may miss those ranks."
            )

        self._mla_worker_id_as0_mode = (
            remote_worker_id_as0
            and metadata.use_mla
            and metadata.world_size > 1
            and metadata.worker_id != 0
        )
        logger.info(f"metadata={metadata}")
        logger.info(
            f"Connected to remote storage at {config.remote_url}, "
            f"remote_mla_worker_id_as_0 mode: {self._mla_worker_id_as0_mode}"
        )

        # TODO(Jiayi): If we want to have cache admission policies,
        # we must make decision (whether to send or not) at the local side

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        # NOTE: Health monitoring is now handled at the LMCacheEngine level
        # through HealthMonitor. RemoteBackend no longer manages its own
        # health monitoring. The HealthMonitor in LMCacheEngine will
        # register RemoteBackendHealthCheck for each RemoteBackend.

        self._setup_metrics()

        self._get_blocking_failed_count = 0
        self._put_failed_count = 0

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is not None:
            prometheus_logger.remote_put_task_num.set_function(
                lambda: len(self.put_tasks)
            )
            prometheus_logger.get_blocking_failed_count.set_function(
                lambda: self._get_blocking_failed_count
            )
            prometheus_logger.put_failed_count.set_function(
                lambda: self._put_failed_count
            )

    def __str__(self):
        return self.__class__.__name__

    def init_connection(self):
        # Initialize connection
        if self.connection is not None:
            return
        if (time.time() - self.failure_time) < self.min_reconnect_interval:
            logger.warning(
                "Connection will not be re-established yet "
                "since it has not been long enough since "
                "the last failure"
            )
            return
        try:
            assert self.config.remote_url is not None
            self.connection = CreateConnector(
                self.config.remote_url,
                self.loop,
                self.local_cpu_backend,
                self.config,
                self.metadata,
            )
            logger.info(
                f"Connection initialized/re-established at {self.config.remote_url}"
            )
        except IrrecoverableException:
            logger.error("Irrecoverable error during connection initialization")
            raise
        except Exception as e:
            with self.lock:
                self.failure_time = time.time()
            logger.warning(f"Failed to initialize/re-establish remote connection: {e}")
            self.connection = None

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        if self.connection is None:
            logger.warning("Connection is None in contains, returning False")
            record_key_event("remote_contains_connection_none", key, pin=pin)
            return False

        # For MLA worker id as 0 mode, use worker_id 0
        raw_key = key
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
            record_key_event(
                "remote_contains_key_rewrite",
                raw_key,
                remote_key=key_to_debug_string(key),
                pin=pin,
            )

        try:
            if self.config.extra_config is not None and self.config.extra_config.get(
                "use_exists_sync", False
            ):
                result = self.connection.exists_sync(key)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    self.connection.exists(key), self.loop
                )
                result = future.result()
            record_key_event(
                "remote_contains_result",
                raw_key,
                remote_key=key_to_debug_string(key),
                result=result,
                pin=pin,
            )
            return result
        except Exception as e:
            logger.warning(f"Remote connection failed in contains: {e}")
            logger.warning("Returning False")
            record_key_event(
                "remote_contains_error",
                raw_key,
                remote_key=key_to_debug_string(key),
                error=f"{type(e).__name__}:{e}",
                pin=pin,
            )
            return False

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_contains, returning 0")
            return 0

        if not self.connection.support_batched_contains():
            return super().batched_contains(keys, pin)

        raw_keys = keys
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]
            if raw_keys:
                record_key_event(
                    "remote_batched_contains_key_rewrite",
                    raw_keys[0],
                    remote_first_key=key_to_debug_string(keys[0]),
                    keys_count=len(keys),
                    pin=pin,
                )

        try:
            hit_chunks = self.connection.batched_contains(keys)
            if raw_keys:
                record_key_event(
                    "remote_batched_contains_result",
                    raw_keys[0],
                    remote_first_key=key_to_debug_string(keys[0]),
                    keys_count=len(keys),
                    hit_chunks=hit_chunks,
                    pin=pin,
                )
                if hit_chunks:
                    record_key_batch_event(
                        "remote_batched_contains_hit",
                        raw_keys[:hit_chunks],
                        remote_first_key=key_to_debug_string(keys[0]),
                        keys_count=len(keys),
                        hit_chunks=hit_chunks,
                        pin=pin,
                    )
            return hit_chunks
        except Exception as e:
            logger.warning(f"Remote connection failed in batched_contains: {e}")
            if raw_keys:
                record_key_event(
                    "remote_batched_contains_error",
                    raw_keys[0],
                    remote_first_key=key_to_debug_string(keys[0]),
                    keys_count=len(keys),
                    error=f"{type(e).__name__}:{e}",
                    pin=pin,
                )
            return 0

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.lock:
            return key in self.put_tasks

    def put_callback(
        self,
        future: Future,
        key: CacheEngineKey,
        submit_time: Optional[float] = None,
    ) -> None:
        with self.lock:
            self.put_tasks.discard(key)
        elapsed_ms = (
            (time.perf_counter() - submit_time) * 1000
            if submit_time is not None
            else None
        )
        try:
            future.result()
        except Exception as e:
            self._put_failed_count += 1
            logger.error(f"Put task failed for key {key}: {e}")
            record_key_event(
                "remote_put_error",
                key,
                elapsed_ms=elapsed_ms,
                error=f"{type(e).__name__}:{e}",
            )
            return
        record_key_event("remote_put_done", key, elapsed_ms=elapsed_ms)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to remote storage asynchronously.

        :param on_complete_callback: Optional callback invoked after the remote
            write completes. Callback exceptions are caught and logged.
        """

        def create_immediate_empty_future() -> Future:
            f: Future = Future()
            f.set_result(None)
            return f

        if self.connection is None:
            logger.warning("Connection is None in submit_put_task, returning None")
            record_key_event("remote_put_connection_none", key)
            return create_immediate_empty_future()

        # If MLA worker id as 0 mode is enabled, skip put tasks
        if self._mla_worker_id_as0_mode:
            record_key_event("remote_put_skip_mla_worker_id_as0", key)
            return create_immediate_empty_future()

        if self.exists_in_put_tasks(key):
            record_key_event("remote_put_skip_duplicate", key)
            return create_immediate_empty_future()

        memory_obj.ref_count_up()

        with self.lock:
            self.put_tasks.add(key)

        record_key_event(
            "remote_put_serialize_begin",
            key,
            memory_obj_type=type(memory_obj).__name__,
        )
        compressed_memory_obj = self.serializer.serialize(memory_obj)
        memory_obj.ref_count_down()
        record_key_event("remote_put_submit", key)

        def put_done_callback(f: Future) -> None:
            self.put_callback(f, key, submit_time)
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(f"on_complete_callback failed for key {key}: {e}")

        # NOTE: No need to do error handling here
        # since the `future` is never waited
        submit_time = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(
            self.connection.put(key, compressed_memory_obj), self.loop
        )
        future.add_done_callback(put_done_callback)
        return future

    def batched_put_callback(
        self,
        future: Future,
        keys: List[CacheEngineKey],
        submit_time: Optional[float] = None,
    ) -> None:
        """
        Callback function for batched put tasks.
        """
        with self.lock:
            self.put_tasks.difference_update(keys)
        elapsed_ms = (
            (time.perf_counter() - submit_time) * 1000
            if submit_time is not None
            else None
        )
        limit = _put_diag_limit(keys)
        try:
            exc = future.exception()
        except Exception as err:
            exc = err
        if exc is not None:
            self._put_failed_count += 1
            record_key_batch_event(
                "remote_batched_put_error",
                keys,
                limit=limit,
                keys_count=len(keys),
                elapsed_ms=elapsed_ms,
                error=f"{type(exc).__name__}:{exc}",
            )
            return
        record_key_batch_event(
            "remote_batched_put_done",
            keys,
            limit=limit,
            keys_count=len(keys),
            elapsed_ms=elapsed_ms,
        )

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to remote storage.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's write completes (not once per batch).
        """
        if self.connection is None:
            logger.warning(
                "Connection is None in batched_submit_put_task, returning None"
            )
            record_key_batch_event(
                "remote_batched_put_connection_none",
                keys,
                limit=_put_diag_limit(keys),
                keys_count=len(keys),
                memory_objs_count=len(memory_objs),
            )
            return
        if self.connection.support_batched_put():
            if self._mla_worker_id_as0_mode:
                record_key_batch_event(
                    "remote_batched_put_skip_mla_worker_id_as0",
                    keys,
                    limit=_put_diag_limit(keys),
                    keys_count=len(keys),
                    memory_objs_count=len(memory_objs),
                )
                return

            key_list = list(keys)
            put_diag_limit = _put_diag_limit(key_list)
            record_key_batch_event(
                "remote_batched_put_begin",
                key_list,
                limit=put_diag_limit,
                keys_count=len(key_list),
                memory_objs_count=len(memory_objs),
                transfer_spec_type=type(transfer_spec).__name__,
            )

            # First, increment reference counts for all objects
            for memory_obj in memory_objs:
                memory_obj.ref_count_up()

            compressed_memory_objs = []
            try:
                for memory_obj in memory_objs:
                    compressed_memory_objs.append(self.serializer.serialize(memory_obj))
            except Exception as exc:
                record_key_batch_event(
                    "remote_batched_put_serialize_error",
                    key_list,
                    limit=put_diag_limit,
                    keys_count=len(key_list),
                    memory_objs_count=len(memory_objs),
                    error=f"{type(exc).__name__}:{exc}",
                )
                raise
            finally:
                # Always decrement reference counts for all objects,
                # regardless of whether serialization succeeded or failed
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()

            submit_time = time.perf_counter()
            if _pd_diag_enabled():
                logger.info(
                    "[LMC_REMOTE_PUT_SUBMIT] key_count=%d object_count=%d "
                    "first_key=%s last_key=%s",
                    len(key_list),
                    len(memory_objs),
                    _pd_diag_key(key_list[0]) if key_list else None,
                    _pd_diag_key(key_list[-1]) if key_list else None,
                )

            def batched_done_callback(f: Future) -> None:
                self.batched_put_callback(f, key_list, submit_time)
                if _pd_diag_enabled():
                    elapsed_ms = (time.perf_counter() - submit_time) * 1000
                    try:
                        exc = f.exception()
                    except Exception as err:
                        exc = err
                    logger.info(
                        "[LMC_REMOTE_PUT_DONE] key_count=%d elapsed_ms=%.4f "
                        "exception=%s first_key=%s last_key=%s",
                        len(key_list),
                        elapsed_ms,
                        repr(exc) if exc else None,
                        _pd_diag_key(key_list[0]) if key_list else None,
                        _pd_diag_key(key_list[-1]) if key_list else None,
                    )
                # Invoke per-key callback for each key in the batch
                if on_complete_callback is not None:
                    for key in key_list:
                        try:
                            on_complete_callback(key)
                        except Exception as e:
                            logger.warning(
                                f"on_complete_callback failed for key {key}: {e}"
                            )

            record_key_batch_event(
                "remote_batched_put_submit",
                key_list,
                limit=put_diag_limit,
                keys_count=len(key_list),
                memory_objs_count=len(memory_objs),
            )
            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_put(key_list, compressed_memory_objs),  # type: ignore
                self.loop,
            )
            future.add_done_callback(batched_done_callback)
        else:
            record_key_batch_event(
                "remote_batched_put_fallback_single",
                keys,
                limit=_put_diag_limit(keys),
                keys_count=len(keys),
                memory_objs_count=len(memory_objs),
            )
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                self.submit_put_task(
                    key, memory_obj, on_complete_callback=on_complete_callback
                )

    @_lmcache_nvtx_annotate
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in get_blocking "
                "(likely scheduler role), returning None"
            )
            return None

        if self.connection is None:
            logger.warning("Connection is None in get_blocking, returning None")
            return None
        # For MLA worker id as 0 mode, use worker_id 0
        if self._mla_worker_id_as0_mode:
            key = key.with_new_worker_id(0)
        t1 = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)

        try:
            memory_obj = future.result(self.config.blocking_timeout_secs)
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning("get blocking timeout, trigger cancel the future task")
                future.cancel()
            logger.warning("Error occurred in get_blocking: %s, return None", e)
            memory_obj = None

        t2 = time.perf_counter()
        self.stats_monitor.update_interval_remote_time_to_get_sync((t2 - t1) * 1000)
        if memory_obj is None:
            self._get_blocking_failed_count += 1
            return None
        decompressed_memory_obj = self.deserializer.deserialize(memory_obj)
        t3 = time.perf_counter()
        logger.debug(
            "Get takes %.6f msec, deserialization takes %.6f msec",
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
        )
        return decompressed_memory_obj

    @property
    def get_blocking_failed_count(self):
        return self._get_blocking_failed_count

    @property
    def put_failed_count(self):
        return self._put_failed_count

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_blocking "
                "(likely scheduler role), returning None list"
            )
            return [None] * len(keys)

        if self.connection is None:
            logger.warning("Connection is None in batched_get_blocking, returning None")
            return [None] * len(keys)

        # For MLA worker id as 0 mode, use worker_id 0
        raw_keys = keys
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]
            if raw_keys:
                record_key_event(
                    "remote_batched_get_key_rewrite",
                    raw_keys[0],
                    remote_first_key=key_to_debug_string(keys[0]),
                    keys_count=len(keys),
                )

        t1 = time.perf_counter()
        # batched get
        if self.connection.support_batched_get():
            future = asyncio.run_coroutine_threadsafe(
                self.connection.batched_get(keys), self.loop
            )
            try:
                memory_objs = future.result(self.config.blocking_timeout_secs)
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logger.warning(
                        "batched get blocking timeout, trigger cancel the future task"
                    )
                    future.cancel()
                else:
                    logger.warning(
                        f"Error occurred in batched_get_blocking: {e}, "
                        f"returning None list"
                    )
                memory_objs = [None] * len(keys)
        else:
            remote_backend_individual_get_stats: dict[
                CacheEngineKey, dict[str, float]
            ] = {}
            retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
            if retrieve_stats is not None:
                retrieve_stats.detailed_metrics[
                    "remote_backend_individual_get_stats"
                ] = remote_backend_individual_get_stats

            futures = [
                asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)
                for key in keys
            ]
            memory_objs = []
            failed = False
            for fut in futures:
                if not failed:
                    try:
                        memory_obj = fut.result(self.config.blocking_timeout_secs)
                    except Exception as e:
                        failed = True
                        if isinstance(e, TimeoutError):
                            logger.warning(
                                "get blocking timeout, trigger cancel the future task"
                            )
                            fut.cancel()
                        else:
                            logger.warning(
                                f"Error occurred in get_blocking: {e}, returning None"
                            )
                        memory_obj = None
                    memory_objs.append(memory_obj)
                else:
                    memory_objs.append(None)
                    fut.cancel()

        t2 = time.perf_counter()
        duration = t2 - t1
        self.stats_monitor.update_interval_remote_time_to_get_sync(duration * 1000)

        retrieve_stats = self.stats_monitor.get_current_retrieve_stats()
        if retrieve_stats is not None:
            retrieve_stats.detailed_metrics[
                "remote_backend_batched_get_blocking_time"
            ] = (
                retrieve_stats.detailed_metrics.get(
                    "remote_backend_batched_get_blocking_time", 0.0
                )
                + duration
            )
        decompressed_memory_objs: list[Optional[MemoryObj]] = []
        error_happened = False
        for memory_obj in memory_objs:
            if memory_obj is None:
                error_happened = True
                decompressed_memory_objs.append(None)
            else:
                decompressed_memory_objs.append(
                    self.deserializer.deserialize(memory_obj)
                )
        if error_happened:
            self._get_blocking_failed_count += 1

        assert len(decompressed_memory_objs) == len(keys), (
            f"keys length: {len(keys)}, "
            f"decompressed memory objs length: {len(decompressed_memory_objs)}"
        )
        if raw_keys:
            hit_count = sum(1 for obj in decompressed_memory_objs if obj is not None)
            record_key_event(
                "remote_batched_get_result",
                raw_keys[0],
                remote_first_key=key_to_debug_string(keys[0]),
                keys_count=len(keys),
                hit_count=hit_count,
                error_happened=error_happened,
            )
            record_key_batch_event(
                "remote_batched_get_key_result",
                raw_keys,
                remote_first_key=key_to_debug_string(keys[0]),
                keys_count=len(keys),
                hit_count=hit_count,
                error_happened=error_happened,
            )
        return decompressed_memory_objs

    async def support_batched_async_contains(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_async_contains()
        )

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if self.connection is None:
            logger.warning("Connection is None in batched_async_contains, returning 0")
            if keys:
                record_key_event(
                    "remote_batched_async_contains_connection_none",
                    keys[0],
                    lookup_id=lookup_id,
                    keys_count=len(keys),
                    pin=pin,
                )
            return 0
        raw_keys = keys
        if self._mla_worker_id_as0_mode:
            keys = [key.with_new_worker_id(0) for key in keys]
            if raw_keys:
                record_key_event(
                    "remote_batched_async_contains_key_rewrite",
                    raw_keys[0],
                    lookup_id=lookup_id,
                    remote_first_key=key_to_debug_string(keys[0]),
                    keys_count=len(keys),
                    pin=pin,
                )

        try:
            assert self.connection.support_batched_async_contains(), (
                f"Connector {self.connection} does not support batched async contains"
            )
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            hit_chunks = await asyncio.wait_for(
                self.connection.batched_async_contains(lookup_id, keys, pin),
                self.config.blocking_timeout_secs,
            )
            if raw_keys:
                record_key_event(
                    "remote_batched_async_contains_result",
                    raw_keys[0],
                    lookup_id=lookup_id,
                    remote_first_key=key_to_debug_string(keys[0]),
                    keys_count=len(keys),
                    hit_chunks=hit_chunks,
                    pin=pin,
                )
            return hit_chunks
        except asyncio.TimeoutError:
            logger.warning("batched_async_contains timed out")
            if raw_keys:
                record_key_event(
                    "remote_batched_async_contains_timeout",
                    raw_keys[0],
                    lookup_id=lookup_id,
                    keys_count=len(keys),
                    pin=pin,
                )
            return 0
        except Exception as e:
            logger.warning(f"Error occurred in batched_async_contains: {e}")
            if raw_keys:
                record_key_event(
                    "remote_batched_async_contains_error",
                    raw_keys[0],
                    lookup_id=lookup_id,
                    keys_count=len(keys),
                    error=f"{type(e).__name__}:{e}",
                    pin=pin,
                )
            return 0

    async def support_batched_get_non_blocking(self) -> bool:
        return (
            self.connection is not None
            and self.connection.support_batched_get_non_blocking()
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        # Check if local_cpu_backend is available (required for memory allocation)
        if self.local_cpu_backend is None:
            logger.warning(
                "local_cpu_backend is None in batched_get_non_blocking "
                "(likely scheduler role), returning empty list"
            )
            return []

        if self.connection is None:
            logger.warning(
                "Connection is None in batched_get_non_blocking, returning empty list"
            )
            return []
        try:
            # warning, this timeout will not actually stop the
            # scheduler from waiting for the result
            return await asyncio.wait_for(
                self.connection.batched_get_non_blocking(lookup_id, keys),
                self.config.blocking_timeout_secs,
            )
        except asyncio.TimeoutError:
            logger.warning("batched_get_non_blocking timed out")
            return []
        except Exception as e:
            logger.warning(f"Error occurred in batched_get_non_blocking: {e}")
            return []

    def pin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support pin. "
            "This method is a no-op and will return True."
        )
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        logger.debug(
            "Remote backend does not support unpin. "
            "This method is a no-op and will return True."
        )
        return True

    def remove(self, key, force=True):
        if self.connection is None:
            logger.warning("Connection is None in remove, returning False")
            return False

        try:
            return self.connection.remove_sync(key)
        except Exception as e:
            logger.exception(
                f"Failed to remove key {key} from remote backend, error: {e}"
            )
            return False

    def get_allocator_backend(self):
        assert self.local_cpu_backend is not None, (
            "local_cpu_backend is required for get_allocator_backend, "
            "should not be called in scheduler role"
        )
        return self.local_cpu_backend

    def close(self):
        try:
            assert self.connection is not None
            future = asyncio.run_coroutine_threadsafe(
                self.connection.close(), self.loop
            )
            future.result()
            logger.info("Remote backend closed.")
        except Exception as e:
            logger.warning(f"Error occurred when closing remote connection: {e}")
