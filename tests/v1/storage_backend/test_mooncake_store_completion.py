# SPDX-License-Identifier: Apache-2.0
"""Mooncake-specific remote store completion semantics."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class _MemoryObj:
    def __init__(self) -> None:
        self.ref_count = 1
        self.raw_tensor = object()
        self.data_ptr = 123

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        self.ref_count -= 1

    def get_size(self) -> int:
        return 16


class _Serializer:
    @staticmethod
    def serialize(memory_obj: _MemoryObj) -> _MemoryObj:
        memory_obj.ref_count_up()
        return memory_obj


class _Connection:
    def __init__(self, requires_completion: bool) -> None:
        self._requires_completion = requires_completion

    @staticmethod
    def support_batched_put() -> bool:
        return True

    def requires_put_completion(self) -> bool:
        return self._requires_completion

    @staticmethod
    async def batched_put(keys, memory_objs) -> None:
        return None


def _key(chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey("test", 1, 0, chunk_hash, torch.float16)


def _make_remote_backend(requires_completion: bool) -> RemoteBackend:
    backend = object.__new__(RemoteBackend)
    backend.connection = _Connection(requires_completion)
    backend.loop = object()
    backend.serializer = _Serializer()
    backend._mla_worker_id_as0_mode = False
    backend.put_tasks = set()
    backend.lock = threading.Lock()
    return backend


def test_remote_backend_returns_only_required_completion(monkeypatch) -> None:
    source_futures: list[Future] = []

    def submit(coroutine, loop) -> Future:
        coroutine.close()
        future: Future = Future()
        source_futures.append(future)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)

    mooncake_future = _make_remote_backend(True).batched_submit_put_task(
        [_key(1)], [_MemoryObj()]
    )
    other_future = _make_remote_backend(False).batched_submit_put_task(
        [_key(2)], [_MemoryObj()]
    )

    assert mooncake_future == [source_futures[0]]
    assert other_future is None
    for future in source_futures:
        future.set_result(None)


@pytest.mark.parametrize("requires_completion", [False, True])
def test_instrumented_put_preserves_connector_failure_policy(
    requires_completion: bool,
) -> None:
    class _FailingConnector:
        @staticmethod
        async def batched_put(keys, memory_objs) -> None:
            raise RuntimeError("write failed")

        @staticmethod
        def requires_put_completion() -> bool:
            return requires_completion

    connector = object.__new__(InstrumentedRemoteConnector)
    connector._connector = _FailingConnector()
    connector._stats_monitor = SimpleNamespace(
        update_interval_remote_time_to_put=lambda value: None,
        update_interval_remote_write_metrics=lambda value: None,
    )
    connector.name = "test"
    memory_obj = _MemoryObj()
    operation = connector.batched_put([_key(1)], [memory_obj])

    if requires_completion:
        with pytest.raises(RuntimeError, match="write failed"):
            asyncio.run(operation)
    else:
        asyncio.run(operation)
    assert memory_obj.ref_count == 0


def test_mooncake_requires_put_completion() -> None:
    connector = object.__new__(MooncakestoreConnector)
    assert connector.requires_put_completion()


def test_mooncake_batch_status_failure_is_not_silenced() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector.store = SimpleNamespace(batch_put_from=lambda *args: [0, -1])

    with pytest.raises(RuntimeError, match="batch_put_from failed"):
        asyncio.run(
            connector._batched_put_zero_copy(
                [_key(1), _key(2)], [_MemoryObj(), _MemoryObj()]
            )
        )


def test_mooncake_timeout_keeps_source_buffer_until_native_put_exits() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector.config = SimpleNamespace(transfer_timeout=0.01)
    connector._inflight_put_tasks = set()
    memory_obj = _MemoryObj()
    release = threading.Event()

    def blocking_put() -> int:
        release.wait()
        return 0

    async def run() -> None:
        try:
            with pytest.raises(TimeoutError, match="timed out"):
                await connector._run_blocking_put(
                    "put_from", blocking_put, (), [memory_obj]
                )
            assert memory_obj.ref_count == 2
        finally:
            release.set()
        while connector._inflight_put_tasks:
            await asyncio.sleep(0.001)

    asyncio.run(run())
    assert memory_obj.ref_count == 1
