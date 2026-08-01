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
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.storage_backend.audit_backend import AuditBackend
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class _MemoryObj:
    def __init__(self, size: int = 16, data_ptr: int = 123) -> None:
        self.ref_count = 1
        self.raw_tensor = object()
        self.data_ptr = data_ptr
        self.size = size

    def ref_count_up(self) -> None:
        self.ref_count += 1

    def ref_count_down(self) -> None:
        self.ref_count -= 1

    def get_size(self) -> int:
        return self.size

    def is_valid(self) -> bool:
        return self.ref_count > 0


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


def _layer_key(chunk_hash: int, layer_id: int) -> LayerCacheEngineKey:
    return LayerCacheEngineKey(
        "test",
        1,
        0,
        chunk_hash,
        torch.float16,
        layer_id=layer_id,
    )


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


def test_mooncake_page_layout_capability_survives_wrappers() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.store = SimpleNamespace(is_exist=lambda _key: 1)
    instrumented = object.__new__(InstrumentedRemoteConnector)
    instrumented._connector = connector
    remote = object.__new__(RemoteBackend)
    remote.connection = instrumented
    remote.dst_device = "cpu"
    remote._mla_worker_id_as0_mode = False
    audit = AuditBackend(remote)

    assert audit.uses_page_first_layout()
    assert audit.contains_page(_layer_key(1, 0))


@pytest.mark.parametrize(
    ("status", "expected"),
    [(-900, False), (-1, False), (0, False), (1, True)],
)
def test_mooncake_exists_accepts_only_explicit_hit_status(
    status: int,
    expected: bool,
) -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = False
    connector.store = SimpleNamespace(is_exist=lambda key: status)
    key = _key(1)

    assert asyncio.run(connector.exists(key)) is expected
    assert connector.exists_sync(key) is expected
    assert asyncio.run(connector.batched_async_contains("lookup-1", [key])) == int(
        expected
    )
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    assert connector.exists_page_sync(_layer_key(1, 0)) is expected


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


def test_mooncake_page_contains_hit_gets_complete_layer_page() -> None:
    class _PageStore:
        def batch_is_exist(self, page_keys):
            assert len(page_keys) == 1
            return [1]

        def batch_get_into_multi_buffers(self, page_keys, ptrs, sizes):
            assert len(page_keys) == 1
            assert ptrs == [[100, 300]]
            assert sizes == [[16, 16]]
            return [32]

        @staticmethod
        def batch_get_into(*args):
            pytest.fail("complete page retrieval must not use legacy layer keys")

    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.save_chunk_meta = False
    connector.store = _PageStore()
    memory_objs = [_MemoryObj(16, 100), _MemoryObj(16, 300)]
    connector._allocate_zero_copy_buffers = lambda _keys: (
        memory_objs,
        [],
        "batched",
    )
    keys = [_layer_key(1, 0), _layer_key(1, 1)]

    assert asyncio.run(connector.batched_async_contains("lookup-1", keys)) == 2
    loaded = asyncio.run(connector.batched_get(keys))

    assert loaded == memory_objs
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)


def test_mooncake_page_put_keeps_partial_tail_in_legacy_layout() -> None:
    class _PageStore:
        def __init__(self) -> None:
            self.page_args = None
            self.legacy_args = None

        def batch_put_from_multi_buffers(self, *args):
            self.page_args = args
            return [0]

        def batch_put_from(self, *args):
            self.legacy_args = args
            return [0, 0]

    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector.local_cpu_backend = SimpleNamespace(
        metadata=SimpleNamespace(chunk_size=4)
    )
    connector._metadata_for_raw_key = lambda _key: ([], [], None, 4)
    connector.store = _PageStore()
    keys = [
        _layer_key(1, 0),
        _layer_key(2, 0),
        _layer_key(1, 1),
        _layer_key(2, 1),
    ]
    memory_objs = [
        _MemoryObj(16, 100),
        _MemoryObj(8, 200),
        _MemoryObj(16, 300),
        _MemoryObj(8, 400),
    ]

    asyncio.run(connector._batched_put_zero_copy(keys, memory_objs))

    assert connector.store.page_args[1] == [[100, 300]]
    assert connector.store.page_args[2] == [[16, 16]]
    assert connector.store.legacy_args[1] == [200, 400]
    assert connector.store.legacy_args[2] == [8, 8]
    assert all(memory_obj.ref_count == 1 for memory_obj in memory_objs)


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


def test_mooncake_cancel_keeps_receive_buffer_until_native_get_exits() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._inflight_get_tasks = set()
    connector._inflight_put_tasks = set()
    closed = []
    connector._unregister_cpu_buffer = lambda: None
    connector.store = SimpleNamespace(close=lambda: closed.append(True))
    memory_obj = _MemoryObj()
    release = threading.Event()

    def blocking_get() -> int:
        release.wait()
        return memory_obj.get_size()

    async def run() -> None:
        task = asyncio.create_task(
            connector._run_blocking_get(blocking_get, (), [memory_obj])
        )
        while not connector._inflight_get_tasks:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The caller releases its reference during cancellation cleanup. The
        # native worker's extra reference must keep the destination alive.
        memory_obj.ref_count_down()
        assert memory_obj.ref_count == 1

        close_task = asyncio.create_task(connector.close())
        await asyncio.sleep(0)
        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done()
        assert memory_obj.ref_count == 1
        assert not closed

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert memory_obj.ref_count == 0
        assert closed == [True]

    asyncio.run(run())
