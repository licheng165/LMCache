# SPDX-License-Identifier: Apache-2.0
"""Mooncake-specific remote store completion semantics."""

# Standard
from concurrent.futures import Future, TimeoutError
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.storage_backend.connector import (
    mooncakestore_connector as mooncake_connector,
)
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


def test_layer_page_timeout_releases_late_result(monkeypatch) -> None:
    page = _MemoryObj()

    class _Connection:
        @staticmethod
        async def batched_get_layer_pages(keys):
            return [page]

    class _LateFuture:
        callback = None
        complete = False

        def result(self, timeout=None):
            if not self.complete:
                raise TimeoutError
            return [page]

        def add_done_callback(self, callback):
            self.callback = callback

        def cancel(self):
            raise AssertionError("timed-out transfer must finish for safe cleanup")

    late = _LateFuture()

    def submit(coroutine, loop):
        coroutine.close()
        return late

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    backend = object.__new__(RemoteBackend)
    backend.connection = _Connection()
    backend.loop = object()
    backend.config = SimpleNamespace(blocking_timeout_secs=0.01)
    backend._mla_worker_id_as0_mode = False

    with pytest.raises(TimeoutError):
        backend.batched_get_layer_pages([_layer_key(1, 0)])
    assert page.ref_count == 1
    late.complete = True
    assert late.callback is not None
    late.callback(late)
    assert page.ref_count == 0


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


def test_mooncake_zero_copy_metadata_reuses_homogeneous_group() -> None:
    def metadata_for_key(
        key: CacheEngineKey,
    ) -> tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int]:
        fmt = (
            MemoryFormat.KV_DSA_INDEX_FMT
            if key.kv_group == 1
            else MemoryFormat.KV_MLA_LATENT_FMT
        )
        return ([torch.Size([4])], [torch.float16], fmt, 8)

    metadata = Mock(side_effect=metadata_for_key)
    backend = SimpleNamespace(
        batched_allocate=Mock(
            side_effect=lambda *args, batch_size, **kwargs: [
                _MemoryObj() for _ in range(batch_size)
            ]
        ),
        allocate=Mock(side_effect=lambda *args: _MemoryObj()),
    )
    connector = object.__new__(MooncakestoreConnector)
    connector._metadata_for_raw_key = metadata
    connector.local_cpu_backend = backend
    connector._page_first_multi_buffer = True

    keys = [_layer_key(1, layer) for layer in range(3)]
    memory_objs, key_metadata, mode = connector._allocate_zero_copy_buffers(keys)

    assert metadata.call_count == 1
    assert all(value is key_metadata[0] for value in key_metadata)
    assert len(memory_objs) == len(keys)
    assert mode == "batched"
    assert backend.batched_allocate.call_args.kwargs["address_backed"] is True

    metadata.reset_mock()
    backend.batched_allocate.reset_mock()
    backend.allocate.reset_mock()
    keys[1].kv_group = 1
    _, key_metadata, mode = connector._allocate_zero_copy_buffers(keys)

    assert metadata.call_count == len(keys)
    assert backend.batched_allocate.call_count == 0
    assert backend.allocate.call_count == len(keys)
    assert [value[2] for value in key_metadata] == [
        MemoryFormat.KV_MLA_LATENT_FMT,
        MemoryFormat.KV_DSA_INDEX_FMT,
        MemoryFormat.KV_MLA_LATENT_FMT,
    ]
    assert mode == "individual"


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


def test_mooncake_zero_copy_put_does_not_require_tensor_view() -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector.config = SimpleNamespace(transfer_timeout=1)
    connector.replica_config = object()
    connector._inflight_put_tasks = set()
    connector._page_first_multi_buffer = False
    connector.save_chunk_meta = False
    connector.store = SimpleNamespace(batch_put_from=lambda *args: [0])
    memory_obj = _MemoryObj()
    memory_obj.has_tensor_storage = True
    del memory_obj.raw_tensor

    asyncio.run(connector.batched_put([_key(1)], [memory_obj]))

    assert memory_obj.ref_count == 1


def test_mooncake_page_get_scatter_returns_layer_objects() -> None:
    class _PageStore:
        def batch_get_into_multi_buffers(self, page_keys, ptrs, sizes):
            assert page_keys == ["page-1", "page-2"]
            assert ptrs == [[100, 200], [300, 400]]
            assert sizes == [[16, 16], [16, 16]]
            return [32, 0]

    connector = object.__new__(MooncakestoreConnector)
    connector._page_num_layers = 2
    connector.store = _PageStore()
    memory_objs = [_MemoryObj(16, address) for address in (100, 200, 300, 400)]
    allocated = list(memory_objs)
    connector._allocate_zero_copy_buffers = lambda _keys: (
        memory_objs,
        [],
        "batched",
    )
    keys = [
        _layer_key(1, 0),
        _layer_key(2, 0),
        _layer_key(1, 1),
        _layer_key(2, 1),
    ]
    expected = [memory_objs[0], None, memory_objs[1], None]

    loaded = asyncio.run(
        connector._batch_get_pages(
            keys,
            [("page-1", [0, 2]), ("page-2", [1, 3])],
        )
    )

    assert loaded == expected
    assert [memory_obj.ref_count for memory_obj in allocated] == [1, 1, 0, 0]


def test_mooncake_layer_page_get_allocates_one_object_per_chunk() -> None:
    class _PageStore:
        def __init__(self) -> None:
            self.args = None

        def batch_get_into_multi_buffers(self, *args):
            self.args = args
            return [sum(sizes) for sizes in args[2]]

    class _Backend:
        def __init__(self) -> None:
            self.allocator = TensorMemoryAllocator(
                torch.zeros(16384, dtype=torch.uint8)
            )
            self.submitted = None

        def batched_allocate_layer_pages(self, *args):
            return self.allocator.batched_allocate_layer_pages(*args)

        def batched_submit_layer_pages(self, keys, pages):
            self.submitted = (keys, pages)

    connector = object.__new__(MooncakestoreConnector)
    connector._layer_merged_pages = True
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.local_cpu_backend = _Backend()
    connector.store = _PageStore()
    metadata_calls = []

    def metadata_for_raw_key(key):
        metadata_calls.append(key)
        return (
            [torch.Size([8])],
            [torch.float16],
            MemoryFormat.KV_MLA_LATENT_FMT,
            16,
        )

    connector._metadata_for_raw_key = metadata_for_raw_key
    keys = [_layer_key(chunk_hash, 0) for chunk_hash in (1, 2)]

    pages = asyncio.run(connector.batched_get_layer_pages(keys))

    assert len(pages) == 2
    assert connector.store.args[2] == [[16, 16], [16, 16]]
    assert connector.store.args[1] == [
        [page.layer_data_ptr(0), page.layer_data_ptr(1)] for page in pages
    ]
    assert len(metadata_calls) == 1
    submitted_keys, submitted_pages = connector.local_cpu_backend.submitted
    assert submitted_pages == pages
    assert submitted_keys == [keys[0].without_layer(), keys[1].without_layer()]
    for page in pages:
        page.ref_count_down()


def test_mooncake_page_grouping_serializes_each_page_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 3
    keys = [
        _layer_key(chunk_hash, layer_id)
        for layer_id in range(3)
        for chunk_hash in (1, 2)
    ]
    page_key = Mock(wraps=mooncake_connector.mooncake_page_key)
    monkeypatch.setattr(mooncake_connector, "mooncake_page_key", page_key)

    groups, legacy_indices = connector._complete_page_groups(keys)

    assert legacy_indices == []
    assert len(groups) == 2
    assert sorted(indices for _, indices in groups) == [[0, 2, 4], [1, 3, 5]]
    assert page_key.call_count == 2


def test_mooncake_page_alias_requires_complete_batch() -> None:
    class _Store:
        @staticmethod
        def batch_is_exist(keys):
            return [int(key.startswith("__lmcache_page_v1__")) for key in keys]

        @staticmethod
        def is_exist(key):
            return int(key.startswith("__lmcache_page_v1__"))

    connector = object.__new__(MooncakestoreConnector)
    connector._page_first_multi_buffer = True
    connector._page_num_layers = 2
    connector.store = _Store()
    keys = [_layer_key(1, layer_id) for layer_id in range(2)]

    assert connector.batched_contains(keys[:1]) == 0
    assert connector.batched_contains(keys) == 2
    assert connector.batched_contains_layer_pages(keys[:1]) == 1
    assert not asyncio.run(connector.exists(keys[0]))

    del _Store.batch_is_exist
    assert connector.batched_contains_layer_pages(keys[:1]) == 1


def test_mooncake_page_put_keeps_partial_tail_in_legacy_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    events = []
    monkeypatch.setattr(
        mooncake_connector, "cold_start_perf_enabled", lambda: True
    )
    monkeypatch.setattr(
        mooncake_connector,
        "cold_start_perf_log",
        lambda _logger, event, **fields: events.append((event, fields)),
    )
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
    event, fields = events[0]
    assert event == "mooncake_page_put"
    assert fields["pages"] == 1
    assert fields["buffers"] == 2
    assert fields["bytes"] == 32
    assert fields["kv_groups"] == [0]
    assert fields["first_page_key"] == connector.store.page_args[0][0]
    assert fields["last_page_key"] == connector.store.page_args[0][0]
    assert fields["legacy_objects"] == 2


def test_mooncake_page_put_selects_each_layer_buffer() -> None:
    class _PageStore:
        def __init__(self) -> None:
            self.page_args = None
            self.ref_count_during_put = None

        def batch_put_from_multi_buffers(self, *args):
            self.page_args = args
            self.ref_count_during_put = page.get_ref_count()
            return [0]

    allocator = TensorMemoryAllocator(torch.zeros(16384, dtype=torch.uint8))
    pages = allocator.batched_allocate_layer_pages(
        [torch.Size([8])],
        [torch.float16],
        batch_size=1,
        num_layers=2,
        fmt=MemoryFormat.KV_MLA_LATENT_FMT,
    )
    assert pages is not None
    page = pages[0]
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
    keys = [_layer_key(1, layer_id) for layer_id in range(2)]

    asyncio.run(connector._batched_put_zero_copy(keys, [page, page]))

    assert connector.store.page_args[1] == [
        [page.layer_data_ptr(layer_id) for layer_id in range(2)]
    ]
    assert connector.store.page_args[2] == [[page.layer_size] * 2]
    assert connector.store.ref_count_during_put == 2
    assert page.get_ref_count() == 1
    page.ref_count_down()


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
