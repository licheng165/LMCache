# SPDX-License-Identifier: Apache-2.0
"""P1: warm tensor cache vs LocalCPU eviction under memory pressure."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from tests.v1.storage_backend.test_local_cpu_backend import (
    create_test_config,
    create_test_key,
    create_test_metadata,
)
from tests.v1.utils import create_test_memory_obj

pytest.importorskip("lmcache_ascend", reason="Ascend engine helpers required")
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


@pytest.fixture
def tiny_local_cpu_backend(memory_allocator):
    config = create_test_config()
    config.max_local_cpu_size = 0.0005
    PinMonitor.GetOrCreate(config)
    LMCStatsMonitor.GetOrCreate()
    metadata = create_test_metadata()
    backend = LocalCPUBackend(
        config=config,
        metadata=metadata,
        dst_device="cpu",
        memory_allocator=memory_allocator,
    )
    yield backend
    PinMonitor.DestroyInstance()


class TestSparseWarmDegradesOnEviction:
    def test_evicted_key_missing_from_hot_cache_but_tensor_cache_flag_stays(
        self, tiny_local_cpu_backend
    ) -> None:
        backend = tiny_local_cpu_backend
        key = create_test_key("evict_a")
        obj = create_test_memory_obj()
        backend.submit_put_task(key, obj)

        assert backend.contains(key)
        backend.remove(key, force=True)
        assert not backend.contains(key)

        cached_tensors = [obj.tensor, obj.tensor]
        assert AscendLMCacheEngine._has_retrieve_data_cache(cached_tensors, None, 2)

    def test_warm_metadata_keeps_stale_location_when_contains_misses(self) -> None:
        """Documents current behavior: contains() miss does not clear cached location."""
        engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
        engine.storage_manager = MagicMock()
        engine.storage_manager.contains.return_value = None
        engine.retrieve_locations = ["LocalCPUBackend", "RemoteBackend"]
        engine.num_layers = 2

        key_a = create_test_key("warm_a")
        key_b = create_test_key("warm_b")
        cached_keys: list[list] = [[key_a], [key_b]]
        cached_starts = [0]
        cached_ends = [256]
        ret_mask = torch.zeros(256, dtype=torch.bool)
        retrieve_kwargs = {
            "_retrieve_metadata_warm": True,
            "cached_retrieve_location": "LocalCPUBackend",
        }

        location, _, _, _ = engine._ensure_retrieve_chunk_metadata(
            tokens=[0] * 256,
            mask=None,
            request_configs=None,
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs=retrieve_kwargs,
        )

        engine.storage_manager.contains.assert_called_once()
        assert location == "LocalCPUBackend"
        assert retrieve_kwargs["cached_retrieve_location"] == "LocalCPUBackend"


class TestPinnedChunksNotEvictedDuringSparseDecode:
    def test_pin_blocks_eviction_candidates(self, tiny_local_cpu_backend) -> None:
        backend = tiny_local_cpu_backend
        key = create_test_key("pin_a")
        obj = create_test_memory_obj(shape=torch.Size([2, 256, 32, 128]))
        backend.submit_put_task(key, obj)
        stored = backend.hot_cache[key]

        backend.pin(key)
        assert stored.is_pinned
        assert not stored.can_evict

        candidates = backend.cache_policy.get_evict_candidates(
            backend.hot_cache, num_candidates=10
        )
        assert key not in candidates

        backend.unpin(key)
        assert not stored.is_pinned
