# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient


class _FakeStore:
    def __init__(self, rets=None):
        self.keys = None
        self.rets = rets

    def batch_is_exist(self, keys):
        self.keys = keys
        return self.rets if self.rets is not None else [1 for _ in keys]


class _FakeTokenDatabase:
    def __init__(self, kv_group=1):
        self.kv_group = kv_group

    def process_tokens(self, token_ids, request_configs=None):
        yield (
            0,
            len(token_ids),
            CacheEngineKey(
                "model",
                1,
                0,
                0xABC,
                torch.bfloat16,
                request_configs=request_configs,
                kv_group=self.kv_group,
            ),
        )

    def _make_key_by_hash(self, chunk_hash, request_configs=None, kv_group=0):
        return CacheEngineKey(
            "model",
            1,
            0,
            chunk_hash,
            torch.bfloat16,
            request_configs=request_configs,
            kv_group=kv_group,
        )


def test_mooncake_lookup_passes_request_configs_to_cache_keys():
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.store = _FakeStore()
    client.token_database = _FakeTokenDatabase()

    hit_tokens = client.lookup(
        [1, 2, 3],
        request_configs={"lmcache.tag.schema": "dsa-index-save-v2"},
    )

    assert hit_tokens == 3
    assert client.store.keys == [
        "model@1@0@abc@bfloat16@1@schema%dsa-index-save-v2"
    ]


def test_mooncake_lookup_requires_dsa_index_group_before_hit():
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(dsa_two_groups=True, use_layerwise=False)
    client.metadata = SimpleNamespace(kv_shape=(2, 1, 256, 1, 1))
    client.store = _FakeStore(rets=[1, 0])
    client.token_database = _FakeTokenDatabase(kv_group=0)

    hit_tokens = client.lookup([1, 2, 3])

    assert hit_tokens == 0
    assert client.store.keys == [
        "model@1@0@abc@bfloat16@0",
        "model@1@0@abc@bfloat16@1",
    ]

    client.store = _FakeStore(rets=[1, 1])
    assert client.lookup([1, 2, 3]) == 3


def test_mooncake_lookup_layerwise_checks_all_layers_and_groups():
    client = MooncakeLookupClient.__new__(MooncakeLookupClient)
    client.config = SimpleNamespace(dsa_two_groups=True, use_layerwise=True)
    client.metadata = SimpleNamespace(kv_shape=(2, 1, 256, 1, 1))
    client.store = _FakeStore(rets=[1, 1, 1, 0])
    client.token_database = _FakeTokenDatabase(kv_group=0)

    assert client.lookup([1, 2, 3]) == 0
    assert client.store.keys == [
        "model@1@0@abc@bfloat16@0@0",
        "model@1@0@abc@bfloat16@0@1",
        "model@1@0@abc@bfloat16@1@0",
        "model@1@0@abc@bfloat16@1@1",
    ]

    client.store = _FakeStore(rets=[1, 1, 1, 1])
    assert client.lookup([1, 2, 3]) == 3
