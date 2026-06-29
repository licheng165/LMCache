# SPDX-License-Identifier: Apache-2.0
"""Unit tests for prefix-load diagnostic helpers."""

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.prefix_load_diag import cache_key_label
import torch


def test_cache_key_label_includes_worker_and_world_size() -> None:
    key = CacheEngineKey(
        model_name="m",
        world_size=8,
        worker_id=3,
        chunk_hash=12345,
        dtype=torch.bfloat16,
    )
    label = cache_key_label(key)
    assert "ws=8" in label
    assert "wid=3" in label
    assert "hash=12345" in label
