# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from typing import Optional

from lmcache.utils import CacheEngineKey


def first_last_layer_keys(
    group_keys: Sequence[CacheEngineKey],
    num_layers: int,
) -> list[CacheEngineKey]:
    """Return the first and last layer key for each KV group."""
    if num_layers <= 0:
        return []
    layer_ids = (0,) if num_layers == 1 else (0, num_layers - 1)
    sampled: list[CacheEngineKey] = []
    for group_key in group_keys:
        sampled.extend(group_key.get_layer(layer_id) for layer_id in layer_ids)
    return sampled


def find_last_sampled_hit(
    sample_count: int,
    exists_at: Callable[[int], bool],
) -> Optional[int]:
    """Return a candidate tail hit under a contiguous-prefix assumption.

    This helper is advisory: it does not inspect intermediate samples. An
    authoritative caller must revalidate every chunk through the candidate,
    as the production pinned lookup path does.
    """
    if sample_count <= 0 or not exists_at(0):
        return None
    for index in range(sample_count - 1, 0, -1):
        if exists_at(index):
            return index
    return 0
