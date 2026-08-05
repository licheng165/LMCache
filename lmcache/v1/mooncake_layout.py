# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.utils import CacheEngineKey


def mooncake_page_key(key: CacheEngineKey, num_layers: int) -> str:
    """Return the versioned Mooncake key for one all-layer token page.

    The key intentionally excludes ``layer_id`` while retaining the model,
    worker, chunk hash, dtype, KV group, and request tags of the source key.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    # Explicit base dispatch omits LayerCacheEngineKey.layer_id without
    # rebuilding and revalidating an equivalent chunk key.
    chunk_key = CacheEngineKey.to_string(key)
    return f"__lmcache_page_v1__@{num_layers}@{chunk_key}"


def mooncake_page_layout_enabled(config: object) -> bool:
    """Return whether page-first Mooncake multi-buffer storage is enabled."""
    extra_config = getattr(config, "extra_config", None) or {}
    return bool(extra_config.get("mooncake_page_first_multi_buffer", False))
