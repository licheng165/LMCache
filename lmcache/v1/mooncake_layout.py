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


def mooncake_layer_pages_enabled(config: object) -> bool:
    """Return whether the experimental LocalCPU layer-page layout is enabled."""
    extra_config = getattr(config, "extra_config", None) or {}
    shared = bool(
        getattr(
            config,
            "enable_shared_cpu_cache",
            extra_config.get("enable_shared_cpu_cache", False),
        )
    )
    return (
        shared
        and bool(getattr(config, "use_layerwise", False))
        and bool(extra_config.get("save_only_first_rank", False))
        and str(getattr(config, "remote_url", "")).startswith("mooncakestore://")
        and mooncake_page_layout_enabled(config)
        and bool(extra_config.get("mooncake_layer_merged_page_objects", False))
    )
