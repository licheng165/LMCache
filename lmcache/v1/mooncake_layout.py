# SPDX-License-Identifier: Apache-2.0

# Standard
import hashlib
import json
import os

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata


MOONCAKE_PAYLOAD_LAYOUT_TAG = "lmcache.tag.payload_v2"


def mooncake_payload_layout(
    config: LMCacheEngineConfig, metadata: LMCacheMetadata
) -> tuple[str, dict[str, object]]:
    """Return a stable, scheduler-visible Mooncake payload schema identity."""
    # Local import avoids coupling the lightweight page-key helpers to backend
    # initialization while reusing the connector's authoritative defaulting.
    from lmcache.v1.storage_backend.connector.base_connector import (
        resolve_save_chunk_meta,
    )

    chunk_size = int(config.chunk_size)
    kv_shape = tuple(int(value) for value in metadata.kv_shape)
    dtype = metadata.kv_dtype
    extra_config = config.extra_config or {}
    model_config_hash = ""
    model_config_path = os.path.join(
        metadata.model_name, "config.json"
    )
    try:
        with open(model_config_path, encoding="utf-8") as model_config:
            canonical_model_config = json.dumps(
                json.load(model_config), sort_keys=True, separators=(",", ":")
            )
            model_config_hash = hashlib.blake2b(
                canonical_model_config.encode(), digest_size=8
            ).hexdigest()
    except (OSError, TypeError, ValueError):
        pass
    raw_token_dims = extra_config.get("mooncake_dsa_raw_token_dims", "")
    if isinstance(raw_token_dims, dict):
        raw_token_dims = tuple(
            sorted((str(key), int(value)) for key, value in raw_token_dims.items())
        )
    elif isinstance(raw_token_dims, (list, tuple)):
        raw_token_dims = tuple(int(value) for value in raw_token_dims)
    else:
        raw_token_dims = str(raw_token_dims)
    descriptor = {
        "version": 2,
        "chunk_size": chunk_size,
        "kv_shape": kv_shape,
        "dtype": str(dtype),
        "model_config": model_config_hash,
        "use_mla": metadata.use_mla,
        "use_layerwise": config.use_layerwise,
        "dsa_two_groups": config.dsa_two_groups,
        "save_chunk_meta": resolve_save_chunk_meta(config),
        "raw_token_dims": raw_token_dims,
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(encoded.encode(), digest_size=8).hexdigest(), descriptor


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
