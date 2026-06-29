# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostics for layerwise prefix cache store/load under TP."""

# Standard
import os
from typing import Any

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def prefix_load_diag_enabled() -> bool:
    """Enable with LMCACHE_DIAG_PREFIX_LOAD=1 (also on when LMCACHE_DIAG_SPARSE_TP=1)."""
    if os.environ.get("LMCACHE_DIAG_SPARSE_TP", "").lower() in ("1", "true", "yes"):
        return True
    return os.environ.get("LMCACHE_DIAG_PREFIX_LOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )


def log_prefix_load_diag(fmt: str, *args) -> None:
    if prefix_load_diag_enabled():
        logger.info("[LMCache-Diag-PrefixLoad] " + fmt, *args)


def cache_key_label(key: Any) -> str:
    """Compact CacheEngineKey summary for logs."""
    return (
        f"ws={getattr(key, 'world_size', '?')} "
        f"wid={getattr(key, 'worker_id', '?')} "
        f"hash={getattr(key, 'chunk_hash', '?')}"
    )
