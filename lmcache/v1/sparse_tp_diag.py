# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostics for sparse attention + tensor parallel issues."""

# Standard
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def sparse_tp_diag_enabled() -> bool:
    """Enable with environment variable LMCACHE_DIAG_SPARSE_TP=1."""
    return os.environ.get("LMCACHE_DIAG_SPARSE_TP", "").lower() in (
        "1",
        "true",
        "yes",
    )


def prefix_load_diag_enabled() -> bool:
    """Also enabled when LMCACHE_DIAG_SPARSE_TP=1."""
    if sparse_tp_diag_enabled():
        return True
    return os.environ.get("LMCACHE_DIAG_PREFIX_LOAD", "").lower() in (
        "1",
        "true",
        "yes",
    )


def log_sparse_tp_diag(fmt: str, *args) -> None:
    if sparse_tp_diag_enabled():
        logger.info("[LMCache-Diag-SparseTP] " + fmt, *args)


def log_prefix_load_diag(fmt: str, *args) -> None:
    # Re-export for callers that already import from sparse_tp_diag.
    if prefix_load_diag_enabled():
        logger.info("[LMCache-Diag-PrefixLoad] " + fmt, *args)
