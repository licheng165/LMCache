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


def log_sparse_tp_diag(fmt: str, *args) -> None:
    if sparse_tp_diag_enabled():
        logger.info("[LMCache-Diag-SparseTP] " + fmt, *args)
