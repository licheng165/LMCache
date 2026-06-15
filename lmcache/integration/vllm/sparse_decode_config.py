# SPDX-License-Identifier: Apache-2.0
"""Configuration for sparse-decode KV retrieve in the vLLM connector."""

from __future__ import annotations

import os

_DEFAULT_SPARSE_DECODE_RETRIEVE_TOKENS = 2048


def _parse_sparse_decode_retrieve_tokens() -> int:
    raw = os.environ.get(
        "LMCACHE_SPARSE_DECODE_RETRIEVE_TOKENS",
        str(_DEFAULT_SPARSE_DECODE_RETRIEVE_TOKENS),
    )
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_SPARSE_DECODE_RETRIEVE_TOKENS
    return value if value > 0 else _DEFAULT_SPARSE_DECODE_RETRIEVE_TOKENS


# Prompt tokens whose slot_mapping is built/transferred per sparse-decode step.
# Override at process start: export LMCACHE_SPARSE_DECODE_RETRIEVE_TOKENS=4096
SPARSE_DECODE_RETRIEVE_TOKENS: int = _parse_sparse_decode_retrieve_tokens()


def sparse_decode_slot_mapping_tokens(prompt_token_count: int) -> int:
    """Slot-mapping length for sparse decode (capped by prompt size)."""
    if prompt_token_count <= 0:
        return 0
    return min(SPARSE_DECODE_RETRIEVE_TOKENS, prompt_token_count)
