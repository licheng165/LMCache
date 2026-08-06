# SPDX-License-Identifier: Apache-2.0
"""Opt-in structured timing for LMCache cold retrieval."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import time
from typing import Any

COLD_START_PERF_ENV = "LMCACHE_COLD_START_PERF"
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_PERF_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "lmcache_cold_start_perf_context", default=None
)


def cold_start_perf_enabled() -> bool:
    return os.environ.get(COLD_START_PERF_ENV, "0").strip().lower() not in (
        _FALSE_VALUES
    )


def cold_start_perf_now() -> float:
    return time.perf_counter()


@contextmanager
def cold_start_perf_scope(**fields: Any) -> Iterator[None]:
    """Attach correlation fields to nested cold-perf events in this task."""
    current = _PERF_CONTEXT.get() or {}
    token = _PERF_CONTEXT.set({**current, **fields})
    try:
        yield
    finally:
        _PERF_CONTEXT.reset(token)


def cold_start_perf_log(
    logger,
    event: str,
    *,
    started: float | None = None,
    **fields: Any,
) -> None:
    if not cold_start_perf_enabled():
        return
    if started is not None:
        fields["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    payload = {
        "schema": 1,
        "event": event,
        "pid": os.getpid(),
        "monotonic_ms": round(time.perf_counter() * 1000, 3),
        **(_PERF_CONTEXT.get() or {}),
        **fields,
    }
    logger.info(
        "[LMCACHE_COLD_PERF] %s",
        json.dumps(payload, default=str, separators=(",", ":")),
    )
