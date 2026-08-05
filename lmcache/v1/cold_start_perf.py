# SPDX-License-Identifier: Apache-2.0
"""Opt-in structured timing for LMCache cold retrieval."""

# Standard
import json
import os
import time
from typing import Any

COLD_START_PERF_ENV = "LMCACHE_COLD_START_PERF"
_FALSE_VALUES = {"", "0", "false", "no", "off"}


def cold_start_perf_enabled() -> bool:
    return os.environ.get(COLD_START_PERF_ENV, "0").strip().lower() not in (
        _FALSE_VALUES
    )


def cold_start_perf_now() -> float:
    return time.perf_counter()


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
        **fields,
    }
    logger.info(
        "[LMCACHE_COLD_PERF] %s",
        json.dumps(payload, default=str, separators=(",", ":")),
    )
