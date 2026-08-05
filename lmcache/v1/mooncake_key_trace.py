# SPDX-License-Identifier: Apache-2.0
# Standard
import json
import os
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Sequence

_TRACE_LOCK = Lock()


def trace_mooncake_keys(
    operation: str,
    keys: Sequence[str],
    results: Any = None,
    **context: Any,
) -> None:
    """Append one timestamped JSON record per physical Mooncake key.

    Tracing is disabled unless ``LMCACHE_MOONCAKE_KEY_TRACE_FILE`` is set.
    ``{pid}`` in the configured path is replaced with the current process ID.
    """
    path = os.getenv("LMCACHE_MOONCAKE_KEY_TRACE_FILE")
    if not path or not keys:
        return

    path = path.replace("{pid}", str(os.getpid()))
    timestamp_ns = time.time_ns()
    call_id = f"{os.getpid()}:{time.monotonic_ns()}"
    values = (
        results
        if not isinstance(results, (str, bytes))
        and hasattr(results, "__len__")
        and hasattr(results, "__getitem__")
        else None
    )
    records = []
    for index, key in enumerate(keys):
        result = (
            values[index]
            if values is not None and index < len(values)
            else results
        )
        records.append(
            json.dumps(
                {
                    "schema": 1,
                    "timestamp": datetime.fromtimestamp(
                        timestamp_ns / 1_000_000_000, timezone.utc
                    ).isoformat(timespec="microseconds"),
                    "timestamp_ns": timestamp_ns,
                    "pid": os.getpid(),
                    "call_id": call_id,
                    "operation": operation,
                    "index": index,
                    "count": len(keys),
                    "key": key,
                    "result": result,
                    **context,
                },
                separators=(",", ":"),
                default=str,
            )
        )

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = memoryview(("\n".join(records) + "\n").encode())
        with _TRACE_LOCK:
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                while payload:
                    payload = payload[os.write(fd, payload) :]
            finally:
                os.close(fd)
    except OSError:
        # Diagnostics must never break a production cache operation.
        return
