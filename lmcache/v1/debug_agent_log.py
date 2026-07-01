# SPDX-License-Identifier: Apache-2.0
"""Compact NDJSON debug logger for agent debug sessions."""
from __future__ import annotations

import json
import os
import time
from typing import Any


def agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    try:
        log_path = os.environ.get(
            "LMCACHE_DEBUG_LOG_PATH",
            os.path.join(os.getcwd(), "debug-d9c30c.log"),
        )
        payload = {
            "sessionId": "d9c30c",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass
    # #endregion
