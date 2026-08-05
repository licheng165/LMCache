# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# First Party
from lmcache.v1.cold_start_perf import (
    COLD_START_PERF_ENV,
    cold_start_perf_log,
    cold_start_perf_now,
)


class _Logger:
    def __init__(self):
        self.records = []

    def info(self, message, payload):
        self.records.append((message, payload))


def test_cold_start_perf_disabled(monkeypatch):
    monkeypatch.delenv(COLD_START_PERF_ENV, raising=False)
    logger = _Logger()

    cold_start_perf_log(logger, "ignored")

    assert logger.records == []


def test_cold_start_perf_structured_log(monkeypatch):
    monkeypatch.setenv(COLD_START_PERF_ENV, "1")
    logger = _Logger()

    cold_start_perf_log(
        logger,
        "stage",
        started=cold_start_perf_now(),
        req_id="request-1",
    )

    message, raw = logger.records[0]
    payload = json.loads(raw)
    assert message == "[LMCACHE_COLD_PERF] %s"
    assert payload["event"] == "stage"
    assert payload["req_id"] == "request-1"
    assert payload["schema"] == 1
    assert payload["elapsed_ms"] >= 0
