# SPDX-License-Identifier: Apache-2.0
# Standard
import json

# First Party
from lmcache.v1.mooncake_key_trace import trace_mooncake_keys


def test_trace_mooncake_keys_records_every_key(tmp_path, monkeypatch) -> None:
    path = tmp_path / "trace-{pid}.jsonl"
    monkeypatch.setenv("LMCACHE_MOONCAKE_KEY_TRACE_FILE", str(path))

    trace_mooncake_keys("lookup", ["key-a", "key-b"], [1, 0], api="test")

    files = list(tmp_path.glob("trace-*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [(record["key"], record["result"]) for record in records] == [
        ("key-a", 1),
        ("key-b", 0),
    ]
    assert records[0]["call_id"] == records[1]["call_id"]
    assert all(record["operation"] == "lookup" for record in records)


def test_trace_mooncake_keys_is_disabled_without_path(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LMCACHE_MOONCAKE_KEY_TRACE_FILE", raising=False)
    trace_mooncake_keys("put", ["key"])
    assert not list(tmp_path.iterdir())
