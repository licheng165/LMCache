# SPDX-License-Identifier: Apache-2.0

# Standard
import os
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from benchmarks.storage_backend_io.mooncake_page_lookup_repro import (
    _client_protocol,
    _load_config,
    _prepare_child_environment,
    _role_device,
    classify_result,
    lmcache_config,
)


def _group(
    page_hits=2,
    layer_hits=72,
    retrieved=2,
    mismatches=None,
    get_attempted=True,
):
    return {
        "page_hits": page_hits,
        "expected_pages": 2,
        "layer_hits": layer_hits,
        "expected_layer_keys": 72,
        "retrieved_pages": retrieved,
        "mismatches": mismatches or [],
        "get_attempted": get_attempted,
    }


@pytest.mark.parametrize(
    ("producer", "consumer", "expected"),
    [
        (
            {"status": "ready", "groups": {"0": _group(page_hits=0)}},
            {"status": "done", "groups": {"0": _group()}},
            "producer_put_not_visible",
        ),
        (
            {"status": "ready", "groups": {"0": _group(layer_hits=0)}},
            {"status": "done", "groups": {"0": _group()}},
            "producer_put_not_visible",
        ),
        (
            {"status": "ready", "groups": {"0": _group()}},
            {"status": "done", "groups": {"0": _group(page_hits=0)}},
            "cross_process_visibility_failure",
        ),
        (
            {"status": "ready", "groups": {"0": _group()}},
            {
                "status": "done",
                "groups": {"0": _group()},
                "scheduler": {"error": "setup failed"},
            },
            "scheduler_lookup_client_error",
        ),
        (
            {"status": "ready", "groups": {"0": _group()}},
            {
                "status": "done",
                "groups": {"0": _group()},
                "scheduler": {"hit_tokens": 0, "expected_tokens": 512},
            },
            "scheduler_lookup_client_failure",
        ),
        (
            {"status": "ready", "groups": {"0": _group()}},
            {"status": "done", "groups": {"0": _group(retrieved=1)}},
            "lookup_visible_get_failed",
        ),
        (
            {"status": "ready", "groups": {"0": _group()}},
            {
                "status": "done",
                "groups": {"0": _group(mismatches=[{"layer": 1}])},
            },
            "payload_mismatch",
        ),
        (
            {"status": "ready", "groups": {"0": _group()}},
            {"status": "done", "groups": {"0": _group()}},
            "ok",
        ),
    ],
)
def test_classify_result(producer, consumer, expected):
    assert classify_result(producer, consumer) == expected


def test_classify_infrastructure_error():
    assert classify_result({"status": "error"}, {"status": "done"}) == (
        "infrastructure_error"
    )


def test_classify_lookup_only_does_not_require_retrieval():
    producer = {"status": "ready", "groups": {"0": _group()}}
    consumer = {
        "status": "done",
        "groups": {"0": _group(retrieved=0, get_attempted=False)},
    }
    assert classify_result(producer, consumer) == "ok"


@pytest.mark.parametrize(("device", "expected"), [("3", "3,4"), ("none", "")])
def test_prepare_child_environment(monkeypatch, device, expected):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    _prepare_child_environment({"mooncake_device": device})
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == expected


def test_prepare_consumer_environment(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    _prepare_child_environment(
        {"mooncake_device": "0", "consumer_mooncake_device": "1"},
        "consumer",
    )
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize(
    ("producer", "consumer", "expected"),
    [("0", "auto", "1"), ("3", "5", "5"), ("none", "auto", "none")],
)
def test_consumer_device_selection(producer, consumer, expected):
    assert (
        _role_device(
            {
                "mooncake_device": producer,
                "consumer_mooncake_device": consumer,
            },
            "consumer",
        )
        == expected
    )


@pytest.mark.parametrize(
    ("protocol", "device", "expected"),
    [
        ("auto", "0", "ascend"),
        ("auto", "none", "tcp"),
        ("tcp", "0", "tcp"),
    ],
)
def test_client_protocol(protocol, device, expected):
    assert (
        _client_protocol(
            {"client_protocol": protocol, "mooncake_device": device}
        )
        == expected
    )


def test_load_config_uses_current_config_class(monkeypatch):
    loaded = SimpleNamespace(extra_config={})

    class CurrentConfig:
        @classmethod
        def from_file(cls, path):
            assert path == "config.yaml"
            return loaded

    monkeypatch.setattr(lmcache_config, "LMCacheEngineConfig", CurrentConfig)
    config = _load_config(
        {
            "config": "config.yaml",
            "client_global_segment_size": 0,
            "prefer_local_alloc": False,
            "client_protocol": "tcp",
        }
    )

    assert config is loaded
    assert config.extra_config["mooncake_layer_merged_page_objects"] is True
