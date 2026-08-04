# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from benchmarks.storage_backend_io.mooncake_page_lookup_repro import (
    classify_result,
)


def _group(page_hits=2, layer_hits=72, retrieved=2, mismatches=None):
    return {
        "page_hits": page_hits,
        "expected_pages": 2,
        "layer_hits": layer_hits,
        "expected_layer_keys": 72,
        "retrieved_pages": retrieved,
        "mismatches": mismatches or [],
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
