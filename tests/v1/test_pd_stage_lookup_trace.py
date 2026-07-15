# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from lmcache.integration.vllm import vllm_v1_adapter as adapter_mod
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl


class _AsyncLookupClient:
    def __init__(self) -> None:
        self._cached_results = iter((-1, None, 64))

    def lookup_cache(self, lookup_id: str):
        return next(self._cached_results)

    def lookup(self, token_ids, lookup_id: str, request_configs):
        return None


def test_async_lookup_trace_aggregates_submit_poll_and_pending(monkeypatch):
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.kv_role = "kv_consumer"
    connector._manager = SimpleNamespace(lookup_client=_AsyncLookupClient())
    connector._requests_priority = {}
    connector._pd_lookup_traces = {}
    connector._pd_lookup_sequence = 0
    connector.skip_last_n_tokens = 0
    connector.load_specs = {}
    connector.config = SimpleNamespace(min_retrieve_tokens=0)

    request = SimpleNamespace(
        request_id="request-1",
        num_tokens=100,
        all_token_ids=list(range(100)),
        prompt_token_ids=list(range(100)),
        sampling_params=object(),
        priority=0,
    )

    clock = iter(range(1, 100))
    logs = []
    monkeypatch.setattr(adapter_mod, "_PD_STAGE_TRACE_ENABLED", True)
    monkeypatch.setattr(adapter_mod, "_PD_STAGE_TRACE_EVERY", 1)
    monkeypatch.setattr(adapter_mod, "_PD_STAGE_TRACE_REQUEST_ID", "")
    monkeypatch.setattr(
        adapter_mod.time,
        "perf_counter_ns",
        lambda: next(clock) * 1_000_000,
    )
    monkeypatch.setattr(
        adapter_mod, "extract_mm_features", lambda request: (None, None)
    )
    monkeypatch.setattr(adapter_mod, "extract_request_configs", lambda params: {})
    monkeypatch.setattr(
        adapter_mod.logger, "info", lambda *args, **kwargs: logs.append(args)
    )

    assert connector.get_num_new_matched_tokens(request, 0) is None
    assert connector.get_num_new_matched_tokens(request, 0) is None
    assert connector.get_num_new_matched_tokens(request, 0) == 64

    submit = next(args for args in logs if "phase=submit" in args[0])
    complete = next(args for args in logs if "phase=complete" in args[0])
    assert submit[3] == "async"
    assert complete[3] == 64
    assert complete[4] == "async"
    assert complete[5] == "cache"
    assert complete[8] == 3
    assert complete[13] == 3
    assert connector._pd_lookup_traces == {}


def test_layer_wait_detail_does_not_double_count_payload_children(monkeypatch):
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    logs = []
    trace = {
        "step": 3,
        "requests": "request-1",
        "start_ns": 0,
        "start_load_ms": 1.0,
        "wait_calls": 1,
        "wait_ns": 100_000_000,
        "wait_by_group_ns": {0: 100_000_000},
        "slowest_wait_ns": 100_000_000,
        "slowest_layer": "model.layers.0.self_attn",
        "slowest_group": 0,
        "internal_ns": {
            "adapter_payload_build_g0": 80_000_000,
            "adapter_detail_payload_selected_multi_row_select_g0": 70_000_000,
            "active_connector_send_g0": 10_000_000,
        },
        "internal_counts": {
            "adapter_payload_build_g0": 1,
            "adapter_detail_payload_selected_multi_row_select_g0": 1,
            "active_connector_send_g0": 1,
        },
        "internal_max_ns": {
            "adapter_payload_build_g0": 80_000_000,
            "adapter_detail_payload_selected_multi_row_select_g0": 70_000_000,
            "active_connector_send_g0": 10_000_000,
        },
        "internal_slowest_layer": {
            "adapter_payload_build_g0": "model.layers.0.self_attn",
            "adapter_detail_payload_selected_multi_row_select_g0": (
                "model.layers.0.self_attn"
            ),
            "active_connector_send_g0": "model.layers.0.self_attn",
        },
        "emitted": False,
    }

    monkeypatch.setattr(adapter_mod, "_PD_STAGE_TRACE_ENABLED", True)
    monkeypatch.setattr(adapter_mod.time, "perf_counter_ns", lambda: 200_000_000)
    monkeypatch.setattr(
        adapter_mod.logger, "info", lambda *args, **kwargs: logs.append(args)
    )

    connector._emit_pd_stage_trace_summary(trace)

    detail = next(args for args in logs if "scope=layer_wait_detail" in args[0])
    assert detail[5] == 80.0
    assert detail[6] == 20.0
    assert detail[7:10] == (80.0, 70.0, 10.0)
    assert "adapter_payload_build_g0:80.000" in detail[10]
    assert (
        "adapter_detail_payload_selected_multi_row_select_g0:70.000"
        in detail[11]
    )
    assert "active_connector_send_g0:10.000" in detail[12]
