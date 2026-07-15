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
