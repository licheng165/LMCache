# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import replace
import hashlib
from types import SimpleNamespace

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

from vllm.v1.core.sched.dsa_types import (
    DSAOperationCommand,
    DSAOperationRef,
    DSAReceiptExpectation,
    DSASourceLease,
    ParticipantIdentity,
    RequestKey,
)

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
    WorkerRetrieveState,
)
from lmcache.v1.cache_engine import LayerwiseStoreResult


def _digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(len(tokens).to_bytes(8, byteorder="big", signed=False))
    for token in tokens:
        digest.update(token.to_bytes(8, byteorder="big", signed=True))
    return digest.hexdigest()


PARTICIPANT = ParticipantIdentity(
    engine_id="engine",
    process_instance_id="worker-0",
    worker_rank=0,
    dp_rank=0,
    pp_rank=0,
    tp_rank=0,
)


def _command(
    request_key: RequestKey,
    operation_id: str,
    *,
    kind: str = "store",
    range_start: int = 0,
    range_end: int = 4,
    route_epoch: int = 0,
    input_generation: str | None = None,
    output_generation: str | None = "generation-1",
    parent_operation_id: str | None = None,
    tokens: list[int] | None = None,
) -> DSAOperationCommand:
    tokens = list(range(range_end)) if tokens is None else tokens
    operation = DSAOperationRef(
        operation_id=operation_id,
        parent_operation_id=parent_operation_id,
        kind=kind,
        obligations=frozenset(
            {"source_activation" if kind == "source_activation" else "promotion"}
        ),
        range_start=range_start,
        range_end=range_end,
        input_generation_id=input_generation,
        output_generation_id=output_generation,
        route_epoch=route_epoch,
    )
    chunks = ((range_start, range_end),)
    receipt_specs = (
        (
            (
                "route_epoch_complete",
                "npu",
                "npu_materialized",
            ),
        )
        if kind == "source_activation"
        else (
            ("storage", "local_cpu", "local_cpu_pinned"),
            ("source_seal", "local_cpu", "local_cpu_pinned"),
        )
    )
    expectations = tuple(
        DSAReceiptExpectation(
            participant=PARTICIPANT,
            receipt_kind=receipt_kind,
            kv_group=0,
            layers=(0,),
            chunks=chunks,
            storage_tier=storage_tier,
            minimum_guarantee=guarantee,
        )
        for receipt_kind, storage_tier, guarantee in receipt_specs
    )
    return DSAOperationCommand(
        request_key=request_key,
        operation=operation,
        accepted_end_at_issue=len(tokens),
        token_prefix_digest=_digest(tokens),
        cache_namespace_fingerprint="namespace",
        expected_receipts=expectations,
    )


class _MemoryObj:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.refs = 1
        self.pins = 0

    def is_valid(self) -> bool:
        return self.refs > 0

    def ref_count_up(self) -> None:
        self.refs += 1

    def ref_count_down(self) -> None:
        self.refs -= 1

    def pin(self) -> bool:
        self.pins += 1
        return True

    def unpin(self) -> bool:
        self.pins -= 1
        return True

    @property
    def is_pinned(self) -> bool:
        return self.pins > 0


class _Engine:
    def __init__(self, *, empty: bool = False, healthy: bool = True) -> None:
        self.metadata = SimpleNamespace(worker_id=0, world_size=1)
        self.num_layers = 1
        self.empty = empty
        self.healthy = healthy
        self.calls: list[dict] = []
        self.unpinned: list[str] = []

    def is_healthy(self) -> bool:
        return self.healthy

    def is_frozen(self) -> bool:
        return False

    def _is_passive(self) -> bool:
        return False

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store_layer(self, token_ids: list[int], **kwargs):
        self.calls.append({"token_ids": list(token_ids), **kwargs})
        mask = kwargs["mask"]
        selected = torch.nonzero(mask, as_tuple=False).reshape(-1)
        start = int(selected[0]) if selected.numel() else len(token_ids)
        end = len(token_ids)
        result = LayerwiseStoreResult(request_id=kwargs["req_id"])
        if not self.empty:
            tensor = torch.arange(end - start, dtype=torch.float32)
            memory_obj = _MemoryObj(tensor)
            result.starts = [start]
            result.ends = [end]
            result.keys = [[object()]]
            result.memory_objs = [[memory_obj]]
            result.tensors = [[tensor]]
            result.chunk_dev_ptrs = [[tensor.data_ptr()]]
            result.chunk_ptrs = [torch.tensor([tensor.data_ptr()], dtype=torch.int64)]

        def storer():
            yield None
            yield result

        return storer()


class _PassiveSharedEngine(_Engine):
    def __init__(
        self,
        *,
        exchange_error: str | None = None,
        fail_kv_group: int | None = None,
    ) -> None:
        super().__init__(empty=True)
        self.exchange_error = exchange_error
        self.fail_kv_group = fail_kv_group
        self.exchange_calls: list[dict] = []
        self.shared_memory_obj: _MemoryObj | None = None
        self.shared_memory_objs: list[_MemoryObj] = []
        self.gpu_connector = SimpleNamespace()

    def _is_passive(self) -> bool:
        return True

    def dsa_store_exchange_role(self) -> str:
        return "passive"

    def exchange_dsa_store_result(self, token_ids: list[int], **kwargs):
        self.exchange_calls.append({"token_ids": list(token_ids), **kwargs})
        if kwargs["source_error"] is not None:
            return None, kwargs["source_error"]
        if self.exchange_error is not None or (
            kwargs["kv_group"] == self.fail_kv_group
        ):
            return None, self.exchange_error or "shared_source_exchange_failed"
        chunks = kwargs["required_chunks"]
        tensor = torch.arange(
            sum(end - start for start, end in chunks),
            dtype=torch.float32,
        )
        self.shared_memory_obj = _MemoryObj(tensor)
        self.shared_memory_objs.append(self.shared_memory_obj)
        return (
            LayerwiseStoreResult(
                request_id=kwargs["request_id"],
                kv_group=kwargs["kv_group"],
                starts=[start for start, _ in chunks],
                ends=[end for _, end in chunks],
                keys=[[object() for _ in chunks]],
                memory_objs=[[self.shared_memory_obj]],
                tensors=[[tensor]],
            ),
            None,
        )


class _Parent:
    def __init__(self, metadata: LMCacheConnectorMetadata) -> None:
        self._connector_metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self._connector_metadata


def _request(command: DSAOperationCommand) -> ReqMeta:
    start = command.operation.range_start
    end = command.operation.range_end
    return ReqMeta(
        req_id=command.request_key.request_id,
        token_ids=list(range(end)),
        slot_mapping=[torch.arange(start, end, dtype=torch.long)],
        save_slot_mapping=[torch.arange(start, end, dtype=torch.long)],
        save_slot_mapping_base=start,
        windowed_sparse_save=True,
        is_last_prefill=True,
        save_spec=SaveSpec(
            skip_leading_tokens=start,
            can_save=True,
            can_save_latent=True,
        ),
        dsa_store_command=command,
        dsa_request_key=command.request_key,
        dsa_route_epoch=command.operation.route_epoch,
    )


def _connector(
    command: DSAOperationCommand,
    *,
    engine: _Engine | None = None,
) -> tuple[LMCacheConnectorV1Impl, LMCacheConnectorMetadata, _Engine]:
    engine = _Engine() if engine is None else engine
    metadata = LMCacheConnectorMetadata(
        requests=[_request(command)],
        dsa_commands=(command,),
    )
    connector = object.__new__(LMCacheConnectorV1Impl)
    connector._parent = _Parent(metadata)
    connector._manager = SimpleNamespace(lmcache_engine=engine)
    connector._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=0, data_parallel_rank=0)
    )
    connector.kv_role = "kv_consumer"
    connector.use_layerwise = True
    connector.enable_sparse_attention = False
    connector.config = SimpleNamespace(
        dsa_two_groups=False,
        blocking_timeout_secs=1,
    )
    connector.device = "cpu"
    connector._lmcache_chunk_size = 4
    connector.num_layers = 1
    kv_cache = torch.zeros(1)
    connector.kv_caches = {"layer0": kv_cache}
    connector._kvcaches_list = [kv_cache]
    connector._latent_layer_names = ["layer0"]
    connector._indexer_layer_names = []
    connector._latent_kvcaches = [kv_cache]
    connector._indexer_kvcaches = []
    connector._layerwise_save_storers = {}
    connector._deferred_latent_pending = set()
    connector._worker_retrieve_state = {}
    connector._worker_retrieve_registry_version = 0
    connector._wait_for_save_done = True
    connector._finished_req_ids_waiting_for_save = set()
    connector._late_finished_sending = set()
    connector._completed_decode_window_saves = {}
    connector._decode_window_save_completed_groups = set()
    connector._decode_window_save_expected_start = {}
    return connector, metadata, engine


def _run_store(
    connector: LMCacheConnectorV1Impl,
) -> tuple:
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    connector.wait_for_save()
    return connector.get_dsa_operation_receipts()


def test_command_store_forces_exact_range_and_waits_for_final_fence() -> None:
    key = RequestKey("scheduler", "req", 1)
    command = _command(key, "store-1")
    connector, _, engine = _connector(command)
    fence_seen = False
    original_finish = connector._finish_save_batch

    def finish(save_context: dict) -> None:
        nonlocal fence_seen
        assert connector.get_dsa_operation_receipts() == ()
        original_finish(save_context)
        fence_seen = True

    connector._finish_save_batch = finish
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert connector.get_dsa_operation_receipts() == ()
    connector.wait_for_save()

    assert fence_seen
    assert engine.calls[0]["offset"] == 0
    assert engine.calls[0]["dsa_command_store"] is True
    assert engine.calls[0]["mask"].tolist() == [True, True, True, True]
    receipts = connector.get_dsa_operation_receipts()
    assert {receipt.receipt_kind for receipt in receipts} == {
        "storage",
        "source_seal",
    }
    assert all(receipt.status == "complete" for receipt in receipts)
    assert all(receipt.covered_chunks == ((0, 4),) for receipt in receipts)
    assert connector.get_dsa_operation_receipts() == ()
    assert connector.get_completed_decode_window_saves() == {}


def test_command_metadata_bypasses_request_save_and_disagg_gates() -> None:
    key = RequestKey("scheduler", "req", 1)
    command = _command(key, "store-forced")
    tracker = RequestTracker(
        req_id="req",
        prompt_len=4,
        token_ids=[0, 1, 2, 3],
        allocated_block_ids=[3],
        skip_save=True,
        disagg_spec=SimpleNamespace(num_transferred_tokens=0),
        dsa_request_key=key,
        dsa_route_authoritative=True,
    )

    request = ReqMeta.from_dsa_store_command(tracker, 4, command, {0})

    assert request.save_spec is not None
    assert request.save_spec.can_save
    assert request.save_spec.skip_leading_tokens == 0
    assert request.disagg_spec is None
    assert request.save_slot_mapping_base == 0
    assert request.save_slot_mapping[0].tolist() == [12, 13, 14, 15]


def test_command_only_store_runs_the_same_wait_and_fence_protocol() -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-command-only")
    connector, _, engine = _connector(command)

    connector.start_load_kv(SimpleNamespace(attn_metadata=None))
    receipts = connector.get_dsa_operation_receipts()

    assert len(engine.calls) == 1
    assert len(receipts) == 2
    assert all(receipt.status == "complete" for receipt in receipts)
    assert connector._wait_for_save_done


def test_dynamic_store_catch_up_uses_prior_generation_manifest() -> None:
    key = RequestKey("scheduler", "req", 1)
    first = _command(key, "store-1")
    connector, metadata, engine = _connector(first)
    assert all(receipt.status == "complete" for receipt in _run_store(connector))

    second = _command(
        key,
        "store-2",
        range_start=4,
        range_end=8,
        input_generation="generation-1",
        output_generation="generation-2",
    )
    metadata.requests = [_request(second)]
    metadata.dsa_commands = (second,)
    receipts = _run_store(connector)

    assert engine.calls[-1]["offset"] == 4
    assert engine.calls[-1]["mask"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert all(receipt.status == "complete" for receipt in receipts)
    sealed = connector._dsa_sealed_sources[(key, "generation-2")]
    assert sealed.groups[0].chunks == ((0, 4), (4, 8))
    assert sealed.range_end == 8
    assert (key, "generation-1") not in connector._dsa_sealed_sources


@pytest.mark.parametrize(
    ("engine", "error_code"),
    [
        (_Engine(empty=True), "empty_store_result"),
        (_Engine(empty=True, healthy=False), "backend_unhealthy"),
    ],
)
def test_empty_or_unhealthy_store_emits_only_failed_receipts(
    engine: _Engine,
    error_code: str,
) -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-failed")
    connector, _, _ = _connector(command, engine=engine)

    receipts = _run_store(connector)

    assert len(receipts) == 2
    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(receipt.error_code == error_code for receipt in receipts)


def test_passive_shared_store_seals_local_source_before_receipts() -> None:
    key = RequestKey("scheduler", "req", 1)
    command = _command(key, "store-passive")
    engine = _PassiveSharedEngine()
    connector, _, _ = _connector(command, engine=engine)

    receipts = _run_store(connector)

    assert len(engine.exchange_calls) == 1
    assert engine.exchange_calls[0]["source_result"] is None
    assert engine.exchange_calls[0]["source_error"] is None
    assert all(receipt.status == "complete" for receipt in receipts)
    source = connector._dsa_sealed_sources[(key, "generation-1")]
    assert source.groups[0].prepared_source.total_tokens == 4
    assert engine.shared_memory_obj is not None
    assert engine.shared_memory_obj.refs == 1
    assert engine.shared_memory_obj.pins == 1


def test_passive_shared_store_failure_does_not_create_generation() -> None:
    key = RequestKey("scheduler", "req", 1)
    command = _command(key, "store-passive-failed")
    engine = _PassiveSharedEngine(exchange_error="shared_source_exchange_failed")
    connector, _, _ = _connector(command, engine=engine)

    receipts = _run_store(connector)

    assert len(engine.exchange_calls) == 1
    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(
        receipt.error_code == "shared_source_exchange_failed"
        for receipt in receipts
    )
    assert (key, "generation-1") not in connector._dsa_sealed_sources


def test_passive_shared_multigroup_failure_releases_earlier_views() -> None:
    key = RequestKey("scheduler", "req", 1)
    command = _command(key, "store-passive-partial")
    command = replace(
        command,
        expected_receipts=(
            *command.expected_receipts,
            *(
                replace(expectation, kv_group=1)
                for expectation in command.expected_receipts
            ),
        ),
    )
    engine = _PassiveSharedEngine(fail_kv_group=1)
    connector, _, _ = _connector(command, engine=engine)

    receipts = _run_store(connector)

    assert len(engine.exchange_calls) == 2
    assert all(receipt.status == "failed" for receipt in receipts)
    assert len(engine.shared_memory_objs) == 1
    assert engine.shared_memory_objs[0].refs == 0
    assert engine.shared_memory_objs[0].pins == 0
    assert (key, "generation-1") not in connector._dsa_sealed_sources


def test_malformed_participant_manifest_keeps_collective_schedule() -> None:
    key = RequestKey("scheduler", "req", 1)
    command = _command(key, "store-malformed-participants")
    participant_1 = replace(
        PARTICIPANT,
        process_instance_id="worker-1",
        worker_rank=1,
        tp_rank=1,
    )
    command = replace(
        command,
        expected_receipts=(
            *command.expected_receipts,
            *(
                replace(
                    expectation,
                    participant=participant_1,
                    kv_group=1,
                )
                for expectation in command.expected_receipts
            ),
        ),
    )
    engine = _PassiveSharedEngine()
    connector, _, _ = _connector(command, engine=engine)

    receipts = _run_store(connector)

    assert [call["kv_group"] for call in engine.exchange_calls] == [0, 1]
    assert len(receipts) == 2
    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(
        receipt.error_code == "malformed_receipt_expectations"
        for receipt in receipts
    )


def test_partial_chunk_result_cannot_publish_success() -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-partial")
    connector, _, engine = _connector(command)
    original_store = engine.store_layer

    def partial_store(token_ids: list[int], **kwargs):
        storer = original_store(token_ids, **kwargs)
        next(storer)
        result = next(storer)
        result.ends = [3]

        def partial():
            yield None
            yield result

        return partial()

    engine.store_layer = partial_store
    receipts = _run_store(connector)

    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(receipt.error_code == "partial_chunk_coverage" for receipt in receipts)


def test_non_cpu_source_requires_resolved_device_pointer_metadata() -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-no-device-ptrs")
    connector, _, engine = _connector(command)
    connector.device = "meta"
    original_store = engine.store_layer

    def store_without_device_ptrs(token_ids: list[int], **kwargs):
        storer = original_store(token_ids, **kwargs)
        next(storer)
        result = next(storer)
        result.chunk_dev_ptrs = []
        result.chunk_ptrs = []

        def incomplete_ptrs():
            yield None
            yield result

        return incomplete_ptrs()

    engine.store_layer = store_without_device_ptrs

    receipts = _run_store(connector)

    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(
        receipt.error_code == "source_device_pointer_coverage_incomplete"
        for receipt in receipts
    )


def test_extra_chunk_boundary_cannot_be_truncated_into_success() -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-extra-boundary")
    connector, _, engine = _connector(command)
    original_store = engine.store_layer

    def extra_boundary_store(token_ids: list[int], **kwargs):
        storer = original_store(token_ids, **kwargs)
        next(storer)
        result = next(storer)
        result.starts.append(99)

        def extra_boundary():
            yield None
            yield result

        return extra_boundary()

    engine.store_layer = extra_boundary_store
    receipts = _run_store(connector)

    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(receipt.error_code == "partial_chunk_coverage" for receipt in receipts)


def test_unsupported_store_expectation_fails_the_entire_local_command() -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-extra-receipt")
    extra = replace(
        command.expected_receipts[0],
        receipt_kind="npu_materialization",
        storage_tier="npu",
        minimum_guarantee="npu_materialized",
    )
    command = replace(
        command,
        expected_receipts=(*command.expected_receipts, extra),
    )
    connector, _, _ = _connector(command)

    receipts = _run_store(connector)

    assert len(receipts) == 3
    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(
        receipt.error_code == "malformed_receipt_expectations" for receipt in receipts
    )


def test_command_only_activation_fences_and_drains_once() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-1")
    connector, metadata, _ = _connector(store)
    _run_store(connector)
    activation = _command(
        key,
        "activation-1",
        kind="source_activation",
        route_epoch=1,
        input_generation="generation-1",
        output_generation="generation-1",
        parent_operation_id="store-1",
    )
    metadata.requests = []
    metadata.dsa_commands = (activation,)
    fenced = False

    def fence() -> None:
        nonlocal fenced
        fenced = True

    connector._fence_dsa_source_activation = fence
    connector.start_load_kv(SimpleNamespace(attn_metadata=None))

    receipts = connector.get_dsa_operation_receipts()
    assert fenced
    assert len(receipts) == 1
    assert receipts[0].status == "complete"
    assert receipts[0].receipt_kind == "route_epoch_complete"
    assert connector._dsa_pending_active_generations[key] == "generation-1"
    assert connector.get_dsa_operation_receipts() == ()


def test_terminal_store_command_replays_identical_receipts() -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-replay")
    connector, metadata, engine = _connector(command)
    original = _run_store(connector)
    store_call_count = len(engine.calls)

    connector._initialize_dsa_commands(metadata)
    replay = connector.get_dsa_operation_receipts()

    assert replay == original
    assert len(engine.calls) == store_call_count


def test_terminal_store_command_rejects_conflicting_replay() -> None:
    command = _command(RequestKey("scheduler", "req", 1), "store-conflict")
    connector, metadata, _ = _connector(command)
    _run_store(connector)
    metadata.dsa_commands = (
        replace(command, cache_namespace_fingerprint="different-namespace"),
    )

    with pytest.raises(RuntimeError, match="Conflicting replay"):
        connector._initialize_dsa_commands(metadata)


def test_authoritative_fallback_rolls_back_pending_activation() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-before-rollback")
    connector, metadata, _ = _connector(store)
    assert all(receipt.status == "complete" for receipt in _run_store(connector))
    connector.get_dsa_operation_receipts()

    activation = _command(
        key,
        "activation-before-rollback",
        kind="source_activation",
        route_epoch=1,
        input_generation="generation-1",
        output_generation="generation-1",
        parent_operation_id="store-before-rollback",
    )
    metadata.requests = []
    metadata.dsa_commands = (activation,)
    connector._process_dsa_activation_commands(metadata)
    source = connector._dsa_sealed_sources[(key, "generation-1")]
    assert source.pending_active

    connector._reconcile_pending_dsa_activation(
        ReqMeta(
            req_id="req",
            token_ids=[0, 1, 2, 3],
            dsa_request_key=key,
            dsa_route_state="promoting",
            dsa_route_authoritative=True,
            dsa_route_epoch=0,
        )
    )
    assert source.pending_active

    connector._reconcile_pending_dsa_activation(
        ReqMeta(
            req_id="req",
            token_ids=[0, 1, 2, 3],
            dsa_request_key=key,
            dsa_route_state="fallback_resident",
            dsa_route_authoritative=True,
            dsa_route_epoch=1,
        )
    )

    assert (key, "generation-1") not in connector._dsa_sealed_sources
    assert key not in connector._dsa_pending_active_generations


def test_authoritative_fallback_retires_successful_store_candidate() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-before-fallback")
    connector, _, _ = _connector(store)
    assert all(receipt.status == "complete" for receipt in _run_store(connector))
    assert (key, "generation-1") in connector._dsa_sealed_sources

    connector._reconcile_pending_dsa_activation(
        ReqMeta(
            req_id="req",
            token_ids=[0, 1, 2, 3],
            dsa_request_key=key,
            dsa_route_state="fallback_resident",
            dsa_route_authoritative=True,
            dsa_route_epoch=1,
        )
    )

    assert (key, "generation-1") not in connector._dsa_sealed_sources
    assert connector.get_dsa_control_events() == ()


def test_authoritative_route_binds_only_the_activated_generation() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-1")
    connector, metadata, _ = _connector(store)
    _run_store(connector)
    activation = _command(
        key,
        "activation-1",
        kind="source_activation",
        route_epoch=1,
        input_generation="generation-1",
        output_generation="generation-1",
        parent_operation_id="store-1",
    )
    metadata.requests = []
    metadata.dsa_commands = (activation,)
    connector.start_load_kv(SimpleNamespace(attn_metadata=None))
    connector.get_dsa_operation_receipts()
    lease = DSASourceLease(
        source_lease_id="lease-active",
        request_key=key,
        execution_seq=2,
        route_epoch=1,
        source_generation_id="generation-1",
    )
    request = ReqMeta(
        req_id="req",
        token_ids=[0, 1, 2, 3],
        is_sparse_decode=True,
        load_spec=LoadSpec(0, 4, True),
        dsa_request_key=key,
        dsa_route_epoch=1,
        dsa_active_generation_id="generation-1",
        dsa_active_token_prefix_digest=store.token_prefix_digest,
        dsa_source_lease=lease,
    )

    connector._bind_dsa_active_source(request)

    state = connector._worker_retrieve_state["req"]
    assert state.dsa_request_key == key
    assert state.dsa_generation_id == "generation-1"
    assert state.prepared_sparse_sources[0].total_tokens == 4
    assert lease.source_lease_id in connector._dsa_source_lease_bindings
    stale = replace(request, dsa_active_generation_id="stale-generation")
    with pytest.raises(RuntimeError, match="unavailable sealed generation"):
        connector._bind_dsa_active_source(stale)
    connector._release_dsa_source_leases((request,))
    assert connector.get_released_dsa_source_leases() == (lease,)
    assert connector.get_released_dsa_source_leases() == ()
    connector._release_dsa_source_leases((request,))
    with pytest.raises(RuntimeError, match="already released"):
        connector._bind_dsa_active_source(request)
    assert lease.source_lease_id not in connector._dsa_source_lease_bindings


def test_invalid_source_lease_cannot_mutate_pending_activation() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-before-invalid-lease")
    connector, metadata, _ = _connector(store)
    _run_store(connector)
    activation = _command(
        key,
        "activation-before-invalid-lease",
        kind="source_activation",
        route_epoch=1,
        input_generation="generation-1",
        output_generation="generation-1",
        parent_operation_id="store-before-invalid-lease",
    )
    metadata.requests = []
    metadata.dsa_commands = (activation,)
    connector._process_dsa_activation_commands(metadata)
    source = connector._dsa_sealed_sources[(key, "generation-1")]
    invalid_lease = DSASourceLease(
        source_lease_id="lease-invalid",
        request_key=key,
        execution_seq=2,
        route_epoch=1,
        source_generation_id="different-generation",
    )
    request = ReqMeta(
        req_id="req",
        token_ids=[0, 1, 2, 3],
        is_sparse_decode=True,
        load_spec=LoadSpec(0, 4, True),
        dsa_request_key=key,
        dsa_route_epoch=1,
        dsa_active_generation_id="generation-1",
        dsa_active_token_prefix_digest=store.token_prefix_digest,
        dsa_source_lease=invalid_lease,
    )

    with pytest.raises(RuntimeError, match="lease identity mismatch"):
        connector._bind_dsa_active_source(request)

    assert connector._dsa_pending_active_generations[key] == "generation-1"
    assert key not in connector._dsa_active_generations
    assert source.pending_active
    assert not source.active
    assert "req" not in connector._worker_retrieve_state


def test_activation_rejects_stale_generation() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-1")
    connector, metadata, _ = _connector(store)
    _run_store(connector)
    activation = _command(
        key,
        "activation-stale",
        kind="source_activation",
        route_epoch=1,
        input_generation="stale-generation",
        output_generation="stale-generation",
        parent_operation_id="store-1",
    )
    metadata.requests = []
    metadata.dsa_commands = (activation,)

    connector.start_load_kv(SimpleNamespace(attn_metadata=None))
    receipts = connector.get_dsa_operation_receipts()

    assert len(receipts) == 1
    assert receipts[0].status == "failed"
    assert receipts[0].error_code == "sealed_generation_unavailable"


def test_activation_requires_the_exact_sealed_group_manifest() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-1")
    connector, metadata, _ = _connector(store)
    _run_store(connector)
    activation = _command(
        key,
        "activation-extra-group",
        kind="source_activation",
        route_epoch=1,
        input_generation="generation-1",
        output_generation="generation-1",
        parent_operation_id="store-1",
    )
    extra_group = replace(activation.expected_receipts[0], kv_group=1)
    activation = replace(
        activation,
        expected_receipts=(*activation.expected_receipts, extra_group),
    )
    metadata.requests = []
    metadata.dsa_commands = (activation,)

    connector.start_load_kv(SimpleNamespace(attn_metadata=None))
    receipts = connector.get_dsa_operation_receipts()

    assert len(receipts) == 2
    assert all(receipt.status == "failed" for receipt in receipts)
    assert all(
        receipt.error_code == "activation_group_manifest_mismatch"
        for receipt in receipts
    )


def test_finished_request_preserves_generation_until_source_lease_releases() -> None:
    key = RequestKey("scheduler", "req", 1)
    store = _command(key, "store-1")
    connector, _, _ = _connector(store)
    _run_store(connector)
    source = connector._dsa_sealed_sources[(key, "generation-1")]
    source.active = True
    connector._dsa_active_generations[key] = "generation-1"
    lease = DSASourceLease(
        source_lease_id="lease-1",
        request_key=key,
        execution_seq=1,
        route_epoch=1,
        source_generation_id="generation-1",
    )
    source.source_lease_ids.add(lease.source_lease_id)
    connector._dsa_source_lease_bindings[lease.source_lease_id] = (
        key,
        "generation-1",
    )
    connector._dsa_bound_source_leases[lease.source_lease_id] = lease
    connector._worker_retrieve_state["req"] = WorkerRetrieveState(
        req_id="req",
        metadata_warm=True,
        dsa_request_key=key,
        dsa_generation_id="generation-1",
    )

    assert connector._retire_dsa_request_sources("req") is False
    assert (key, "generation-1") in connector._dsa_sealed_sources
    request = ReqMeta(
        req_id="req",
        token_ids=[],
        dsa_request_key=key,
        dsa_active_generation_id="generation-1",
        dsa_source_lease=lease,
    )
    connector._release_dsa_source_leases((request,))

    assert (key, "generation-1") not in connector._dsa_sealed_sources
    assert "req" not in connector._worker_retrieve_state


def test_retirement_releases_unleased_candidate_generations() -> None:
    key = RequestKey("scheduler", "req", 1)
    first = _command(key, "store-1")
    connector, metadata, _ = _connector(first)
    _run_store(connector)
    second = _command(
        key,
        "store-2",
        range_start=4,
        range_end=8,
        input_generation="generation-1",
        output_generation="generation-2",
    )
    metadata.requests = [_request(second)]
    metadata.dsa_commands = (second,)
    _run_store(connector)
    source = connector._dsa_sealed_sources[(key, "generation-2")]
    source.active = True
    connector._dsa_active_generations[key] = "generation-2"
    lease = DSASourceLease(
        source_lease_id="lease-2",
        request_key=key,
        execution_seq=2,
        route_epoch=1,
        source_generation_id="generation-2",
    )
    source.source_lease_ids.add(lease.source_lease_id)
    connector._dsa_source_lease_bindings[lease.source_lease_id] = (
        key,
        "generation-2",
    )
    connector._dsa_bound_source_leases[lease.source_lease_id] = lease

    assert connector._retire_dsa_request_sources("req") is False

    assert (key, "generation-1") not in connector._dsa_sealed_sources
    assert (key, "generation-2") in connector._dsa_sealed_sources


def test_scheduler_forwards_command_only_metadata_and_rejects_stale_key() -> None:
    key = RequestKey("scheduler", "req", 1)
    tracker = RequestTracker(
        req_id="req",
        prompt_len=4,
        token_ids=[0, 1, 2, 3],
        allocated_block_ids=[0],
        dsa_request_key=key,
        dsa_route_authoritative=True,
        dsa_route_epoch=0,
    )
    connector = object.__new__(LMCacheConnectorV1Impl)
    connector._request_trackers = {"req": tracker}
    connector._block_size = 4
    command = _command(
        key,
        "activation",
        kind="source_activation",
        route_epoch=1,
        input_generation="generation-1",
        output_generation="generation-1",
        parent_operation_id="store-1",
    )
    output = SimpleNamespace(
        dsa_commands=(command,),
        dsa_routes={},
        dsa_data_compatibility_fingerprint="namespace",
    )
    metadata = LMCacheConnectorMetadata()

    connector._attach_dsa_commands(metadata, output)
    assert metadata.dsa_commands == (command,)
    assert metadata.requests == []

    stale = replace(
        command,
        request_key=RequestKey("scheduler", "req", 2),
    )
    output.dsa_commands = (stale,)
    with pytest.raises(RuntimeError, match="unknown or stale"):
        connector._attach_dsa_commands(LMCacheConnectorMetadata(), output)
