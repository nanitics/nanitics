"""Resume behavior for workflows nested inside other workflows.

A :class:`Workflow` nested via :class:`WorkflowStep` must resume at its own
suspension point rather than re-executing from the top. These tests exercise
the full suspend/resume cycle through :class:`DurableRun` /
:class:`ResumeService` for every orchestrator that can contain a nested
workflow, and assert the two properties that motivated the change:

- A nested ``Conditional``'s router is **not** re-invoked on resume.
- A non-deterministic router cannot re-route the resumed run onto a
  different branch than the one originally selected.

The leaf suspension is driven by :class:`ApprovalGate` +
:class:`DurableHumanInputProvider`, the same primitives used by
``tests/test_durable_resume.py`` — no agent-resume wiring required.
"""

from __future__ import annotations

import pytest

from nanitics.collaboration.approval_gate import ApprovalGate
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.protocol import HumanDecision, HumanInputResponse
from nanitics.composition import (
    DAG,
    Conditional,
    FunctionStep,
    InMemoryCheckpointStore,
    Loop,
    MapReduce,
    Parallel,
    Pipeline,
    Sequential,
)
from nanitics.composition.durability.models import (
    CheckpointVersionError,
    RunCheckpoint,
    SuspensionInfo,
)
from nanitics.composition.durability.resume import (
    DurableRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
)
from nanitics.composition.durability.resume import (
    SuspendedRun as _SuspendedRun,
)
from nanitics.composition.orchestration.adapters import WorkflowStep
from nanitics.composition.orchestration.dag import DAGNode
from nanitics.composition.orchestration.pipeline import Stage
from nanitics.hitl import InMemoryHitlRequestStore
from tests.testing_helpers import make_emitter

_RUN_ID = "nested-resume-test"


def _echo(name: str) -> FunctionStep:
    async def fn(value: object) -> object:
        return value

    return FunctionStep(name=name, fn=fn)


def _approve(request_id: str) -> HumanInputResponse:
    return HumanInputResponse(request_id=request_id, decision=HumanDecision.APPROVE)


def _gate(provider: DurableHumanInputProvider, name: str = "review") -> ApprovalGate:
    return ApprovalGate(provider=provider, name=name, run_id=_RUN_ID)


def _service(
    hitl_store: InMemoryHitlRequestStore,
    checkpoint_store: InMemoryCheckpointStore,
    build: object,
) -> ResumeService:
    def factory(ctx: ResumeContext) -> DurableRun:
        return DurableRun(
            build(),  # type: ignore[operator]
            hitl_store=ctx.hitl_store,
            checkpoint_store=ctx.checkpoint_store,
        )

    return ResumeService(hitl_store=hitl_store, checkpoint_store=checkpoint_store, factory=factory)


class TestConditionalNestingReportedCase:
    async def test_router_not_reinvoked_on_resume(self) -> None:
        """The motivating case: a nested Conditional's router runs exactly once."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        router_calls: list[object] = []

        def router(value: object) -> str:
            router_calls.append(value)
            return "hot"

        def build() -> Sequential:
            branch = Sequential(
                name="branch",
                steps=[_gate(provider), _echo("after-gate")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            cond = Conditional(
                name="cond",
                router=router,
                branches={"hot": WorkflowStep(branch)},
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Sequential(
                name="outer",
                steps=[WorkflowStep(cond)],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)
        assert len(router_calls) == 1

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )

        assert isinstance(result, ResumeResult)
        assert result.output == "payload"
        # The router must NOT have been called again on resume.
        assert len(router_calls) == 1

    async def test_nondeterministic_router_keeps_original_branch(self) -> None:
        """A router that would now route differently still resumes the original branch."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        calls = {"n": 0}

        def router(_value: object) -> str:
            calls["n"] += 1
            return "gated" if calls["n"] == 1 else "ungated"

        async def tag_b(_value: object) -> str:
            return "B-RAN"

        def build() -> Sequential:
            gated = Sequential(
                name="gated-branch",
                steps=[_gate(provider), _echo("a-tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            cond = Conditional(
                name="cond",
                router=router,
                branches={
                    "gated": WorkflowStep(gated),
                    "ungated": FunctionStep(name="b", fn=tag_b),
                },
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Sequential(
                name="outer",
                steps=[WorkflowStep(cond)],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )
        assert isinstance(result, ResumeResult)
        # Had the router re-run, it would return "ungated" and yield "B-RAN".
        assert result.output == "payload"


class TestSequentialNesting:
    async def test_pre_suspension_step_not_reexecuted(self) -> None:
        """A step before the nested workflow runs once across start + resume."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        before_calls = {"n": 0}

        async def count_before(value: object) -> object:
            before_calls["n"] += 1
            return value

        def build() -> Sequential:
            inner = Sequential(
                name="inner",
                steps=[_gate(provider), _echo("inner-tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Sequential(
                name="outer",
                steps=[
                    FunctionStep(name="before", fn=count_before),
                    WorkflowStep(inner),
                    _echo("outer-tail"),
                ],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)
        assert before_calls["n"] == 1

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )
        assert isinstance(result, ResumeResult)
        assert result.output == "payload"
        # "before" must not run a second time on resume.
        assert before_calls["n"] == 1


class TestParallelNesting:
    async def test_suspended_branch_resumes_nested_siblings_restored(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> Parallel:
            gated = Sequential(
                name="gated",
                steps=[_gate(provider), _echo("gated-tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Parallel(
                name="outer",
                steps=[_echo("sibling"), WorkflowStep(gated)],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )
        assert isinstance(result, ResumeResult)
        # Both branches contribute "payload" (echo sibling + resumed nested gate).
        assert result.output == ["payload", "payload"]


class TestTwoLevelNestingCheckpointShape:
    async def test_persisted_checkpoint_is_recursive(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> Sequential:
            branch = Sequential(
                name="branch",
                steps=[_gate(provider), _echo("tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            cond = Conditional(
                name="cond",
                router=lambda _v: "a",
                branches={"a": WorkflowStep(branch)},
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Sequential(
                name="outer",
                steps=[WorkflowStep(cond)],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        await durable.start("payload")

        loaded = await checkpoint_store.load(_RUN_ID)
        assert loaded is not None
        outer_state = loaded.state
        assert outer_state["orchestrator_type"] == "sequential"
        assert outer_state["suspended_step_index"] == 0
        cond_state = outer_state["nested_checkpoint"]
        assert cond_state["orchestrator_type"] == "conditional"
        assert cond_state["selected_branch"] == "a"
        branch_state = cond_state["nested_checkpoint"]
        assert branch_state["orchestrator_type"] == "sequential"
        assert branch_state["suspended_step_index"] == 0
        # No agent_checkpoint anywhere — the leaf is a gate, not an agent.
        assert "agent_checkpoint" not in branch_state


class TestResumeThenSuspendAgain:
    async def test_two_nested_gates_each_resume_renests(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> Sequential:
            branch = Sequential(
                name="branch",
                steps=[
                    _gate(provider, name="first"),
                    _gate(provider, name="second"),
                    _echo("tail"),
                ],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            cond = Conditional(
                name="cond",
                router=lambda _v: "a",
                branches={"a": WorkflowStep(branch)},
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Sequential(
                name="outer",
                steps=[WorkflowStep(cond)],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        first = await durable.start("payload")
        assert isinstance(first, _SuspendedRun)

        service = _service(hitl_store, checkpoint_store, build)
        second = await service.resume(_RUN_ID, _approve(first.pending_request.request_id))
        assert isinstance(second, _SuspendedRun)
        assert second.pending_request.request_id != first.pending_request.request_id

        final = await service.resume(_RUN_ID, _approve(second.pending_request.request_id))
        assert isinstance(final, ResumeResult)
        assert final.output == "payload"


class TestDAGNesting:
    async def test_nested_node_resumes(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> DAG:
            gated = Sequential(
                name="gated",
                steps=[_gate(provider), _echo("tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return DAG(
                name="outer",
                nodes={
                    "root": DAGNode(step=_echo("root")),
                    "gated": DAGNode(step=WorkflowStep(gated), depends_on=["root"]),
                },
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )
        assert isinstance(result, ResumeResult)
        assert result.output == "payload"


class TestLoopNesting:
    async def test_nested_loop_body_resumes(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> Loop:
            body = Sequential(
                name="body",
                steps=[_gate(provider), _echo("tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Loop(
                name="outer",
                step=WorkflowStep(body),
                condition=lambda _result, _iteration: True,
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )
        assert isinstance(result, ResumeResult)
        assert result.output == "payload"


class TestPipelineNesting:
    async def test_nested_stage_resumes(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> Pipeline:
            gated = Sequential(
                name="gated",
                steps=[_gate(provider), _echo("tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return Pipeline(
                name="outer",
                stages=[Stage(_echo("first")), Stage(WorkflowStep(gated))],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )
        assert isinstance(result, ResumeResult)
        assert result.output == "payload"


class TestMapReduceNesting:
    async def test_nested_item_resumes(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> MapReduce:
            gated = Sequential(
                name="gated",
                steps=[_gate(provider), _echo("tail")],
                emitter=make_emitter(),
                run_id=_RUN_ID,
            )
            return MapReduce(
                name="outer",
                step=WorkflowStep(gated),
                splitter=lambda value: [value],
                reducer=lambda results: [r.output for r in results],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        suspended = await durable.start("payload")
        assert isinstance(suspended, _SuspendedRun)

        result = await _service(hitl_store, checkpoint_store, build).resume(
            _RUN_ID, _approve(suspended.pending_request.request_id)
        )
        assert isinstance(result, ResumeResult)
        assert result.output == ["payload"]


class TestSchemaVersionGate:
    async def test_stale_v2_checkpoint_rejected(self) -> None:
        """A checkpoint from the previous schema version is refused on resume."""
        provider = DurableHumanInputProvider(request_store=InMemoryHitlRequestStore())
        outer = Sequential(
            name="outer",
            steps=[WorkflowStep(Sequential(name="inner", steps=[_gate(provider)], emitter=make_emitter()))],
            emitter=make_emitter(),
            run_id=_RUN_ID,
        )
        stale = RunCheckpoint(
            run_id=_RUN_ID,
            checkpoint_type="orchestration",
            schema_version=2,
            state={"suspended_step_index": 0},
            suspension_info=SuspensionInfo(
                suspension_id="s",
                request_id="r",
                request_type="approval",
                prompt="?",
            ),
        )
        with pytest.raises(CheckpointVersionError):
            await outer.execute("payload", resume_from=stale)
