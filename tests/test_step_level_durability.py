"""Tests for step-level durability (orchestration-level slice).

Covers the opt-in per-step cursor checkpoint + journal write, and the non-HITL
resume entrypoints (``DurableRun.resume_from_checkpoint`` /
``ResumeService.resume_interrupted``) that re-drive an interrupted run from the
last completed step without re-executing it. Crash is simulated with a step that
raises a plain exception on its first call — execution unwinds after the prior
step's cursor checkpoint has been written, exactly as a worker dying mid-run.

The agent-internal (tool-call granularity) slice is separate; see the
implementation plan. These tests exercise the agent-agnostic orchestration path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from nanitics.capabilities.errors.handler import ErrorHandler
from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.protocol import HumanDecision, HumanInputResponse
from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.resume import (
    DurableRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
    SuspendedRun,
)
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.adapters import FunctionStep, WorkflowStep
from nanitics.composition.orchestration.dag import DAG, DAGNode
from nanitics.composition.orchestration.loop import Loop
from nanitics.composition.orchestration.parallel import Parallel
from nanitics.composition.orchestration.protocol import StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.hitl import InMemoryHitlRequestStore
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.infrastructure.observability.events import Usage
from nanitics.strategies import ReActAgent, tool
from nanitics.strategies.tools import FunctionTool
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter, make_response, make_step


def counting_step(name: str, calls: dict[str, int], *, fail_first: bool = False) -> FunctionStep:
    """A step that records each invocation; optionally raises on its first call.

    The first-call raise simulates a crash *after* the preceding step's cursor
    checkpoint was written but before this step completed.
    """

    async def fn(x: object) -> str:
        calls[name] = calls.get(name, 0) + 1
        if fail_first and calls[name] == 1:
            raise RuntimeError(f"simulated crash in {name}")
        return f"{x}->{name}"

    return FunctionStep(name=name, fn=fn)


def _suspension() -> SuspensionInfo:
    return SuspensionInfo(
        suspension_id="sus-1",
        request_id="req-1",
        request_type="approval",
        prompt="Approve?",
    )


class TestStepCheckpointWrite:
    async def test_cursor_and_journal_written_after_each_step(self) -> None:
        calls: dict[str, int] = {}
        steps = [counting_step("s0", calls), counting_step("s1", calls)]
        store = InMemoryCheckpointStore()
        seq = Sequential(
            name="w",
            steps=steps,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        result = await seq.execute("in")

        assert result.output == "in->s0->s1"
        journal = await store.load_journal("r")
        assert [rec.step_path for rec in journal] == ["sequential#0:s0", "sequential#1:s1"]
        assert all(rec.step_kind == "orchestration_step" for rec in journal)
        assert journal[0].result["output"] == "in->s0"
        latest = await store.load("r")
        assert latest is not None
        assert latest.checkpoint_reason == "step"
        assert latest.suspension_info is None
        assert latest.state["suspended_step_index"] == 2

    async def test_no_checkpoints_when_disabled(self) -> None:
        calls: dict[str, int] = {}
        steps = [counting_step("s0", calls), counting_step("s1", calls)]
        store = InMemoryCheckpointStore()
        seq = Sequential(
            name="w",
            steps=steps,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
        )  # step_checkpoints defaults False

        await seq.execute("in")

        assert await store.load("r") is None
        assert await store.load_journal("r") == []


class TestOrchestrationResume:
    async def test_resume_from_cursor_skips_completed_steps(self) -> None:
        calls: dict[str, int] = {}
        steps = [counting_step("s0", calls), counting_step("s1", calls, fail_first=True)]
        store = InMemoryCheckpointStore()
        seq = Sequential(
            name="w",
            steps=steps,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        with pytest.raises(RuntimeError, match="simulated crash in s1"):
            await seq.execute("in")
        assert calls == {"s0": 1, "s1": 1}

        cursor = await store.load("r")
        assert cursor is not None
        result = await seq.execute("in", resume_from=cursor)

        assert calls == {"s0": 1, "s1": 2}  # s0 not re-run
        assert result.output == "in->s0->s1"

    async def test_resume_from_final_cursor_finalizes_without_rerun(self) -> None:
        calls: dict[str, int] = {}
        steps = [counting_step("s0", calls)]
        store = InMemoryCheckpointStore()
        seq = Sequential(
            name="w",
            steps=steps,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        await seq.execute("in")
        assert calls == {"s0": 1}
        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.state["suspended_step_index"] == 1  # == len(steps)

        result = await seq.execute("in", resume_from=cursor)

        assert calls == {"s0": 1}  # fully-completed run re-finalizes, no re-run
        assert result.output == "in->s0"


class TestDurableRunResumeFromCheckpoint:
    async def test_resumes_interrupted_run(self) -> None:
        calls: dict[str, int] = {}
        steps = [counting_step("s0", calls), counting_step("s1", calls, fail_first=True)]
        store = InMemoryCheckpointStore()
        seq = Sequential(
            name="w",
            steps=steps,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        with pytest.raises(RuntimeError):
            await seq.execute("in")

        durable = DurableRun(
            seq,
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            run_id="r",
        )
        result = await durable.resume_from_checkpoint()

        assert isinstance(result, ResumeResult)
        assert result.output == "in->s0->s1"
        assert calls == {"s0": 1, "s1": 2}

    async def test_rejects_hitl_suspension(self) -> None:
        store = InMemoryCheckpointStore()
        await store.save(
            RunCheckpoint(
                run_id="r",
                checkpoint_type="orchestration",
                state={"original_input": "x"},
                suspension_info=_suspension(),
            )
        )
        seq = Sequential(
            name="w",
            steps=[make_step("s0")],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        durable = DurableRun(seq, hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, run_id="r")

        with pytest.raises(ValueError, match="suspended awaiting human input"):
            await durable.resume_from_checkpoint()

    async def test_rejects_when_no_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()
        seq = Sequential(
            name="w",
            steps=[make_step("s0")],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        durable = DurableRun(seq, hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, run_id="r")

        with pytest.raises(ValueError, match="No checkpoint to resume"):
            await durable.resume_from_checkpoint()


class TestResumeServiceResumeInterrupted:
    async def test_resumes_interrupted_run(self) -> None:
        calls: dict[str, int] = {}
        steps = [counting_step("s0", calls), counting_step("s1", calls, fail_first=True)]
        store = InMemoryCheckpointStore()
        hitl = InMemoryHitlRequestStore()
        seq = Sequential(
            name="w",
            steps=steps,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        with pytest.raises(RuntimeError):
            await seq.execute("in")

        def factory(ctx: ResumeContext) -> DurableRun:
            return DurableRun(seq, hitl_store=ctx.hitl_store, checkpoint_store=ctx.checkpoint_store, run_id="r")

        service = ResumeService(hitl_store=hitl, checkpoint_store=store, factory=factory)
        result = await service.resume_interrupted("r")

        assert isinstance(result, ResumeResult)
        assert result.output == "in->s0->s1"
        assert calls == {"s0": 1, "s1": 2}

    async def test_rejects_hitl_suspension(self) -> None:
        store = InMemoryCheckpointStore()
        await store.save(
            RunCheckpoint(
                run_id="r",
                checkpoint_type="orchestration",
                state={},
                suspension_info=_suspension(),
            )
        )

        def factory(ctx: ResumeContext) -> DurableRun:  # pragma: no cover
            raise AssertionError("factory must not run for a suspension checkpoint")

        service = ResumeService(hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, factory=factory)
        with pytest.raises(ValueError, match="HITL suspension awaiting"):
            await service.resume_interrupted("r")

    async def test_rejects_when_no_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()

        def factory(ctx: ResumeContext) -> DurableRun:  # pragma: no cover
            raise AssertionError("factory must not run when no checkpoint exists")

        service = ResumeService(hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, factory=factory)
        with pytest.raises(ValueError, match="No checkpoint for run_id"):
            await service.resume_interrupted("missing")

    async def test_rejects_factory_returning_non_durable_run(self) -> None:
        calls: dict[str, int] = {}
        steps = [counting_step("s0", calls), counting_step("s1", calls, fail_first=True)]
        store = InMemoryCheckpointStore()
        seq = Sequential(
            name="w",
            steps=steps,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        with pytest.raises(RuntimeError):
            await seq.execute("in")

        def factory(ctx: ResumeContext) -> DurableRun:
            return "not a durable run"  # type: ignore[return-value]

        service = ResumeService(hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, factory=factory)
        with pytest.raises(TypeError, match="must return a DurableRun"):
            await service.resume_interrupted("r")


# ---------------------------------------------------------------------------
# Phase 2b — agent-internal (tool-call granularity) crash-resume for ReAct.
#
# A single ReAct agent checkpoints after each *completed* tool batch via the
# orchestration-provided sink, so an interrupted run resumes from the last
# completed batch without re-firing tools already done (they replay from the
# agent's message history). The in-flight batch at crash time is the one-step
# replay window (design-rationale §2): its side effects may repeat.
# ---------------------------------------------------------------------------


def _act_tool(calls: dict[str, int]) -> FunctionTool:
    """A side-effecting tool whose calls are counted by the ``calls`` spy.

    ``act("boom")`` raises on its *first* invocation only, simulating a crash
    mid-batch; on a later (resume) invocation it succeeds — letting a test
    observe the in-flight batch repeating exactly once.
    """

    @tool("act", "Perform a labelled side effect and record the call")
    async def act(label: str) -> str:
        calls[label] = calls.get(label, 0) + 1
        if label == "boom" and calls[label] == 1:
            raise RuntimeError("simulated crash mid-batch")
        return f"did {label}"

    return act


def _tool_call(call_id: str, label: str) -> ToolCall:
    return ToolCall(id=call_id, name="act", arguments={"label": label})


def _react_agent(
    tools: list[FunctionTool],
    responses: list[object],
    *,
    name: str = "react",
) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(responses),
        emitter=make_emitter(),
        system_prompt="You are a test agent.",
        tools=tools,
        error_handler=ErrorHandler.fail_fast(),  # tool errors propagate as a crash
    )


class TestReActCrashResume:
    async def test_completed_tools_fire_once_inflight_batch_repeats(self) -> None:
        """A single ReAct agent crashes on its third batch; resume re-fires only
        the in-flight batch.

        Original run: ``act(a)`` and ``act(b)`` complete (each sinks a cursor),
        then ``act(boom)`` raises → crash. Resume loads the last completed-batch
        cursor (after ``act(b)``); the restored message history already holds the
        ``a``/``b`` results, so re-entering the loop replays them without
        re-dispatch. The resumed LLM re-issues the ``boom`` batch — the one-step
        replay window — which now succeeds, then finishes.
        """
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()

        agent = _react_agent(
            [_act_tool(calls)],
            [
                make_response("step a", tool_calls=[_tool_call("c1", "a")]),
                make_response("step b", tool_calls=[_tool_call("c2", "b")]),
                make_response("step boom", tool_calls=[_tool_call("c3", "boom")]),
            ],
        )
        durable = DurableRun(
            agent,
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        with pytest.raises(ToolExecutionError, match="simulated crash"):
            await durable.start("go")

        # Two completed batches were journaled; the latest cursor points at the
        # agent step (index 0) and carries a completed-batch agent snapshot.
        journal = await store.load_journal("r")
        assert [rec.step_path for rec in journal] == [
            "sequential#0:react/turn#1",
            "sequential#0:react/turn#2",
        ]
        assert all(rec.step_kind == "tool_call" for rec in journal)
        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.checkpoint_reason == "step"
        assert cursor.suspension_info is None
        assert cursor.state["suspended_step_index"] == 0
        assert "suspended_tool_index" not in cursor.state["agent_checkpoint"]

        # Resume with a fresh agent (the factory pattern Studio's worker uses).
        resumed = _react_agent(
            [_act_tool(calls)],
            [
                make_response("retry boom", tool_calls=[_tool_call("c4", "boom")]),
                make_response("all done"),
            ],
        )
        durable_resume = DurableRun(
            resumed,
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        result = await durable_resume.resume_from_checkpoint()

        assert isinstance(result, ResumeResult)
        assert result.output == "all done"
        # Completed tools fired exactly once across the original run + resume;
        # only the in-flight batch at crash (``boom``) repeated — the one-step
        # replay window of design-rationale §2.
        assert calls["a"] == 1
        assert calls["b"] == 1
        assert calls["boom"] == 2

    async def test_crash_resume_restores_limiter_and_working_memory(self) -> None:
        """Crash-resume restores the tool-call limiter count and working memory
        from the completed-batch snapshot.

        Covers the two restore branches distinct from ``_execute_resume``: an
        agent configured with ``max_tool_calls`` and ``working_memory`` writes
        both into its snapshot after the first completed batch, and the resumed
        agent re-seeds them before continuing.
        """
        from nanitics.capabilities.memory.working_memory import InMemoryWorkingMemory

        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()

        agent = ReActAgent(
            name="react",
            llm_client=MockLLMClient(
                [
                    make_response(
                        "<working_memory>## Notes\nseeded</working_memory>",
                        tool_calls=[_tool_call("c1", "a")],
                    ),
                    make_response("step boom", tool_calls=[_tool_call("c2", "boom")]),
                ]
            ),
            emitter=make_emitter(),
            system_prompt="You are a test agent.",
            tools=[_act_tool(calls)],
            error_handler=ErrorHandler.fail_fast(),
            max_tool_calls=10,
            working_memory=InMemoryWorkingMemory(),
        )
        durable = DurableRun(
            agent, hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, run_id="r", step_checkpoints=True
        )
        with pytest.raises(ToolExecutionError, match="simulated crash"):
            await durable.start("go")

        cursor = await store.load("r")
        assert cursor is not None
        snapshot = cursor.state["agent_checkpoint"]
        assert "seeded" in snapshot["working_memory"]
        assert snapshot["tool_call_limiter_count"] == 1

        resumed_wm = InMemoryWorkingMemory()
        resumed = ReActAgent(
            name="react",
            llm_client=MockLLMClient([make_response("done")]),
            emitter=make_emitter(),
            system_prompt="You are a test agent.",
            tools=[_act_tool(calls)],
            error_handler=ErrorHandler.fail_fast(),
            max_tool_calls=10,
            working_memory=resumed_wm,
        )
        durable_resume = DurableRun(
            resumed, hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, run_id="r", step_checkpoints=True
        )
        result = await durable_resume.resume_from_checkpoint()

        assert isinstance(result, ResumeResult)
        assert result.output == "done"
        assert "seeded" in (resumed_wm.read() or "")  # working memory re-seeded on resume
        assert calls["a"] == 1  # completed batch fired once

    async def test_resume_via_resume_interrupted(self) -> None:
        """The same crash resumes through ``ResumeService.resume_interrupted``."""
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        hitl = InMemoryHitlRequestStore()

        agent = _react_agent(
            [_act_tool(calls)],
            [
                make_response("step a", tool_calls=[_tool_call("c1", "a")]),
                make_response("step boom", tool_calls=[_tool_call("c2", "boom")]),
            ],
        )
        durable = DurableRun(agent, hitl_store=hitl, checkpoint_store=store, run_id="r", step_checkpoints=True)
        with pytest.raises(ToolExecutionError, match="simulated crash"):
            await durable.start("go")

        def factory(ctx: ResumeContext) -> DurableRun:
            resumed = _react_agent([_act_tool(calls)], [make_response("done")])
            return DurableRun(
                resumed,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
                run_id="r",
                step_checkpoints=True,
            )

        service = ResumeService(hitl_store=hitl, checkpoint_store=store, factory=factory)
        result = await service.resume_interrupted("r")

        assert isinstance(result, ResumeResult)
        assert result.output == "done"
        # ``act(a)`` completed and journaled (turn#1 cursor); resume restores it
        # from message history and does not re-fire it. The crashed ``boom``
        # batch never wrote a cursor, and the resumed LLM takes a different,
        # tool-free path (the §1 non-determinism boundary), so ``boom`` fired
        # exactly once (the original crash) and is not replayed.
        assert calls["a"] == 1
        assert calls["boom"] == 1

    async def test_no_sink_when_step_checkpoints_disabled(self) -> None:
        """``step_checkpoints=False`` (default) writes no agent checkpoints and
        leaves the sink unset — byte-for-byte today's behaviour."""
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()

        agent = _react_agent(
            [_act_tool(calls)],
            [
                make_response("step a", tool_calls=[_tool_call("c1", "a")]),
                make_response("final"),
            ],
        )
        durable = DurableRun(
            agent,
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            run_id="r",
        )  # step_checkpoints defaults False

        result = await durable.start("go")

        assert isinstance(result, ResumeResult)
        assert result.output == "final"
        assert calls["a"] == 1
        assert agent._checkpoint_sink is None  # sink never injected
        assert await store.load("r") is None  # no cursor written
        assert await store.load_journal("r") == []  # no journal entries


class TestHITLMidAgentRegression:
    async def test_hitl_suspension_resumes_via_existing_path(self) -> None:
        """A mid-agent HITL suspension still resumes through ``_execute_resume``
        even with ``step_checkpoints=True``.

        ``act(a)`` completes first (writing a step cursor), then an
        approval-wrapped tool suspends mid-batch. The HITL checkpoint is the
        latest (carries ``suspension_info``); resume routes through
        ``ResumeService.resume`` and the agent's state-shape discriminator picks
        the suspended-batch path because the snapshot carries
        ``suspended_tool_index``.
        """
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        hitl = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl)

        first = ReActAgent(
            name="react",
            llm_client=MockLLMClient(
                [
                    make_response("do a", tool_calls=[_tool_call("c1", "a")]),
                    make_response(
                        "need approval",
                        tool_calls=[ToolCall(id="c2", name="approve", arguments={"label": "z"})],
                    ),
                ]
            ),
            emitter=make_emitter(),
            system_prompt="You are a test agent.",
            tools=[
                _act_tool(calls),
                ApprovalWrappedTool(tool=_approve_tool(calls), provider=provider),
            ],
            error_handler=ErrorHandler.fail_fast(),
            run_id="r",
        )
        durable = DurableRun(first, hitl_store=hitl, checkpoint_store=store, run_id="r", step_checkpoints=True)
        suspended = await durable.start("go")
        assert isinstance(suspended, SuspendedRun)
        assert calls["a"] == 1  # the completed batch ran once

        # Latest checkpoint is the HITL suspension, not a step cursor.
        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.suspension_info is not None
        assert "suspended_tool_index" in cursor.state["agent_checkpoint"]

        resumed_agents: list[ReActAgent] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            agent = ReActAgent(
                name="react",
                llm_client=MockLLMClient([make_response("approved and finished")]),
                emitter=make_emitter(),
                system_prompt="You are a test agent.",
                tools=[
                    _act_tool(calls),
                    ApprovalWrappedTool(tool=_approve_tool(calls), provider=provider),
                ],
                error_handler=ErrorHandler.fail_fast(),
                run_id="r",
            )
            resumed_agents.append(agent)
            return DurableRun(
                agent,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
                run_id="r",
                step_checkpoints=True,
            )

        service = ResumeService(hitl_store=hitl, checkpoint_store=store, factory=factory)
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)

        assert isinstance(result, ResumeResult)
        assert result.output == "approved and finished"
        assert calls["z"] == 1  # the approved tool ran exactly once on resume
        assert resumed_agents[0]._resume_state is None


def _approve_tool(calls: dict[str, int]) -> FunctionTool:
    @tool("approve", "A tool that requires human approval before running")
    async def approve(label: str) -> str:
        calls[label] = calls.get(label, 0) + 1
        return f"approved {label}"

    return approve


# ---------------------------------------------------------------------------
# Phase 2c — orchestration-level per-iteration cursor for the Loop orchestrator.
#
# Loop mirrors Sequential's 2a step cursor over iterations: after each
# *continuing* iteration (one that did not satisfy the stop condition) a cursor
# pointing at the next iteration plus a journal record are written, so an
# interrupted loop resumes at iteration+1 without re-running completed
# iterations. The in-flight iteration at crash time may repeat (the one-step
# replay window).
# ---------------------------------------------------------------------------


def _loop_body(fire_counts: dict[str, int], *, crash_input: str | None = None) -> FunctionStep:
    """A loop body that counts invocations per input and chains its output.

    Each iteration receives a distinct input (the prior output), so
    ``fire_counts`` keyed by input is a per-iteration fire count. With
    ``crash_input`` set, the body raises the *first* time it sees that input
    (simulating a crash mid-iteration) and succeeds on the re-run.
    """

    async def fn(x: object) -> str:
        key = str(x)
        fire_counts[key] = fire_counts.get(key, 0) + 1
        if crash_input is not None and key == crash_input and fire_counts[key] == 1:
            raise RuntimeError(f"crash on input {key!r}")
        return f"{key}->o"

    return FunctionStep(name="body", fn=fn)


def _stop_at(n: int) -> Callable[[StepResult, int], bool]:
    return lambda _result, iteration: iteration >= n


class TestLoopStepDurability:
    async def test_cursor_and_journal_written_after_each_continuing_iteration(self) -> None:
        fire_counts: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        loop = Loop(
            name="w",
            step=_loop_body(fire_counts),
            condition=_stop_at(3),
            max_iterations=5,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        result = await loop.execute("in")

        assert result.output == "in->o->o->o"  # 3 iterations chained
        journal = await store.load_journal("r")
        # Only the two *continuing* iterations journal; the stopping one does not.
        assert [rec.step_path for rec in journal] == ["loop#1", "loop#2"]
        assert all(rec.step_kind == "orchestration_step" for rec in journal)
        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.checkpoint_reason == "step"
        assert cursor.suspension_info is None
        assert cursor.state["iteration"] == 3  # points at the next iteration

    async def test_resume_skips_completed_iterations(self) -> None:
        fire_counts: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        loop = Loop(
            name="w",
            step=_loop_body(fire_counts, crash_input="in->o->o"),  # crash entering iteration 3
            condition=_stop_at(3),
            max_iterations=5,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        with pytest.raises(RuntimeError, match="crash on input"):
            await loop.execute("in")
        assert fire_counts == {"in": 1, "in->o": 1, "in->o->o": 1}

        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.state["iteration"] == 3
        result = await loop.execute("in", resume_from=cursor)

        assert result.output == "in->o->o->o"
        # Iterations 1 and 2 completed and journaled — not re-run on resume.
        # Iteration 3 was in flight at crash — it repeats exactly once (the
        # one-step replay window of design-rationale §2).
        assert fire_counts == {"in": 1, "in->o": 1, "in->o->o": 2}

    async def test_no_checkpoints_when_disabled(self) -> None:
        fire_counts: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        loop = Loop(
            name="w",
            step=_loop_body(fire_counts),
            condition=_stop_at(3),
            max_iterations=5,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
        )  # step_checkpoints defaults False

        await loop.execute("in")

        assert await store.load("r") is None
        assert await store.load_journal("r") == []

    async def test_durable_run_resumes_interrupted_loop(self) -> None:
        fire_counts: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        loop = Loop(
            name="w",
            step=_loop_body(fire_counts, crash_input="in->o->o"),
            condition=_stop_at(3),
            max_iterations=5,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        durable = DurableRun(loop, hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, run_id="r")
        with pytest.raises(RuntimeError, match="crash on input"):
            await durable.start("in")

        result = await durable.resume_from_checkpoint()

        assert isinstance(result, ResumeResult)
        assert result.output == "in->o->o->o"
        assert fire_counts == {"in": 1, "in->o": 1, "in->o->o": 2}

    async def test_nested_workflow_body_resumes_from_step_cursor(self) -> None:
        """A Loop whose body is a nested ``WorkflowStep`` resumes from a step
        cursor (which carries no ``nested_checkpoint``) by running the iteration
        fresh — exercising the softened resume branch that previously asserted a
        nested frame was always present.
        """
        fire_counts: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        inner = Sequential(
            name="inner",
            steps=[_loop_body(fire_counts, crash_input="in->o")],  # crash entering iteration 2
            emitter=make_emitter(),
        )
        loop = Loop(
            name="w",
            step=WorkflowStep(inner),
            condition=_stop_at(2),
            max_iterations=5,
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        with pytest.raises(RuntimeError, match="crash on input"):
            await loop.execute("in")
        assert fire_counts == {"in": 1, "in->o": 1}

        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.state["iteration"] == 2
        assert "nested_checkpoint" not in cursor.state  # a step cursor, not a suspension
        result = await loop.execute("in", resume_from=cursor)

        assert result.output == "in->o->o"
        assert fire_counts == {"in": 1, "in->o": 2}  # iteration 2 (in flight) repeats once


# ---------------------------------------------------------------------------
# Phase 2c — orchestration-level step cursor for the concurrent orchestrators
# (Parallel + DAG).
#
# Unlike Sequential/Loop there is no single integer cursor: branches/nodes run
# concurrently and complete in arbitrary order. The completed set on resume is
# therefore reconstructed from the append-only *journal* (an order-independent
# union keyed by step path), NOT from the cursor checkpoint — so concurrent
# completions cannot clobber each other and the latest-by-created_at cursor need
# only carry ``original_input``. Completed+journaled branches/nodes run at most
# once; the branches/nodes in flight at crash (up to the degree of concurrency)
# may each repeat — the concurrent generalization of the one-step replay window.
# Agent-internal per-branch tool-call durability is OUT of this slice (whole
# branch/node = one durable step), mirroring the Loop slice.
# ---------------------------------------------------------------------------


def _usage_step(name: str, calls: dict[str, int]) -> FunctionStep:
    """A branch that returns a usage-bearing ``StepResult``.

    Exercises the ``usage is not None`` arms of the serialize/restore round-trip
    that the concurrent orchestrators use for journal records.
    """

    async def fn(x: object) -> StepResult:
        calls[name] = calls.get(name, 0) + 1
        return StepResult(output=f"{x}->{name}", usage=Usage(input_tokens=1, output_tokens=2))

    return FunctionStep(name=name, fn=fn)


def _branch_crashing_after(
    name: str,
    calls: dict[str, int],
    *,
    store: InMemoryCheckpointStore,
    run_id: str,
    after: int,
) -> FunctionStep:
    """A branch that crashes its first call, but only once ``after`` peer journal
    records exist.

    Polling the shared journal makes "peers completed + journaled, this branch
    in flight" deterministic under concurrency: the branch yields until its peers
    have been journaled by the orchestrator's drain loop, then raises — so a crash
    cannot pre-empt the peers' journal writes regardless of task scheduling order.
    """

    async def fn(x: object) -> str:
        while len(await store.load_journal(run_id)) < after:
            await asyncio.sleep(0)
        calls[name] = calls.get(name, 0) + 1
        if calls[name] == 1:
            raise RuntimeError(f"simulated crash in {name}")
        return f"{x}->{name}"

    return FunctionStep(name=name, fn=fn)


def _suspending_then_ok(name: str, calls: dict[str, int]) -> FunctionStep:
    """A branch/node that suspends for HITL on its first call, succeeds on resume."""

    async def fn(x: object) -> str:
        calls[name] = calls.get(name, 0) + 1
        if calls[name] == 1:
            raise SuspendExecution(suspension_info=_suspension())
        return f"{x}->{name}"

    return FunctionStep(name=name, fn=fn)


class TestParallelStepDurability:
    async def test_cursor_and_journal_written_after_each_branch(self) -> None:
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        par = Parallel(
            name="w",
            steps=[counting_step("a", calls), _usage_step("b", calls), counting_step("c", calls)],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        result = await par.execute("in")

        assert result.output == ["in->a", "in->b", "in->c"]
        journal = await store.load_journal("r")
        assert {rec.step_path for rec in journal} == {"parallel#0:a", "parallel#1:b", "parallel#2:c"}
        assert all(rec.step_kind == "orchestration_step" for rec in journal)
        # The usage-bearing branch round-trips its usage into the journal.
        b_rec = next(r for r in journal if r.step_path == "parallel#1:b")
        assert b_rec.result["usage"]["input_tokens"] == 1
        a_rec = next(r for r in journal if r.step_path == "parallel#0:a")
        assert a_rec.result["usage"] is None
        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.checkpoint_reason == "step"
        assert cursor.suspension_info is None
        # The cursor is thin — it carries only original_input, not the completed
        # set (that comes from the journal union on resume).
        assert cursor.state == {"orchestrator_type": "parallel", "original_input": "in"}

    async def test_resume_skips_completed_branches_inflight_repeats(self) -> None:
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        # "a" (usage None) and "u" (usage-bearing) complete and journal; "b"
        # crashes once the two peers are journaled — deterministically in flight.
        par = Parallel(
            name="w",
            steps=[
                counting_step("a", calls),
                _usage_step("u", calls),
                _branch_crashing_after("b", calls, store=store, run_id="r", after=2),
            ],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        with pytest.raises(RuntimeError, match="simulated crash in b"):
            await par.execute("in")
        assert calls == {"a": 1, "u": 1, "b": 1}

        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.suspension_info is None  # a step cursor, routes to step-resume
        result = await par.execute("in", resume_from=cursor)

        assert result.output == ["in->a", "in->u", "in->b"]
        # "a"/"u" completed + journaled → restored, not re-run. "b" was in flight
        # at crash → repeats exactly once (the one-step replay window).
        assert calls == {"a": 1, "u": 1, "b": 2}

    async def test_resume_after_full_completion_reruns_nothing(self) -> None:
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        par = Parallel(
            name="w",
            steps=[counting_step("a", calls), counting_step("b", calls)],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        await par.execute("in")
        assert calls == {"a": 1, "b": 1}
        cursor = await store.load("r")
        assert cursor is not None

        # All branches are in the journal — resume finalizes without re-launching.
        result = await par.execute("in", resume_from=cursor)
        assert result.output == ["in->a", "in->b"]
        assert calls == {"a": 1, "b": 1}

    async def test_no_checkpoints_when_disabled(self) -> None:
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        par = Parallel(
            name="w",
            steps=[counting_step("a", calls), counting_step("b", calls)],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
        )  # step_checkpoints defaults False

        await par.execute("in")

        assert await store.load("r") is None
        assert await store.load_journal("r") == []

    async def test_hitl_suspension_resumes_via_existing_path(self) -> None:
        """A mid-orchestration HITL suspension still resumes through the
        unchanged suspend branch even with ``step_checkpoints=True``.

        "a" completes (writing a step cursor); "b" suspends. The latest checkpoint
        is the HITL suspension (carries ``suspension_info``), so resume routes to
        the suspend branch — which re-runs only the suspended branch.
        """
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        par = Parallel(
            name="w",
            steps=[counting_step("a", calls), _suspending_then_ok("b", calls)],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        with pytest.raises(SuspendExecution):
            await par.execute("in")

        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.suspension_info is not None  # HITL suspension, not a step cursor
        assert cursor.state["suspended_branch"] == "b"
        assert "a" in cursor.state["completed_branches"]

        result = await par.execute("in", resume_from=cursor)
        assert result.output == ["in->a", "in->b"]
        assert calls == {"a": 1, "b": 2}  # only the suspended branch re-ran


class TestDAGStepDurability:
    async def test_cursor_and_journal_written_after_each_node(self) -> None:
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        dag = DAG(
            name="w",
            nodes={
                "a": DAGNode(step=counting_step("a", calls)),
                "b": DAGNode(step=counting_step("b", calls), depends_on=["a"]),
            },
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        result = await dag.execute("in")

        assert result.output == "in->a->b"
        journal = await store.load_journal("r")
        assert {rec.step_path for rec in journal} == {"dag#a", "dag#b"}
        assert all(rec.step_kind == "orchestration_step" for rec in journal)
        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.checkpoint_reason == "step"
        assert cursor.suspension_info is None
        assert cursor.state == {"orchestrator_type": "dag", "original_input": "in"}

    async def test_resume_skips_completed_nodes_inflight_repeats(self) -> None:
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        # Two concurrent sources complete + journal; the terminal node (which
        # depends on both) crashes once — deterministically in flight, since its
        # dependencies must be journaled before it becomes ready.
        dag = DAG(
            name="w",
            nodes={
                "a1": DAGNode(step=counting_step("a1", calls)),
                "a2": DAGNode(step=_usage_step("a2", calls)),
                "d": DAGNode(step=_crashing_node("d", calls), depends_on=["a1", "a2"]),
            },
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        with pytest.raises(RuntimeError, match="simulated crash in d"):
            await dag.execute("in")
        assert calls == {"a1": 1, "a2": 1, "d": 1}
        journal = await store.load_journal("r")
        assert {rec.step_path for rec in journal} == {"dag#a1", "dag#a2"}  # d never journaled

        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.suspension_info is None
        result = await dag.execute("in", resume_from=cursor)

        assert result.output == "d-done"
        # The two sources are restored from the journal (not re-run); the terminal
        # node was in flight at crash → repeats exactly once.
        assert calls == {"a1": 1, "a2": 1, "d": 2}

    async def test_no_checkpoints_when_disabled(self) -> None:
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        dag = DAG(
            name="w",
            nodes={
                "a": DAGNode(step=counting_step("a", calls)),
                "b": DAGNode(step=counting_step("b", calls), depends_on=["a"]),
            },
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
        )  # step_checkpoints defaults False

        await dag.execute("in")

        assert await store.load("r") is None
        assert await store.load_journal("r") == []

    async def test_hitl_suspension_resumes_via_existing_path(self) -> None:
        """A mid-DAG HITL suspension still resumes through the unchanged suspend
        branch even with ``step_checkpoints=True``."""
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        dag = DAG(
            name="w",
            nodes={
                "a": DAGNode(step=counting_step("a", calls)),
                "b": DAGNode(step=_suspending_then_ok("b", calls), depends_on=["a"]),
            },
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )

        with pytest.raises(SuspendExecution):
            await dag.execute("in")

        cursor = await store.load("r")
        assert cursor is not None
        assert cursor.suspension_info is not None
        assert cursor.state["suspended_node"] == "b"
        assert "a" in cursor.state["completed_nodes"]

        result = await dag.execute("in", resume_from=cursor)
        assert result.output == "in->a->b"
        assert calls == {"a": 1, "b": 2}  # only the suspended node re-ran


def _crashing_node(name: str, calls: dict[str, int]) -> FunctionStep:
    """A DAG node that crashes its first call and succeeds on resume.

    Its dependencies are journaled before it becomes ready (the drain loop
    journals a node before unblocking its dependents), so a crash here leaves the
    dependencies completed + journaled deterministically.
    """

    async def fn(x: object) -> str:
        calls[name] = calls.get(name, 0) + 1
        if calls[name] == 1:
            raise RuntimeError(f"simulated crash in {name}")
        return f"{name}-done"

    return FunctionStep(name=name, fn=fn)


class TestCheckpointLoadTieBreak:
    async def test_suspension_wins_created_at_tie_over_step_cursor(self) -> None:
        """On an equal ``created_at``, ``load()`` returns the HITL suspension, not
        the step cursor.

        With ``step_checkpoints`` enabled, a branch/node completing in the same
        instant a sibling suspends writes a step cursor and a suspension
        checkpoint that can share a microsecond ``created_at``. The suspension
        must win so the pending run routes to HITL ``resume`` rather than being
        rejected by ``resume_interrupted`` and left un-resumable. The step cursor
        is saved FIRST here — the insertion order a naive ``max()`` would return.
        """
        from datetime import UTC, datetime

        store = InMemoryCheckpointStore()
        ts = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)
        step_cursor = RunCheckpoint(
            run_id="r",
            checkpoint_type="orchestration",
            state={"orchestrator_type": "parallel", "original_input": "in"},
            suspension_info=None,
            checkpoint_reason="step",
            created_at=ts,
        )
        hitl = RunCheckpoint(
            run_id="r",
            checkpoint_type="orchestration",
            state={"suspended_branch": "b", "completed_branches": {}, "original_input": "in"},
            suspension_info=_suspension(),
            created_at=ts,
        )

        await store.save(step_cursor)  # inserted first → loses a naive created_at max()
        await store.save(hitl)

        loaded = await store.load("r")
        assert loaded is not None
        assert loaded.suspension_info is not None
        assert loaded.checkpoint_id == hitl.checkpoint_id
