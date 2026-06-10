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
from nanitics.composition.orchestration.adapters import FunctionStep
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.hitl import InMemoryHitlRequestStore
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.errors import ToolExecutionError
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
