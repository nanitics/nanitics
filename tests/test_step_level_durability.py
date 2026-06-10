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

from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.resume import (
    DurableRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
)
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.orchestration.adapters import FunctionStep
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.hitl import InMemoryHitlRequestStore
from tests.testing_helpers import make_emitter, make_step


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
