import asyncio
from typing import Any

import pytest

from nanitics import CancellationToken
from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.parallel import FailurePolicy, Parallel
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.infrastructure.observability.events import (
    WorkflowCompleteEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


# ── Construction Tests ─────────────────────────────────────


class TestParallelConstruction:
    def test_empty_steps_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="at least one step"):
            Parallel(name="empty", steps=[], emitter=emitter)

    def test_satisfies_step_protocol(self) -> None:
        emitter = make_emitter()
        par = Parallel(name="par", steps=[make_step("a")], emitter=emitter)
        assert isinstance(par, Step)


# ── Execution Tests ────────────────────────────────────────


class TestParallelExecution:
    async def test_concurrent_execution(self) -> None:
        """Verify steps run concurrently via timing."""
        timestamps: list[float] = []

        async def slow_step(x):
            timestamps.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return x

        emitter = make_emitter()
        par = Parallel(
            name="concurrent",
            steps=[make_step("s1", slow_step), make_step("s2", slow_step)],
            emitter=emitter,
        )
        await par.execute("input")
        # Both steps should have started nearly at the same time
        assert abs(timestamps[0] - timestamps[1]) < 0.03

    async def test_default_aggregation(self) -> None:
        async def step_a(x):
            return "a"

        async def step_b(x):
            return "b"

        emitter = make_emitter()
        par = Parallel(
            name="agg",
            steps=[make_step("a", step_a), make_step("b", step_b)],
            emitter=emitter,
        )
        result = await par.execute("input")
        assert result.output == ["a", "b"]

    async def test_custom_aggregator(self) -> None:
        async def num(x):
            return x

        def sum_aggregator(results: list[StepResult]) -> Any:
            return sum(r.output for r in results)

        emitter = make_emitter()
        par = Parallel(
            name="sum",
            steps=[make_step("a", num), make_step("b", num)],
            emitter=emitter,
            aggregator=sum_aggregator,
        )
        result = await par.execute(5)
        assert result.output == 10

    async def test_all_steps_receive_same_input(self) -> None:
        received = []

        async def capture(x):
            received.append(x)
            return x

        emitter = make_emitter()
        par = Parallel(
            name="same-input",
            steps=[make_step("a", capture), make_step("b", capture)],
            emitter=emitter,
        )
        await par.execute("shared")
        assert received == ["shared", "shared"]


# ── Failure Policy Tests ───────────────────────────────────


class TestParallelFailurePolicy:
    async def test_all_or_nothing_one_fails(self) -> None:
        async def ok(x):
            return x

        async def fail(x):
            raise RuntimeError("boom")

        emitter = make_emitter()
        par = Parallel(
            name="aon",
            steps=[make_step("ok", ok), make_step("fail", fail)],
            emitter=emitter,
            failure_policy=FailurePolicy.ALL_OR_NOTHING,
        )
        with pytest.raises(RuntimeError, match="boom"):
            await par.execute("input")

    async def test_best_effort_partial_results(self) -> None:
        async def ok(x):
            return "success"

        async def fail(x):
            raise RuntimeError("boom")

        emitter = make_emitter()
        par = Parallel(
            name="best",
            steps=[make_step("ok", ok), make_step("fail", fail)],
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )
        result = await par.execute("input")
        assert result.output == ["success"]
        assert result.metadata["failed_steps"] == ["fail"]

    async def test_best_effort_all_succeed(self) -> None:
        async def ok(x):
            return x

        emitter = make_emitter()
        par = Parallel(
            name="all-ok",
            steps=[make_step("a", ok), make_step("b", ok)],
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )
        result = await par.execute("val")
        assert result.output == ["val", "val"]
        assert "failed_steps" not in result.metadata


# ── Cancellation Tests ─────────────────────────────────────


class TestParallelCancellation:
    async def test_cancellation_before_start(self) -> None:
        token = CancellationToken()
        token.cancel()

        emitter = make_emitter()
        par = Parallel(
            name="cancel",
            steps=[make_step("a")],
            emitter=emitter,
            cancellation_token=token,
        )
        with pytest.raises(Exception, match="cancelled"):
            await par.execute("input")


# ── Event Emission Tests ───────────────────────────────────


class TestParallelEvents:
    async def test_emits_workflow_events(self) -> None:
        async def noop(x):
            return x

        emitter = make_emitter()
        par = Parallel(
            name="events",
            steps=[make_step("s1", noop), make_step("s2", noop)],
            emitter=emitter,
        )
        await par.execute("input")

        start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].workflow_type == "parallel"
        assert start_events[0].step_count == 2

        step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
        assert len(step_events) == 2

        complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]
        assert len(complete_events) == 1

    async def test_error_event_on_all_or_nothing_failure(self) -> None:
        async def fail(x):
            raise ValueError("fail")

        emitter = make_emitter()
        par = Parallel(
            name="err",
            steps=[make_step("bad", fail)],
            emitter=emitter,
        )
        with pytest.raises(ValueError):
            await par.execute("input")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1


# ── ALL_OR_NOTHING with pending tasks ──────────────────────


class TestParallelAllOrNothingCancellation:
    async def test_cancels_pending_tasks_on_failure(self) -> None:
        """When one step fails while others are still running, pending tasks are cancelled."""
        slow_cancelled = asyncio.Event()

        async def fast_fail(x: object) -> None:
            raise RuntimeError("fast boom")

        async def slow_step(x: object) -> str:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                slow_cancelled.set()
                raise
            return "never"

        emitter = make_emitter()
        par = Parallel(
            name="cancel-pending",
            steps=[make_step("slow", slow_step), make_step("fast", fast_fail)],
            emitter=emitter,
            failure_policy=FailurePolicy.ALL_OR_NOTHING,
        )
        with pytest.raises(RuntimeError, match="fast boom"):
            await par.execute("input")

        assert slow_cancelled.is_set()


# ── Suspension with checkpoint_data ────────────────────────


class TestParallelSuspensionCheckpointData:
    async def test_checkpoint_data_included_in_state(self) -> None:
        """SuspendExecution with checkpoint_data includes it in the saved checkpoint."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def suspend(x: object) -> None:
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="test-suspension",
                    request_id="test-request",
                    request_type="approval",
                    prompt="Approve?",
                    agent_name="test-agent",
                ),
                checkpoint_data={"agent_state": "paused"},
            )

        par = Parallel(
            name="par-cp-data",
            steps=[make_step("A", suspend)],
            emitter=emitter,
            checkpoint_store=store,
            run_id="test-run",
        )

        with pytest.raises(SuspendExecution):
            await par.execute("input")

        cp = await store.load("test-run")
        assert cp is not None
        assert cp.state["agent_checkpoint"] == {"agent_state": "paused"}
