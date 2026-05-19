import pytest

from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.loop import Loop
from nanitics.composition.orchestration.protocol import Step
from nanitics.infrastructure.observability.events import (
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from nanitics.safety import CancellationToken
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


# ── Construction Tests ─────────────────────────────────────


class TestLoopConstruction:
    def test_satisfies_step_protocol(self) -> None:
        emitter = make_emitter()
        loop = Loop(
            name="loop",
            step=make_step("s"),
            condition=lambda r, i: True,
            emitter=emitter,
        )
        assert isinstance(loop, Step)

    def test_name_property(self) -> None:
        emitter = make_emitter()
        loop = Loop(
            name="my-loop",
            step=make_step("s"),
            condition=lambda r, i: True,
            emitter=emitter,
        )
        assert loop.name == "my-loop"


# ── Termination Tests ─────────────────────────────────────


class TestLoopTermination:
    async def test_terminate_on_condition(self) -> None:
        async def increment(x):
            return x + 1

        emitter = make_emitter()
        loop = Loop(
            name="until-3",
            step=make_step("inc", increment),
            condition=lambda r, i: r.output >= 3,
            max_iterations=10,
            emitter=emitter,
        )
        result = await loop.execute(0)
        assert result.output == 3
        assert result.metadata["iterations"] == 3

    async def test_reach_max_iterations(self) -> None:
        """Returns last result, does not raise."""

        async def increment(x):
            return x + 1

        emitter = make_emitter()
        loop = Loop(
            name="max-iter",
            step=make_step("inc", increment),
            condition=lambda r, i: False,  # Never stop
            max_iterations=3,
            emitter=emitter,
        )
        result = await loop.execute(0)
        assert result.output == 3
        assert result.metadata["terminated"] == "iteration_limit"
        assert result.metadata["iterations"] == 3

    async def test_single_iteration(self) -> None:
        """Condition True on first result."""
        emitter = make_emitter()
        loop = Loop(
            name="single",
            step=make_step("s"),
            condition=lambda r, i: True,
            emitter=emitter,
        )
        result = await loop.execute("input")
        assert result.output == "input"
        assert result.metadata["iterations"] == 1


# ── Feedback Loop Tests ────────────────────────────────────


class TestLoopFeedback:
    async def test_output_becomes_next_input(self) -> None:
        received_inputs = []

        async def track(x):
            received_inputs.append(x)
            return x + 1

        emitter = make_emitter()
        loop = Loop(
            name="feedback",
            step=make_step("track", track),
            condition=lambda r, i: i >= 3,
            emitter=emitter,
        )
        result = await loop.execute(0)
        assert received_inputs == [0, 1, 2]
        assert result.output == 3


# ── Async Condition Tests ──────────────────────────────────


class TestLoopAsyncCondition:
    async def test_async_condition(self) -> None:
        async def increment(x):
            return x + 1

        async def condition(result, iteration):
            return result.output >= 2

        emitter = make_emitter()
        loop = Loop(
            name="async-cond",
            step=make_step("inc", increment),
            condition=condition,
            emitter=emitter,
        )
        result = await loop.execute(0)
        assert result.output == 2
        assert result.metadata["iterations"] == 2


# ── Cancellation Tests ─────────────────────────────────────


class TestLoopCancellation:
    async def test_cancellation_between_iterations(self) -> None:
        token = CancellationToken()
        call_count = 0

        async def step_fn(x):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                token.cancel()
            return x + 1

        emitter = make_emitter()
        loop = Loop(
            name="cancel",
            step=make_step("s", step_fn),
            condition=lambda r, i: False,
            max_iterations=10,
            emitter=emitter,
            cancellation_token=token,
        )
        result = await loop.execute(0)
        assert call_count == 2
        assert result.metadata["terminated"] == "cancelled"

    async def test_cancellation_before_start(self) -> None:
        token = CancellationToken()
        token.cancel()

        emitter = make_emitter()
        loop = Loop(
            name="cancel-pre",
            step=make_step("s"),
            condition=lambda r, i: True,
            emitter=emitter,
            cancellation_token=token,
        )
        with pytest.raises(Exception, match="cancelled"):
            await loop.execute("input")


# ── Metadata Tests ─────────────────────────────────────────


class TestLoopMetadata:
    async def test_iteration_count_in_metadata(self) -> None:
        async def increment(x):
            return x + 1

        emitter = make_emitter()
        loop = Loop(
            name="meta",
            step=make_step("inc", increment),
            condition=lambda r, i: i >= 2,
            emitter=emitter,
        )
        result = await loop.execute(0)
        assert result.metadata["iterations"] == 2


# ── Event Emission Tests ───────────────────────────────────


class TestLoopEvents:
    async def test_event_emission_per_iteration(self) -> None:
        async def increment(x):
            return x + 1

        emitter = make_emitter()
        loop = Loop(
            name="events",
            step=make_step("inc", increment),
            condition=lambda r, i: i >= 3,
            emitter=emitter,
        )
        await loop.execute(0)

        step_complete_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
        assert len(step_complete_events) == 3

    async def test_workflow_lifecycle_events(self) -> None:
        emitter = make_emitter()
        loop = Loop(
            name="lifecycle",
            step=make_step("s"),
            condition=lambda r, i: True,
            emitter=emitter,
        )
        await loop.execute("input")

        event_types = [e.event_type for e in emitter.events]
        assert "workflow.start" in event_types
        assert "workflow.step.complete" in event_types
        assert "workflow.complete" in event_types

    async def test_start_event_metadata(self) -> None:
        emitter = make_emitter()
        loop = Loop(
            name="meta-events",
            step=make_step("s"),
            condition=lambda r, i: True,
            emitter=emitter,
        )
        await loop.execute("input")

        start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].workflow_type == "loop"
        assert start_events[0].step_count == 1

    async def test_error_emits_workflow_error_event(self) -> None:
        async def fail(x):
            raise RuntimeError("loop step failed")

        emitter = make_emitter()
        loop = Loop(
            name="error-events",
            step=make_step("fail", fail),
            condition=lambda r, i: True,
            emitter=emitter,
        )
        with pytest.raises(RuntimeError, match="loop step failed"):
            await loop.execute("input")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1


# ── Suspension with checkpoint_data ────────────────────────


class TestLoopSuspensionCheckpointData:
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

        loop = Loop(
            name="loop-cp-data",
            step=make_step("A", suspend),
            condition=lambda r, i: True,
            emitter=emitter,
            checkpoint_store=store,
            run_id="test-run",
        )

        with pytest.raises(SuspendExecution):
            await loop.execute("input")

        cp = await store.load("test-run")
        assert cp is not None
        assert cp.state["agent_checkpoint"] == {"agent_state": "paused"}
