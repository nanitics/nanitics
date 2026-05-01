import pytest

from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.conditional import Conditional
from nanitics.composition.orchestration.protocol import Step
from nanitics.infrastructure.observability.events import (
    WorkflowErrorEvent,
    WorkflowStartEvent,
)
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


# ── Construction Tests ─────────────────────────────────────


class TestConditionalConstruction:
    def test_empty_branches_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="at least one branch"):
            Conditional(
                name="empty",
                router=lambda x: "a",
                branches={},
                emitter=emitter,
            )

    def test_satisfies_step_protocol(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="cond",
            router=lambda x: "a",
            branches={"a": make_step("a")},
            emitter=emitter,
        )
        assert isinstance(cond, Step)

    def test_name_property(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="my-cond",
            router=lambda x: "a",
            branches={"a": make_step("a")},
            emitter=emitter,
        )
        assert cond.name == "my-cond"


# ── Routing Tests ──────────────────────────────────────────


class TestConditionalRouting:
    async def test_sync_router(self) -> None:
        async def upper(x):
            return x.upper()

        async def lower(x):
            return x.lower()

        emitter = make_emitter()
        cond = Conditional(
            name="route",
            router=lambda x: "upper" if x == "UP" else "lower",
            branches={
                "upper": make_step("upper", upper),
                "lower": make_step("lower", lower),
            },
            emitter=emitter,
        )
        result = await cond.execute("UP")
        assert result.output == "UP"
        assert result.metadata["selected_branch"] == "upper"

    async def test_async_router(self) -> None:
        async def route(x):
            return "b"

        emitter = make_emitter()
        cond = Conditional(
            name="async-route",
            router=route,
            branches={
                "a": make_step("a"),
                "b": make_step("b"),
            },
            emitter=emitter,
        )
        result = await cond.execute("input")
        assert result.metadata["selected_branch"] == "b"

    async def test_default_branch_fallback(self) -> None:
        async def default_fn(x):
            return "default-output"

        emitter = make_emitter()
        cond = Conditional(
            name="default",
            router=lambda x: "unknown",
            branches={"a": make_step("a")},
            default=make_step("fallback", default_fn),
            emitter=emitter,
        )
        result = await cond.execute("input")
        assert result.output == "default-output"
        assert result.metadata["selected_branch"] == "default(unknown)"

    async def test_missing_branch_no_default_raises(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="no-default",
            router=lambda x: "missing",
            branches={"a": make_step("a"), "b": make_step("b")},
            emitter=emitter,
        )
        with pytest.raises(ValueError, match="unknown branch 'missing'"):
            await cond.execute("input")

    async def test_branch_receives_workflow_input(self) -> None:
        received = []

        async def capture(x):
            received.append(x)
            return x

        emitter = make_emitter()
        cond = Conditional(
            name="input-pass",
            router=lambda x: "cap",
            branches={"cap": make_step("cap", capture)},
            emitter=emitter,
        )
        await cond.execute("my-input")
        assert received == ["my-input"]

    async def test_selected_branch_in_metadata(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="meta",
            router=lambda x: "chosen",
            branches={"chosen": make_step("chosen"), "other": make_step("other")},
            emitter=emitter,
        )
        result = await cond.execute("input")
        assert result.metadata["selected_branch"] == "chosen"


# ── Event Emission Tests ───────────────────────────────────


class TestConditionalEvents:
    async def test_event_emission(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="events",
            router=lambda x: "a",
            branches={"a": make_step("a")},
            emitter=emitter,
        )
        await cond.execute("input")

        event_types = [e.event_type for e in emitter.events]
        assert "workflow.start" in event_types
        assert "workflow.step.complete" in event_types
        assert "workflow.complete" in event_types

    async def test_start_event_metadata(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="meta-events",
            router=lambda x: "a",
            branches={"a": make_step("a"), "b": make_step("b")},
            emitter=emitter,
        )
        await cond.execute("input")

        start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].workflow_type == "conditional"
        assert start_events[0].step_count == 2

    async def test_error_emits_workflow_error_event(self) -> None:
        async def fail(x):
            raise RuntimeError("branch failed")

        emitter = make_emitter()
        cond = Conditional(
            name="error-events",
            router=lambda x: "fail",
            branches={"fail": make_step("fail", fail)},
            emitter=emitter,
        )
        with pytest.raises(RuntimeError, match="branch failed"):
            await cond.execute("input")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "RuntimeError"


# ── Resume with default branch ─────────────────────────────


class TestConditionalResumeDefault:
    async def test_resume_falls_back_to_default_step(self) -> None:
        """When resuming and the stored branch is not in branches dict, the default step is used."""
        executed: list[str] = []

        async def default_fn(x: object) -> str:
            executed.append("default")
            return "default-output"

        emitter = make_emitter()
        cond = Conditional(
            name="resume-default",
            router=lambda x: "a",
            branches={"a": make_step("a")},
            default=make_step("fallback", default_fn),
            emitter=emitter,
        )

        checkpoint = RunCheckpoint(
            run_id="test-run",
            checkpoint_type="orchestration",
            state={
                "orchestrator_type": "conditional",
                "selected_branch": "unknown-branch",
                "original_input": "input",
            },
            suspension_info=SuspensionInfo(
                suspension_id="test-suspension",
                request_id="test-request",
                request_type="approval",
                prompt="Approve?",
            ),
        )

        result = await cond.execute("input", resume_from=checkpoint)
        assert executed == ["default"]
        assert result.output == "default-output"


# ── Suspension with checkpoint_data ────────────────────────


class TestConditionalSuspensionCheckpointData:
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

        cond = Conditional(
            name="cond-cp-data",
            router=lambda x: "a",
            branches={"a": make_step("a", suspend)},
            emitter=emitter,
            checkpoint_store=store,
            run_id="test-run",
        )

        with pytest.raises(SuspendExecution):
            await cond.execute("input")

        cp = await store.load("test-run")
        assert cp is not None
        assert cp.state["agent_checkpoint"] == {"agent_state": "paused"}
