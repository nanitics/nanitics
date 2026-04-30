import pytest
from pydantic import BaseModel

from nanitics import CancellationToken
from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.pipeline import (
    Pipeline,
    PipelineContractError,
    Stage,
)
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.infrastructure.observability.events import (
    WorkflowCompleteEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


# ── Pydantic models for contract tests ─────────────────────


class NumberInput(BaseModel):
    value: int


class NumberOutput(BaseModel):
    value: int
    doubled: int


class TextInput(BaseModel):
    text: str


# ── Construction ───────────────────────────────────────────


class TestPipelineConstruction:
    def test_empty_stages_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="at least one stage"):
            Pipeline(name="empty", stages=[], emitter=emitter)

    def test_satisfies_step_protocol(self) -> None:
        emitter = make_emitter()
        stage = Stage(make_step("a"))
        pipeline = Pipeline(name="p", stages=[stage], emitter=emitter)
        assert isinstance(pipeline, Step)

    def test_name_property(self) -> None:
        emitter = make_emitter()
        stage = Stage(make_step("a"))
        pipeline = Pipeline(name="my-pipeline", stages=[stage], emitter=emitter)
        assert pipeline.name == "my-pipeline"

    def test_stage_delegates_name(self) -> None:
        step = make_step("inner-step")
        stage = Stage(step)
        assert stage.name == "inner-step"


# ── Execution ──────────────────────────────────────────────


class TestPipelineExecution:
    async def test_single_stage_passthrough(self) -> None:
        emitter = make_emitter()

        async def double(x):
            return x * 2

        stage = Stage(make_step("double", double))
        pipeline = Pipeline(name="p", stages=[stage], emitter=emitter)

        result = await pipeline.execute(5)
        assert result.output == 10

    async def test_multi_stage_chaining(self) -> None:
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        async def multiply_three(x):
            return x * 3

        pipeline = Pipeline(
            name="chain",
            stages=[
                Stage(make_step("add", add_one)),
                Stage(make_step("mul", multiply_three)),
            ],
            emitter=emitter,
        )

        result = await pipeline.execute(2)
        # (2 + 1) * 3 = 9
        assert result.output == 9

    async def test_output_forwarding_between_stages(self) -> None:
        emitter = make_emitter()
        received_inputs: list = []

        async def capture_and_transform(x):
            received_inputs.append(x)
            return f"transformed-{x}"

        async def capture_final(x):
            received_inputs.append(x)
            return f"final-{x}"

        pipeline = Pipeline(
            name="forward",
            stages=[
                Stage(make_step("first", capture_and_transform)),
                Stage(make_step("second", capture_final)),
            ],
            emitter=emitter,
        )

        result = await pipeline.execute("input")
        assert received_inputs == ["input", "transformed-input"]
        assert result.output == "final-transformed-input"

    async def test_metadata_includes_intermediate_results(self) -> None:
        emitter = make_emitter()

        async def step_a(x):
            return x + 1

        async def step_b(x):
            return x * 2

        pipeline = Pipeline(
            name="meta",
            stages=[
                Stage(make_step("a", step_a)),
                Stage(make_step("b", step_b)),
            ],
            emitter=emitter,
        )

        result = await pipeline.execute(5)
        intermediate = result.metadata["intermediate_results"]
        assert isinstance(intermediate["a"], StepResult)
        assert isinstance(intermediate["b"], StepResult)
        assert intermediate["a"].output == 6
        assert intermediate["b"].output == 12
        assert result.metadata["total_steps_executed"] == 2


# ── Contract Validation ────────────────────────────────────


class TestPipelineContracts:
    async def test_input_contract_violation(self) -> None:
        emitter = make_emitter()
        stage = Stage(
            make_step("typed"),
            input_type=NumberInput,
        )
        pipeline = Pipeline(name="p", stages=[stage], emitter=emitter)

        with pytest.raises(PipelineContractError) as exc_info:
            await pipeline.execute("not a number input")

        err = exc_info.value
        assert err.stage_name == "typed"
        assert err.stage_index == 0
        assert err.direction == "input"
        assert err.expected_type == "NumberInput"

    async def test_output_contract_violation(self) -> None:
        emitter = make_emitter()

        async def returns_wrong_type(x):
            return "not a NumberOutput"

        stage = Stage(
            make_step("typed", returns_wrong_type),
            output_type=NumberOutput,
        )
        pipeline = Pipeline(name="p", stages=[stage], emitter=emitter)

        with pytest.raises(PipelineContractError) as exc_info:
            await pipeline.execute({"value": 5})

        err = exc_info.value
        assert err.stage_name == "typed"
        assert err.stage_index == 0
        assert err.direction == "output"
        assert err.expected_type == "NumberOutput"

    async def test_valid_contracts_pass(self) -> None:
        emitter = make_emitter()

        async def transform(x):
            return {"value": x["value"], "doubled": x["value"] * 2}

        stage = Stage(
            make_step("typed", transform),
            input_type=NumberInput,
            output_type=NumberOutput,
        )
        pipeline = Pipeline(name="p", stages=[stage], emitter=emitter)

        result = await pipeline.execute({"value": 5})
        assert result.output == {"value": 5, "doubled": 10}

    async def test_stages_without_contracts_work_like_sequential(self) -> None:
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        pipeline = Pipeline(
            name="untyped",
            stages=[
                Stage(make_step("a", add_one)),
                Stage(make_step("b", add_one)),
            ],
            emitter=emitter,
        )

        result = await pipeline.execute(0)
        assert result.output == 2

    async def test_input_contract_on_second_stage(self) -> None:
        emitter = make_emitter()

        async def produce_text(x):
            return {"text": f"hello-{x}"}

        pipeline = Pipeline(
            name="p",
            stages=[
                Stage(make_step("first", produce_text)),
                Stage(make_step("second"), input_type=NumberInput),
            ],
            emitter=emitter,
        )

        with pytest.raises(PipelineContractError) as exc_info:
            await pipeline.execute("anything")

        err = exc_info.value
        assert err.stage_index == 1
        assert err.direction == "input"


# ── Cancellation ───────────────────────────────────────────


class TestPipelineCancellation:
    async def test_respects_cancellation_between_stages(self) -> None:
        emitter = make_emitter()
        token = CancellationToken()

        executed: list[str] = []

        async def stage_one(x):
            executed.append("one")
            token.cancel()
            return x

        async def stage_two(x):
            executed.append("two")
            return x

        pipeline = Pipeline(
            name="cancel",
            stages=[
                Stage(make_step("one", stage_one)),
                Stage(make_step("two", stage_two)),
            ],
            emitter=emitter,
            cancellation_token=token,
        )

        result = await pipeline.execute("input")
        assert executed == ["one"]
        assert result.metadata.get("terminated") == "cancelled"
        assert result.metadata["total_steps_executed"] == 1


# ── Events ─────────────────────────────────────────────────


class TestPipelineEvents:
    async def test_emits_workflow_events(self) -> None:
        emitter = make_emitter()

        pipeline = Pipeline(
            name="evented",
            stages=[
                Stage(make_step("a")),
                Stage(make_step("b")),
            ],
            emitter=emitter,
        )

        await pipeline.execute("x")

        events = emitter.events
        start_events = [e for e in events if isinstance(e, WorkflowStartEvent)]
        step_events = [e for e in events if isinstance(e, WorkflowStepCompleteEvent)]
        complete_events = [e for e in events if isinstance(e, WorkflowCompleteEvent)]

        assert len(start_events) == 1
        assert start_events[0].workflow_type == "pipeline"
        assert start_events[0].step_count == 2

        assert len(step_events) == 2
        assert step_events[0].step_name == "a"
        assert step_events[0].step_index == 0
        assert step_events[1].step_name == "b"
        assert step_events[1].step_index == 1

        assert len(complete_events) == 1
        assert complete_events[0].workflow_type == "pipeline"

    async def test_emits_error_event_on_failure(self) -> None:
        emitter = make_emitter()

        async def fail(x):
            raise RuntimeError("boom")

        pipeline = Pipeline(
            name="err",
            stages=[Stage(make_step("fail", fail))],
            emitter=emitter,
        )

        with pytest.raises(RuntimeError):
            await pipeline.execute("x")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].workflow_type == "pipeline"
        assert error_events[0].error_message == "boom"


# ── Error Propagation ─────────────────────────────────────


class TestPipelineErrorPropagation:
    async def test_stage_error_propagates(self) -> None:
        emitter = make_emitter()

        async def fail(x):
            raise ValueError("stage failed")

        pipeline = Pipeline(
            name="p",
            stages=[
                Stage(make_step("ok")),
                Stage(make_step("fail", fail)),
            ],
            emitter=emitter,
        )

        with pytest.raises(ValueError, match="stage failed"):
            await pipeline.execute("input")

    async def test_contract_error_is_nanitics_error(self) -> None:
        from nanitics.infrastructure.errors import NaniticsError

        emitter = make_emitter()
        stage = Stage(make_step("typed"), input_type=NumberInput)
        pipeline = Pipeline(name="p", stages=[stage], emitter=emitter)

        with pytest.raises(NaniticsError):
            await pipeline.execute("invalid")


# ── Suspension with checkpoint_data ────────────────────────


class TestPipelineSuspensionCheckpointData:
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

        pipeline = Pipeline(
            name="pipe-cp-data",
            stages=[Stage(make_step("A", suspend))],
            emitter=emitter,
            checkpoint_store=store,
            run_id="test-run",
        )

        with pytest.raises(SuspendExecution):
            await pipeline.execute("input")

        cp = await store.load("test-run")
        assert cp is not None
        assert cp.state["agent_checkpoint"] == {"agent_state": "paused"}


class TestPipelineStageDirectExecute:
    async def test_passthrough_returns_wrapped_step_result(self) -> None:
        stage = Stage(make_step("direct"))
        result = await stage.execute("hello")
        assert result.output == "hello"
