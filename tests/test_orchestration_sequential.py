import pytest

from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.orchestration.adapters import AgentStep, FunctionStep
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.observability.events import (
    Usage,
    WorkflowCompleteEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from nanitics.safety import CancellationToken
from nanitics.strategies import ReasoningAgent
from tests.testing_helpers import make_emitter, make_response, make_step, make_usage

# ── Helpers ────────────────────────────────────────────────


# ── Construction Tests ─────────────────────────────────────


class TestSequentialConstruction:
    def test_empty_steps_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="at least one step"):
            Sequential(name="empty", steps=[], emitter=emitter)

    def test_satisfies_step_protocol(self) -> None:
        emitter = make_emitter()
        seq = Sequential(name="seq", steps=[make_step("a")], emitter=emitter)
        assert isinstance(seq, Step)

    def test_name_property(self) -> None:
        emitter = make_emitter()
        seq = Sequential(name="my-seq", steps=[make_step("a")], emitter=emitter)
        assert seq.name == "my-seq"


# ── Execution Tests ────────────────────────────────────────


class TestSequentialExecution:
    async def test_two_step_chain(self) -> None:
        async def add_one(x):
            return x + 1

        async def double(x):
            return x * 2

        emitter = make_emitter()
        seq = Sequential(
            name="chain",
            steps=[make_step("add", add_one), make_step("double", double)],
            emitter=emitter,
        )
        result = await seq.execute(5)
        assert result.output == 12  # (5+1)*2

    async def test_three_step_chain(self) -> None:
        async def step_a(x):
            return x + "A"

        async def step_b(x):
            return x + "B"

        async def step_c(x):
            return x + "C"

        emitter = make_emitter()
        seq = Sequential(
            name="abc",
            steps=[
                make_step("a", step_a),
                make_step("b", step_b),
                make_step("c", step_c),
            ],
            emitter=emitter,
        )
        result = await seq.execute("")
        assert result.output == "ABC"

    async def test_output_chaining(self) -> None:
        """Step N+1 receives step N's output."""
        received_inputs = []

        async def capture(x):
            received_inputs.append(x)
            return x + 1

        emitter = make_emitter()
        seq = Sequential(
            name="chain",
            steps=[make_step("s1", capture), make_step("s2", capture)],
            emitter=emitter,
        )
        await seq.execute(10)
        assert received_inputs == [10, 11]

    async def test_intermediate_results_in_metadata(self) -> None:
        async def add_one(x):
            return x + 1

        emitter = make_emitter()
        seq = Sequential(
            name="meta",
            steps=[make_step("first", add_one), make_step("second", add_one)],
            emitter=emitter,
        )
        result = await seq.execute(0)
        intermediate = result.metadata["intermediate_results"]
        assert isinstance(intermediate["first"], StepResult)
        assert isinstance(intermediate["second"], StepResult)
        assert intermediate["first"].output == 1
        assert intermediate["second"].output == 2

    async def test_error_propagation(self) -> None:
        """Step 2 fails, workflow fails."""
        call_count = 0

        async def ok(x):
            nonlocal call_count
            call_count += 1
            return x

        async def fail(x):
            raise RuntimeError("step failed")

        emitter = make_emitter()
        seq = Sequential(
            name="fail",
            steps=[make_step("ok", ok), make_step("fail", fail)],
            emitter=emitter,
        )
        with pytest.raises(RuntimeError, match="step failed"):
            await seq.execute("input")
        assert call_count == 1


# ── Cancellation Tests ─────────────────────────────────────


class TestSequentialCancellation:
    async def test_cancellation_between_steps(self) -> None:
        token = CancellationToken()
        executed = []

        async def step_fn(x):
            executed.append(True)
            token.cancel()  # Cancel after first step
            return x + 1

        async def should_not_run(x):
            executed.append(True)
            return x

        emitter = make_emitter()
        seq = Sequential(
            name="cancel",
            steps=[make_step("s1", step_fn), make_step("s2", should_not_run)],
            emitter=emitter,
            cancellation_token=token,
        )
        result = await seq.execute(0)
        assert len(executed) == 1
        assert result.metadata["terminated"] == "cancelled"
        assert result.output == 1


# ── Event Emission Tests ───────────────────────────────────


class TestSequentialEvents:
    async def test_event_emission_order(self) -> None:
        emitter = make_emitter()

        async def noop(x):
            return x

        seq = Sequential(
            name="events",
            steps=[make_step("s1", noop), make_step("s2", noop)],
            emitter=emitter,
        )
        await seq.execute("input")

        workflow_events = [
            e
            for e in emitter.events
            if isinstance(
                e,
                (
                    WorkflowStartEvent,
                    WorkflowStepCompleteEvent,
                    WorkflowCompleteEvent,
                ),
            )
        ]
        assert isinstance(workflow_events[0], WorkflowStartEvent)
        assert workflow_events[0].workflow_type == "sequential"
        assert workflow_events[0].step_count == 2
        assert isinstance(workflow_events[1], WorkflowStepCompleteEvent)
        assert workflow_events[1].step_name == "s1"
        assert workflow_events[1].step_index == 0
        assert isinstance(workflow_events[2], WorkflowStepCompleteEvent)
        assert workflow_events[2].step_name == "s2"
        assert workflow_events[2].step_index == 1
        assert isinstance(workflow_events[3], WorkflowCompleteEvent)

    async def test_error_event_on_step_failure(self) -> None:
        async def fail(x):
            raise ValueError("boom")

        emitter = make_emitter()
        seq = Sequential(
            name="err",
            steps=[make_step("bad", fail)],
            emitter=emitter,
        )
        with pytest.raises(ValueError):
            await seq.execute("input")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "ValueError"


# ── Usage Aggregation Tests ────────────────────────────────


def _make_agent_step(name: str, input_tokens: int, output_tokens: int, emitter):
    client = MockLLMClient([make_response("ok", usage=make_usage(input_tokens, output_tokens))])
    agent = ReasoningAgent(
        name=name,
        llm_client=client,
        emitter=emitter,
        system_prompt="test",
    )
    return AgentStep(agent)


class TestSequentialUsageAggregation:
    async def test_three_agent_steps_sum_usage(self) -> None:
        emitter = make_emitter()
        seq = Sequential(
            name="agg",
            steps=[
                _make_agent_step("a", 1, 2, emitter),
                _make_agent_step("b", 3, 4, emitter),
                _make_agent_step("c", 5, 6, emitter),
            ],
            emitter=emitter,
        )
        result = await seq.execute("task")
        assert result.usage == Usage(input_tokens=9, output_tokens=12)

    async def test_mixed_agent_and_function_steps(self) -> None:
        emitter = make_emitter()

        async def passthrough(x):
            return x

        seq = Sequential(
            name="mix",
            steps=[
                _make_agent_step("a", 2, 3, emitter),
                FunctionStep(name="noop", fn=passthrough),
                _make_agent_step("b", 4, 5, emitter),
            ],
            emitter=emitter,
        )
        result = await seq.execute("task")
        assert result.usage == Usage(input_tokens=6, output_tokens=8)

    async def test_all_function_steps_yields_none_usage(self) -> None:
        emitter = make_emitter()

        async def passthrough(x):
            return x

        seq = Sequential(
            name="fns",
            steps=[
                FunctionStep(name="a", fn=passthrough),
                FunctionStep(name="b", fn=passthrough),
                FunctionStep(name="c", fn=passthrough),
            ],
            emitter=emitter,
        )
        result = await seq.execute("task")
        assert result.usage is None

    async def test_cancellation_returns_partial_usage(self) -> None:
        emitter = make_emitter()
        token = CancellationToken()

        client = MockLLMClient([make_response("ok", usage=make_usage(7, 8))])
        agent = ReasoningAgent(
            name="first",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
        )

        async def cancel_then_pass(x):
            token.cancel()
            return x

        seq = Sequential(
            name="partial",
            steps=[
                AgentStep(agent),
                FunctionStep(name="canceller", fn=cancel_then_pass),
                _make_agent_step("would-not-run", 100, 200, emitter),
            ],
            emitter=emitter,
            cancellation_token=token,
        )
        result = await seq.execute("task")
        assert result.metadata["terminated"] == "cancelled"
        assert result.usage == Usage(input_tokens=7, output_tokens=8)


class TestSequentialUsageCheckpointResume:
    async def test_resume_with_usage_in_checkpoint(self) -> None:
        emitter = make_emitter()

        seq = Sequential(
            name="resume",
            steps=[
                _make_agent_step("s1", 1, 1, emitter),
                _make_agent_step("s2", 2, 2, emitter),
            ],
            emitter=emitter,
        )

        checkpoint = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="orchestration",
            state={
                "orchestrator_type": "sequential",
                "suspended_step_index": 1,
                "completed_results": {
                    "s1": {
                        "output": "ok",
                        "metadata": {},
                        "usage": Usage(input_tokens=10, output_tokens=20).model_dump(),
                    },
                },
                "last_output": "ok",
                "original_input": "task",
            },
            suspension_info=SuspensionInfo(
                suspension_id="sid",
                request_id="rid",
                request_type="hitl",
                prompt="p",
            ),
        )
        result = await seq.execute("task", resume_from=checkpoint)
        assert result.usage == Usage(input_tokens=12, output_tokens=22)

    async def test_resume_pre_bump_checkpoint_missing_usage_key(self) -> None:
        emitter = make_emitter()

        seq = Sequential(
            name="resume-pre",
            steps=[
                _make_agent_step("s1", 1, 1, emitter),
                _make_agent_step("s2", 3, 5, emitter),
            ],
            emitter=emitter,
        )

        # Synthesized "pre-bump" checkpoint dict: state has no "usage" key.
        checkpoint = RunCheckpoint(
            run_id="run-2",
            checkpoint_type="orchestration",
            state={
                "orchestrator_type": "sequential",
                "suspended_step_index": 1,
                "completed_results": {
                    "s1": {"output": "ok", "metadata": {}},
                },
                "last_output": "ok",
                "original_input": "task",
            },
            suspension_info=SuspensionInfo(
                suspension_id="sid",
                request_id="rid",
                request_type="hitl",
                prompt="p",
            ),
        )
        result = await seq.execute("task", resume_from=checkpoint)
        # Restored step contributes None (missing key); only post-resume agent counts.
        assert result.usage == Usage(input_tokens=3, output_tokens=5)
