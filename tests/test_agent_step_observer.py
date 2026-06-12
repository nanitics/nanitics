"""Tests for ``AgentStep`` lifecycle observers (``StepObserver``).

An observer attaches ``on_start`` / ``on_complete`` boundary hooks to an
``AgentStep`` so a consumer can persist per-step progress without wrapping the
agent in a custom ``Step``. The headline property: because the step stays a
first-class ``AgentStep``, the orchestration checkpoint sink still reaches the
agent, so tool-batch crash-resume keeps working — the durability a custom-step
wrapper would have forfeited.
"""

from __future__ import annotations

from typing import Any

import pytest

from nanitics.capabilities.errors.handler import ErrorHandler
from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.composition.durability.resume import DurableRun, ResumeResult
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.adapters import AgentStep
from nanitics.composition.orchestration.protocol import StepObserver, StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.hitl import InMemoryHitlRequestStore
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.strategies import ReActAgent, tool
from nanitics.strategies.tools import FunctionTool
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter, make_response


class _RecordingObserver:
    """Records the lifecycle calls it receives, in order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.inputs: list[Any] = []
        self.results: list[StepResult] = []

    async def on_start(self, input: Any) -> None:
        self.events.append("start")
        self.inputs.append(input)

    async def on_complete(self, result: StepResult) -> None:
        self.events.append("complete")
        self.results.append(result)


@tool("add", "Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


def _react_agent(
    tools: list[FunctionTool],
    responses: list[object],
    *,
    name: str = "react",
    run_id: str | None = None,
) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(responses),
        emitter=make_emitter(),
        system_prompt="You are a test agent.",
        tools=tools,
        error_handler=ErrorHandler.fail_fast(),
        run_id=run_id,
    )


def _act_tool(calls: dict[str, int]) -> FunctionTool:
    """A side-effecting tool counted by ``calls``; ``act("boom")`` raises on its
    first invocation only, simulating a crash mid-batch."""

    @tool("act", "Perform a labelled side effect and record the call")
    async def act(label: str) -> str:
        calls[label] = calls.get(label, 0) + 1
        if label == "boom" and calls[label] == 1:
            raise RuntimeError("simulated crash mid-batch")
        return f"did {label}"

    return act


def _tool_call(call_id: str, label: str) -> ToolCall:
    return ToolCall(id=call_id, name="act", arguments={"label": label})


class TestAgentStepObserver:
    def test_recording_observer_satisfies_protocol(self) -> None:
        assert isinstance(_RecordingObserver(), StepObserver)

    async def test_observer_fires_around_bound_agent_step(self) -> None:
        obs = _RecordingObserver()
        agent = _react_agent([], [make_response("hello")])
        workflow = Sequential(name="wf", steps=[AgentStep(agent, observer=obs)], emitter=make_emitter())

        result = await workflow.execute("hi")

        assert obs.events == ["start", "complete"]
        assert obs.inputs == ["hi"]
        assert isinstance(obs.results[0], StepResult)
        assert obs.results[0].output == "hello"
        assert result.metadata["intermediate_results"]["react"].output == "hello"

    async def test_observer_fires_on_standalone_execute(self) -> None:
        obs = _RecordingObserver()
        agent = _react_agent([], [make_response("done")])
        step = AgentStep(agent, observer=obs)

        result = await step.execute("x")

        assert obs.events == ["start", "complete"]
        assert obs.inputs == ["x"]
        assert result.output == "done"
        assert obs.results[0] is result

    async def test_observer_on_complete_skipped_on_suspension(self) -> None:
        obs = _RecordingObserver()
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped = ApprovalWrappedTool(tool=add_tool, provider=provider)
        agent = _react_agent(
            [wrapped],
            [make_response("adding", tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})])],
            run_id="r",
        )
        workflow = Sequential(
            name="wf",
            steps=[AgentStep(agent, observer=obs)],
            emitter=make_emitter(),
            checkpoint_store=InMemoryCheckpointStore(),
            run_id="r",
        )

        with pytest.raises(SuspendExecution):
            await workflow.execute("add 1 and 2")

        # on_start fired before the agent ran; the step suspended before
        # completing, so on_complete never fired.
        assert obs.events == ["start"]

    async def test_observer_preserves_tool_batch_durability(self) -> None:
        """An ``AgentStep`` carrying an observer still receives the
        orchestration checkpoint sink, so the agent journals each completed tool
        batch and a crash resumes at tool-batch granularity — the property a
        custom-step wrapper forfeits."""
        calls: dict[str, int] = {}
        store = InMemoryCheckpointStore()
        obs = _RecordingObserver()

        agent = _react_agent(
            [_act_tool(calls)],
            [
                make_response("step a", tool_calls=[_tool_call("c1", "a")]),
                make_response("step b", tool_calls=[_tool_call("c2", "b")]),
                make_response("step boom", tool_calls=[_tool_call("c3", "boom")]),
            ],
        )
        workflow = Sequential(
            name="wf",
            steps=[AgentStep(agent, observer=obs)],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        durable = DurableRun(workflow, hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, run_id="r")

        with pytest.raises(ToolExecutionError, match="simulated crash"):
            await durable.start("go")

        # The sink reached the agent: completed batches were journaled per turn.
        journal = await store.load_journal("r")
        assert [rec.step_path for rec in journal] == [
            "sequential#0:react/turn#1",
            "sequential#0:react/turn#2",
        ]
        # on_start fired before the agent ran; the crash skipped on_complete.
        assert obs.events == ["start"]

        resumed = _react_agent(
            [_act_tool(calls)],
            [
                make_response("retry boom", tool_calls=[_tool_call("c4", "boom")]),
                make_response("all done"),
            ],
        )
        obs2 = _RecordingObserver()
        workflow2 = Sequential(
            name="wf",
            steps=[AgentStep(resumed, observer=obs2)],
            emitter=make_emitter(),
            checkpoint_store=store,
            run_id="r",
            step_checkpoints=True,
        )
        durable2 = DurableRun(workflow2, hitl_store=InMemoryHitlRequestStore(), checkpoint_store=store, run_id="r")

        result = await durable2.resume_from_checkpoint()

        assert isinstance(result, ResumeResult)
        assert result.output == "all done"
        # Completed tools fired once; only the in-flight batch at crash repeated.
        assert calls["a"] == 1
        assert calls["b"] == 1
        assert calls["boom"] == 2
        # The resumed step completed, so the observer saw the full lifecycle.
        assert obs2.events == ["start", "complete"]
