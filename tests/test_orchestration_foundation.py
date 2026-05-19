from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.adapters import AgentStep, FunctionStep
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.composition.orchestration.workflow import Workflow
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.observability.emitter import InMemoryEmitter as ConcreteInMemoryEmitter
from nanitics.infrastructure.observability.events import (
    RunCompleteEvent,
    RunFailedEvent,
    RunStartEvent,
    RunSuspendedEvent,
    SpanEndEvent,
    SpanStartEvent,
    WorkflowCompleteEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStructureEvent,
)
from nanitics.infrastructure.observability.storage import InMemoryPersistentTraceStore
from nanitics.safety import CancellationToken
from nanitics.strategies import (
    ReActAgent,
    ReasoningAgent,
)
from nanitics.tracing import InMemoryEmitter
from tests.testing_helpers import make_emitter, make_response

# ── Helpers ────────────────────────────────────────────────


# ── Concrete Workflow Subclass for Testing ABC ─────────────


class StubWorkflow(Workflow):
    """Minimal concrete Workflow for testing ABC behavior."""

    def __init__(
        self,
        *,
        name: str = "test-workflow",
        emitter: InMemoryEmitter,
        cancellation_token: CancellationToken | None = None,
        run_fn: Callable[..., Any] | None = None,
        workflow_type: str = "test",
        step_count: int = 1,
        run_id: str | None = None,
        trace_store: Any = None,
    ) -> None:
        super().__init__(
            name=name,
            emitter=emitter,
            cancellation_token=cancellation_token,
            run_id=run_id,
            trace_store=trace_store,
        )
        self._run_fn = run_fn
        self._wf_type = workflow_type
        self._step_count_val = step_count

    def _workflow_type(self) -> str:
        return self._wf_type

    def _step_count(self) -> int:
        return self._step_count_val

    def _get_step_definitions(self) -> list:
        return []

    async def _run(self, input, *, resume_from=None):
        if self._run_fn:
            return await self._run_fn(input)
        return StepResult(output=input)


# ── Protocol Conformance Tests ─────────────────────────────


class TestStepProtocol:
    def test_agent_step_satisfies_protocol(self) -> None:
        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="proto-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
        )
        step = AgentStep(agent)
        assert isinstance(step, Step)

    def test_function_step_satisfies_protocol(self) -> None:
        async def noop(x):
            return x

        step = FunctionStep(name="noop", fn=noop)
        assert isinstance(step, Step)

    def test_workflow_satisfies_protocol(self) -> None:
        emitter = make_emitter()
        wf = StubWorkflow(emitter=emitter)
        assert isinstance(wf, Step)


# ── StepResult Tests ───────────────────────────────────────


class TestStepResult:
    def test_defaults(self) -> None:
        result = StepResult()
        assert result.output is None
        assert result.metadata == {}

    def test_with_values(self) -> None:
        result = StepResult(output="hello", metadata={"key": "value"})
        assert result.output == "hello"
        assert result.metadata == {"key": "value"}

    def test_frozen(self) -> None:
        result = StepResult(output="x")
        with pytest.raises(ValidationError):
            result.output = "y"


# ── Workflow ABC Tests ─────────────────────────────────────


class TestWorkflowABC:
    async def test_execute_creates_span(self) -> None:
        emitter = make_emitter()
        wf = StubWorkflow(name="span-test", emitter=emitter)
        await wf.execute("input")

        span_starts = [e for e in emitter.events if isinstance(e, SpanStartEvent)]
        span_ends = [e for e in emitter.events if isinstance(e, SpanEndEvent)]
        assert any(s.name == "span-test" for s in span_starts)
        assert any(s.name == "span-test" for s in span_ends)

    async def test_execute_emits_start_and_complete_events(self) -> None:
        emitter = make_emitter()
        wf = StubWorkflow(
            name="event-test",
            emitter=emitter,
            workflow_type="test",
            step_count=2,
        )
        await wf.execute("input")

        start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].workflow_name == "event-test"
        assert start_events[0].workflow_type == "test"
        assert start_events[0].step_count == 2

        complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].workflow_name == "event-test"

    async def test_execute_emits_error_event_on_failure(self) -> None:
        async def failing_run(input):
            raise ValueError("boom")

        emitter = make_emitter()
        wf = StubWorkflow(name="error-test", emitter=emitter, run_fn=failing_run)

        with pytest.raises(ValueError, match="boom"):
            await wf.execute("input")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "ValueError"
        assert error_events[0].error_message == "boom"

    async def test_execute_checks_cancellation(self) -> None:
        token = CancellationToken()
        token.cancel()
        emitter = make_emitter()
        wf = StubWorkflow(name="cancel-test", emitter=emitter, cancellation_token=token)

        with pytest.raises(Exception, match="cancelled"):
            await wf.execute("input")

    async def test_execute_returns_run_result(self) -> None:
        emitter = make_emitter()
        wf = StubWorkflow(name="result-test", emitter=emitter)
        result = await wf.execute("hello")
        assert result.output == "hello"


# ── AgentStep Tests ────────────────────────────────────────


class TestAgentStep:
    async def test_wraps_reasoning_agent(self) -> None:
        client = MockLLMClient([make_response("agent output")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="reasoning-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
        )
        step = AgentStep(agent)
        assert step.name == "reasoning-agent"

        result = await step.execute("some task")
        assert result.output == "agent output"
        assert "total_steps" in result.metadata
        assert "termination_reason" in result.metadata
        assert "usage" in result.metadata

    async def test_wraps_react_agent(self) -> None:
        client = MockLLMClient([make_response("react output")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[],
        )
        step = AgentStep(agent)
        result = await step.execute("task")
        assert result.output == "react output"

    async def test_converts_non_string_input_to_string(self) -> None:
        client = MockLLMClient([make_response("done")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="str-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
        )
        step = AgentStep(agent)
        result = await step.execute(42)
        assert result.output == "done"

    async def test_forwards_parsed_output_when_available(self) -> None:
        class TestFindings(BaseModel):
            topic: str
            key_points: list[str]

        findings_json = '{"topic": "testing", "key_points": ["point1", "point2"]}'
        client = MockLLMClient(
            [
                make_response("I found some things about testing."),
                make_response(findings_json),
            ]
        )
        emitter = make_emitter()
        agent = ReActAgent(
            name="structured-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[],
            output_schema=TestFindings,
        )
        step = AgentStep(agent)
        result = await step.execute("research testing")

        assert isinstance(result.output, TestFindings)
        assert result.output.topic == "testing"
        assert result.output.key_points == ["point1", "point2"]
        assert result.metadata["text_output"] == findings_json
        assert result.metadata["total_steps"] == 2
        assert result.metadata["termination_reason"] == "complete"

    async def test_preserves_text_output_when_no_parsed(self) -> None:
        client = MockLLMClient([make_response("plain text output")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="text-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[],
        )
        step = AgentStep(agent)
        result = await step.execute("task")

        assert result.output == "plain text output"
        assert "text_output" not in result.metadata
        assert result.metadata["total_steps"] == 1


# ── FunctionStep Tests ─────────────────────────────────────


class TestFunctionStep:
    async def test_async_function(self) -> None:
        async def double(x):
            return x * 2

        step = FunctionStep(name="double", fn=double)
        assert step.name == "double"

        result = await step.execute(5)
        assert result.output == 10

    async def test_returns_step_result_directly(self) -> None:
        async def custom(x):
            return StepResult(output=x, metadata={"custom": True})

        step = FunctionStep(name="custom", fn=custom)
        result = await step.execute("data")
        assert result.output == "data"
        assert result.metadata["custom"] is True

    async def test_wraps_non_step_result(self) -> None:
        async def plain(x):
            return {"key": "value"}

        step = FunctionStep(name="plain", fn=plain)
        result = await step.execute("ignored")
        assert result.output == {"key": "value"}
        assert result.metadata == {}

    async def test_none_return(self) -> None:
        async def void(x):
            return None

        step = FunctionStep(name="void", fn=void)
        result = await step.execute("ignored")
        assert result.output is None


# ── Run Lifecycle Tests ────────────────────────────────────


class TestRunLifecycleEvents:
    """Test that workflow execution emits run lifecycle events."""

    async def test_successful_run_emits_start_and_complete(self) -> None:
        emitter = make_emitter()
        wf = StubWorkflow(name="run-test", emitter=emitter)
        await wf.execute("input")

        starts = [e for e in emitter.events if isinstance(e, RunStartEvent)]
        assert len(starts) == 1
        assert starts[0].workflow_name == "run-test"
        assert starts[0].run_id == wf._run_id

        completes = [e for e in emitter.events if isinstance(e, RunCompleteEvent)]
        assert len(completes) == 1
        assert completes[0].run_id == wf._run_id
        assert completes[0].duration_ms >= 0

    async def test_failed_run_emits_start_and_failed(self) -> None:
        async def failing(x):
            raise ValueError("test error")

        emitter = make_emitter()
        wf = StubWorkflow(name="fail-test", emitter=emitter, run_fn=failing)

        with pytest.raises(ValueError, match="test error"):
            await wf.execute("input")

        starts = [e for e in emitter.events if isinstance(e, RunStartEvent)]
        assert len(starts) == 1

        failures = [e for e in emitter.events if isinstance(e, RunFailedEvent)]
        assert len(failures) == 1
        assert failures[0].run_id == wf._run_id
        assert failures[0].error_type == "ValueError"
        assert failures[0].error_message == "test error"

        # No RunCompleteEvent on failure
        completes = [e for e in emitter.events if isinstance(e, RunCompleteEvent)]
        assert len(completes) == 0

    async def test_suspended_run_emits_start_and_suspended(self) -> None:
        suspension_info = SuspensionInfo(
            suspension_id="susp-1",
            suspension_type="hitl",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        )

        async def suspending(x):
            raise SuspendExecution(suspension_info=suspension_info)

        emitter = make_emitter()
        wf = StubWorkflow(name="suspend-test", emitter=emitter, run_fn=suspending)

        with pytest.raises(SuspendExecution):
            await wf.execute("input")

        starts = [e for e in emitter.events if isinstance(e, RunStartEvent)]
        assert len(starts) == 1

        suspensions = [e for e in emitter.events if isinstance(e, RunSuspendedEvent)]
        assert len(suspensions) == 1
        assert suspensions[0].run_id == wf._run_id
        assert suspensions[0].suspension_id == "susp-1"

    async def test_run_id_auto_generated(self) -> None:
        emitter = make_emitter()
        wf = StubWorkflow(emitter=emitter)
        assert wf._run_id is not None
        assert len(wf._run_id) > 0

    async def test_run_id_explicit(self) -> None:
        emitter = make_emitter()
        wf = StubWorkflow(emitter=emitter, run_id="custom-run-id")
        assert wf._run_id == "custom-run-id"

        await wf.execute("input")
        starts = [e for e in emitter.events if isinstance(e, RunStartEvent)]
        assert starts[0].run_id == "custom-run-id"

    async def test_cancelled_run_does_not_register_in_store(self) -> None:
        """A pre-cancelled workflow should not register a run in the store."""
        token = CancellationToken()
        token.cancel()
        emitter = make_emitter()
        store = InMemoryPersistentTraceStore()
        wf = StubWorkflow(
            name="cancel-store",
            emitter=emitter,
            cancellation_token=token,
            trace_store=store,
            run_id="cancelled-run",
        )

        with pytest.raises(Exception, match="cancelled"):
            await wf.execute("input")

        run = await store.get_run("cancelled-run")
        assert run is None

        # No run lifecycle events emitted
        run_starts = [e for e in emitter.events if isinstance(e, RunStartEvent)]
        assert len(run_starts) == 0


class TestRunLifecycleWithStore:
    """Test that workflow updates trace store for run lifecycle."""

    async def test_successful_run_registers_and_completes_in_store(self) -> None:
        emitter = make_emitter()
        store = InMemoryPersistentTraceStore()
        wf = StubWorkflow(name="store-test", emitter=emitter, trace_store=store, run_id="run-1")
        await wf.execute("input")

        run = await store.get_run("run-1")
        assert run is not None
        assert run.status == "completed"
        assert run.trace_id == "test-trace"
        assert run.metadata["workflow_name"] == "store-test"

    async def test_failed_run_updates_store_with_error(self) -> None:
        async def failing(x):
            raise RuntimeError("store fail")

        emitter = make_emitter()
        store = InMemoryPersistentTraceStore()
        wf = StubWorkflow(name="fail-store", emitter=emitter, trace_store=store, run_id="run-2", run_fn=failing)

        with pytest.raises(RuntimeError):
            await wf.execute("input")

        run = await store.get_run("run-2")
        assert run is not None
        assert run.status == "failed"
        assert run.error == "store fail"

    async def test_suspended_run_updates_store(self) -> None:
        suspension_info = SuspensionInfo(
            suspension_id="susp-2",
            suspension_type="hitl",
            request_id="req-2",
            request_type="approval",
            prompt="Approve?",
        )

        async def suspending(x):
            raise SuspendExecution(suspension_info=suspension_info)

        emitter = make_emitter()
        store = InMemoryPersistentTraceStore()
        wf = StubWorkflow(name="suspend-store", emitter=emitter, trace_store=store, run_id="run-3", run_fn=suspending)

        with pytest.raises(SuspendExecution):
            await wf.execute("input")

        run = await store.get_run("run-3")
        assert run is not None
        assert run.status == "suspended"

    async def test_no_store_does_not_error(self) -> None:
        """Workflow works fine without a trace_store — no store calls made."""
        emitter = make_emitter()
        wf = StubWorkflow(name="no-store-test", emitter=emitter)
        result = await wf.execute("input")
        assert result.output == "input"


# ── E2E Integration Test: Workflow with Agents ─────────────


class TestRunLifecycleIntegration:
    """End-to-end: Sequential workflow with agents produces connected trace
    with agent metadata, workflow structure, and run lifecycle events."""

    async def test_sequential_workflow_full_trace(self) -> None:
        from nanitics.composition.orchestration.sequential import Sequential

        emitter = InMemoryEmitter(trace_id="e2e-trace")
        store = InMemoryPersistentTraceStore()

        client_a = MockLLMClient([make_response("output-a")])
        client_b = MockLLMClient([make_response("output-b")])

        agent_a = ReasoningAgent(
            name="agent-a", llm_client=client_a, emitter=InMemoryEmitter(trace_id="unused-a"), system_prompt="test"
        )
        agent_b = ReasoningAgent(
            name="agent-b", llm_client=client_b, emitter=InMemoryEmitter(trace_id="unused-b"), system_prompt="test"
        )

        wf = Sequential(
            name="e2e-workflow",
            steps=[AgentStep(agent_a), AgentStep(agent_b)],
            emitter=emitter,
            trace_store=store,
            run_id="e2e-run",
        )

        result = await wf.execute("start")
        assert result.output == "output-b"

        # 1. All events share the same trace_id. Inner-agent child emitters
        # forward their events into the outer emitter's ``events`` list, so
        # ``emitter.events`` is authoritative.
        all_events = emitter.events
        assert isinstance(agent_a._emitter, ConcreteInMemoryEmitter)
        assert isinstance(agent_b._emitter, ConcreteInMemoryEmitter)
        trace_ids = {e.trace_id for e in all_events}
        assert trace_ids == {"e2e-trace"}

        # 2. Workflow structure event present
        structure_events = [e for e in emitter.events if isinstance(e, WorkflowStructureEvent)]
        assert len(structure_events) == 1
        assert structure_events[0].workflow_type == "sequential"
        assert len(structure_events[0].steps) == 2

        # 3. Run lifecycle events present
        run_starts = [e for e in emitter.events if isinstance(e, RunStartEvent)]
        assert len(run_starts) == 1
        assert run_starts[0].run_id == "e2e-run"
        assert run_starts[0].workflow_name == "e2e-workflow"

        run_completes = [e for e in emitter.events if isinstance(e, RunCompleteEvent)]
        assert len(run_completes) == 1
        assert run_completes[0].run_id == "e2e-run"

        # 4. Store has completed run
        run = await store.get_run("e2e-run")
        assert run is not None
        assert run.status == "completed"
        assert run.trace_id == "e2e-trace"

        # 5. Agent metadata present (from 1B)
        from nanitics.infrastructure.observability.events import AgentStartEvent

        agent_starts = [evt for evt in all_events if isinstance(evt, AgentStartEvent)]
        assert len(agent_starts) == 2
        agent_names = {e.agent_name for e in agent_starts}
        assert agent_names == {"agent-a", "agent-b"}
        # All agent events share the same trace_id
        for e in agent_starts:
            assert e.trace_id == "e2e-trace"
            assert e.agent_type == "reasoning"
