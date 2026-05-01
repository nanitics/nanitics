"""Tests for safety event emission (iteration limit and cancellation)."""

from nanitics import (
    CancellationToken,
    ExecutionResult,
    LLMResponse,
    MockLLMClient,
    MockSandbox,
    ToolCall,
    tool,
)
from nanitics.core.agents.codeact import CodeActAgent
from nanitics.core.agents.react import ReActAgent
from nanitics.infrastructure import (
    SafetyCancellationEvent,
    SafetyIterationLimitEvent,
)
from tests.testing_helpers import make_emitter, make_response, make_usage


@tool(name="add", description="Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


_tc_counter = 0


def _next_tc_id() -> str:
    global _tc_counter
    _tc_counter += 1
    return f"tc_{_tc_counter}"


def make_code_response(*code_blocks: str) -> LLMResponse:
    tool_calls = [ToolCall(id=_next_tc_id(), name="execute_code", arguments={"code": code}) for code in code_blocks]
    return LLMResponse(
        content=None,
        tool_calls=tool_calls,
        usage=make_usage(),
        model="test-model",
        stop_reason="tool_use",
    )


def make_exec_result(stdout: str = "", success: bool = True) -> ExecutionResult:
    return ExecutionResult(
        stdout=stdout,
        stderr="",
        return_value=None,
        success=success,
        error=None,
        duration_ms=10.0,
    )


# ──────────────────────────────────────────────────────────
# Iteration Limit Events
# ──────────────────────────────────────────────────────────


class TestIterationLimitEvents:
    async def test_react_emits_iteration_limit_event(self) -> None:
        """ReActAgent emits SafetyIterationLimitEvent when hitting limit."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        responses = [make_response(content="step", tool_calls=[tool_call]) for _ in range(4)]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            max_iterations=3,
        )

        result = await agent.run("Loop forever")

        assert result.termination_reason == "iteration_limit"

        limit_events = [e for e in emitter.events if isinstance(e, SafetyIterationLimitEvent)]
        assert len(limit_events) == 1
        event = limit_events[0]
        assert event.agent_name == "react-agent"
        assert event.max_iterations == 3
        assert event.current_iteration == 4
        assert event.step_number == 3

    async def test_codeact_emits_iteration_limit_event(self) -> None:
        """CodeActAgent emits SafetyIterationLimitEvent when hitting limit."""
        responses = [make_code_response("step()") for _ in range(5)]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result() for _ in range(3)])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            max_iterations=3,
        )

        result = await agent.run("Loop forever")

        assert result.termination_reason == "iteration_limit"

        limit_events = [e for e in emitter.events if isinstance(e, SafetyIterationLimitEvent)]
        assert len(limit_events) == 1
        event = limit_events[0]
        assert event.agent_name == "codeact-agent"
        assert event.max_iterations == 3
        assert event.current_iteration == 4
        assert event.step_number == 3


# ──────────────────────────────────────────────────────────
# Cancellation Events
# ──────────────────────────────────────────────────────────


class TestCancellationEvents:
    async def test_react_emits_cancellation_event(self) -> None:
        """ReActAgent emits SafetyCancellationEvent when cancelled."""
        token = CancellationToken()
        token.cancel()
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            cancellation_token=token,
        )

        result = await agent.run("Do something")

        assert result.termination_reason == "cancelled"

        cancel_events = [e for e in emitter.events if isinstance(e, SafetyCancellationEvent)]
        assert len(cancel_events) == 1
        event = cancel_events[0]
        assert event.agent_name == "react-agent"
        assert event.step_number == 0

    async def test_codeact_emits_cancellation_event(self) -> None:
        """CodeActAgent emits SafetyCancellationEvent when cancelled."""
        token = CancellationToken()
        token.cancel()
        client = MockLLMClient([make_response(content="answer")])
        sandbox = MockSandbox([])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            cancellation_token=token,
        )

        result = await agent.run("Do something")

        assert result.termination_reason == "cancelled"

        cancel_events = [e for e in emitter.events if isinstance(e, SafetyCancellationEvent)]
        assert len(cancel_events) == 1
        event = cancel_events[0]
        assert event.agent_name == "codeact-agent"
        assert event.step_number == 0
