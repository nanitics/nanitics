import pytest
from pydantic import BaseModel

from nanitics.capabilities.errors.handler import ErrorHandler
from nanitics.errors import (
    ToolExecutionError,
    ToolParameterError,
)
from nanitics.infrastructure import AgentErrorEvent, AgentStartEvent, LLMRequestEvent, MockLLMClient
from nanitics.infrastructure.errors import LLMRateLimitError, ToolNotFoundError
from nanitics.infrastructure.observability.events import (
    ErrorCorrectionEvent,
    ErrorDegradationEvent,
    ErrorRetryEvent,
    SafetyToolCallLimitEvent,
)
from nanitics.safety import CancellationToken
from nanitics.strategies import (
    ReActAgent,
    ReasoningAgent,
    ToolContext,
    tool,
)
from nanitics.strategies.tools import ToolRegistry
from nanitics.tracing import (
    InMemoryEmitter,
    Message,
    ToolCall,
)


@tool(name="add", description="Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


@tool(name="multiply", description="Multiply two numbers")
async def multiply_tool(a: int, b: int) -> str:
    return str(a * b)


@tool(name="failing", description="Always fails")
async def failing_tool() -> str:
    raise ValueError("intentional error")


# ──────────────────────────────────────────────────────────
# ReasoningAgent Tests
# ──────────────────────────────────────────────────────────


class TestAgentEmitStep:
    async def test_emit_step_threads_artifact(self) -> None:
        from nanitics.infrastructure.observability.events import AgentStepEvent

        emitter = InMemoryEmitter(trace_id="t")
        agent = ReasoningAgent(
            name="reasoning",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
        )
        # _emit_step is a protected helper on the base Agent class; calling
        # it directly is the most focused way to verify that the artifact
        # kwarg threads into the AgentStepEvent payload.
        agent._emit_step(1, artifact={"a": 1})
        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        assert step_events[0].step_number == 1
        assert step_events[0].artifact == {"a": 1}
        assert step_events[0].thought is None
        assert step_events[0].action is None
        assert step_events[0].observation is None

    async def test_emit_step_artifact_defaults_none(self) -> None:
        from nanitics.infrastructure.observability.events import AgentStepEvent

        emitter = InMemoryEmitter(trace_id="t")
        agent = ReasoningAgent(
            name="reasoning",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
        )
        agent._emit_step(1, thought="free text")
        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        assert step_events[0].artifact is None
        assert step_events[0].thought == "free text"


class TestSupportsDynamicTools:
    def test_base_agent_returns_false(self) -> None:
        agent = ReasoningAgent(
            name="reasoning",
            llm_client=MockLLMClient([make_response()]),
            emitter=InMemoryEmitter(trace_id="t"),
            system_prompt="test",
        )
        assert agent.supports_dynamic_tools is False

    def test_react_agent_returns_true(self) -> None:
        agent = ReActAgent(
            name="react",
            llm_client=MockLLMClient([make_response()]),
            emitter=InMemoryEmitter(trace_id="t"),
            system_prompt="test",
            tools=[],
        )
        assert agent.supports_dynamic_tools is True

    def test_update_tool_state_raises_on_agent_without_tools(self) -> None:
        """ReasoningAgent inherits base Agent.update_tool_state which raises."""
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="test",
        )
        with pytest.raises(NotImplementedError, match="ReasoningAgent does not support tool state"):
            agent.update_tool_state("k", "v")


class TestReasoningAgent:
    async def test_successful_completion(self) -> None:
        client = MockLLMClient([make_response(content="Hello world")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        result = await agent.run("Hi")

        assert result.output == "Hello world"
        assert result.total_steps == 1
        assert result.termination_reason == "complete"

    async def test_messages_list(self) -> None:
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        result = await agent.run("question")

        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "question"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "answer"

    async def test_usage(self) -> None:
        usage = make_usage(input_tokens=20, output_tokens=10)
        client = MockLLMClient([make_response(usage=usage)])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        result = await agent.run("Hi")

        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 10
        assert result.usage.total_tokens == 30

    async def test_cancellation_before_execution(self) -> None:
        """ReasoningAgent doesn't check cancellation itself (single step),
        but the base class doesn't check before _execute either.
        Cancellation is checked by the loop-based agents.
        For ReasoningAgent, we verify that if cancelled before run(),
        it still completes since it doesn't check cancellation."""
        token = CancellationToken()
        token.cancel()
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            cancellation_token=token,
        )

        # ReasoningAgent has no loop, so it completes even when cancelled
        result = await agent.run("Hi")
        assert result.termination_reason == "complete"

    async def test_event_emission(self) -> None:
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        await agent.run("Hi")

        event_types = [e.event_type for e in emitter.events]
        assert "span.start" in event_types
        assert "agent.start" in event_types
        assert "llm.request" in event_types
        assert "llm.response" in event_types
        assert "agent.step" in event_types
        assert "agent.complete" in event_types
        assert "span.end" in event_types

    async def test_event_emission_order(self) -> None:
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        await agent.run("Hi")

        event_types = [e.event_type for e in emitter.events]
        assert event_types == [
            "span.start",  # agent span opens
            "agent.start",
            "llm.request",
            "llm.response",
            "agent.step",
            "agent.complete",
            "span.end",  # agent span closes
        ]

    async def test_start_event_has_no_tools(self) -> None:
        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        await agent.run("Hi")

        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].tools_available == []

    async def test_name_property(self) -> None:
        agent = ReasoningAgent(
            name="my-agent",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="prompt",
        )
        assert agent.name == "my-agent"

    async def test_structured_output_schema_passed_to_llm(self) -> None:
        """Verify output_schema is passed through to MockLLMClient.generate()."""

        class MathAnswer(BaseModel):
            result: int
            explanation: str

        json_content = '{"result": 42, "explanation": "The answer"}'
        client = MockLLMClient([make_response(content=json_content)])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Extract data.",
            output_schema=MathAnswer,
        )

        result = await agent.run("What is 6 * 7?")

        # Verify schema was passed to generate()
        assert len(client.calls) == 1
        assert client.calls[0]["output_schema"] is MathAnswer

        # Verify result contains the JSON string and parsed model
        assert result.output == json_content
        assert result.total_steps == 1
        assert result.termination_reason == "complete"
        assert result.parsed is not None
        assert isinstance(result.parsed, MathAnswer)
        assert result.parsed.result == 42
        assert result.parsed.explanation == "The answer"

    async def test_structured_output_event_includes_schema(self) -> None:
        """Verify LLMRequestEvent includes output_schema dict."""

        class ContactInfo(BaseModel):
            name: str
            email: str | None = None

        client = MockLLMClient([make_response(content='{"name": "Alice"}')])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Extract.",
            output_schema=ContactInfo,
        )

        await agent.run("Extract contact info")

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        assert len(request_events) == 1
        assert request_events[0].output_schema is not None
        assert "properties" in request_events[0].output_schema
        assert "name" in request_events[0].output_schema["properties"]

    async def test_without_output_schema(self) -> None:
        """ReasoningAgent without output_schema returns plain text."""
        client = MockLLMClient([make_response(content="plain answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
        )

        result = await agent.run("Hi")

        assert result.output == "plain answer"
        assert result.parsed is None
        assert len(client.calls) == 1
        assert client.calls[0]["output_schema"] is None

    async def test_streaming_emits_token_events(self) -> None:
        """ReasoningAgent(streaming=True) emits LLMTokenEvent during generation."""
        from nanitics.infrastructure.observability.events import LLMTokenEvent

        client = MockLLMClient([make_response(content="hello world")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            streaming=True,
        )

        result = await agent.run("Hi")

        assert result.output == "hello world"
        token_events = [e for e in emitter.events if isinstance(e, LLMTokenEvent)]
        assert len(token_events) > 0
        reconstructed = "".join(e.token for e in token_events)
        assert "hello" in reconstructed
        assert "world" in reconstructed

    async def test_non_nanitics_error_emits_error_with_message_metadata(self) -> None:
        """Non-NaniticsError produces error event with message-only metadata."""
        from nanitics.infrastructure import LLMResponse

        client = MockLLMClient([])

        async def failing_generate(**kwargs: object) -> LLMResponse:
            raise RuntimeError("unexpected failure")

        client.generate = failing_generate  # type: ignore[assignment]
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
        )
        with pytest.raises(RuntimeError):
            await agent.run("Hi")
        error_events = [e for e in emitter.events if isinstance(e, AgentErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_metadata == {"message": "unexpected failure"}

    async def test_truncation_triggers_revision_with_evaluator(self) -> None:
        """Truncated response triggers revision when evaluator present."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class _AcceptEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            LLMResponse(content="partial", tool_calls=[], usage=make_usage(), model="test", stop_reason="max_tokens"),
            make_response(content="full answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            output_evaluator=_AcceptEval(),
        )
        result = await agent.run("test")
        assert result.output == "full answer"

    async def test_truncation_during_revision_loop(self) -> None:
        """Truncation during revision loop (after evaluator REVISE) triggers truncation handling."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        call_count = 0

        class _ReviseOnceEval:
            @property
            def max_revisions(self) -> int:
                return 3

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        feedback="Improve it",
                        evaluator_name="test",
                    )
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            make_response(content="first attempt"),
            LLMResponse(content="truncated", tool_calls=[], usage=make_usage(), model="test", stop_reason="max_tokens"),
            make_response(content="final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            output_evaluator=_ReviseOnceEval(),
        )
        result = await agent.run("test")
        assert result.output == "final answer"
        assert result.termination_reason == "complete"

    async def test_per_iteration_emission_with_reasoning_text(self) -> None:
        """ReasoningAgent with evaluator-driven revisions emits one step per LLM call.

        The evaluator returns REVISE twice, then ACCEPT. Three LLM responses are
        scripted, each with a distinct ``reasoning_text``. The agent must emit
        three ``AgentStepEvent`` whose ``thought`` fields match the scripted
        ``reasoning_text`` values and whose ``artifact`` fields match the
        scripted ``parsed.model_dump()``.
        """
        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import AgentStepEvent
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            value: int

        call_count = 0

        class _TwiceReviseEval:
            @property
            def max_revisions(self) -> int:
                return 5

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        feedback="revise please",
                        evaluator_name="test",
                    )
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            LLMResponse(
                content='{"value": 1}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                parsed=Answer(value=1),
                reasoning_text="reasoning one",
            ),
            LLMResponse(
                content='{"value": 2}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                parsed=Answer(value=2),
                reasoning_text="reasoning two",
            ),
            LLMResponse(
                content='{"value": 3}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                parsed=Answer(value=3),
                reasoning_text="reasoning three",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            output_schema=Answer,
            output_evaluator=_TwiceReviseEval(),
        )

        result = await agent.run("test")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 3
        assert result.total_steps == 3
        assert step_events[0].thought == "reasoning one"
        assert step_events[0].artifact == {"value": 1}
        assert step_events[1].thought == "reasoning two"
        assert step_events[1].artifact == {"value": 2}
        assert step_events[2].thought == "reasoning three"
        assert step_events[2].artifact == {"value": 3}

    async def test_per_iteration_emission_without_reasoning_text(self) -> None:
        """ReasoningAgent with ``reasoning_text=None`` emits ``thought=None``
        but still emits the artifact from ``parsed``."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import AgentStepEvent

        class Answer(BaseModel):
            value: int

        responses = [
            LLMResponse(
                content='{"value": 42}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                parsed=Answer(value=42),
                reasoning_text=None,
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            output_schema=Answer,
        )

        await agent.run("test")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        assert step_events[0].thought is None
        assert step_events[0].artifact == {"value": 42}


# ──────────────────────────────────────────────────────────
# ReActAgent Tests
# ──────────────────────────────────────────────────────────


class TestReActAgent:
    async def test_single_step_completion(self) -> None:
        """LLM returns content with no tool calls on first response."""
        client = MockLLMClient([make_response(content="direct answer")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result = await agent.run("What is 1+1?")

        assert result.output == "direct answer"
        assert result.total_steps == 1
        assert result.termination_reason == "complete"

    async def test_initial_messages_prepended(self) -> None:
        """initial_messages are prepended before the current user input."""
        client = MockLLMClient([make_response(content="I remember you said hello")])
        emitter = make_emitter()
        history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            initial_messages=history,
        )

        result = await agent.run("Do you remember what I said?")

        assert result.output == "I remember you said hello"
        # Messages should contain: history (2) + current user (1) + assistant (1)
        assert len(result.messages) == 4
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "Hi there!"
        assert result.messages[2].role == "user"
        assert result.messages[2].content == "Do you remember what I said?"
        assert result.messages[3].role == "assistant"
        assert result.messages[3].content == "I remember you said hello"

    async def test_initial_messages_none_by_default(self) -> None:
        """Without initial_messages, agent starts with just the user input."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result = await agent.run("Question")

        # Messages should contain: user (1) + assistant (1)
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Question"

    async def test_multi_step_tool_use(self) -> None:
        """LLM returns tool call, then final answer."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})
        responses = [
            make_response(content="Let me add", tool_calls=[tool_call]),
            make_response(content="The answer is 5"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result = await agent.run("Add 2 and 3")

        assert result.output == "The answer is 5"
        assert result.total_steps == 2
        assert result.termination_reason == "complete"

        # Check message history includes tool result
        tool_result_msgs = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].content == "5"
        assert tool_result_msgs[0].tool_call_id == "tc1"

    async def test_multiple_tool_calls_in_one_response(self) -> None:
        """LLM returns 2 tool calls in one response. Both executed sequentially."""
        tool_calls = [
            ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4}),
        ]
        responses = [
            make_response(content="Computing", tool_calls=tool_calls),
            make_response(content="Results: 3 and 12"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool, multiply_tool],
        )

        result = await agent.run("Compute both")

        assert result.output == "Results: 3 and 12"
        assert result.total_steps == 2
        tool_result_msgs = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_result_msgs) == 2
        assert tool_result_msgs[0].content == "3"
        assert tool_result_msgs[1].content == "12"

    async def test_iteration_limit(self) -> None:
        """LLM always returns tool calls. Agent hits limit."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        # 4 responses: 3 with tool calls (hit limit at 3), never reaches 4th
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
        assert result.total_steps == 3
        assert result.output is None

    async def test_cancellation(self) -> None:
        """Token is cancelled before first step."""
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
        assert result.total_steps == 0
        assert result.output is None

    async def test_tool_error_propagation(self) -> None:
        """Tool raises exception with fail-fast handler.

        Agent emits error event and re-raises.
        """
        tool_call = ToolCall(id="tc1", name="failing", arguments={})
        client = MockLLMClient(
            [
                make_response(content="Let me try", tool_calls=[tool_call]),
            ]
        )
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[failing_tool],
            error_handler=ErrorHandler.fail_fast(),
        )

        with pytest.raises(ToolExecutionError):
            await agent.run("Do something")

        error_events = [e for e in emitter.events if isinstance(e, AgentErrorEvent)]
        assert len(error_events) == 1

    async def test_event_emission_multi_step(self) -> None:
        """Verify full event sequence for a multi-step run."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
        responses = [
            make_response(content="thinking", tool_calls=[tool_call]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        await agent.run("Add 1+2")

        event_types = [e.event_type for e in emitter.events]
        assert event_types == [
            "span.start",  # agent span
            "agent.start",
            # step 1
            "span.start",  # step-1 span
            "llm.request",
            "llm.response",
            "tool.invoke",
            "tool.result",
            "agent.step",
            "span.end",  # step-1 span end
            # step 2
            "span.start",  # step-2 span
            "llm.request",
            "llm.response",
            "agent.step",
            "span.end",  # step-2 span end
            # completion
            "agent.complete",
            "span.end",  # agent span end
        ]

    async def test_agent_reusability(self) -> None:
        """Call run() twice on same agent. Both should work correctly."""
        responses = [
            make_response(content="first answer"),
            make_response(content="second answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result1 = await agent.run("First")
        result2 = await agent.run("Second")

        assert result1.output == "first answer"
        assert result1.total_steps == 1
        assert result2.output == "second answer"
        assert result2.total_steps == 1

    async def test_no_tools_provided(self) -> None:
        """Agent with empty tool list."""
        client = MockLLMClient([make_response(content="no tools needed")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[],
        )

        result = await agent.run("Hi")

        assert result.output == "no tools needed"
        assert result.total_steps == 1

    async def test_tools_available_in_start_event(self) -> None:
        """Start event lists tool names."""
        client = MockLLMClient([make_response(content="ok")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool, multiply_tool],
        )

        await agent.run("Hi")

        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert sorted(start_events[0].tools_available) == ["add", "multiply"]

    async def test_usage_aggregation(self) -> None:
        """Usage should be aggregated across all LLM calls."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
        usage1 = make_usage(input_tokens=10, output_tokens=5)
        usage2 = make_usage(input_tokens=20, output_tokens=15)
        responses = [
            make_response(content="thinking", tool_calls=[tool_call], usage=usage1),
            make_response(content="done", usage=usage2),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result = await agent.run("Add")

        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 50

    async def test_tool_state_available_in_tools(self) -> None:
        """ReActAgent with tool_state makes state accessible to tools via ToolContext."""
        captured_state: list[dict] = []

        @tool(name="state_reader", description="Reads state")
        async def state_reader(context: ToolContext) -> str:
            captured_state.append(dict(context.state))
            return f"max_depth={context.state.get('max_depth', 'missing')}"

        tool_call = ToolCall(id="tc1", name="state_reader", arguments={})
        responses = [
            make_response(content="Reading state", tool_calls=[tool_call]),
            make_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[state_reader],
            tool_state={"max_depth": 3, "user_id": "abc"},
        )

        result = await agent.run("Read the state")

        assert result.termination_reason == "complete"
        assert len(captured_state) == 1
        assert captured_state[0] == {"max_depth": 3, "user_id": "abc"}

    async def test_update_tool_state_adds_key_visible_to_tools(self) -> None:
        """update_tool_state adds a new key accessible via ToolContext.state."""
        captured_state: list[dict] = []

        @tool(name="state_reader", description="Reads state")
        async def state_reader(context: ToolContext) -> str:
            captured_state.append(dict(context.state))
            return "ok"

        tool_call = ToolCall(id="tc1", name="state_reader", arguments={})
        responses = [
            make_response(content="Reading state", tool_calls=[tool_call]),
            make_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[state_reader],
            tool_state={"a": 1},
        )

        agent.update_tool_state("b", 2)
        result = await agent.run("Read the state")

        assert result.termination_reason == "complete"
        assert len(captured_state) == 1
        assert captured_state[0] == {"a": 1, "b": 2}

    async def test_update_tool_state_overwrites_existing_key(self) -> None:
        """update_tool_state replaces an existing key's value."""
        captured_state: list[dict] = []

        @tool(name="state_reader", description="Reads state")
        async def state_reader(context: ToolContext) -> str:
            captured_state.append(dict(context.state))
            return "ok"

        tool_call = ToolCall(id="tc1", name="state_reader", arguments={})
        responses = [
            make_response(content="Reading state", tool_calls=[tool_call]),
            make_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[state_reader],
            tool_state={"a": 1},
        )

        agent.update_tool_state("a", 99)
        result = await agent.run("Read the state")

        assert result.termination_reason == "complete"
        assert captured_state[0]["a"] == 99

    async def test_thought_carries_reasoning_text_on_tool_use_step(self) -> None:
        """Tool-use step emits ``thought == response.reasoning_text`` (not content)."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import AgentStepEvent

        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})
        responses = [
            LLMResponse(
                content="prose before tool call",
                tool_calls=[tool_call],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                reasoning_text="reasoning about the add",
            ),
            make_response(content="5"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )
        await agent.run("Add 2+3")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        # Two step events: step 1 (tool-use), step 2 (final content).
        assert len(step_events) == 2
        assert step_events[0].thought == "reasoning about the add"
        assert step_events[0].action == "add"
        assert step_events[0].artifact is None

    async def test_thought_carries_reasoning_text_on_final_content_step(self) -> None:
        """Final content step emits ``thought == response.reasoning_text`` and
        ``observation == response.content`` — on the terminal no-tool path
        the agent's 'observation' is the final answer it produced."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import AgentStepEvent

        responses = [
            LLMResponse(
                content="direct answer",
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                reasoning_text=None,
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )
        await agent.run("Hi")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        # reasoning_text is None on a plain final answer — thought is None.
        assert step_events[0].thought is None
        # Terminal no-tool-calls path: observation carries the final content
        # so the step is not entirely null for consumers (Observatory UI,
        # trace analyzers) rendering agent.step detail panels.
        assert step_events[0].observation == "direct answer"
        assert step_events[0].action is None
        assert step_events[0].artifact is None

    async def test_react_terminal_step_populates_observation_from_content(self) -> None:
        """Closes observability W2: on a minimal vanilla ReAct run
        (max_iterations=1, tools=[]) with reasoning_text=None, the single
        emitted AgentStepEvent must populate observation from content so the
        event is not all-None."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import AgentStepEvent

        responses = [
            LLMResponse(
                content="OK",
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                reasoning_text=None,
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Reply with a single short word.",
            tools=[],
            max_iterations=1,
        )
        await agent.run("Say OK.")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        step = step_events[0]
        # Regression guard: W2 required that no step has all four fields None.
        assert step.observation == "OK"
        assert step.thought is None
        assert step.action is None
        assert step.artifact is None

    async def test_react_reasoning_and_terminal_content_do_not_duplicate(self) -> None:
        """When the model returns both reasoning_text (Anthropic thinking
        block) and content (final answer), thought and observation carry
        semantically distinct pieces — they are not the same string."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import AgentStepEvent

        responses = [
            LLMResponse(
                content="answer",
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                reasoning_text="thinking...",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[],
            max_iterations=1,
        )
        await agent.run("Prompt.")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        step = step_events[0]
        assert step.thought == "thinking..."
        assert step.observation == "answer"
        # Contract: thought and observation never carry the same string.
        assert step.thought != step.observation


# ──────────────────────────────────────────────────────────
# Error Handling Integration Tests
# ──────────────────────────────────────────────────────────


from nanitics.capabilities.memory.working_memory import InMemoryWorkingMemory

# ──────────────────────────────────────────────────────────
# Prompt Composition Tests
# ──────────────────────────────────────────────────────────


class TestPromptComposition:
    def test_react_agent_with_working_memory_composes_prompt(self) -> None:
        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        wm = InMemoryWorkingMemory()
        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a detective.",
            tools=[add_tool],
            working_memory=wm,
        )

        assert "You are a detective." in agent._system_prompt
        assert "<working_memory>" in agent._system_prompt
        assert "[Working Memory]" in agent._system_prompt

    def test_react_agent_without_working_memory_uses_base_prompt(self) -> None:
        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are helpful.",
            tools=[add_tool],
        )

        assert agent._system_prompt.startswith("You are helpful.")
        assert "autonomously" in agent._system_prompt
        assert "not visible" not in agent._system_prompt

    def test_composed_prompt_base_section_comes_first(self) -> None:
        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        wm = InMemoryWorkingMemory()
        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Base instructions here.",
            tools=[add_tool],
            working_memory=wm,
        )

        assert agent._system_prompt.startswith("Base instructions here.")

    def test_react_agent_with_external_prompt_contributor(self) -> None:
        class PlanningContributor:
            def system_prompt_section(self) -> tuple[str, str] | None:
                return ("planning", "Always plan before acting.")

        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Base prompt.",
            tools=[add_tool],
            prompt_contributors=[PlanningContributor()],
        )

        assert "Base prompt." in agent._system_prompt
        assert "Always plan before acting." in agent._system_prompt

    def test_prompt_contributor_returning_none_is_skipped(self) -> None:
        class SilentContributor:
            def system_prompt_section(self) -> tuple[str, str] | None:
                return None

        class ActiveContributor:
            def system_prompt_section(self) -> tuple[str, str] | None:
                return ("planning", "Always plan before acting.")

        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Base prompt.",
            tools=[add_tool],
            prompt_contributors=[SilentContributor(), ActiveContributor()],
        )

        assert "Always plan before acting." in agent._system_prompt

    def test_react_agent_external_contributors_compose_with_working_memory(self) -> None:
        class EpisodicContributor:
            def system_prompt_section(self) -> tuple[str, str] | None:
                return ("episodic", "Recall past experiences.")

        client = MockLLMClient([make_response()])
        emitter = make_emitter()
        wm = InMemoryWorkingMemory()
        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Base prompt.",
            tools=[add_tool],
            working_memory=wm,
            prompt_contributors=[EpisodicContributor()],
        )

        prompt = agent._system_prompt
        assert "Base prompt." in prompt
        assert "<working_memory>" in prompt
        assert "Recall past experiences." in prompt
        # Working memory should come before external contributors
        wm_pos = prompt.index("<working_memory>")
        episodic_pos = prompt.index("Recall past experiences.")
        assert wm_pos < episodic_pos

    async def test_agent_threads_builder_sections_to_llm_client(self) -> None:
        """``Agent._call_llm`` passes ``builder.build_sections()`` into the
        LLM client's ``system_prompt_sections`` parameter.

        Without this, cacheable structured sections never reach the
        provider and every prompt caches as one flat block — defeating
        the per-section ``cacheable`` flag the builder carries.
        """

        class StableContributor:
            def system_prompt_section(self) -> tuple[str, str] | None:
                return ("planning", "Always plan before acting.")

        client = MockLLMClient([make_response(content="ok")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Base prompt.",
            prompt_contributors=[StableContributor()],
        )

        await agent.run("ping")

        sections = client.calls[-1]["system_prompt_sections"]
        assert sections is not None, "Agent must thread sections to LLMClient.generate()."
        # Expected three sections, in insertion order: base, environment, planning.
        assert len(sections) == 3
        assert sections[0].content == "Base prompt."
        assert "autonomously" in sections[1].content
        assert "not visible" not in sections[1].content
        assert sections[2].content == "Always plan before acting."
        # ``cacheable`` defaults to True for every section the agent adds.
        for section in sections:
            assert section.cacheable is True


from typing import Literal

from tests.testing_helpers import make_emitter, make_response, make_usage


@tool(name="strict_search", description="Search with strict params")
async def strict_search_tool(
    category: Literal["electronics", "books"],
) -> str:
    return f"Results for {category}"


class TestErrorHandlingIntegration:
    async def test_self_correction_on_tool_parameter_error(self) -> None:
        """Tool parameter error → correction prompt → LLM adjusts → succeeds."""
        # First call: LLM sends invalid category, tool raises ToolParameterError
        # Error is fed back as correction prompt, LLM gets another turn
        # Second call: LLM sends correct category
        bad_call = ToolCall(id="tc1", name="strict_search", arguments={"category": "kitchen"})
        good_call = ToolCall(id="tc2", name="strict_search", arguments={"category": "electronics"})
        responses = [
            make_response(content="Searching kitchen", tool_calls=[bad_call]),
            make_response(content="Searching electronics", tool_calls=[good_call]),
            make_response(content="Found results for electronics"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[strict_search_tool],
            error_handler=ErrorHandler(),
        )

        result = await agent.run("Search kitchen appliances")
        assert result.termination_reason == "complete"
        # Verify correction event was emitted
        correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
        assert len(correction_events) == 1
        assert correction_events[0].error_type == "ToolParameterError"

    async def test_correction_budget_exhaustion_degrades(self) -> None:
        """Correction budget exhaustion → degradation message → LLM wraps up."""
        bad_call = ToolCall(id="tc1", name="strict_search", arguments={"category": "invalid"})
        # LLM keeps sending bad calls, eventually handler degrades
        responses = [
            make_response(content="try 1", tool_calls=[bad_call]),
            make_response(content="try 2", tool_calls=[bad_call]),
            make_response(content="try 3", tool_calls=[bad_call]),
            make_response(content="try 4", tool_calls=[bad_call]),
            make_response(content="I could not find the results"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        handler = ErrorHandler(max_corrections=3, max_total_corrections=3)
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[strict_search_tool],
            error_handler=handler,
        )

        result = await agent.run("Search something")

        assert result.output == "I could not find the results"
        assert result.termination_reason == "complete"
        # Should have correction events followed by a degradation event
        correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
        degradation_events = [e for e in emitter.events if isinstance(e, ErrorDegradationEvent)]
        assert len(correction_events) == 3
        assert len(degradation_events) == 1

    async def test_fail_fast_tool_error_raises(self) -> None:
        """ErrorHandler.fail_fast(): tool error propagates immediately."""
        bad_call = ToolCall(id="tc1", name="strict_search", arguments={"category": "invalid"})
        client = MockLLMClient(
            [
                make_response(content="searching", tool_calls=[bad_call]),
            ]
        )
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[strict_search_tool],
            error_handler=ErrorHandler.fail_fast(),
        )

        with pytest.raises(ToolParameterError):
            await agent.run("Search")

    async def test_llm_rate_limit_retry_succeeds(self) -> None:
        """LLM rate limit error → retry succeeds on second attempt."""
        call_count = 0
        original_client = MockLLMClient(
            [
                make_response(content="answer"),
                make_response(content="answer"),
                make_response(content="answer"),
            ]
        )

        async def flaky_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on calls 1 and 2 so retry_with_backoff sees the error too
            if call_count <= 2:
                raise LLMRateLimitError("rate limited", retry_after=0.0)
            return await MockLLMClient.generate(original_client, **kwargs)

        original_client.generate = flaky_generate  # type: ignore[method-assign]
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=original_client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            error_handler=ErrorHandler(),
        )

        result = await agent.run("Hi")

        assert result.output == "answer"
        retry_events = [e for e in emitter.events if isinstance(e, ErrorRetryEvent)]
        assert len(retry_events) >= 1

    async def test_default_handler_tool_error_with_correction_events(self) -> None:
        """Verify ErrorCorrectionEvent appears in emitter events."""
        bad_call = ToolCall(id="tc1", name="strict_search", arguments={"category": "wrong"})
        good_call = ToolCall(id="tc2", name="strict_search", arguments={"category": "books"})
        responses = [
            make_response(content="trying", tool_calls=[bad_call]),
            make_response(content="fixed", tool_calls=[good_call]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[strict_search_tool],
            error_handler=ErrorHandler(),
        )

        await agent.run("Search")

        correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
        assert len(correction_events) == 1
        assert "strict_search" in correction_events[0].correction_prompt

    async def test_tool_not_found_error_corrected(self) -> None:
        """ToolNotFoundError from hallucinated tool name → correction → LLM retries with valid tool."""
        hallucinated_call = ToolCall(id="tc1", name="working_memory", arguments={})
        valid_call = ToolCall(id="tc2", name="strict_search", arguments={"category": "books"})
        responses = [
            make_response(content="Updating memory", tool_calls=[hallucinated_call]),
            make_response(content="Let me search instead", tool_calls=[valid_call]),
            make_response(content="Found results."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[strict_search_tool],
            error_handler=ErrorHandler(),
        )

        result = await agent.run("Search for books")

        assert result.output == "Found results."
        correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
        assert len(correction_events) == 1
        assert correction_events[0].error_type == "ToolNotFoundError"
        assert "working_memory" in correction_events[0].correction_prompt
        assert "strict_search" in correction_events[0].correction_prompt
        assert correction_events[0].attempt == 1
        assert correction_events[0].max_attempts == 3

    async def test_schema_violation_corrected_on_retry(self) -> None:
        """_call_llm catches LLMSchemaViolationError, appends correction, retries."""

        class Output(BaseModel):
            answer: int

        responses = [
            make_response(content="not valid json"),  # triggers LLMSchemaViolationError
            make_response(content='{"answer": 42}'),  # valid
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_schema=Output,
            error_handler=ErrorHandler.default(),
        )

        result = await agent.run("What is 6 * 7?")
        assert result.output == '{"answer": 42}'

        correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
        assert len(correction_events) == 1
        assert correction_events[0].error_type == "LLMSchemaViolationError"

    async def test_schema_violation_raises_when_budget_exhausted(self) -> None:
        """Repeated schema violations exhaust budget and raise."""
        from nanitics.infrastructure.errors import LLMSchemaViolationError

        class Output(BaseModel):
            answer: int

        # All responses are invalid — will exhaust the correction budget
        responses = [make_response(content="bad") for _ in range(10)]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_schema=Output,
            error_handler=ErrorHandler(max_corrections=2, max_total_corrections=5),
        )

        with pytest.raises(LLMSchemaViolationError):
            await agent.run("What is 6 * 7?")

        correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
        assert len(correction_events) == 2  # max_corrections=2

    async def test_schema_violation_no_correction_without_error_handler(self) -> None:
        """With ErrorHandler.fail_fast(), schema violations propagate immediately."""
        from nanitics.infrastructure.errors import LLMSchemaViolationError

        class Output(BaseModel):
            answer: int

        client = MockLLMClient([make_response(content="bad")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            output_schema=Output,
            error_handler=ErrorHandler.fail_fast(),
        )

        with pytest.raises(LLMSchemaViolationError):
            await agent.run("What is 6 * 7?")

        correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
        assert len(correction_events) == 0


# ──────────────────────────────────────────────────────────
# Context Management Integration Tests
# ──────────────────────────────────────────────────────────


class TestContextManagementIntegration:
    async def test_react_agent_with_context_manager_sends_managed_messages(
        self,
    ) -> None:
        """Verify managed (truncated) messages are sent to the LLM."""
        from nanitics.capabilities.context.manager import ContextManager
        from nanitics.capabilities.context.token_counter import EstimateTokenCounter
        from nanitics.capabilities.context.truncation import TruncationPolicy

        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
        responses = [
            make_response(content="step", tool_calls=[tool_call]),
            make_response(content="step", tool_calls=[tool_call]),
            make_response(content="step", tool_calls=[tool_call]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        context_manager = ContextManager(
            context_limit=300,
            reserve_tokens=20,
            threshold=0.5,
            token_counter=EstimateTokenCounter(),
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            context_manager=context_manager,
        )

        result = await agent.run("Add numbers")

        assert result.output == "done"
        assert result.termination_reason == "complete"
        # The full message history is preserved in AgentResult
        assert len(result.messages) > 4

    async def test_react_agent_without_context_manager_unchanged(self) -> None:
        """Default None context_manager doesn't affect existing behavior."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result = await agent.run("Hi")

        assert result.output == "answer"
        assert result.total_steps == 1
        # All messages preserved
        assert len(result.messages) == 2

    async def test_agent_result_contains_full_messages(self) -> None:
        """AgentResult.messages contains full (unmanaged) message list."""
        from nanitics.capabilities.context.manager import ContextManager
        from nanitics.capabilities.context.truncation import TruncationPolicy

        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        responses = [
            make_response(content="s1", tool_calls=[tool_call]),
            make_response(content="s2", tool_calls=[tool_call]),
            make_response(content="final"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        context_manager = ContextManager(
            context_limit=300,
            reserve_tokens=20,
            threshold=0.5,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            context_manager=context_manager,
        )

        result = await agent.run("Do math")

        # Full unmanaged history: user, assistant+tc, tool_result, assistant+tc, tool_result, assistant
        assert len(result.messages) == 6
        assert result.messages[0].role == "user"
        assert result.messages[-1].role == "assistant"


# ──────────────────────────────────────────────────────────
# ToolRegistry.unregister Tests
# ──────────────────────────────────────────────────────────


class TestToolRegistryUnregister:
    def test_unregister_removes_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(add_tool)
        assert registry.has("add")
        registry.unregister("add")
        assert not registry.has("add")

    def test_unregister_unknown_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.unregister("nonexistent")

    def test_unregister_then_reregister(self) -> None:
        registry = ToolRegistry()
        registry.register(add_tool)
        registry.unregister("add")
        registry.register(add_tool)
        assert registry.has("add")


# ──────────────────────────────────────────────────────────
# Agent.add_tools / remove_tools Tests
# ──────────────────────────────────────────────────────────


class TestAgentToolInjection:
    def test_base_agent_add_tools_returns_empty(self) -> None:
        agent = ReasoningAgent(
            name="reasoning",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="test",
        )
        result = agent.add_tools([add_tool])
        assert result == []

    def test_base_agent_remove_tools_is_noop(self) -> None:
        agent = ReasoningAgent(
            name="reasoning",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="test",
        )
        # Should not raise
        agent.remove_tools(["anything"])

    def test_react_add_tools_registers(self) -> None:
        agent = ReActAgent(
            name="react",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[],
        )
        added = agent.add_tools([add_tool, multiply_tool])
        assert sorted(added) == ["add", "multiply"]
        assert "add" in agent._get_tools_available()
        assert "multiply" in agent._get_tools_available()

    def test_react_add_tools_skips_duplicates(self) -> None:
        agent = ReActAgent(
            name="react",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[add_tool],
        )
        added = agent.add_tools([add_tool, multiply_tool])
        assert added == ["multiply"]

    def test_react_remove_tools(self) -> None:
        agent = ReActAgent(
            name="react",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[add_tool, multiply_tool],
        )
        agent.remove_tools(["add", "multiply"])
        assert agent._get_tools_available() == []

    def test_react_add_then_remove(self) -> None:
        agent = ReActAgent(
            name="react",
            llm_client=MockLLMClient([make_response()]),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[],
        )
        added = agent.add_tools([add_tool])
        assert added == ["add"]
        agent.remove_tools(added)
        assert "add" not in agent._get_tools_available()


# ──────────────────────────────────────────────────────────
# ReActAgent Structured Output Tests
# ──────────────────────────────────────────────────────────


class AnalysisResult(BaseModel):
    answer: str
    confidence: float


class TestReActStructuredOutput:
    async def test_react_structured_output_basic(self) -> None:
        """Tool use followed by structured final call populates parsed."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})
        responses = [
            make_response(content="Let me compute", tool_calls=[tool_call]),
            make_response(content="The sum is 5"),
            make_response(content='{"answer": "5", "confidence": 0.95}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            output_schema=AnalysisResult,
        )

        result = await agent.run("Add 2 and 3")

        assert result.parsed is not None
        assert isinstance(result.parsed, AnalysisResult)
        assert result.parsed.answer == "5"
        assert result.parsed.confidence == 0.95
        assert result.output == '{"answer": "5", "confidence": 0.95}'
        assert result.termination_reason == "complete"
        assert result.total_steps == 3  # tool step + answer step + structured step

        # Verify final call used output_schema, not tools
        assert client.calls[-1]["output_schema"] is AnalysisResult
        assert client.calls[-1]["tools"] is None

    async def test_react_structured_final_step_carries_artifact(self) -> None:
        """Structured final step emits ``artifact == parsed.model_dump()`` and
        ``thought == response.reasoning_text``."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import AgentStepEvent

        responses = [
            make_response(content="Direct answer"),
            LLMResponse(
                content='{"answer": "42", "confidence": 0.9}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
                parsed=AnalysisResult(answer="42", confidence=0.9),
                reasoning_text="chose to structure the answer",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            output_schema=AnalysisResult,
        )

        await agent.run("Question")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 2
        # Structured final step (last one) carries the artifact + reasoning_text.
        final_step = step_events[-1]
        assert final_step.artifact == {"answer": "42", "confidence": 0.9}
        assert final_step.thought == "chose to structure the answer"

    async def test_react_structured_output_no_tool_use(self) -> None:
        """LLM returns no tool calls immediately, then structured final call."""
        responses = [
            make_response(content="I know the answer"),
            make_response(content='{"answer": "42", "confidence": 1.0}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            output_schema=AnalysisResult,
        )

        result = await agent.run("What is 42?")

        assert result.parsed is not None
        assert isinstance(result.parsed, AnalysisResult)
        assert result.parsed.answer == "42"
        assert result.total_steps == 2  # answer step + structured step
        assert result.termination_reason == "complete"

    async def test_react_structured_output_multi_step(self) -> None:
        """Multiple tool-use steps, then structured output."""
        tc1 = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
        tc2 = ToolCall(id="tc2", name="multiply", arguments={"a": 3, "b": 4})
        responses = [
            make_response(content="Step 1", tool_calls=[tc1]),
            make_response(content="Step 2", tool_calls=[tc2]),
            make_response(content="Done computing"),
            make_response(content='{"answer": "3 and 12", "confidence": 0.9}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool, multiply_tool],
            output_schema=AnalysisResult,
        )

        result = await agent.run("Compute both")

        assert result.parsed is not None
        assert isinstance(result.parsed, AnalysisResult)
        assert result.total_steps == 4  # 2 tool steps + answer + structured
        assert result.termination_reason == "complete"

    async def test_react_structured_output_schema_in_event(self) -> None:
        """Verify LLMRequestEvent includes schema on final call."""
        responses = [
            make_response(content="answer"),
            make_response(content='{"answer": "x", "confidence": 0.5}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            output_schema=AnalysisResult,
        )

        await agent.run("Test")

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        # First call: tools, no schema. Second call: schema, no tools.
        assert len(request_events) == 2
        assert request_events[0].output_schema is None
        assert request_events[1].output_schema is not None
        assert "properties" in request_events[1].output_schema
        assert "answer" in request_events[1].output_schema["properties"]

    async def test_react_no_structured_output_on_iteration_limit(self) -> None:
        """Loop hits iteration limit — no structured final call, parsed is None."""
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
            output_schema=AnalysisResult,
        )

        result = await agent.run("Loop forever")

        assert result.termination_reason == "iteration_limit"
        assert result.parsed is None
        # Only 3 LLM calls (the loop steps), no structured final call
        assert len(client.calls) == 3

    async def test_react_no_structured_output_on_cancellation(self) -> None:
        """Agent cancelled — no structured final call, parsed is None."""
        token = CancellationToken()
        token.cancel()
        responses = [
            make_response(content="answer"),
            make_response(content='{"answer": "x", "confidence": 0.5}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            cancellation_token=token,
            output_schema=AnalysisResult,
        )

        result = await agent.run("Do something")

        assert result.termination_reason == "cancelled"
        assert result.parsed is None
        # No LLM calls made at all
        assert len(client.calls) == 0

    async def test_react_structured_output_with_evaluator(self) -> None:
        """Evaluator revises structured output, not tools."""
        from nanitics.capabilities.evaluation import EvaluationResult as EvalResult
        from nanitics.capabilities.evaluation import EvaluationVerdict

        class ReviseOnceEvaluator:
            def __init__(self) -> None:
                self._call_count = 0

            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: object) -> EvalResult:
                self._call_count += 1
                if self._call_count == 1:
                    return EvalResult(
                        verdict=EvaluationVerdict.REVISE,
                        score=0.3,
                        feedback="Be more specific",
                        evaluator_name="test",
                    )
                return EvalResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=0.9,
                    feedback=None,
                    evaluator_name="test",
                )

        responses = [
            make_response(content="answer"),
            # First structured attempt (will be revised)
            make_response(content='{"answer": "vague", "confidence": 0.3}'),
            # Re-enter tool loop after REVISE
            make_response(content="revised analysis"),
            # Second structured attempt (accepted)
            make_response(content='{"answer": "specific", "confidence": 0.9}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            output_schema=AnalysisResult,
            output_evaluator=ReviseOnceEvaluator(),
        )

        result = await agent.run("Question")

        assert result.parsed is not None
        assert result.parsed.answer == "specific"
        assert result.termination_reason == "complete"
        # 4 calls: tool loop + structured (revised) + tool loop re-entry + structured (accepted)
        assert len(client.calls) == 4
        # Revision structured call uses schema, not tools
        assert client.calls[3]["output_schema"] is AnalysisResult
        assert client.calls[3]["tools"] is None

    async def test_react_without_output_schema_unchanged(self) -> None:
        """Without output_schema, behavior is identical to before."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})
        responses = [
            make_response(content="Computing", tool_calls=[tool_call]),
            make_response(content="The answer is 5"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result = await agent.run("Add 2 and 3")

        assert result.output == "The answer is 5"
        assert result.parsed is None
        assert result.total_steps == 2
        assert result.termination_reason == "complete"

    async def test_react_structured_output_capability_reported(self) -> None:
        """ReActAgent with output_schema reports 'structured_output' capability."""
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=MockLLMClient([make_response(), make_response(content='{"answer": "x", "confidence": 0.5}')]),
            emitter=emitter,
            system_prompt="test",
            tools=[],
            output_schema=AnalysisResult,
        )

        await agent.run("test")

        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert "structured_output" in start_events[0].capabilities
        assert "tool_use" in start_events[0].capabilities

    async def test_react_truncation_triggers_revision(self) -> None:
        """Truncated response (max_tokens) triggers revision loop in ReAct."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class _AcceptEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            LLMResponse(content="partial", tool_calls=[], usage=make_usage(), model="test", stop_reason="max_tokens"),
            make_response(content="complete answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_evaluator=_AcceptEval(),
        )
        result = await agent.run("test")
        assert result.output == "complete answer"

    async def test_react_truncation_exceeds_max_revisions(self) -> None:
        """Repeated truncation beyond max_revisions → evaluation_failed."""
        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class _AcceptEval:
            @property
            def max_revisions(self) -> int:
                return 1

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            LLMResponse(content="partial1", tool_calls=[], usage=make_usage(), model="test", stop_reason="max_tokens"),
            LLMResponse(content="partial2", tool_calls=[], usage=make_usage(), model="test", stop_reason="max_tokens"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_evaluator=_AcceptEval(),
        )
        result = await agent.run("test")
        assert result.termination_reason == "evaluation_failed"

    async def test_react_evaluator_error_skips_evaluation(self) -> None:
        """EVALUATOR_ERROR verdict → evaluation_skipped."""
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class _ErrorEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.EVALUATOR_ERROR, evaluator_name="test")

        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_evaluator=_ErrorEval(),
        )
        result = await agent.run("test")
        assert result.termination_reason == "evaluation_skipped"

    async def test_react_reject_without_revision_budget(self) -> None:
        """REJECT verdict with max_revisions=0 → evaluation_failed."""
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class _RejectEval:
            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.REJECT, evaluator_name="test")

        client = MockLLMClient([make_response(content="bad")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_evaluator=_RejectEval(),
        )
        result = await agent.run("test")
        assert result.termination_reason == "evaluation_failed"


# ──────────────────────────────────────────────────────────
# ReAct Structured Output Evaluation Paths
# ──────────────────────────────────────────────────────────


class TestReActStructuredOutputEvaluation:
    """Tests for truncation/evaluator paths in the structured final call (output_schema)."""

    async def test_structured_truncation_triggers_revision(self) -> None:
        """Truncated structured response → revision loop."""
        from pydantic import BaseModel

        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        class _AcceptEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            # Main loop: direct answer (no tool calls)
            make_response(content="analysis complete"),
            # Structured call: truncated
            LLMResponse(
                content='{"text": "trunc"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="max_tokens",
            ),
            # Re-enter tool loop after truncation REVISE
            make_response(content="revised analysis"),
            # Revision: complete structured output
            LLMResponse(
                content='{"text": "full answer"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_AcceptEval(),
        )
        result = await agent.run("test")
        assert result.termination_reason == "complete"

    async def test_structured_truncation_in_revision_loop(self) -> None:
        """Truncation during structured revision loop."""
        from pydantic import BaseModel

        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        call_count = 0

        class _ReviseOnceEval:
            @property
            def max_revisions(self) -> int:
                return 3

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        feedback="Improve",
                        evaluator_name="test",
                    )
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            make_response(content="analysis"),
            # Structured call: normal → evaluator says REVISE
            LLMResponse(
                content='{"text": "v1"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            # Re-enter tool loop after REVISE
            make_response(content="revised analysis"),
            # Structured call: truncated → REVISE
            LLMResponse(
                content='{"text": "v2"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="max_tokens",
            ),
            # Re-enter tool loop after truncation REVISE
            make_response(content="revised analysis 2"),
            # Another structured call: complete → ACCEPT
            LLMResponse(
                content='{"text": "v3"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_ReviseOnceEval(),
        )
        result = await agent.run("test")
        assert result.termination_reason == "complete"

    async def test_structured_evaluator_error(self) -> None:
        """EVALUATOR_ERROR in structured output path → evaluation_skipped."""
        from pydantic import BaseModel

        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        class _ErrorEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.EVALUATOR_ERROR, evaluator_name="test")

        responses = [
            make_response(content="analysis"),
            make_response(content='{"text": "answer"}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_ErrorEval(),
        )
        result = await agent.run("test")
        assert result.termination_reason == "evaluation_skipped"

    async def test_structured_reject(self) -> None:
        """REJECT in structured output path → evaluation_failed."""
        from pydantic import BaseModel

        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        class _RejectEval:
            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(verdict=EvaluationVerdict.REJECT, evaluator_name="test")

        responses = [
            make_response(content="analysis"),
            make_response(content='{"text": "bad"}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_RejectEval(),
        )
        result = await agent.run("test")
        assert result.termination_reason == "evaluation_failed"

    async def test_revise_reenters_tool_loop_with_tool_calls(self) -> None:
        """REVISE re-enters the tool loop and agent can call tools during revision."""
        from pydantic import BaseModel

        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        call_count = 0

        class _ReviseOnceEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        feedback="Need to verify with a tool call",
                        evaluator_name="test",
                    )
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
        responses = [
            # 1. Tool loop: direct answer (no tool calls)
            make_response(content="initial analysis"),
            # 2. Structured output call → evaluator returns REVISE
            LLMResponse(
                content='{"text": "v1"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            # 3. Re-enter tool loop: agent calls a tool
            make_response(content="let me verify", tool_calls=[tool_call]),
            # 4. Tool loop: agent produces final text
            make_response(content="verified analysis"),
            # 5. Structured output call → evaluator returns ACCEPT
            LLMResponse(
                content='{"text": "v2 verified"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_ReviseOnceEval(),
        )
        result = await agent.run("test")

        assert result.termination_reason == "complete"
        assert result.output == '{"text": "v2 verified"}'
        # Verify tool was called during revision
        tool_result_msgs = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].content == "3"

    async def test_revise_without_tool_calls_during_revision(self) -> None:
        """REVISE re-enters tool loop; agent produces text only (no tools) then structured output."""
        from pydantic import BaseModel

        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        call_count = 0

        class _ReviseOnceEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        feedback="Improve quality",
                        evaluator_name="test",
                    )
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            # 1. Tool loop: direct answer
            make_response(content="initial analysis"),
            # 2. Structured output → REVISE
            LLMResponse(
                content='{"text": "v1"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            # 3. Re-enter tool loop: text only (no tools)
            make_response(content="improved analysis"),
            # 4. Structured output → ACCEPT
            LLMResponse(
                content='{"text": "v2 improved"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_ReviseOnceEval(),
        )
        result = await agent.run("test")

        assert result.termination_reason == "complete"
        assert result.output == '{"text": "v2 improved"}'

    async def test_revision_count_carries_across_cycles(self) -> None:
        """revision_count carries across tool-loop/structured-output cycles; max_revisions=2 means 2 total."""
        from pydantic import BaseModel

        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        class _AlwaysReviseEval:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.REVISE,
                    feedback="Not good enough",
                    evaluator_name="test",
                )

        responses = [
            # Cycle 1: tool loop → structured → REVISE (revision_count 0→1)
            make_response(content="analysis 1"),
            LLMResponse(
                content='{"text": "v1"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            # Cycle 2: tool loop → structured → REVISE (revision_count 1→2)
            make_response(content="analysis 2"),
            LLMResponse(
                content='{"text": "v2"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            # Cycle 3: tool loop → structured → REVISE but budget exhausted (revision_count == max_revisions)
            make_response(content="analysis 3"),
            LLMResponse(
                content='{"text": "v3"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_AlwaysReviseEval(),
        )
        result = await agent.run("test")

        # Budget exhausted on 3rd evaluation: verdict is REVISE but no more revisions allowed
        assert result.termination_reason == "evaluation_failed"
        assert result.output == '{"text": "v3"}'

    async def test_iteration_budget_consumed_during_revision_tool_loop(self) -> None:
        """Iteration budget is consumed during revision tool loops; exhaustion yields iteration_limit."""
        from pydantic import BaseModel

        from nanitics.infrastructure import LLMResponse
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        class _AlwaysReviseEval:
            @property
            def max_revisions(self) -> int:
                return 5

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.REVISE,
                    feedback="Keep trying",
                    evaluator_name="test",
                )

        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        responses = [
            # Cycle 1: tool call + answer (2 iterations) + structured → REVISE
            make_response(content="thinking", tool_calls=[tool_call]),
            make_response(content="analysis 1"),
            LLMResponse(
                content='{"text": "v1"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            # Cycle 2: re-enter tool loop with 1 iteration + structured → REVISE
            make_response(content="analysis 2"),
            LLMResponse(
                content='{"text": "v2"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            # Cycle 3: re-enter tool loop; iteration 4 hits limit (max_iterations=3)
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_AlwaysReviseEval(),
            max_iterations=3,
        )
        result = await agent.run("test")

        assert result.termination_reason == "iteration_limit"

    async def test_evaluation_events_emitted_correctly(self) -> None:
        """EvaluationEvent and EvaluationRevisionEvent emitted for each cycle with correct attempt numbers."""
        from pydantic import BaseModel

        from nanitics.infrastructure import LLMResponse
        from nanitics.infrastructure.observability.events import EvaluationEvent, EvaluationRevisionEvent
        from nanitics.strategies.agents.evaluation import EvaluationContext, EvaluationResult, EvaluationVerdict

        class Answer(BaseModel):
            text: str

        call_count = 0

        class _ReviseOnceEval:
            @property
            def max_revisions(self) -> int:
                return 3

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        feedback="Needs improvement",
                        evaluator_name="test",
                    )
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, evaluator_name="test")

        responses = [
            make_response(content="analysis"),
            LLMResponse(
                content='{"text": "v1"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
            make_response(content="revised analysis"),
            LLMResponse(
                content='{"text": "v2"}',
                tool_calls=[],
                usage=make_usage(),
                model="test",
                stop_reason="end_turn",
            ),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            output_schema=Answer,
            output_evaluator=_ReviseOnceEval(),
        )
        result = await agent.run("test")

        assert result.termination_reason == "complete"

        eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
        revision_events = [e for e in emitter.events if isinstance(e, EvaluationRevisionEvent)]

        # Two evaluation events: first REVISE (attempt 0), then ACCEPT (attempt 1)
        assert len(eval_events) == 2
        assert eval_events[0].verdict == "revise"
        assert eval_events[0].revision_attempt == 0
        assert eval_events[1].verdict == "accept"
        assert eval_events[1].revision_attempt == 1

        # One revision event for the REVISE cycle
        assert len(revision_events) == 1
        assert revision_events[0].feedback == "Needs improvement"
        assert revision_events[0].revision_attempt == 0
        assert revision_events[0].max_revisions == 3


# ──────────────────────────────────────────────────────────
# ReActAgent Tool Call Limit Tests
# ──────────────────────────────────────────────────────────


class TestReActToolCallLimit:
    async def test_tool_call_limit_terminates_loop(self) -> None:
        """Agent stops after reaching the tool call limit."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        # 4 responses with 2 tool calls each — limit of 3 should stop after 2 steps
        responses = [make_response(content="step", tool_calls=[tool_call, tool_call]) for _ in range(4)]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            max_iterations=10,
            max_tool_calls=3,
        )

        result = await agent.run("Loop")

        assert result.termination_reason == "tool_call_limit"
        # Step 1: 2 calls (total 2, within limit)
        # Step 2: 2 calls (total 4, exceeds 3 -> break)
        assert result.total_steps == 2

    async def test_tool_call_limit_emits_event(self) -> None:
        """Safety event is emitted when tool call limit is reached."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        # Each step has 2 tool calls, limit of 3 -> step 2 exceeds
        responses = [make_response(content="step", tool_calls=[tool_call, tool_call]) for _ in range(5)]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            max_tool_calls=3,
        )

        result = await agent.run("Loop")

        assert result.termination_reason == "tool_call_limit"
        limit_events = [e for e in emitter.events if isinstance(e, SafetyToolCallLimitEvent)]
        assert len(limit_events) == 1
        assert limit_events[0].current_tool_calls == 4  # 2+2 exceeded limit of 3
        assert limit_events[0].max_tool_calls == 3
        assert limit_events[0].agent_name == "react-agent"

    async def test_no_tool_call_limit_by_default(self) -> None:
        """Default max_tool_calls=None does not limit tool calls."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        # 3 tool-call responses + 1 final text response
        responses = [
            make_response(content="step", tool_calls=[tool_call]),
            make_response(content="step", tool_calls=[tool_call]),
            make_response(content="step", tool_calls=[tool_call]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
        )

        result = await agent.run("Do things")

        assert result.termination_reason == "complete"
        assert result.output == "done"
        assert result.total_steps == 4

    async def test_combined_limits_iteration_wins(self) -> None:
        """When max_iterations triggers first, that terminates the loop."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        responses = [make_response(content="step", tool_calls=[tool_call]) for _ in range(5)]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            max_iterations=2,
            max_tool_calls=100,
        )

        result = await agent.run("Loop")

        assert result.termination_reason == "iteration_limit"
        assert result.total_steps == 2

    async def test_combined_limits_tool_call_wins(self) -> None:
        """When max_tool_calls triggers first, that terminates the loop."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        # Each step has 3 tool calls — tool call limit of 3 stops after 2 steps
        # Step 1: 3 calls, total=3 (at limit, not over)
        # Step 2: 3 calls, total=6 (exceeds 3, break)
        responses = [make_response(content="step", tool_calls=[tool_call, tool_call, tool_call]) for _ in range(5)]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            max_iterations=10,
            max_tool_calls=3,
        )

        result = await agent.run("Loop")

        assert result.termination_reason == "tool_call_limit"
        assert result.total_steps == 2

    async def test_batch_exceeds_limit_completes_then_stops(self) -> None:
        """A batch that exceeds the limit still completes; next iteration is prevented."""
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 1})
        # First response has 4 tool calls, limit is 3 — all 4 execute, then loop stops
        responses = [
            make_response(content="step", tool_calls=[tool_call] * 4),
            make_response(content="should not reach"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[add_tool],
            max_tool_calls=3,
        )

        result = await agent.run("Loop")

        assert result.termination_reason == "tool_call_limit"
        assert result.total_steps == 1

    async def test_checkpoint_includes_tool_call_count(self) -> None:
        """Checkpoint state includes the tool call limiter count."""
        emitter = make_emitter()
        tool_call = ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})
        responses = [
            make_response(content="step", tool_calls=[tool_call]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        agent = ReActAgent(
            name="react-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            tools=[add_tool],
            max_tool_calls=10,
        )

        result = await agent.run("Do things")

        assert result.termination_reason == "complete"
        # After one tool call batch of size 1, limiter should track it
        assert agent._tool_call_limiter is not None
        # Verify the limiter was used (count was reset at start, then incremented)
        # After completion the count reflects usage during the run
        # We verify the _build_checkpoint_state method includes the field

        state = agent._build_checkpoint_state(
            messages=[],
            step_number=1,
            revision_count=0,
            usages=[],
            tool_calls=[tool_call],
            completed_tool_results={},
            suspended_tool_index=0,
        )
        assert "tool_call_limiter_count" in state
        assert isinstance(state["tool_call_limiter_count"], int)


class TestReActRunId:
    async def test_run_id_kwarg_populates_tool_context(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="capture", description="Capture context")
        async def capture_tool(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        tc = ToolCall(id="tc-abc", name="capture", arguments={})
        responses = [
            make_response(content="calling", tool_calls=[tc]),
            make_response(content="done"),
        ]
        agent = ReActAgent(
            name="react-agent",
            llm_client=MockLLMClient(responses),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[capture_tool],
            run_id="r-1",
        )

        await agent.run("go")
        assert captured[0] is not None
        assert captured[0].run_id == "r-1"
        assert captured[0].tool_call_id == "tc-abc"

    async def test_tool_state_run_id_without_kwarg(self) -> None:
        captured: list[ToolContext | None] = []

        @tool(name="capture", description="Capture context")
        async def capture_tool(context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        tc = ToolCall(id="tc-1", name="capture", arguments={})
        responses = [
            make_response(content="calling", tool_calls=[tc]),
            make_response(content="done"),
        ]
        agent = ReActAgent(
            name="react-agent",
            llm_client=MockLLMClient(responses),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[capture_tool],
            tool_state={"run_id": "r-2"},
        )

        await agent.run("go")
        assert captured[0] is not None
        assert captured[0].run_id == "r-2"
