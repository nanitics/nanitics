from nanitics.infrastructure import (
    AgentStartEvent,
    AgentStepEvent,
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.observability.events import (
    CodeExecutionEvent,
    CodeExecutionResultEvent,
)
from nanitics.safety import (
    CancellationToken,
    ExecutionResult,
    MockSandbox,
)
from nanitics.strategies import tool
from nanitics.strategies.agents.codeact import CodeActAgent
from nanitics.tracing import (
    ToolCall,
    Usage,
)
from tests.testing_helpers import make_emitter, make_usage

_tc_counter = 0


def _next_tc_id() -> str:
    global _tc_counter
    _tc_counter += 1
    return f"tc_{_tc_counter}"


def make_llm_response(
    content: str = "response",
    usage: Usage | None = None,
    stop_reason: str = "end_turn",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=usage or make_usage(),
        model="test-model",
        stop_reason=stop_reason,
    )


def make_code_response(
    *code_blocks: str,
    content: str | None = None,
    usage: Usage | None = None,
) -> LLMResponse:
    """Create an LLMResponse with execute_code tool calls."""
    tool_calls = [ToolCall(id=_next_tc_id(), name="execute_code", arguments={"code": code}) for code in code_blocks]
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage or make_usage(),
        model="test-model",
        stop_reason="tool_use",
    )


def make_exec_result(
    stdout: str = "",
    stderr: str = "",
    return_value: str | None = None,
    success: bool = True,
    error: str | None = None,
    duration_ms: float = 10.0,
) -> ExecutionResult:
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        return_value=return_value,
        success=success,
        error=error,
        duration_ms=duration_ms,
    )


@tool(name="search", description="Search for information")
async def search_tool(query: str) -> str:
    return f"Results for: {query}"


# ──────────────────────────────────────────────────────────
# Basic Execution
# ──────────────────────────────────────────────────────────


class TestCodeActAgentBasic:
    async def test_direct_answer_no_code(self) -> None:
        """LLM responds with plain text (no code blocks) → immediate final answer."""
        client = MockLLMClient([make_llm_response(content="The answer is 42")])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("What is 6 * 7?")

        assert result.output == "The answer is 42"
        assert result.total_steps == 1
        assert result.termination_reason == "complete"

    async def test_single_code_block_then_answer(self) -> None:
        """LLM writes code → executes → LLM gives final answer."""
        responses = [
            make_code_response("print(6 * 7)", content="Let me compute:"),
            make_llm_response(content="The answer is 42"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox(
            [
                make_exec_result(stdout="42\n", return_value="42"),
            ]
        )
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Compute 6 * 7")

        assert result.output == "The answer is 42"
        assert result.total_steps == 2
        assert result.termination_reason == "complete"

    async def test_multiple_iterations(self) -> None:
        """LLM writes code twice before answering."""
        responses = [
            make_code_response("x = 10"),
            make_code_response("print(x * 2)"),
            make_llm_response(content="The result is 20"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox(
            [
                make_exec_result(return_value="10"),
                make_exec_result(stdout="20\n", return_value="20"),
            ]
        )
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Compute 10 * 2")

        assert result.output == "The result is 20"
        assert result.total_steps == 3
        assert result.termination_reason == "complete"

    async def test_multiple_code_blocks_in_single_response(self) -> None:
        """LLM returns two tool calls in one response — both executed."""
        responses = [
            make_code_response("x = 5", "print(x + 1)"),
            make_llm_response(content="The result is 6"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox(
            [
                make_exec_result(return_value="5"),
                make_exec_result(stdout="6\n"),
            ]
        )
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Compute 5 + 1")

        assert result.output == "The result is 6"
        assert result.total_steps == 2

    async def test_text_with_code_examples_no_tool_calls_is_final_answer(self) -> None:
        """LLM returns text containing code (no tool calls) → treated as final answer, not executed."""
        content_with_code = "Here's how you can do it:\n```python\nprint(6 * 7)\n```\nThis prints 42."
        client = MockLLMClient([make_llm_response(content=content_with_code)])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("How do I multiply?")

        assert result.output == content_with_code
        assert result.total_steps == 1
        assert result.termination_reason == "complete"
        # Sandbox was never called — code in text is not executed
        assert sandbox._index == 0


# ──────────────────────────────────────────────────────────
# Observation Formatting
# ──────────────────────────────────────────────────────────


class TestObservationFormatting:
    async def test_stdout_in_observation(self) -> None:
        """Execution stdout appears in observation message."""
        responses = [
            make_code_response("print('hello')"),
            make_llm_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result(stdout="hello\n")])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Print hello")

        # Observation is in the tool_result message after the assistant code message
        observation_msg = result.messages[2]  # user(input), assistant(tool_calls), tool_result
        assert observation_msg.role == "tool_result"
        assert isinstance(observation_msg.content, str)
        assert "[Execution output]" in observation_msg.content
        assert "hello" in observation_msg.content

    async def test_return_value_in_observation(self) -> None:
        responses = [
            make_code_response("2 + 2"),
            make_llm_response(content="4"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result(return_value="4")])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Compute 2+2")

        observation_msg = result.messages[2]
        assert isinstance(observation_msg.content, str)
        assert "[Return value]" in observation_msg.content
        assert "4" in observation_msg.content

    async def test_error_in_observation(self) -> None:
        """Execution error appears in observation for self-correction."""
        responses = [
            make_code_response("1/0"),
            make_code_response("print('fixed')"),
            make_llm_response(content="Fixed it"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox(
            [
                make_exec_result(
                    success=False,
                    error="ZeroDivisionError: division by zero",
                ),
                make_exec_result(stdout="fixed\n"),
            ]
        )
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Do something")

        assert result.output == "Fixed it"
        assert result.total_steps == 3
        # First observation contains the error
        observation_msg = result.messages[2]
        assert observation_msg.role == "tool_result"
        assert isinstance(observation_msg.content, str)
        assert "[Execution error]" in observation_msg.content
        assert "ZeroDivisionError" in observation_msg.content

    async def test_no_output_observation(self) -> None:
        """Code that produces no output gets a placeholder."""
        responses = [
            make_code_response("x = 1"),
            make_llm_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result()])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Set x")

        observation_msg = result.messages[2]
        assert isinstance(observation_msg.content, str)
        assert "[Execution completed with no output]" in observation_msg.content

    async def test_observation_truncation(self) -> None:
        """Long output is truncated to max_observation_length."""
        long_output = "x" * 200
        responses = [
            make_code_response("print('x' * 200)"),
            make_llm_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result(stdout=long_output)])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            max_observation_length=50,
        )

        result = await agent.run("Print a lot")

        observation_msg = result.messages[2]
        assert isinstance(observation_msg.content, str)
        assert "... (output truncated)" in observation_msg.content

    async def test_error_with_partial_output(self) -> None:
        """Error observation includes partial stdout captured before failure."""
        responses = [
            make_code_response("fail()"),
            make_llm_response(content="Understood"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox(
            [
                make_exec_result(
                    success=False,
                    error="NameError: name 'fail' is not defined",
                    stdout="partial output\n",
                ),
            ]
        )
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Do something")

        observation_msg = result.messages[2]
        assert isinstance(observation_msg.content, str)
        assert "[Execution error]" in observation_msg.content
        assert "[Partial output]" in observation_msg.content
        assert "partial output" in observation_msg.content


# ──────────────────────────────────────────────────────────
# Iteration Limits & Cancellation
# ──────────────────────────────────────────────────────────


class TestLimitsAndCancellation:
    async def test_iteration_limit(self) -> None:
        """Agent stops when iteration limit is reached."""
        responses = [make_code_response("step()") for _ in range(5)]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result() for _ in range(3)])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            max_iterations=3,
        )

        result = await agent.run("Loop forever")

        assert result.termination_reason == "iteration_limit"
        assert result.total_steps == 3
        assert result.output is None

    async def test_cancellation(self) -> None:
        """Agent stops when cancellation token is set."""
        token = CancellationToken()
        token.cancel()
        client = MockLLMClient([make_llm_response()])
        sandbox = MockSandbox([])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            cancellation_token=token,
        )

        result = await agent.run("Do something")

        assert result.termination_reason == "cancelled"
        assert result.output is None
        assert result.total_steps == 0


# ──────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────


class TestCodeActEvents:
    async def test_event_emission_order_no_code(self) -> None:
        """Plain-text response emits standard agent lifecycle events."""
        client = MockLLMClient([make_llm_response(content="answer")])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        await agent.run("Hi")

        event_types = [e.event_type for e in emitter.events]
        assert event_types == [
            "span.start",
            "agent.start",
            "span.start",  # step-1 span
            "llm.request",
            "llm.response",
            "agent.step",
            "span.end",  # step-1 span
            "agent.complete",
            "span.end",
        ]

    async def test_code_execution_events(self) -> None:
        """Code execution emits CodeExecutionEvent and CodeExecutionResultEvent."""
        responses = [
            make_code_response("print('hi')"),
            make_llm_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result(stdout="hi\n")])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        await agent.run("Print hi")

        exec_events = [e for e in emitter.events if isinstance(e, CodeExecutionEvent)]
        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]

        assert len(exec_events) == 1
        assert exec_events[0].agent_name == "codeact"
        assert exec_events[0].code == "print('hi')"
        assert exec_events[0].step_number == 1

        assert len(result_events) == 1
        assert result_events[0].stdout == "hi\n"
        assert result_events[0].success is True
        assert result_events[0].step_number == 1

    async def test_step_event_includes_thought_action_observation(self) -> None:
        """AgentStepEvent for a code step carries ``thought == response.reasoning_text``
        (sourced from the provider's reasoning extraction, not from ``content``),
        plus the concatenated code in ``action`` and the execution output in
        ``observation``."""
        responses = [
            LLMResponse(
                content="prose before code",
                tool_calls=[ToolCall(id=_next_tc_id(), name="execute_code", arguments={"code": "print(42)"})],
                usage=make_usage(),
                model="test-model",
                stop_reason="tool_use",
                reasoning_text="reasoning about the computation",
            ),
            LLMResponse(
                content="42",
                tool_calls=[],
                usage=make_usage(),
                model="test-model",
                stop_reason="end_turn",
                reasoning_text=None,
            ),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result(stdout="42\n")])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        await agent.run("Compute 42")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 2

        # First step: code execution — thought is the provider-extracted reasoning
        # text, NOT the assistant_content. content ("prose before code") belongs
        # on LLMResponseEvent, not on AgentStepEvent.thought.
        code_step = step_events[0]
        assert code_step.thought == "reasoning about the computation"
        assert code_step.action == "print(42)"
        assert code_step.observation is not None
        assert "42" in code_step.observation

        # Second step: final answer — reasoning_text is None, so thought is None.
        # observation carries the model's final content on the terminal no-code
        # step so the event isn't all-None for trace readers.
        answer_step = step_events[1]
        assert answer_step.thought is None
        assert answer_step.observation == "42"
        assert answer_step.action is None

    async def test_codeact_terminal_step_populates_observation_from_content(self) -> None:
        """Closes observability W2 for CodeAct: on a run whose sole response
        is a plain text final answer (no code call) with reasoning_text=None,
        the emitted AgentStepEvent populates observation from content so the
        event is not all-None. Mirrors the ReAct terminal-step guarantee."""
        responses = [
            LLMResponse(
                content="OK",
                tool_calls=[],
                usage=make_usage(),
                model="test-model",
                stop_reason="end_turn",
                reasoning_text=None,
            ),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Reply with a single short word.",
            sandbox=sandbox,
            max_iterations=1,
        )

        await agent.run("Say OK.")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        assert len(step_events) == 1
        step = step_events[0]
        assert step.observation == "OK"
        assert step.thought is None
        assert step.action is None
        assert step.artifact is None

    async def test_start_event_no_tools(self) -> None:
        """AgentStartEvent reports no tools when none are configured."""
        client = MockLLMClient([make_llm_response()])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        await agent.run("Hi")

        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].tools_available == []

    async def test_start_event_with_tools(self) -> None:
        """AgentStartEvent reports tool names when tools are configured."""
        # Need a stub init response for tool stubs
        client = MockLLMClient([make_llm_response()])
        emitter = make_emitter()
        sandbox = MockSandbox(
            [
                make_exec_result(),  # stub initialization
            ]
        )
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            tools=[search_tool],
        )

        await agent.run("Hi")

        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert "search" in start_events[0].tools_available


# ──────────────────────────────────────────────────────────
# Tool Stubs
# ──────────────────────────────────────────────────────────


class TestToolStubSetup:
    async def test_tool_stubs_executed_in_sandbox(self) -> None:
        """When tools are provided, stub code is executed as the first sandbox call."""
        client = MockLLMClient([make_llm_response(content="answer")])
        emitter = make_emitter()
        sandbox = MockSandbox(
            [
                make_exec_result(),  # stub initialization
            ]
        )
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            tools=[search_tool],
        )

        result = await agent.run("Hi")

        assert result.output == "answer"
        # MockSandbox consumed 1 response (stub init)
        assert sandbox._index == 1

    async def test_no_stub_execution_without_tools(self) -> None:
        """Without tools, no stub initialization execute() call happens."""
        client = MockLLMClient([make_llm_response(content="answer")])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Hi")

        assert result.output == "answer"
        assert sandbox._index == 0

    async def test_tool_docs_in_system_prompt(self) -> None:
        """Tool documentation is appended to system prompt when tools provided."""
        client = MockLLMClient([make_llm_response()])
        emitter = make_emitter()
        sandbox = MockSandbox([make_exec_result()])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            tools=[search_tool],
        )

        assert "Available Functions" in agent._system_prompt
        assert "search" in agent._system_prompt

    async def test_code_instructions_in_system_prompt(self) -> None:
        """Code execution instructions are in the system prompt."""
        client = MockLLMClient([make_llm_response()])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        assert "Code Execution Environment" in agent._system_prompt


# ──────────────────────────────────────────────────────────
# Output Evaluation
# ──────────────────────────────────────────────────────────


class TestCodeActEvaluation:
    async def test_evaluation_accept(self) -> None:
        """Output evaluator accepts the answer → complete."""
        from nanitics.capabilities.evaluation import EvaluationResult, EvaluationVerdict

        class AcceptEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output, context):
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=1.0,
                    feedback=None,
                    evaluator_name="test",
                )

        client = MockLLMClient([make_llm_response(content="good answer")])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            output_evaluator=AcceptEvaluator(),
        )

        result = await agent.run("Question")

        assert result.output == "good answer"
        assert result.termination_reason == "complete"

    async def test_evaluation_revise(self) -> None:
        """Output evaluator requests revision → agent retries."""
        from nanitics.capabilities.evaluation import EvaluationResult, EvaluationVerdict

        call_count = 0

        class ReviseOnceEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output, context):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        score=0.3,
                        feedback="Be more specific",
                        evaluator_name="test",
                    )
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=0.9,
                    feedback=None,
                    evaluator_name="test",
                )

        responses = [
            make_llm_response(content="vague answer"),
            make_llm_response(content="specific answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            output_evaluator=ReviseOnceEvaluator(),
        )

        result = await agent.run("Question")

        assert result.output == "specific answer"
        assert result.termination_reason == "complete"
        assert result.total_steps == 2

    async def test_evaluation_reject(self) -> None:
        """Output evaluator rejects → evaluation_failed termination."""
        from nanitics.capabilities.evaluation import EvaluationResult, EvaluationVerdict

        class RejectEvaluator:
            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output, context):
                return EvaluationResult(
                    verdict=EvaluationVerdict.REJECT,
                    score=0.0,
                    feedback="Unacceptable",
                    evaluator_name="test",
                )

        client = MockLLMClient([make_llm_response(content="bad answer")])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            output_evaluator=RejectEvaluator(),
        )

        result = await agent.run("Question")

        assert result.output == "bad answer"
        assert result.termination_reason == "evaluation_failed"


# ──────────────────────────────────────────────────────────
# Messages
# ──────────────────────────────────────────────────────────


class TestCodeActMessages:
    async def test_message_structure_with_code(self) -> None:
        """Messages: user(input), assistant(tool_calls), tool_result, assistant(answer)."""
        responses = [
            make_code_response("print(1)"),
            make_llm_response(content="The answer is 1"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result(stdout="1\n")])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Print 1")

        assert len(result.messages) == 4
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Print 1"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].tool_calls is not None
        assert len(result.messages[1].tool_calls) == 1
        assert result.messages[1].tool_calls[0].name == "execute_code"
        assert result.messages[2].role == "tool_result"
        assert isinstance(result.messages[2].content, str)
        assert "[Execution output]" in result.messages[2].content
        assert result.messages[3].role == "assistant"
        assert result.messages[3].content == "The answer is 1"

    async def test_usage_aggregation(self) -> None:
        """Usage is aggregated across all LLM calls."""
        usage1 = make_usage(input_tokens=100, output_tokens=50)
        usage2 = make_usage(input_tokens=200, output_tokens=100)
        responses = [
            make_code_response("x = 1", usage=usage1),
            make_llm_response(content="Done", usage=usage2),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result()])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Do something")

        assert result.usage.input_tokens == 300
        assert result.usage.output_tokens == 150
        assert result.usage.total_tokens == 450


# ──────────────────────────────────────────────────────────
# Context Manager Reset
# ──────────────────────────────────────────────────────────


class TestContextManagerReset:
    async def test_context_manager_reset_called(self) -> None:
        """Context manager reset() is called at the start of _execute."""
        from nanitics.capabilities.context import ContextManager
        from nanitics.capabilities.context.truncation import TruncationPolicy

        cm = ContextManager(
            context_limit=100_000,
            truncation=TruncationPolicy(preserve_recent=5),
        )
        reset_called = False
        original_reset = cm.reset

        def tracking_reset() -> None:
            nonlocal reset_called
            reset_called = True
            original_reset()

        cm.reset = tracking_reset  # type: ignore[method-assign]

        client = MockLLMClient([make_llm_response(content="answer")])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            context_manager=cm,
        )

        await agent.run("Hello")

        assert reset_called


# ──────────────────────────────────────────────────────────
# Truncation + Evaluator Paths
# ──────────────────────────────────────────────────────────


class TestCodeActTruncation:
    async def test_truncation_triggers_revision_with_evaluator(self) -> None:
        """Truncated response with evaluator and revision budget → retries."""
        from nanitics.capabilities.evaluation import EvaluationResult, EvaluationVerdict

        class AcceptEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output, context):
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=1.0,
                    feedback=None,
                    evaluator_name="test",
                )

        responses = [
            make_llm_response(content="truncated...", stop_reason="max_tokens"),
            make_llm_response(content="complete answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            output_evaluator=AcceptEvaluator(),
        )

        result = await agent.run("Question")

        assert result.output == "complete answer"
        assert result.termination_reason == "complete"

    async def test_truncation_exceeds_max_revisions(self) -> None:
        """Truncated responses that exceed max_revisions → evaluation_failed."""
        from nanitics.capabilities.evaluation import EvaluationResult, EvaluationVerdict

        class AcceptEvaluator:
            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output, context):
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=1.0,
                    feedback=None,
                    evaluator_name="test",
                )

        responses = [
            make_llm_response(content="truncated...", stop_reason="max_tokens"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            output_evaluator=AcceptEvaluator(),
        )

        result = await agent.run("Question")

        assert result.output == "truncated..."
        assert result.termination_reason == "evaluation_failed"

    async def test_evaluator_error_skips_evaluation(self) -> None:
        """EVALUATOR_ERROR verdict → evaluation_skipped termination."""
        from nanitics.capabilities.evaluation import EvaluationResult, EvaluationVerdict

        class ErrorEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output, context):
                return EvaluationResult(
                    verdict=EvaluationVerdict.EVALUATOR_ERROR,
                    score=None,
                    feedback="Something broke",
                    evaluator_name="test",
                )

        client = MockLLMClient([make_llm_response(content="answer")])
        emitter = make_emitter()
        sandbox = MockSandbox([])
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
            output_evaluator=ErrorEvaluator(),
        )

        result = await agent.run("Question")

        assert result.output == "answer"
        assert result.termination_reason == "evaluation_skipped"


# ──────────────────────────────────────────────────────────
# Format Observation Edge Cases
# ──────────────────────────────────────────────────────────


class TestFormatObservationEdgeCases:
    async def test_failed_execution_with_no_error_details(self) -> None:
        """Failed exec with no error, no stdout → '[Execution failed with no error details]'."""
        responses = [
            make_code_response("crash()"),
            make_llm_response(content="Done"),
        ]
        client = MockLLMClient(responses)
        sandbox = MockSandbox([make_exec_result(success=False)])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            sandbox=sandbox,
        )

        result = await agent.run("Do something")

        observation_msg = result.messages[2]
        assert isinstance(observation_msg.content, str)
        assert "[Execution failed with no error details]" in observation_msg.content


class TestCodeActRunId:
    def test_run_id_kwarg_populates_registry_tool_state(self) -> None:
        client = MockLLMClient([make_llm_response()])
        sandbox = MockSandbox([make_exec_result()])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            sandbox=sandbox,
            tools=[search_tool],
            run_id="r-1",
        )

        assert agent._tool_registry is not None
        assert agent._tool_registry._tool_state.get("run_id") == "r-1"

    def test_tool_state_run_id_without_kwarg(self) -> None:
        client = MockLLMClient([make_llm_response()])
        sandbox = MockSandbox([make_exec_result()])
        emitter = make_emitter()
        agent = CodeActAgent(
            name="codeact",
            llm_client=client,
            emitter=emitter,
            system_prompt="test",
            sandbox=sandbox,
            tools=[search_tool],
            tool_state={"run_id": "r-2"},
        )

        assert agent._tool_registry is not None
        assert agent._tool_registry._tool_state.get("run_id") == "r-2"
