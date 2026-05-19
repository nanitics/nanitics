"""Integration tests for CodeActAgent + DockerSandbox — requires Docker daemon."""

import pytest

from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.observability.events import (
    CodeExecutionEvent,
    CodeExecutionResultEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.safety.sandbox.docker import DockerSandbox
from nanitics.safety.sandbox.protocol import SandboxConfig
from nanitics.strategies import tool
from nanitics.strategies.agents.codeact import CodeActAgent
from nanitics.tracing import (
    ToolCall,
    Usage,
)
from tests.testing_helpers import make_emitter, make_usage


def _docker_available() -> bool:
    try:
        import docker as docker_lib

        client = docker_lib.from_env()  # type: ignore[attr-defined]
        client.ping()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available"),
]


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


_tc_counter = 0


def _next_tc_id() -> str:
    global _tc_counter
    _tc_counter += 1
    return f"tc_{_tc_counter}"


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


@tool(name="add", description="Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


@tool(name="greet", description="Greet a person by name")
async def greet_tool(name: str) -> str:
    return f"Hello, {name}!"


# ──────────────────────────────────────────────────────────
# Full Loop: CodeActAgent + DockerSandbox
# ──────────────────────────────────────────────────────────


class TestCodeActDockerIntegration:
    async def test_single_code_execution_and_answer(self) -> None:
        """Agent writes code, sandbox executes it, agent answers."""
        responses = [
            make_code_response(
                "result = 6 * 7\nprint(result)",
                content="Let me calculate:",
            ),
            make_llm_response(content="The answer is 42."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        async with DockerSandbox() as sandbox:
            agent = CodeActAgent(
                name="calc",
                llm_client=client,
                emitter=emitter,
                system_prompt="You are a calculator.",
                sandbox=sandbox,
            )
            result = await agent.run("What is 6 * 7?")

        assert result.output == "The answer is 42."
        assert result.total_steps == 2
        assert result.termination_reason == "complete"

        # Verify code execution events were emitted
        exec_events = [e for e in emitter.events if isinstance(e, CodeExecutionEvent)]
        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]
        assert len(exec_events) == 1
        assert len(result_events) == 1
        assert result_events[0].success is True
        assert "42" in result_events[0].stdout


# ──────────────────────────────────────────────────────────
# Tool Bridge: Agent Tools Callable from Sandbox
# ──────────────────────────────────────────────────────────


class TestCodeActToolBridgeIntegration:
    async def test_tool_call_roundtrip(self) -> None:
        """Agent code calls a tool function, result flows back correctly."""
        responses = [
            make_code_response(
                'result = add(a=3, b=4)\nprint(f"Sum is {result}")',
            ),
            make_llm_response(content="The sum of 3 and 4 is 7."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        async with DockerSandbox() as sandbox:
            agent = CodeActAgent(
                name="tool-agent",
                llm_client=client,
                emitter=emitter,
                system_prompt="You are helpful.",
                sandbox=sandbox,
                tools=[add_tool],
            )
            result = await agent.run("What is 3 + 4?")

        assert result.output == "The sum of 3 and 4 is 7."
        assert result.termination_reason == "complete"

        # Verify tool events were emitted via ToolRegistry
        tool_invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        tool_result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(tool_invoke_events) == 1
        assert tool_invoke_events[0].tool_name == "add"
        assert len(tool_result_events) == 1

        # Verify code execution events
        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is True
        assert "Sum is 7" in result_events[0].stdout

    async def test_multiple_tool_calls_in_code(self) -> None:
        """Code calls multiple tools in sequence within one execution."""
        responses = [
            make_code_response(
                'r1 = add(a=10, b=20)\nr2 = greet(name="Alice")\nprint(f"{r1} / {r2}")',
            ),
            make_llm_response(content="Done."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        async with DockerSandbox() as sandbox:
            agent = CodeActAgent(
                name="multi-tool",
                llm_client=client,
                emitter=emitter,
                system_prompt="You are helpful.",
                sandbox=sandbox,
                tools=[add_tool, greet_tool],
            )
            result = await agent.run("Add 10+20 and greet Alice")

        assert result.output == "Done."

        tool_invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        assert len(tool_invoke_events) == 2
        tool_names = {e.tool_name for e in tool_invoke_events}
        assert tool_names == {"add", "greet"}

        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]
        assert result_events[0].success is True
        assert "30" in result_events[0].stdout
        assert "Hello, Alice!" in result_events[0].stdout


# ──────────────────────────────────────────────────────────
# Multi-Iteration: Agent Refines Across Steps
# ──────────────────────────────────────────────────────────


class TestCodeActMultiIteration:
    async def test_multi_step_computation(self) -> None:
        """Agent writes code, sees output, writes more code, produces final answer."""
        responses = [
            make_code_response(
                "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
                content="First, let me define a function:",
            ),
            make_code_response(
                "result = factorial(5)\nprint(result)",
                content="Now compute:",
            ),
            make_llm_response(content="The factorial of 5 is 120."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        async with DockerSandbox() as sandbox:
            agent = CodeActAgent(
                name="factorial-agent",
                llm_client=client,
                emitter=emitter,
                system_prompt="You are a mathematician.",
                sandbox=sandbox,
            )
            result = await agent.run("What is 5 factorial?")

        assert result.output == "The factorial of 5 is 120."
        assert result.total_steps == 3
        assert result.termination_reason == "complete"

        # Both code execution steps should have succeeded
        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]
        assert len(result_events) == 2
        assert all(e.success for e in result_events)
        # Second step should show 120
        assert "120" in result_events[1].stdout

    async def test_state_persists_across_iterations(self) -> None:
        """Variables set in one iteration are accessible in the next."""
        responses = [
            make_code_response("data = [1, 2, 3, 4, 5]"),
            make_code_response("total = sum(data)\nprint(total)"),
            make_llm_response(content="The sum is 15."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        async with DockerSandbox() as sandbox:
            agent = CodeActAgent(
                name="state-agent",
                llm_client=client,
                emitter=emitter,
                system_prompt="You are helpful.",
                sandbox=sandbox,
            )
            result = await agent.run("Sum 1-5")

        assert result.output == "The sum is 15."
        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]
        assert len(result_events) == 2
        assert result_events[1].success is True
        assert "15" in result_events[1].stdout


# ──────────────────────────────────────────────────────────
# Error Self-Correction: Agent Sees Traceback, Fixes Code
# ──────────────────────────────────────────────────────────


class TestCodeActErrorSelfCorrection:
    async def test_error_then_correction(self) -> None:
        """Code produces error, agent sees traceback, writes corrected code."""
        responses = [
            # First attempt: has a NameError
            make_code_response("print(undefined_variable)"),
            # Agent sees the traceback and corrects
            make_code_response(
                "defined_variable = 42\nprint(defined_variable)",
                content="I see the error. Let me fix:",
            ),
            make_llm_response(content="The value is 42."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        async with DockerSandbox() as sandbox:
            agent = CodeActAgent(
                name="correction-agent",
                llm_client=client,
                emitter=emitter,
                system_prompt="You are helpful.",
                sandbox=sandbox,
            )
            result = await agent.run("Print the variable")

        assert result.output == "The value is 42."
        assert result.total_steps == 3
        assert result.termination_reason == "complete"

        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]
        assert len(result_events) == 2

        # First execution failed
        assert result_events[0].success is False
        assert result_events[0].error is not None
        assert "NameError" in result_events[0].error

        # Second execution succeeded
        assert result_events[1].success is True
        assert "42" in result_events[1].stdout

        # Verify the observation with traceback was sent to the LLM
        observation_msg = result.messages[2]  # user(input), assistant(tool_calls), tool_result
        assert isinstance(observation_msg.content, str)
        assert "[Execution error]" in observation_msg.content
        assert "NameError" in observation_msg.content


# ──────────────────────────────────────────────────────────
# Timeout: Code Exceeds Timeout Limit
# ──────────────────────────────────────────────────────────


class TestCodeActTimeout:
    async def test_code_exceeds_timeout(self) -> None:
        """Code that runs too long produces a timeout error observation."""
        responses = [
            make_code_response("import time\ntime.sleep(30)"),
            make_llm_response(content="The code timed out."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        config = SandboxConfig(timeout=3.0)

        async with DockerSandbox(config) as sandbox:
            agent = CodeActAgent(
                name="timeout-agent",
                llm_client=client,
                emitter=emitter,
                system_prompt="You are helpful.",
                sandbox=sandbox,
            )
            result = await agent.run("Sleep for a while")

        assert result.output == "The code timed out."
        assert result.termination_reason == "complete"

        result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert result_events[0].error is not None
        assert "timed out" in result_events[0].error.lower()

        # Verify the timeout error appeared in the observation
        observation_msg = result.messages[2]
        assert isinstance(observation_msg.content, str)
        assert "[Execution error]" in observation_msg.content
        assert "timed out" in observation_msg.content.lower()
