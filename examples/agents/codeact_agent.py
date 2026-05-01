"""CodeActAgent: code as action, sandbox execution, self-correction, and tool bridge.

Demonstrates the CodeActAgent — the agent type that writes Python code instead of
selecting tools. Shows the sandbox execution loop, observation formatting, error
recovery via tracebacks, stateful computation, and bridging SDK tools as callable
functions in the code environment.

Contrast with examples/agents/react_agent.py:
  ReAct:    LLM selects tool name + parameters → tool executes → result returned
  CodeAct:  LLM writes Python code → sandbox executes → stdout/return_value/error returned

Related guide: docs/guides/agent-types.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    CodeActAgent,
    ExecutionResult,
    MockLLMClient,
    MockSandbox,
    ToolCall,
    tool,
)
from nanitics.infrastructure import (
    CodeExecutionEvent,
    CodeExecutionResultEvent,
)

# --- Local helpers for CodeAct-specific patterns ---


def make_code_response(code: str, content: str | None = None):
    """Create an LLMResponse with a single execute_code tool call."""
    return make_response(
        content=content or "",
        tool_calls=[ToolCall(id=f"tc-code-{hash(code) % 10000}", name="execute_code", arguments={"code": code})],
        stop_reason="tool_use",
    )


def make_exec_result(
    stdout: str = "",
    return_value: str | None = None,
    success: bool = True,
    error: str | None = None,
) -> ExecutionResult:
    """Create an ExecutionResult with sensible defaults."""
    return ExecutionResult(
        stdout=stdout,
        stderr="",
        return_value=return_value,
        success=success,
        error=error,
        duration_ms=1.0,
    )


async def main() -> None:
    # --- Section 1: Direct Answer (No Code Execution) ---
    print("--- Section 1: Direct Answer (No Code Execution) ---")

    client = MockLLMClient(
        responses=[
            make_response("The answer is 42."),
        ]
    )
    sandbox = MockSandbox(responses=[])
    emitter = make_emitter("codeact-s1")

    agent = CodeActAgent(
        name="direct-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        sandbox=sandbox,
    )

    result = await agent.run("What is the meaning of life?")

    assert result.output == "The answer is 42.", f"Expected direct answer, got: {result.output}"
    assert result.total_steps == 1, f"Expected 1 step, got: {result.total_steps}"
    assert result.termination_reason == "complete"

    # No tool-result messages means no code execution happened
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    assert tool_results == [], "Sandbox should not have been called"

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Termination: {result.termination_reason}")
    print("✓ CodeAct doesn't force code execution — LLM answers directly when it can")

    # --- Section 2: Single Code Block → Observe → Answer ---
    print("\n--- Section 2: Single Code Block → Observe → Answer ---")

    client = MockLLMClient(
        responses=[
            make_code_response("print(6 * 7)", content="Let me calculate that."),
            make_response("The answer is 42."),
        ]
    )
    sandbox = MockSandbox(
        responses=[
            make_exec_result(stdout="42\n", return_value="42"),
        ]
    )
    emitter = make_emitter("codeact-s2")

    agent = CodeActAgent(
        name="calc-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a calculator.",
        sandbox=sandbox,
    )

    result = await agent.run("What is 6 times 7?")

    assert result.output == "The answer is 42."
    assert result.total_steps == 2, f"Expected 2 steps, got: {result.total_steps}"
    assert result.termination_reason == "complete"

    # Verify observation formatting in conversation
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    assert len(tool_results) == 1
    assert "[Execution output]" in tool_results[0].content
    assert "42" in tool_results[0].content
    assert "[Return value]" in tool_results[0].content

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Observation: {tool_results[0].content!r:.80}")
    print("✓ Core CodeAct cycle: write code → sandbox executes → observe output → answer")

    # --- Section 3: Error → Self-Correction ---
    print("\n--- Section 3: Error → Self-Correction ---")

    client = MockLLMClient(
        responses=[
            make_code_response("1 / 0", content="Let me compute."),
            make_code_response("print(42 / 6)", content="Let me fix that."),
            make_response("The answer is 7.0."),
        ]
    )
    sandbox = MockSandbox(
        responses=[
            make_exec_result(success=False, error="ZeroDivisionError: division by zero"),
            make_exec_result(stdout="7.0\n", success=True),
        ]
    )
    emitter = make_emitter("codeact-s3")

    agent = CodeActAgent(
        name="self-correcting-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a calculator.",
        sandbox=sandbox,
    )

    result = await agent.run("Divide 42 by 6.")

    assert result.output == "The answer is 7.0."
    assert result.total_steps == 3, f"Expected 3 steps, got: {result.total_steps}"

    # First observation contains the error
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    assert "[Execution error]" in tool_results[0].content
    assert "ZeroDivisionError" in tool_results[0].content

    # Second observation contains the successful output
    assert "[Execution output]" in tool_results[1].content

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Error observation: {tool_results[0].content!r:.60}")
    print(f"  Fixed observation: {tool_results[1].content!r:.60}")
    print("✓ Tracebacks give precise error signals — agent self-corrects")

    # --- Section 4: Multiple Iterations (Stateful Computation) ---
    print("\n--- Section 4: Multiple Iterations (Stateful Computation) ---")

    client = MockLLMClient(
        responses=[
            make_code_response("data = [10, 20, 30, 40, 50]"),
            make_code_response("result = sum(data) / len(data)\nprint(f'Average: {result}')"),
            make_response("The average is 30.0."),
        ]
    )
    sandbox = MockSandbox(
        responses=[
            make_exec_result(return_value="[10, 20, 30, 40, 50]"),
            make_exec_result(stdout="Average: 30.0\n"),
        ]
    )
    emitter = make_emitter("codeact-s4")

    agent = CodeActAgent(
        name="stateful-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a data analyst.",
        sandbox=sandbox,
    )

    result = await agent.run("What is the average of 10, 20, 30, 40, 50?")

    assert result.output == "The average is 30.0."
    assert result.total_steps == 3, f"Expected 3 steps, got: {result.total_steps}"

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print("✓ Sandbox persists state across iterations — variables carry over between code blocks")

    # --- Section 5: Tool Bridge — SDK Tools as Functions ---
    print("\n--- Section 5: Tool Bridge — SDK Tools as Functions ---")

    @tool("lookup_price", "Look up the price of an item")
    async def lookup_price(item: str) -> str:
        return "$9.99"

    client = MockLLMClient(
        responses=[
            make_code_response("price = lookup_price(item='widget')\nprint(f'Price: {price}')"),
            make_response("A widget costs $9.99."),
        ]
    )
    sandbox = MockSandbox(
        responses=[
            # First result: automatic tool stub initialization
            make_exec_result(success=True),
            # Second result: actual code execution
            make_exec_result(stdout="Price: $9.99\n"),
        ]
    )
    emitter = make_emitter("codeact-s5")

    agent = CodeActAgent(
        name="tool-bridge-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a shopping assistant.",
        sandbox=sandbox,
        tools=[lookup_price],
    )

    result = await agent.run("How much does a widget cost?")

    assert result.output == "A widget costs $9.99."

    # The bridged tool was invoked from the sandboxed code — verify the tool
    # result text made it through the conversation's tool-result messages.
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    assert any("$9.99" in m.content for m in tool_results), "Expected bridged tool output in observation"

    print(f"  Output: {result.output}")
    print("✓ SDK tools become callable functions in the sandbox via the tool bridge")

    # --- Section 6: Code Execution Events ---
    print("\n--- Section 6: Code Execution Events ---")

    client = MockLLMClient(
        responses=[
            make_code_response("print('hello world')"),
            make_response("Done."),
        ]
    )
    sandbox = MockSandbox(
        responses=[
            make_exec_result(stdout="hello world\n"),
        ]
    )
    emitter = make_emitter("codeact-s6")

    agent = CodeActAgent(
        name="event-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        sandbox=sandbox,
    )

    await agent.run("Say hello.")

    # Filter events by type
    exec_events = [e for e in emitter.events if isinstance(e, CodeExecutionEvent)]
    result_events = [e for e in emitter.events if isinstance(e, CodeExecutionResultEvent)]

    assert len(exec_events) == 1, f"Expected 1 CodeExecutionEvent, got: {len(exec_events)}"
    assert exec_events[0].agent_name == "event-agent"
    assert exec_events[0].code == "print('hello world')"
    assert exec_events[0].step_number == 1

    assert len(result_events) == 1, f"Expected 1 CodeExecutionResultEvent, got: {len(result_events)}"
    assert result_events[0].agent_name == "event-agent"
    assert result_events[0].stdout == "hello world\n"
    assert result_events[0].success is True
    assert result_events[0].step_number == 1
    assert result_events[0].duration_ms >= 0

    # Events are emitted in order: execution → result
    exec_idx = emitter.events.index(exec_events[0])
    result_idx = emitter.events.index(result_events[0])
    assert exec_idx < result_idx, "CodeExecutionEvent should precede CodeExecutionResultEvent"

    print(f"  CodeExecutionEvent: agent={exec_events[0].agent_name}, code={exec_events[0].code!r}")
    print(f"  CodeExecutionResultEvent: stdout={result_events[0].stdout!r}, success={result_events[0].success}")
    print(f"  Event order: execution (index {exec_idx}) → result (index {result_idx})")
    print("✓ Event system provides full visibility into code execution")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
