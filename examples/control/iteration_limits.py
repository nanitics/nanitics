"""Iteration and tool call limits: standalone limiter APIs, error inspection, and agent safety integration.

Demonstrates IterationLimiter and ToolCallLimiter as standalone safety primitives — construction,
step counting, limit enforcement via errors, reset/restore — and then shows how ReActAgent uses
them internally to prevent infinite loops and unbounded tool usage.

Related guide: docs/guides/safety.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    AgentIterationLimitError,
    AgentToolCallLimitError,
    IterationLimiter,
    MockLLMClient,
    ReActAgent,
    ToolCall,
    ToolCallLimiter,
    tool,
)
from nanitics.infrastructure import (
    SafetyIterationLimitEvent,
    SafetyToolCallLimitEvent,
)


@tool("get_weather", "Get the current weather for a city")
async def get_weather(city: str) -> str:
    return f"Sunny, 22°C in {city}"


async def main() -> None:
    # --- Section 1: IterationLimiter — Construction and Properties ---
    print("--- Section 1: IterationLimiter — Construction and Properties ---")

    limiter = IterationLimiter(max_iterations=5)

    assert limiter.max_iterations == 5
    assert limiter.current_iteration == 0
    assert limiter.remaining == 5

    # Invalid construction raises ValueError
    for invalid in (0, -1):
        try:
            IterationLimiter(max_iterations=invalid)
            assert False, f"Should have raised ValueError for max_iterations={invalid}"
        except ValueError:
            pass

    print(f"  max_iterations: {limiter.max_iterations}")
    print(f"  current_iteration: {limiter.current_iteration}")
    print(f"  remaining: {limiter.remaining}")
    print("✓ IterationLimiter constructed; invalid values rejected")

    # --- Section 2: step() — Counting and Limit Enforcement ---
    print("\n--- Section 2: step() — Counting and Limit Enforcement ---")

    limiter = IterationLimiter(max_iterations=3)

    # Step 3 times — within limit
    for i in range(3):
        limiter.step()
        assert limiter.current_iteration == i + 1
        assert limiter.remaining == 3 - (i + 1)

    assert limiter.current_iteration == 3
    assert limiter.remaining == 0

    # Fourth step exceeds limit — step() increments before checking
    try:
        limiter.step()
        assert False, "Should have raised AgentIterationLimitError"
    except AgentIterationLimitError as e:
        assert e.iteration_count == 4
        assert e.iteration_limit == 3

    print(f"  After 3 steps: current={3}, remaining={0}")
    print("  4th step raised AgentIterationLimitError (count=4, limit=3)")
    print("✓ step() increments counter; exceeding limit raises error with inspection attributes")

    # --- Section 3: reset() and restore() ---
    print("\n--- Section 3: reset() and restore() ---")

    limiter = IterationLimiter(max_iterations=5)

    # Step 3 times, then reset
    for _ in range(3):
        limiter.step()
    assert limiter.current_iteration == 3

    limiter.reset()
    assert limiter.current_iteration == 0
    assert limiter.remaining == 5

    # Restore to a specific count
    limiter.restore(2)
    assert limiter.current_iteration == 2
    assert limiter.remaining == 3

    # Invalid restore values raise ValueError
    try:
        limiter.restore(-1)
        assert False, "Should have raised ValueError for negative count"
    except ValueError:
        pass

    try:
        limiter.restore(6)
        assert False, "Should have raised ValueError for count > max_iterations"
    except ValueError:
        pass

    print(f"  After reset: current={0}, remaining={5}")
    print(f"  After restore(2): current={2}, remaining={3}")
    print("✓ reset() clears counter; restore() sets arbitrary position; invalid values rejected")

    # --- Section 4: Agent Hitting Iteration Limit ---
    print("\n--- Section 4: Agent Hitting Iteration Limit ---")

    # Every response includes a tool call — agent never produces a final answer
    responses = [
        make_response(
            f"Checking attempt {i + 1}...",
            tool_calls=[ToolCall(id=f"tc-{i}", name="get_weather", arguments={"city": "Nowhere"})],
            stop_reason="tool_use",
        )
        for i in range(5)
    ]
    client = MockLLMClient(responses=responses)
    emitter = make_emitter("iter-limit")

    agent = ReActAgent(
        name="limited-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
        max_iterations=2,
    )

    result = await agent.run("Keep checking the weather forever.")

    assert result.termination_reason == "iteration_limit"
    assert result.total_steps == 2
    assert result.output is None

    # Verify SafetyIterationLimitEvent was emitted
    limit_events = [e for e in emitter.events if isinstance(e, SafetyIterationLimitEvent)]
    assert len(limit_events) == 1
    limit_event = limit_events[0]
    assert limit_event.agent_name == "limited-agent"
    assert limit_event.max_iterations == 2

    print(f"  Termination reason: {result.termination_reason}")
    print(f"  Steps taken: {result.total_steps}")
    print(f"  Output: {result.output}")
    print(f"  SafetyIterationLimitEvent: agent={limit_event.agent_name}, max_iterations={limit_event.max_iterations}")
    print("✓ Agent stopped at iteration limit; SafetyIterationLimitEvent emitted")

    # --- Section 5: ToolCallLimiter — Construction and Batch Counting ---
    print("\n--- Section 5: ToolCallLimiter — Construction and Batch Counting ---")

    tc_limiter = ToolCallLimiter(max_tool_calls=5)

    assert tc_limiter.max_tool_calls == 5
    assert tc_limiter.current_tool_calls == 0
    assert tc_limiter.remaining == 5

    # Negative max_tool_calls raises ValueError
    try:
        ToolCallLimiter(max_tool_calls=-1)
        assert False, "Should have raised ValueError for max_tool_calls=-1"
    except ValueError:
        pass

    # max_tool_calls=0 is valid: it permits no tool calls (the very first
    # tool-batch dispatch raises AgentToolCallLimitError at the limiter).
    tc_limiter_zero = ToolCallLimiter(max_tool_calls=0)
    assert tc_limiter_zero.max_tool_calls == 0
    print("✓ ToolCallLimiter(max_tool_calls=0) constructs (zero permits no tool calls)")

    # Batch counting — step() accepts a count (number of tool calls in one LLM response)
    tc_limiter.step(2)  # 2 tool calls in first response
    assert tc_limiter.current_tool_calls == 2
    assert tc_limiter.remaining == 3

    tc_limiter.step(3)  # 3 more — now at exactly 5 (the limit), still OK
    assert tc_limiter.current_tool_calls == 5
    assert tc_limiter.remaining == 0

    # Next batch exceeds limit
    try:
        tc_limiter.step(1)
        assert False, "Should have raised AgentToolCallLimitError"
    except AgentToolCallLimitError as e:
        assert e.tool_call_count == 6
        assert e.tool_call_limit == 5

    # step(0) is a no-op (LLM response with no tool calls)
    tc_limiter2 = ToolCallLimiter(max_tool_calls=1)
    tc_limiter2.step(0)
    assert tc_limiter2.current_tool_calls == 0

    print(f"  After step(2) + step(3): current={5}, remaining={0}")
    print("  step(1) raised AgentToolCallLimitError (count=6, limit=5)")
    print("  step(0) is a no-op")
    print("✓ ToolCallLimiter counts tool calls across batches; exceeding limit raises error")

    # --- Section 6: ToolCallLimiter — reset() and restore() ---
    print("\n--- Section 6: ToolCallLimiter — reset() and restore() ---")

    tc_limiter = ToolCallLimiter(max_tool_calls=10)
    tc_limiter.step(7)
    assert tc_limiter.current_tool_calls == 7

    tc_limiter.reset()
    assert tc_limiter.current_tool_calls == 0
    assert tc_limiter.remaining == 10

    tc_limiter.restore(4)
    assert tc_limiter.current_tool_calls == 4
    assert tc_limiter.remaining == 6

    # Invalid restore values raise ValueError
    try:
        tc_limiter.restore(-1)
        assert False, "Should have raised ValueError for negative count"
    except ValueError:
        pass

    try:
        tc_limiter.restore(11)
        assert False, "Should have raised ValueError for count > max_tool_calls"
    except ValueError:
        pass

    print(f"  After reset: current={0}, remaining={10}")
    print(f"  After restore(4): current={4}, remaining={6}")
    print("✓ reset() and restore() work the same as IterationLimiter")

    # --- Section 7: Agent Hitting Tool Call Limit ---
    print("\n--- Section 7: Agent Hitting Tool Call Limit ---")

    # Each response requests 2 tool calls — agent will hit the tool call limit of 3
    responses = [
        make_response(
            f"Batch {i + 1}...",
            tool_calls=[
                ToolCall(id=f"tc-{i}-a", name="get_weather", arguments={"city": "Paris"}),
                ToolCall(id=f"tc-{i}-b", name="get_weather", arguments={"city": "London"}),
            ],
            stop_reason="tool_use",
        )
        for i in range(5)
    ]
    client = MockLLMClient(responses=responses)
    emitter = make_emitter("tc-limit")

    agent = ReActAgent(
        name="tc-limited-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
        max_tool_calls=3,
        max_iterations=10,  # high — tool call limit should trigger first
    )

    result = await agent.run("Check weather in many cities.")

    # Step 1: 2 tool calls (total=2, within limit)
    # Step 2: 2 tool calls (total=4, exceeds limit of 3 — step completes, then stops)
    assert result.termination_reason == "tool_call_limit"
    assert result.total_steps == 2
    assert result.output is None

    # Verify SafetyToolCallLimitEvent was emitted
    tc_events = [e for e in emitter.events if isinstance(e, SafetyToolCallLimitEvent)]
    assert len(tc_events) == 1
    tc_event = tc_events[0]
    assert tc_event.agent_name == "tc-limited-agent"
    assert tc_event.max_tool_calls == 3

    print(f"  Termination reason: {result.termination_reason}")
    print(f"  Steps taken: {result.total_steps}")
    print(f"  SafetyToolCallLimitEvent: agent={tc_event.agent_name}, max_tool_calls={tc_event.max_tool_calls}")
    print("✓ Agent stopped at tool call limit; SafetyToolCallLimitEvent emitted")


if __name__ == "__main__":
    asyncio.run(main())
