"""Multi-tool packages with shared per-run state.

Sibling tools that mutate a shared counter through `ToolContext.state`. The
factory returns `((tool_a, tool_b), state_dict)`; the consumer registers the
tools and threads `state_dict` into the agent's `tool_state`. A second factory
call yields a fresh state dict — the shape is per-run, not module-global.

Related guide: docs/guides/tools.md
"""

import asyncio
from typing import Any

from examples.helpers import make_emitter, make_response
from nanitics import (
    FunctionTool,
    MockLLMClient,
    ReActAgent,
    ToolCall,
    ToolContext,
    tool,
)


def create_counter_tools() -> tuple[tuple[FunctionTool, FunctionTool], dict[str, Any]]:
    """Build a pair of sibling tools sharing a per-run counter dict.

    Returns ``((increment, read), state)``. Both tools read or mutate
    ``state["count"]`` through ``ToolContext.state``; the consumer threads
    ``state`` into the agent via ``tool_state=state``. Each call to this
    factory yields a fresh dict — there is no module-global shared state.
    """
    state: dict[str, Any] = {"count": 0}

    @tool("increment_counter", "Increment the run-scoped counter by an amount.")
    async def increment_counter(amount: int, context: ToolContext) -> str:
        context.state["count"] += amount
        return f"Counter incremented by {amount}; total now {context.state['count']}."

    @tool("read_counter", "Read the current run-scoped counter value.")
    async def read_counter(context: ToolContext) -> str:
        return f"count: {context.state['count']}"

    return ((increment_counter, read_counter), state)


async def main() -> None:
    # --- Section 1: Both tools share the same state dict ---
    print("--- Section 1: Sibling tools share state through ToolContext.state ---")

    tools, state = create_counter_tools()
    assert state == {"count": 0}, "Factory returns a fresh zeroed counter."

    client = MockLLMClient(
        responses=[
            make_response(
                "I will increment the counter twice and then read it.",
                tool_calls=[ToolCall(id="tc-1", name="increment_counter", arguments={"amount": 3})],
                stop_reason="tool_use",
            ),
            make_response(
                "Continuing.",
                tool_calls=[ToolCall(id="tc-2", name="increment_counter", arguments={"amount": 4})],
                stop_reason="tool_use",
            ),
            make_response(
                "Reading the counter.",
                tool_calls=[ToolCall(id="tc-3", name="read_counter", arguments={})],
                stop_reason="tool_use",
            ),
            make_response("The counter is 7."),
        ]
    )

    agent = ReActAgent(
        name="counter-agent",
        llm_client=client,
        emitter=make_emitter("multi-tool-s1"),
        system_prompt="You operate a counter through the available tools.",
        tools=list(tools),
        tool_state=state,
    )

    result = await agent.run("Increment the counter by 3, then by 4, and tell me the total.")

    assert result.termination_reason == "complete"
    assert state["count"] == 7, "Both tools read and wrote the same shared dict."
    assert "7" in result.output, f"Output should reflect the read; got {result.output!r}"
    print(f"  state after run: {state}")
    print(f"  agent output: {result.output}")

    # --- Section 2: A second factory call yields fresh state ---
    print("\n--- Section 2: A second factory call yields fresh state ---")

    (tool_a2, tool_b2), state2 = create_counter_tools()
    assert state2 == {"count": 0}, "Second factory call: state is independent."
    assert state2 is not state, "The two factory returns share no aliasing."

    client2 = MockLLMClient(
        responses=[
            make_response(
                "Reading.",
                tool_calls=[ToolCall(id="tc-r1", name="read_counter", arguments={})],
                stop_reason="tool_use",
            ),
            make_response("The counter is 0."),
        ]
    )
    agent2 = ReActAgent(
        name="counter-agent-2",
        llm_client=client2,
        emitter=make_emitter("multi-tool-s2"),
        system_prompt="You operate a counter through the available tools.",
        tools=[tool_a2, tool_b2],
        tool_state=state2,
    )

    result2 = await agent2.run("What is the counter currently?")
    assert state2 == {"count": 0}, "No mutation: read-only run leaves the dict unchanged."
    assert "0" in result2.output, f"Fresh state should read as zero; got {result2.output!r}"
    print(f"  state2 after run: {state2}")
    print(f"  agent2 output: {result2.output}")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
