"""ReAct agent lifecycle: tools, multi-turn conversation, result inspection, and event tracing.

Demonstrates creating a ReActAgent with tools, running it with MockLLMClient, inspecting
AgentResult (output, steps, termination, usage), examining conversation flow, and observing
emitted events. This is the bridge from standalone components to an integrated agent loop.

Related guide: docs/guides/getting-started.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    AgentCompleteEvent,
    AgentStartEvent,
    AgentStepEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    MockLLMClient,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import ToolCall

# --- Shared tools ---


@tool("get_weather", "Get the current weather for a city")
async def get_weather(city: str) -> str:
    return f"Sunny, 22°C in {city}"


@tool("calculate", "Evaluate a math expression")
async def calculate(expression: str) -> str:
    return f"Result of {expression} is 4"


async def main() -> None:
    # --- Section 1: Simple Agent — Direct Answer (No Tool Calls) ---
    print("--- Section 1: Simple Agent — Direct Answer ---")

    client = MockLLMClient(
        responses=[
            make_response("It's sunny and 22°C."),
        ]
    )
    emitter = make_emitter("react-s1")

    agent = ReActAgent(
        name="weather-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful weather assistant.",
        tools=[get_weather],
    )

    result = await agent.run("What's the weather?")

    assert result.output == "It's sunny and 22°C.", f"Expected direct answer, got: {result.output}"
    assert result.total_steps == 1, f"Expected 1 step, got: {result.total_steps}"
    assert result.termination_reason == "complete"
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Usage: {result.usage.input_tokens} in / {result.usage.output_tokens} out")
    print("✓ Simplest agent run — LLM answers directly without calling tools")

    # --- Section 2: Tool Use — Multi-Step Conversation ---
    print("\n--- Section 2: Tool Use — Multi-Step Conversation ---")

    client = MockLLMClient(
        responses=[
            make_response(
                "Let me check the weather.",
                tool_calls=[ToolCall(id="tc-1", name="get_weather", arguments={"city": "Amsterdam"})],
                stop_reason="tool_use",
            ),
            make_response("The weather in Amsterdam is sunny at 22°C."),
        ]
    )
    emitter = make_emitter("react-s2")

    agent = ReActAgent(
        name="weather-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful weather assistant.",
        tools=[get_weather],
    )

    result = await agent.run("What's the weather in Amsterdam?")

    assert result.output == "The weather in Amsterdam is sunny at 22°C."
    assert result.total_steps == 2, f"Expected 2 steps, got: {result.total_steps}"
    assert result.termination_reason == "complete"

    # Inspect conversation history
    messages = result.messages
    assert messages[0].role == "user"
    assert messages[0].content == "What's the weather in Amsterdam?"

    assert messages[1].role == "assistant"
    assert messages[1].tool_calls is not None
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0].name == "get_weather"

    assert messages[2].role == "tool_result"
    assert messages[2].tool_call_id == "tc-1"
    assert "Sunny, 22°C in Amsterdam" in messages[2].content

    assert messages[3].role == "assistant"
    assert messages[3].content == "The weather in Amsterdam is sunny at 22°C."

    print("  Conversation flow:")
    for msg in messages:
        role = msg.role
        if msg.tool_calls:
            print(f"    {role}: {msg.content} → tool_calls: {[tc.name for tc in msg.tool_calls]}")
        elif msg.tool_call_id:
            print(f"    {role} (id={msg.tool_call_id}): {msg.content}")
        else:
            print(f"    {role}: {msg.content}")
    print("✓ ReAct loop: LLM calls tool → observes result → produces final answer")

    # --- Section 3: Multiple Tool Calls in One Turn ---
    print("\n--- Section 3: Multiple Tool Calls in One Turn ---")

    client = MockLLMClient(
        responses=[
            make_response(
                "Let me check the weather and do the calculation.",
                tool_calls=[
                    ToolCall(id="tc-w", name="get_weather", arguments={"city": "Paris"}),
                    ToolCall(id="tc-c", name="calculate", arguments={"expression": "2 + 2"}),
                ],
                stop_reason="tool_use",
            ),
            make_response("Paris is sunny at 22°C, and 2 + 2 = 4."),
        ]
    )
    emitter = make_emitter("react-s3")

    agent = ReActAgent(
        name="multi-tool-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant with weather and math tools.",
        tools=[get_weather, calculate],
    )

    result = await agent.run("What's the weather in Paris? Also, what's 2 + 2?")

    assert result.output == "Paris is sunny at 22°C, and 2 + 2 = 4."
    assert result.total_steps == 2, f"Expected 2 steps, got: {result.total_steps}"

    # Both tool results should be in the conversation
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    assert len(tool_results) == 2, f"Expected 2 tool results, got: {len(tool_results)}"
    tool_call_ids = {m.tool_call_id for m in tool_results}
    assert tool_call_ids == {"tc-w", "tc-c"}

    print(f"  Output: {result.output}")
    print(f"  Tool results: {len(tool_results)} (ids: {tool_call_ids})")
    print("✓ LLM batches multiple tool calls in one response")

    # --- Section 4: Multi-Turn Tool Use ---
    print("\n--- Section 4: Multi-Turn Tool Use ---")

    client = MockLLMClient(
        responses=[
            make_response(
                "I'll check Amsterdam first.",
                tool_calls=[ToolCall(id="tc-a", name="get_weather", arguments={"city": "Amsterdam"})],
                stop_reason="tool_use",
            ),
            make_response(
                "Now let me check Berlin.",
                tool_calls=[ToolCall(id="tc-b", name="get_weather", arguments={"city": "Berlin"})],
                stop_reason="tool_use",
            ),
            make_response("Amsterdam is sunny at 22°C and Berlin is also sunny at 22°C. Both are great!"),
        ]
    )
    emitter = make_emitter("react-s4")

    agent = ReActAgent(
        name="comparison-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful weather comparison assistant.",
        tools=[get_weather],
    )

    result = await agent.run("Compare the weather in Amsterdam and Berlin.")

    assert result.total_steps == 3, f"Expected 3 steps, got: {result.total_steps}"
    assert result.termination_reason == "complete"

    tool_results = [m for m in result.messages if m.role == "tool_result"]
    assert len(tool_results) == 2

    # Usage is aggregated across all 3 LLM calls
    assert result.usage.input_tokens == 30  # 3 calls × 10 tokens (make_usage default)
    assert result.usage.output_tokens == 15  # 3 calls × 5 tokens

    print(f"  Steps: {result.total_steps}")
    print(f"  Tool calls across turns: {len(tool_results)}")
    print(f"  Aggregated usage: {result.usage.input_tokens} in / {result.usage.output_tokens} out")
    print(f"  Output: {result.output}")
    print("✓ Agent loop continues across multiple turns; usage is aggregated")

    # --- Section 5: Inspecting Events ---
    print("\n--- Section 5: Inspecting Events ---")

    # Fresh run for clean event trace
    client = MockLLMClient(
        responses=[
            make_response(
                "Let me check.",
                tool_calls=[ToolCall(id="tc-e", name="get_weather", arguments={"city": "Tokyo"})],
                stop_reason="tool_use",
            ),
            make_response("Tokyo is sunny at 22°C."),
        ]
    )
    emitter = make_emitter("react-s5")

    agent = ReActAgent(
        name="event-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
    )

    await agent.run("What's the weather in Tokyo?")

    # Filter events by type
    starts = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
    assert len(starts) == 1
    assert starts[0].agent_name == "event-agent"
    assert "get_weather" in starts[0].tools_available
    assert starts[0].task_input == "What's the weather in Tokyo?"

    llm_requests = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
    llm_responses = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
    assert len(llm_requests) == 2, f"Expected 2 LLM requests, got: {len(llm_requests)}"
    assert len(llm_responses) == 2

    tool_invokes = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
    tool_results_ev = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
    assert len(tool_invokes) == 1
    assert tool_invokes[0].tool_name == "get_weather"
    assert tool_invokes[0].parameters == {"city": "Tokyo"}
    assert len(tool_results_ev) == 1
    assert tool_results_ev[0].success is True

    steps = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
    assert len(steps) == 2, f"Expected 2 step events, got: {len(steps)}"

    completes = [e for e in emitter.events if isinstance(e, AgentCompleteEvent)]
    assert len(completes) == 1
    assert completes[0].termination_reason == "complete"
    assert completes[0].total_steps == 2

    print("  Event timeline:")
    for event in emitter.events:
        event_type = event.event_type
        if isinstance(event, AgentStartEvent):
            print(f"    {event_type}: agent={event.agent_name}, tools={event.tools_available}")
        elif isinstance(event, LLMRequestEvent):
            print(f"    {event_type}: {len(event.messages)} messages")
        elif isinstance(event, LLMResponseEvent):
            print(f"    {event_type}: content={event.content!r:.50}, usage={event.usage}")
        elif isinstance(event, ToolInvokeEvent):
            print(f"    {event_type}: {event.tool_name}({event.parameters})")
        elif isinstance(event, ToolResultEvent):
            print(f"    {event_type}: {event.tool_name} success={event.success}")
        elif isinstance(event, AgentStepEvent):
            print(f"    {event_type}: step {event.step_number}")
        elif isinstance(event, AgentCompleteEvent):
            print(f"    {event_type}: reason={event.termination_reason}, steps={event.total_steps}")
        else:
            print(f"    {event_type}")
    print("✓ Emitter captures complete execution trace")

    # --- Section 6: Iteration Limit ---
    print("\n--- Section 6: Iteration Limit ---")

    # Every response includes a tool call — agent never reaches a final answer
    infinite_responses = [
        make_response(
            f"Checking again (attempt {i + 1})...",
            tool_calls=[ToolCall(id=f"tc-loop-{i}", name="get_weather", arguments={"city": "Nowhere"})],
            stop_reason="tool_use",
        )
        for i in range(5)
    ]
    client = MockLLMClient(responses=infinite_responses)
    emitter = make_emitter("react-s6")

    agent = ReActAgent(
        name="limited-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
        max_iterations=3,
    )

    result = await agent.run("Keep checking the weather forever.")

    assert result.termination_reason == "iteration_limit", f"Expected iteration_limit, got: {result.termination_reason}"
    assert result.total_steps == 3, f"Expected 3 steps, got: {result.total_steps}"
    assert result.output is None, f"Expected no output, got: {result.output}"

    print(f"  Termination reason: {result.termination_reason}")
    print(f"  Steps taken: {result.total_steps}")
    print(f"  Output: {result.output}")
    print("✓ max_iterations stops the agent; output is None when limit reached")

    # --- Section 7: MockLLMClient — Inspecting Calls ---
    print("\n--- Section 7: MockLLMClient — Inspecting Calls ---")

    client = MockLLMClient(
        responses=[
            make_response(
                "Let me look that up.",
                tool_calls=[ToolCall(id="tc-i", name="get_weather", arguments={"city": "London"})],
                stop_reason="tool_use",
            ),
            make_response("London is sunny at 22°C."),
        ]
    )
    emitter = make_emitter("react-s7")

    agent = ReActAgent(
        name="inspectable-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a weather assistant.",
        tools=[get_weather],
    )

    await agent.run("What's the weather in London?")

    # MockLLMClient records every call
    assert len(client.calls) == 2, f"Expected 2 LLM calls, got: {len(client.calls)}"

    # First call: just the user message
    first_call = client.calls[0]
    assert "weather assistant" in first_call["system_prompt"]
    assert len(first_call["messages"]) >= 1  # At least the user message
    assert first_call["messages"][0].role == "user"
    assert first_call["tools"] is not None
    assert any(t.name == "get_weather" for t in first_call["tools"])

    # Second call: includes tool result from first step
    second_call = client.calls[1]
    tool_result_msgs = [m for m in second_call["messages"] if m.role == "tool_result"]
    assert len(tool_result_msgs) >= 1, "Second call should include tool result"

    print(f"  Total LLM calls: {len(client.calls)}")
    print(f"  Call 1 system_prompt: ...{first_call['system_prompt'][-40:]}")
    print(f"  Call 1 messages: {len(first_call['messages'])} message(s)")
    print(f"  Call 1 tools: {[t.name for t in first_call['tools']]}")
    print(f"  Call 2 messages: {len(second_call['messages'])} message(s) (includes tool result)")
    print("✓ MockLLMClient.calls records everything sent to the LLM")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
