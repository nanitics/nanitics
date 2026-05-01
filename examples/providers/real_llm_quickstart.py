"""Real-LLM quickstart: the bridge from MockLLMClient to a live provider.

The shortest path from ``pip install nanitics`` to a full agent run against a
real LLM, complete with one tool call, a printed trace, and a cost summary. Every other
example in this gallery is hermetic by design — this is the deliberate exception. With
``ANTHROPIC_API_KEY`` unset the example prints a skip message and exits; with the key set
it runs a single ReActAgent iteration against ``claude-haiku-4-5`` (the cheapest Claude
model) so a first-run costs fractions of a cent.

Usage:
    pip install nanitics
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python examples/providers/real_llm_quickstart.py

Related guide: docs/guides/getting-started.md
Related example: examples/agents/react_agent.py (the MockLLMClient counterpart).
"""

import asyncio
import os
import sys
import time


async def main() -> None:
    # --- Hermetic skip guard ---
    # With no key, exit cleanly so tests/test_examples.py collects and passes.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("SKIPPED: set ANTHROPIC_API_KEY to run this example.")
        print("  pip install nanitics")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    # Import heavy dependencies only after the guard — keeps the skip path fast
    # (no anthropic SDK import cost for CI runs that don't carry a key).
    from nanitics import (
        AnthropicLLMClient,
        InMemoryEmitter,
        ReActAgent,
        tool,
    )
    from nanitics.infrastructure import (
        AgentCompleteEvent,
        AgentStartEvent,
        LLMRequestEvent,
        LLMResponseEvent,
        ToolInvokeEvent,
        ToolResultEvent,
    )

    # --- Section 1: Minimal Agent With a Real Tool Call ---
    print("--- Section 1: Minimal Agent With a Real Tool Call ---")

    @tool("today_date", "Return today's date in ISO 8601 format.")
    async def today_date() -> str:
        # No network, no subprocess — just a deterministic local value the agent can read.
        return time.strftime("%Y-%m-%d")

    # Cheapest Claude model — keeps a first-run near zero cost while exercising the full loop.
    client = AnthropicLLMClient(model="claude-haiku-4-5")
    emitter = InMemoryEmitter(trace_id="real-llm-quickstart")

    agent = ReActAgent(
        name="date-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt=(
            "You are a helpful assistant. If the user asks about the date, call the "
            "``today_date`` tool and then answer using the tool's result."
        ),
        tools=[today_date],
    )

    start_wall = time.perf_counter()
    result = await agent.run("What is today's date?")
    elapsed_s = time.perf_counter() - start_wall

    # Real LLMs are non-deterministic: assert shape (non-empty output, complete), not content.
    assert isinstance(result.output, str) and result.output, "Expected a non-empty string output"
    assert result.termination_reason == "complete", f"Unexpected termination: {result.termination_reason}"

    print(f"  Agent output: {result.output}")
    print(f"  Steps: {result.total_steps}, termination: {result.termination_reason}")

    # --- Section 2: Print the Trace ---
    print("\n--- Section 2: Trace ---")

    # A compact, readable projection of the emitted events. This is the "what observability
    # do I get for free" moment — every LLM call, every tool, every span is already captured.
    for event in emitter.events:
        if isinstance(event, AgentStartEvent):
            print(f"  [agent.start]    agent={event.agent_name} task={event.task_input!r}")
        elif isinstance(event, LLMRequestEvent):
            print(f"  [llm.request]    model={event.model_name}")
        elif isinstance(event, LLMResponseEvent):
            usage = event.usage
            print(
                f"  [llm.response]   model={event.model_name} "
                f"in={usage.input_tokens} out={usage.output_tokens} total={usage.total_tokens}"
            )
        elif isinstance(event, ToolInvokeEvent):
            print(f"  [tool.invoke]    tool={event.tool_name} params={event.parameters}")
        elif isinstance(event, ToolResultEvent):
            print(f"  [tool.result]    tool={event.tool_name} success={event.success}")
        elif isinstance(event, AgentCompleteEvent):
            print(f"  [agent.complete] steps={event.total_steps} reason={event.termination_reason}")

    # --- Section 3: Cost and Duration Summary ---
    print("\n--- Section 3: Cost Summary ---")

    llm_responses = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
    total_input = sum(e.usage.input_tokens for e in llm_responses)
    total_output = sum(e.usage.output_tokens for e in llm_responses)
    total_tokens = sum(e.usage.total_tokens for e in llm_responses)

    print(
        f"  Run used {total_tokens} tokens "
        f"({total_input} in / {total_output} out) "
        f"across {len(llm_responses)} LLM call(s), total duration {elapsed_s:.1f}s"
    )

    # Guardrail: a first-run against haiku + one tool call is ~2 LLM turns (call tool, then
    # respond using the tool result). Staying under a few thousand tokens keeps cost in the
    # fractions-of-a-cent range. Raise this ceiling only if the SDK's agent loop legitimately
    # grows — not to paper over accidental iteration blow-ups.
    assert total_tokens < 5000, (
        f"Unexpected token usage ({total_tokens}); the quickstart is meant to stay tiny. "
        "Check the system prompt or the number of iterations."
    )

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    # Adopter-friendly error surface for common first-run misconfigurations.
    # Set NANITICS_DEBUG=1 to see the full traceback instead of the one-liner.
    try:
        asyncio.run(main())
    except Exception as e:
        if os.environ.get("NANITICS_DEBUG") == "1":
            raise
        from nanitics.infrastructure.errors import LLMProviderError

        if isinstance(e, LLMProviderError):
            print(f"\nLLM provider error: {e}", file=sys.stderr)
            print(
                "  Re-run with NANITICS_DEBUG=1 for the full traceback.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
