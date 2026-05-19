"""OpenAI LLM client: drop-in provider behind the same agent loop.

Demonstrates that `OpenAILLMClient` is an `LLMClient` like any other — the agent loop, tool use,
and routing are unchanged when swapping providers. Section 1 runs an agent against `MockLLMClient`
to keep the example hermetic. Section 2 contains a commented block showing the real-API code path
(requires `OPENAI_API_KEY`). Section 3 shows `RoutingLLMClient` dispatching across an OpenAI mock
and an Anthropic mock to make the cross-provider story concrete.

Related guide: docs/guides/core-concepts.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    MockLLMClient,
    RoutingLLMClient,
    RuleBasedRouting,
)
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import (
    Message,
    ToolCall,
)


@tool("lookup_price", "Look up the current price of a stock ticker")
async def lookup_price(ticker: str) -> str:
    return f"{ticker.upper()}: $123.45"


async def main() -> None:
    # --- Section 1: ReActAgent with a MockLLMClient (always runs) ---
    print("--- Section 1: Agent loop with mocked OpenAI-style responses ---")

    client = MockLLMClient(
        responses=[
            make_response(
                "Let me look that up.",
                tool_calls=[ToolCall(id="tc-1", name="lookup_price", arguments={"ticker": "ACME"})],
                stop_reason="tool_use",
                model="gpt-4o-mini",
            ),
            make_response("ACME is currently trading at $123.45.", model="gpt-4o-mini"),
        ]
    )

    agent = ReActAgent(
        name="price-agent",
        llm_client=client,
        emitter=make_emitter("openai-mock"),
        system_prompt="You are a helpful financial assistant.",
        tools=[lookup_price],
    )

    result = await agent.run("What is ACME trading at?")
    assert result.output == "ACME is currently trading at $123.45."
    assert result.total_steps == 2
    assert result.termination_reason == "complete"

    print(f"  Output: {result.output}")
    print(f"  Steps:  {result.total_steps}")
    print("✓ Same ReActAgent shape — only the client identity changes when moving to OpenAI")

    # --- Section 2: Real OpenAI API (commented out) ---
    print("\n--- Section 2: Real-API code path (commented out — requires OPENAI_API_KEY) ---")
    print("  See the source of this file for the runnable block.")
    # To run against the real OpenAI API, install nanitics and uncomment:
    #
    #     pip install nanitics
    #
    # ----------------------------------------------------------------------------
    # from nanitics import OpenAILLMClient
    #
    # real_client = OpenAILLMClient(model="gpt-4o-mini")  # reads OPENAI_API_KEY env var
    # agent = ReActAgent(
    #     name="price-agent",
    #     llm_client=real_client,
    #     emitter=make_emitter("openai-real"),
    #     system_prompt="You are a helpful financial assistant.",
    #     tools=[lookup_price],
    # )
    # result = await agent.run("What is ACME trading at?")
    # print(result.output)
    # ----------------------------------------------------------------------------

    # --- Section 3: RoutingLLMClient across OpenAI and Anthropic (mocked) ---
    print("\n--- Section 3: Cross-provider routing with mocked clients ---")

    openai_mock = MockLLMClient(
        responses=[make_response("Answer from OpenAI.", model="gpt-4o-mini")],
    )
    anthropic_mock = MockLLMClient(
        responses=[make_response("Answer from Anthropic.", model="claude-haiku-4-5")],
    )

    # Route to OpenAI when there are no tools, Anthropic when tools are present.
    strategy = RuleBasedRouting(rule=lambda ctx: "anthropic" if ctx.tools else "openai")
    router = RoutingLLMClient(
        clients={"openai": openai_mock, "anthropic": anthropic_mock},
        strategy=strategy,
    )

    no_tools = await router.generate(
        system_prompt="You are helpful.",
        messages=[Message(role="user", content="Quick question")],
    )
    assert no_tools.model == "gpt-4o-mini"
    assert len(openai_mock.calls) == 1
    assert len(anthropic_mock.calls) == 0

    from nanitics.infrastructure import ToolSchema

    with_tools = await router.generate(
        system_prompt="You are helpful.",
        messages=[Message(role="user", content="Use a tool")],
        tools=[ToolSchema(name="search", description="Search", parameters={})],
    )
    assert with_tools.model == "claude-haiku-4-5"
    assert len(anthropic_mock.calls) == 1

    print(f"  No tools  → {no_tools.model}")
    print(f"  With tools → {with_tools.model}")
    print("✓ RoutingLLMClient dispatches across providers through the same protocol")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
