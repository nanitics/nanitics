"""LiteLLM adapter client: one client, 100+ providers behind the same agent loop.

Demonstrates that `LiteLLMClient` is an `LLMClient` like any other — the agent loop, tool use,
and routing are unchanged when the underlying provider changes. Section 1 runs an agent against
`MockLLMClient` with a provider-prefixed model string to keep the example hermetic. Section 2
contains a commented block showing the real-API code path across three different providers
(OpenAI, Anthropic, Bedrock) — all via the same `LiteLLMClient` class. Section 3 shows a
`RoutingLLMClient` dispatching across a `LiteLLMClient` mock and an Anthropic mock to make
concrete the "use LiteLLM as the long-tail catch-all, native clients for primary providers"
pattern.

The example file does NOT import the real `litellm` package at module top-level — the import
lives inside the commented Section 2 block, so this file runs without the `[litellm]` extra.

Related guide: docs/guides/core-concepts.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    Message,
    MockLLMClient,
    ReActAgent,
    RoutingLLMClient,
    RuleBasedRouting,
    ToolCall,
    tool,
)


@tool("lookup_price", "Look up the current price of a stock ticker")
async def lookup_price(ticker: str) -> str:
    return f"{ticker.upper()}: $123.45"


async def main() -> None:
    # --- Section 1: ReActAgent with a MockLLMClient using a LiteLLM-style model string ---
    print("--- Section 1: Agent loop with a provider-prefixed (LiteLLM-style) model string ---")

    client = MockLLMClient(
        responses=[
            make_response(
                "Let me look that up.",
                tool_calls=[ToolCall(id="tc-1", name="lookup_price", arguments={"ticker": "ACME"})],
                stop_reason="tool_use",
                model="openai/gpt-4o-mini",
            ),
            make_response("ACME is currently trading at $123.45.", model="openai/gpt-4o-mini"),
        ]
    )

    agent = ReActAgent(
        name="price-agent",
        llm_client=client,
        emitter=make_emitter("litellm-mock"),
        system_prompt="You are a helpful financial assistant.",
        tools=[lookup_price],
    )

    result = await agent.run("What is ACME trading at?")
    assert result.output == "ACME is currently trading at $123.45."
    assert result.total_steps == 2
    assert result.termination_reason == "complete"

    print(f"  Output: {result.output}")
    print(f"  Steps:  {result.total_steps}")
    print("  Model:  openai/gpt-4o-mini  (LiteLLM prefixes identify the underlying provider)")
    print("✓ Same ReActAgent shape — LiteLLMClient is one LLMClient covering 100+ providers")

    # --- Section 2: Real LiteLLM API across three providers (commented out) ---
    print("\n--- Section 2: Real-API code path (commented out — requires LiteLLM + provider keys) ---")
    print("  See the source of this file for the runnable block.")
    # To run against real LiteLLM-supported providers, install the optional extra and uncomment.
    # A single `LiteLLMClient` class covers any provider — only the model string changes.
    #
    #     pip install nanitics[litellm]
    #
    # ----------------------------------------------------------------------------
    # from nanitics import LiteLLMClient
    #
    # # (A) OpenAI — reads OPENAI_API_KEY env var
    # openai_client = LiteLLMClient(model="openai/gpt-4o-mini")
    #
    # # (B) Anthropic — reads ANTHROPIC_API_KEY env var
    # anthropic_client = LiteLLMClient(model="anthropic/claude-haiku-4-5")
    #
    # # (C) Bedrock — provider-specific parameters passed via extra_kwargs
    # bedrock_client = LiteLLMClient(
    #     model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    #     extra_kwargs={"aws_region_name": "us-east-1"},
    # )
    #
    # for real_client in (openai_client, anthropic_client, bedrock_client):
    #     agent = ReActAgent(
    #         name="price-agent",
    #         llm_client=real_client,
    #         emitter=make_emitter(f"litellm-real-{real_client.model}"),
    #         system_prompt="You are a helpful financial assistant.",
    #         tools=[lookup_price],
    #     )
    #     result = await agent.run("What is ACME trading at?")
    #     print(f"{real_client.model}: {result.output}")
    # ----------------------------------------------------------------------------

    # --- Section 3: RoutingLLMClient — LiteLLM as the long-tail catch-all ---
    print("\n--- Section 3: Routing across a LiteLLM catch-all and a native client (mocked) ---")

    anthropic_mock = MockLLMClient(
        responses=[make_response("Answer from native Anthropic.", model="claude-haiku-4-5")],
    )
    litellm_mock = MockLLMClient(
        responses=[make_response("Answer from LiteLLM (Bedrock).", model="bedrock/anthropic.claude-3-5")],
    )

    # Primary provider: native Anthropic. Fallback / long-tail: LiteLLM (here simulating Bedrock).
    # Rule: when the request carries tools, use the native client (tighter error classification);
    # otherwise use LiteLLM as the catch-all.
    strategy = RuleBasedRouting(rule=lambda ctx: "anthropic" if ctx.tools else "litellm")
    router = RoutingLLMClient(
        clients={"anthropic": anthropic_mock, "litellm": litellm_mock},
        strategy=strategy,
    )

    no_tools = await router.generate(
        system_prompt="You are helpful.",
        messages=[Message(role="user", content="Quick question")],
    )
    assert no_tools.model == "bedrock/anthropic.claude-3-5"
    assert len(litellm_mock.calls) == 1
    assert len(anthropic_mock.calls) == 0

    from nanitics import ToolSchema

    with_tools = await router.generate(
        system_prompt="You are helpful.",
        messages=[Message(role="user", content="Use a tool")],
        tools=[ToolSchema(name="search", description="Search", parameters={})],
    )
    assert with_tools.model == "claude-haiku-4-5"
    assert len(anthropic_mock.calls) == 1

    print(f"  No tools  → {no_tools.model}   (LiteLLM catch-all)")
    print(f"  With tools → {with_tools.model}  (native Anthropic)")
    print("✓ Use native clients where they exist; LiteLLM fills in the long tail")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
