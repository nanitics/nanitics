"""LLM routing: dispatch requests across multiple model clients with strategies and observability.

Demonstrates `RoutingLLMClient`, `RuleBasedRouting`, `CostBudgetRouting`, custom `RoutingStrategy`,
and `ModelRoutingEvent` — all at the `generate()` level. Uses multiple `MockLLMClient` instances to
simulate model tiers. No agent integration — `RoutingLLMClient` implements the `LLMClient` protocol,
so it's a drop-in replacement anywhere an LLM client is expected.

Related guide: docs/guides/core-concepts.md
"""

import asyncio

from examples.helpers import make_emitter, make_response, make_usage
from nanitics.infrastructure import (
    CostBudgetRouting,
    MockLLMClient,
    ModelRoutingEvent,
    RoutingContext,
    RoutingLLMClient,
    RoutingStrategy,
    RuleBasedRouting,
    ToolSchema,
)
from nanitics.tracing import Message


async def main() -> None:
    # --- Section 1: Rule-Based Routing by Request Context ---
    print("--- Section 1: Rule-Based Routing by Request Context ---")

    # RuleBasedRouting delegates the routing decision to a user-defined function
    # that receives a RoutingContext — the full request metadata.
    strategy = RuleBasedRouting(rule=lambda ctx: "tool-capable" if ctx.tools else "fast")

    # When tools are present, route to the tool-capable model
    context_with_tools = RoutingContext(
        system_prompt="You are a helpful assistant.",
        messages=[Message(role="user", content="Search for info")],
        tools=[ToolSchema(name="search", description="Search the web", parameters={})],
        output_schema=None,
    )
    assert strategy.select(context_with_tools) == "tool-capable"

    # Without tools, route to the fast model
    context_no_tools = RoutingContext(
        system_prompt="You are a helpful assistant.",
        messages=[Message(role="user", content="What is 2+2?")],
        tools=None,
        output_schema=None,
    )
    assert strategy.select(context_no_tools) == "fast"

    # Routing can inspect any RoutingContext property — output_schema, messages, etc.
    strategy_by_schema = RuleBasedRouting(rule=lambda ctx: "structured" if ctx.output_schema else "general")
    assert strategy_by_schema.select(context_no_tools) == "general"

    print("  tools present    → 'tool-capable'")
    print("  no tools         → 'fast'")
    print("  no output_schema → 'general'")
    print("✓ RuleBasedRouting inspects RoutingContext to select a client key")

    # --- Section 2: RoutingLLMClient Dispatch ---
    print("\n--- Section 2: RoutingLLMClient Dispatch ---")

    # Two mock clients simulating different model tiers
    fast_client = MockLLMClient(
        responses=[
            make_response("Fast answer.", model="fast-model"),
        ]
    )
    smart_client = MockLLMClient(
        responses=[
            make_response("Thorough analysis with reasoning.", model="smart-model"),
        ]
    )

    strategy = RuleBasedRouting(rule=lambda ctx: "smart" if ctx.tools else "fast")

    router = RoutingLLMClient(
        clients={"fast": fast_client, "smart": smart_client},
        strategy=strategy,
    )

    # No tools → fast client
    response = await router.generate(
        system_prompt="You are helpful.",
        messages=[Message(role="user", content="Quick question")],
    )
    assert response.model == "fast-model"
    assert len(fast_client.calls) == 1
    assert len(smart_client.calls) == 0

    # With tools → smart client
    response = await router.generate(
        system_prompt="You are helpful.",
        messages=[Message(role="user", content="Use tools")],
        tools=[ToolSchema(name="search", description="Search", parameters={})],
    )
    assert response.model == "smart-model"
    assert len(smart_client.calls) == 1

    print(f"  No tools  → {fast_client.calls[0]['messages'][0].content!r} → fast-model")
    print(f"  With tools → {smart_client.calls[0]['messages'][0].content!r} → smart-model")
    print("✓ RoutingLLMClient dispatches generate() calls to the correct backing client")

    # --- Section 3: Default Fallback for Unknown Strategy Keys ---
    print("\n--- Section 3: Default Fallback for Unknown Strategy Keys ---")

    fallback_client = MockLLMClient(
        responses=[
            make_response("Fallback response.", model="fallback-model"),
        ]
    )

    # Strategy returns "unknown" — not in the clients dict
    strategy = RuleBasedRouting(rule=lambda ctx: "unknown")

    # Without default → ValueError
    router_no_default = RoutingLLMClient(
        clients={"primary": fallback_client},
        strategy=strategy,
    )
    try:
        await router_no_default.generate(
            system_prompt="test",
            messages=[Message(role="user", content="test")],
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown" in str(e)
        print(f"  No default  → ValueError: {e}")

    # With default → falls back to the specified client
    fallback_client = MockLLMClient(
        responses=[
            make_response("Fallback response.", model="fallback-model"),
        ]
    )
    router_with_default = RoutingLLMClient(
        clients={"primary": fallback_client},
        strategy=strategy,
        default="primary",
    )
    response = await router_with_default.generate(
        system_prompt="test",
        messages=[Message(role="user", content="test")],
    )
    assert response.model == "fallback-model"
    print(f"  With default → routed to 'primary': {response.model}")
    print("✓ default= catches unknown strategy keys instead of raising")

    # --- Section 4: CostBudgetRouting Threshold Transitions ---
    print("\n--- Section 4: CostBudgetRouting Threshold Transitions ---")

    # Budget of 1000 tokens, two transition thresholds
    strategy = CostBudgetRouting(
        budget=1000,
        thresholds=[
            (0.5, "medium"),  # above 50% usage → switch to medium
            (0.8, "cheap"),  # above 80% usage → switch to cheap
        ],
        default="expensive",  # use expensive until a threshold is hit
    )

    # Initially: 0 used, 1000 remaining → default
    assert strategy.select(context_no_tools) == "expensive"
    assert strategy.used == 0
    assert strategy.remaining == 1000

    # Simulate 600 tokens used → 60% → crosses 0.5 threshold → "medium"
    strategy.on_response("expensive", make_response("r1", usage=make_usage(400, 200)))
    assert strategy.used == 600
    assert strategy.remaining == 400
    assert strategy.select(context_no_tools) == "medium"

    # Simulate 300 more tokens → 900 total → 90% → crosses 0.8 threshold → "cheap"
    strategy.on_response("medium", make_response("r2", usage=make_usage(200, 100)))
    assert strategy.used == 900
    assert strategy.remaining == 100
    assert strategy.select(context_no_tools) == "cheap"

    print("  0/1000 used   → 'expensive' (default)")
    print("  600/1000 used → 'medium'    (crossed 50%)")
    print("  900/1000 used → 'cheap'     (crossed 80%)")
    print("✓ CostBudgetRouting transitions through tiers as tokens are consumed")

    # --- Section 5: Cost Budget with Automatic Tracking via RoutingLLMClient ---
    print("\n--- Section 5: Cost Budget with Automatic Tracking ---")

    strategy = CostBudgetRouting(
        budget=100,
        thresholds=[(0.6, "economy")],
        default="premium",
    )

    premium_client = MockLLMClient(
        responses=[
            make_response("Premium r1.", model="premium-model", usage=make_usage(30, 20)),
            make_response("Premium r2.", model="premium-model", usage=make_usage(20, 15)),
        ]
    )
    economy_client = MockLLMClient(
        responses=[
            make_response("Economy r1.", model="economy-model", usage=make_usage(5, 5)),
        ]
    )

    router = RoutingLLMClient(
        clients={"premium": premium_client, "economy": economy_client},
        strategy=strategy,
    )

    messages = [Message(role="user", content="question")]

    # First call: 0% used → premium. Response has 50 tokens → 50% used.
    r1 = await router.generate(system_prompt="test", messages=messages)
    assert r1.model == "premium-model"
    assert strategy.used == 50

    # Second call: 50% used → still premium. Response has 35 tokens → 85% used.
    r2 = await router.generate(system_prompt="test", messages=messages)
    assert r2.model == "premium-model"
    assert strategy.used == 85

    # Third call: 85% used → crosses 60% threshold → economy
    r3 = await router.generate(system_prompt="test", messages=messages)
    assert r3.model == "economy-model"
    assert strategy.used == 95

    print(f"  After call 1: {strategy.used - 45} → 50 tokens used → premium")
    print("  After call 2: 50 → 85 tokens used → premium (still under threshold at call time)")
    print("  After call 3: 85 → 95 tokens used → economy (crossed 60% at call time)")
    print("✓ RoutingLLMClient calls on_response() automatically — no manual tracking needed")

    # --- Section 6: Custom RoutingStrategy ---
    print("\n--- Section 6: Custom RoutingStrategy ---")

    # Implement RoutingStrategy as a class — any object with select(RoutingContext) -> str
    class MessageCountRouting:
        """Routes based on conversation length: short → fast, long → smart."""

        def __init__(self, *, threshold: int) -> None:
            self.threshold = threshold

        def select(self, context: RoutingContext) -> str:
            return "smart" if len(context.messages) >= self.threshold else "fast"

    strategy = MessageCountRouting(threshold=3)

    fast_client = MockLLMClient(
        responses=[
            make_response("Fast.", model="fast-model"),
        ]
    )
    smart_client = MockLLMClient(
        responses=[
            make_response("Smart.", model="smart-model"),
        ]
    )

    router = RoutingLLMClient(
        clients={"fast": fast_client, "smart": smart_client},
        strategy=strategy,
    )

    # Short conversation → fast
    response = await router.generate(
        system_prompt="test",
        messages=[Message(role="user", content="hi")],
    )
    assert response.model == "fast-model"

    # Longer conversation → smart
    response = await router.generate(
        system_prompt="test",
        messages=[
            Message(role="user", content="first"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="follow-up"),
        ],
    )
    assert response.model == "smart-model"

    # Verify it satisfies the RoutingStrategy protocol
    assert isinstance(strategy, RoutingStrategy)

    print(f"  1 message  → fast-model  (below threshold={strategy.threshold})")
    print("  3 messages → smart-model (at or above threshold)")
    print("✓ Any class with select(RoutingContext) -> str satisfies RoutingStrategy")

    # --- Section 7: Routing Events via InMemoryEmitter ---
    print("\n--- Section 7: Routing Events via InMemoryEmitter ---")

    emitter = make_emitter("routing-events")

    fast_client = MockLLMClient(
        responses=[
            make_response("Event test.", model="fast-model"),
        ]
    )

    strategy = RuleBasedRouting(rule=lambda ctx: "fast")

    router = RoutingLLMClient(
        clients={"fast": fast_client},
        strategy=strategy,
        emitter=emitter,
    )

    await router.generate(
        system_prompt="test",
        messages=[Message(role="user", content="trigger event")],
    )

    # Find the routing event
    routing_events = [e for e in emitter.events if isinstance(e, ModelRoutingEvent)]
    assert len(routing_events) == 1

    event = routing_events[0]
    assert event.strategy_name == "RuleBasedRouting"
    assert event.selected_key == "fast"
    assert event.available_keys == ["fast"]
    assert event.trace_id == "routing-events"

    print(f"  strategy_name:  {event.strategy_name}")
    print(f"  selected_key:   {event.selected_key}")
    print(f"  available_keys: {event.available_keys}")
    print(f"  trace_id:       {event.trace_id}")
    print("✓ ModelRoutingEvent emitted for every routing decision — inspectable via emitter")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
