"""Real-provider validation for ``RoutingLLMClient``, routing strategies, and ``InstrumentedLLMClient``.

Each test wires up real Anthropic clients at two different model tiers
(Haiku for the "fast/cheap" tier, a distinct Haiku instance as the
"smart/expensive" tier — same family, separate instances — so we can assert
per-route dispatch by identity via ``call_count`` without mocking the
protocol). Routing decisions are asserted through ``ModelRoutingEvent`` and
by observing which underlying client actually produced the response.

Acceptance criteria:
  - Rule-based routing: a rule keyed on ``RoutingContext.tools`` routes
    tool-bearing requests to the ``smart`` client and tool-free requests
    to the ``fast`` client. ``ModelRoutingEvent`` is emitted with the
    correct ``selected_key`` for each call.
  - Fallback: when the strategy returns an unknown key, the configured
    ``default`` is selected; ``ModelRoutingEvent.selected_key`` reports
    the default.
  - Cost-budget routing: after a real generation pushes ``used`` above
    the threshold, the next call's ``ModelRoutingEvent.selected_key``
    switches to the cheaper tier (budget-exceeded routing).
  - Instrumented client: wrapping a real Anthropic client with
    ``InstrumentedLLMClient`` emits exactly one ``LLMRequestEvent`` and
    one ``LLMResponseEvent`` per ``generate()`` call, each carrying the
    supplied ``label``, a non-empty ``model_name``, and positive token
    counts in ``usage``.
"""

from __future__ import annotations

import os

import pytest

from nanitics import (
    CostBudgetRouting,
    InMemoryEmitter,
    InstrumentedLLMClient,
    Message,
    RoutingLLMClient,
    RuleBasedRouting,
    ToolSchema,
)
from nanitics.infrastructure import (
    LLMRequestEvent,
    LLMResponseEvent,
    ModelRoutingEvent,
)
from nanitics.infrastructure.llm.anthropic import AnthropicLLMClient
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


def _two_real_clients() -> tuple[AnthropicLLMClient, AnthropicLLMClient]:
    """Two real Anthropic clients at the same (cheap) tier.

    Distinct instances — ``call_count`` on each proves per-route dispatch.
    Using the same model family keeps the test fast and deterministic while
    still exercising the real network path on both clients.
    """
    api_key = os.environ["ANTHROPIC_API_KEY"]
    fast = AnthropicLLMClient(model="claude-haiku-4-5-20251001", api_key=api_key)
    smart = AnthropicLLMClient(model="claude-haiku-4-5-20251001", api_key=api_key)
    return fast, smart


class _CountingClient:
    """Decorator over a real ``LLMClient`` that counts ``generate()`` calls.

    The routing assertions need to prove which underlying client handled
    each call. Wrapping the real client is safer than mocking the protocol —
    the real network path still runs and usage numbers are real.
    """

    def __init__(self, inner: AnthropicLLMClient) -> None:
        self._inner = inner
        self.call_count = 0

    @property
    def model(self) -> str | None:
        return self._inner.model

    async def generate(self, **kwargs: object) -> object:
        self.call_count += 1
        return await self._inner.generate(**kwargs)  # type: ignore[arg-type]


@pytest.mark.quick
async def test_rule_based_routing_real(traced_emitter: InMemoryEmitter) -> None:
    fast_raw, smart_raw = _two_real_clients()
    fast = _CountingClient(fast_raw)
    smart = _CountingClient(smart_raw)

    strategy = RuleBasedRouting(rule=lambda ctx: "smart" if ctx.tools else "fast")
    router = RoutingLLMClient(
        clients={"fast": fast, "smart": smart},  # type: ignore[dict-item]
        strategy=strategy,
        emitter=traced_emitter,
    )

    # Tool-free → fast
    await run_with_retry(
        lambda: router.generate(
            system_prompt="Answer in one short sentence.",
            messages=[Message(role="user", content="Say hi.")],
        ),
        max_attempts=2,
    )
    # Tool-bearing → smart
    await run_with_retry(
        lambda: router.generate(
            system_prompt="Answer in one short sentence. Do not call any tool.",
            messages=[Message(role="user", content="Acknowledge that you see a tool.")],
            tools=[ToolSchema(name="noop", description="A no-op tool.", parameters={"type": "object"})],
        ),
        max_attempts=2,
    )

    assert fast.call_count == 1, f"Expected fast.call_count==1, got {fast.call_count}"
    assert smart.call_count == 1, f"Expected smart.call_count==1, got {smart.call_count}"

    routing_events = [e for e in traced_emitter.events if isinstance(e, ModelRoutingEvent)]
    assert len(routing_events) == 2, f"Expected 2 ModelRoutingEvents, got {len(routing_events)}"
    assert routing_events[0].selected_key == "fast"
    assert routing_events[1].selected_key == "smart"
    for event in routing_events:
        assert event.strategy_name == "RuleBasedRouting"
        assert set(event.available_keys) == {"fast", "smart"}


@pytest.mark.quick
async def test_routing_fallback_to_default(traced_emitter: InMemoryEmitter) -> None:
    fast_raw, _ = _two_real_clients()
    primary = _CountingClient(fast_raw)

    # Strategy returns a key that isn't in the clients dict — default must
    # catch it and route to "primary".
    strategy = RuleBasedRouting(rule=lambda ctx: "does-not-exist")
    router = RoutingLLMClient(
        clients={"primary": primary},  # type: ignore[dict-item]
        strategy=strategy,
        default="primary",
        emitter=traced_emitter,
    )

    await run_with_retry(
        lambda: router.generate(
            system_prompt="Answer in one short sentence.",
            messages=[Message(role="user", content="Reply with OK.")],
        ),
        max_attempts=2,
    )

    assert primary.call_count == 1, "Default fallback should have invoked the primary client exactly once."
    assert_trace_contains(
        traced_emitter,
        ModelRoutingEvent,
        predicate=lambda e: e.selected_key == "primary",
    )


@pytest.mark.quick
async def test_cost_budget_routing_real(traced_emitter: InMemoryEmitter) -> None:
    premium_raw, economy_raw = _two_real_clients()
    premium = _CountingClient(premium_raw)
    economy = _CountingClient(economy_raw)

    # Budget small enough that a single real Anthropic call's usage pushes
    # the ratio above 50%, flipping subsequent requests to "economy". A
    # minimal prompt/response lands in the low-20s of total tokens, so
    # budget=20 gives comfortable margin above the 0.5 threshold.
    strategy = CostBudgetRouting(
        budget=20,
        thresholds=[(0.5, "economy")],
        default="premium",
    )
    router = RoutingLLMClient(
        clients={"premium": premium, "economy": economy},  # type: ignore[dict-item]
        strategy=strategy,
        emitter=traced_emitter,
    )

    # First call: 0 used → premium. The real response will populate usage.
    await run_with_retry(
        lambda: router.generate(
            system_prompt="Answer in one short sentence.",
            messages=[Message(role="user", content="Reply with OK.")],
        ),
        max_attempts=2,
    )
    assert premium.call_count == 1
    assert economy.call_count == 0
    # Verify usage was recorded from the real response. The downstream
    # economy.call_count assertion checks that used/budget crossed 0.5.
    assert strategy.used > 0, f"Expected real call to populate strategy.used from response usage; got {strategy.used}"

    # Second call: budget exceeded → economy.
    await run_with_retry(
        lambda: router.generate(
            system_prompt="Answer in one short sentence.",
            messages=[Message(role="user", content="Reply with OK again.")],
        ),
        max_attempts=2,
    )
    assert economy.call_count == 1, (
        f"Expected budget-exceeded routing to switch to economy; got economy.call_count={economy.call_count}"
    )

    routing_events = [e for e in traced_emitter.events if isinstance(e, ModelRoutingEvent)]
    assert len(routing_events) == 2, f"Expected 2 ModelRoutingEvents, got {len(routing_events)}"
    assert routing_events[0].selected_key == "premium"
    assert routing_events[1].selected_key == "economy"
    for event in routing_events:
        assert event.strategy_name == "CostBudgetRouting"


@pytest.mark.quick
async def test_instrumented_client_emits_events(traced_emitter: InMemoryEmitter) -> None:
    raw_client = make_llm_client("anthropic")
    instrumented = InstrumentedLLMClient(
        client=raw_client,
        emitter=traced_emitter,
        label="validation-95",
    )

    response = await run_with_retry(
        lambda: instrumented.generate(
            system_prompt="Reply with a single short word.",
            messages=[Message(role="user", content="Say OK.")],
        ),
        max_attempts=2,
    )

    requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent)]
    responses = [e for e in traced_emitter.events if isinstance(e, LLMResponseEvent)]
    assert len(requests) == 1, f"Expected 1 LLMRequestEvent, got {len(requests)}"
    assert len(responses) == 1, f"Expected 1 LLMResponseEvent, got {len(responses)}"

    req = requests[0]
    resp = responses[0]
    assert req.label == "validation-95"
    assert resp.label == "validation-95"
    assert req.model_name, "Request event must carry a non-empty model_name."
    assert resp.model_name, "Response event must carry a non-empty model_name."
    assert resp.usage.input_tokens > 0, f"Expected positive input_tokens, got {resp.usage.input_tokens}"
    assert resp.usage.output_tokens > 0, f"Expected positive output_tokens, got {resp.usage.output_tokens}"
    assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens
    assert resp.duration_ms > 0, "duration_ms must be positive."
    assert response.usage.total_tokens == resp.usage.total_tokens
