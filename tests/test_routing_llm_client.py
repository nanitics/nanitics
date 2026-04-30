from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanitics.infrastructure.llm.anthropic import AnthropicLLMClient
from nanitics.infrastructure.llm.litellm import LiteLLMClient
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.openai import OpenAILLMClient
from nanitics.infrastructure.llm.protocol import LLMClient, LLMResponse, Message
from nanitics.infrastructure.llm.routing import (
    CostBudgetRouting,
    RoutingContext,
    RoutingLLMClient,
    RuleBasedRouting,
)
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import ModelRoutingEvent, Usage


def _response(total_tokens: int = 100, model: str = "mock") -> LLMResponse:
    return LLMResponse(
        content="ok",
        tool_calls=[],
        usage=Usage(input_tokens=total_tokens // 2, output_tokens=total_tokens // 2),
        model=model,
        stop_reason="end_turn",
    )


def _message() -> list[Message]:
    return [Message(role="user", content="hello")]


# --- RoutingLLMClient dispatches to correct client ---


class TestRoutingLLMClientDispatch:
    @pytest.mark.anyio
    async def test_dispatches_to_selected_client(self) -> None:
        client_a = MockLLMClient(responses=[_response(model="a")])
        client_b = MockLLMClient(responses=[_response(model="b")])
        strategy = RuleBasedRouting(rule=lambda ctx: "b")

        router = RoutingLLMClient(
            clients={"a": client_a, "b": client_b},
            strategy=strategy,
        )

        result = await router.generate(system_prompt="sys", messages=_message())
        assert result.model == "b"
        assert len(client_a.calls) == 0
        assert len(client_b.calls) == 1

    @pytest.mark.anyio
    async def test_reasoning_text_survives_pass_through(self) -> None:
        """``LLMResponse.reasoning_text`` must survive the routing wrapper."""
        inner = MockLLMClient(
            responses=[_response(model="a")],
            reasoning_texts=["reasoning-from-model"],
        )
        router = RoutingLLMClient(
            clients={"a": inner},
            strategy=RuleBasedRouting(rule=lambda ctx: "a"),
        )
        result = await router.generate(system_prompt="sys", messages=_message())
        assert result.reasoning_text == "reasoning-from-model"

    @pytest.mark.anyio
    async def test_dispatches_to_different_clients_per_call(self) -> None:
        client_a = MockLLMClient(responses=[_response(model="a")])
        client_b = MockLLMClient(responses=[_response(model="b")])
        calls: list[str] = []

        def alternating(ctx: RoutingContext) -> str:
            calls.append("call")
            return "a" if len(calls) % 2 == 1 else "b"

        router = RoutingLLMClient(
            clients={"a": client_a, "b": client_b},
            strategy=RuleBasedRouting(rule=alternating),
        )

        r1 = await router.generate(system_prompt="sys", messages=_message())
        r2 = await router.generate(system_prompt="sys", messages=_message())
        assert r1.model == "a"
        assert r2.model == "b"


# --- RuleBasedRouting ---


class TestRuleBasedRouting:
    def test_returns_key_based_on_output_schema(self) -> None:
        strategy = RuleBasedRouting(rule=lambda ctx: "light" if ctx.output_schema else "frontier")
        ctx_with = RoutingContext(system_prompt="s", messages=[], tools=None, output_schema=Usage)
        ctx_without = RoutingContext(system_prompt="s", messages=[], tools=None, output_schema=None)
        assert strategy.select(ctx_with) == "light"
        assert strategy.select(ctx_without) == "frontier"

    def test_returns_key_based_on_tools(self) -> None:
        from nanitics.infrastructure.llm.protocol import ToolSchema

        strategy = RuleBasedRouting(rule=lambda ctx: "mid" if ctx.tools else "frontier")
        tool = ToolSchema(name="t", description="d", parameters={})
        ctx_with_tools = RoutingContext(system_prompt="s", messages=[], tools=[tool], output_schema=None)
        ctx_without_tools = RoutingContext(system_prompt="s", messages=[], tools=None, output_schema=None)
        assert strategy.select(ctx_with_tools) == "mid"
        assert strategy.select(ctx_without_tools) == "frontier"


# --- CostBudgetRouting ---


class TestCostBudgetRouting:
    def test_starts_with_default(self) -> None:
        strategy = CostBudgetRouting(
            budget=1000,
            thresholds=[(0.7, "mid"), (0.9, "light")],
            default="frontier",
        )
        ctx = RoutingContext(system_prompt="s", messages=[], tools=None, output_schema=None)
        assert strategy.select(ctx) == "frontier"

    def test_transitions_through_tiers(self) -> None:
        strategy = CostBudgetRouting(
            budget=1000,
            thresholds=[(0.7, "mid"), (0.9, "light")],
            default="frontier",
        )
        ctx = RoutingContext(system_prompt="s", messages=[], tools=None, output_schema=None)

        # Under 70% — still frontier
        strategy.on_response("frontier", _response(total_tokens=600))
        assert strategy.select(ctx) == "frontier"

        # Now at 70% — transitions to mid
        strategy.on_response("frontier", _response(total_tokens=100))
        assert strategy.select(ctx) == "mid"

        # Now at 90% — transitions to light
        strategy.on_response("mid", _response(total_tokens=200))
        assert strategy.select(ctx) == "light"

    def test_used_and_remaining_properties(self) -> None:
        strategy = CostBudgetRouting(
            budget=1000,
            thresholds=[(0.7, "mid")],
            default="frontier",
        )
        assert strategy.used == 0
        assert strategy.remaining == 1000

        strategy.on_response("frontier", _response(total_tokens=300))
        assert strategy.used == 300
        assert strategy.remaining == 700


# --- on_response called for CostBudgetRouting, not for RuleBasedRouting ---


class TestOnResponseFeedback:
    @pytest.mark.anyio
    async def test_on_response_called_for_cost_budget(self) -> None:
        client = MockLLMClient(responses=[_response(total_tokens=50)])
        strategy = CostBudgetRouting(budget=1000, thresholds=[], default="main")

        router = RoutingLLMClient(clients={"main": client}, strategy=strategy)
        await router.generate(system_prompt="sys", messages=_message())
        assert strategy.used == 50

    @pytest.mark.anyio
    async def test_on_response_not_called_for_rule_based(self) -> None:
        client = MockLLMClient(responses=[_response()])
        strategy = RuleBasedRouting(rule=lambda ctx: "main")

        router = RoutingLLMClient(clients={"main": client}, strategy=strategy)
        await router.generate(system_prompt="sys", messages=_message())
        # No error — RuleBasedRouting has no on_response and RoutingLLMClient handles it gracefully


# --- Unknown key handling ---


class TestUnknownKey:
    @pytest.mark.anyio
    async def test_unknown_key_raises_without_default(self) -> None:
        client = MockLLMClient(responses=[_response()])
        strategy = RuleBasedRouting(rule=lambda ctx: "nonexistent")

        router = RoutingLLMClient(clients={"main": client}, strategy=strategy)
        with pytest.raises(ValueError, match="nonexistent"):
            await router.generate(system_prompt="sys", messages=_message())

    @pytest.mark.anyio
    async def test_unknown_key_uses_default(self) -> None:
        client = MockLLMClient(responses=[_response(model="fallback")])
        strategy = RuleBasedRouting(rule=lambda ctx: "nonexistent")

        router = RoutingLLMClient(
            clients={"main": client},
            strategy=strategy,
            default="main",
        )
        result = await router.generate(system_prompt="sys", messages=_message())
        assert result.model == "fallback"


# --- Error propagation ---


class TestErrorPropagation:
    @pytest.mark.anyio
    async def test_error_from_client_propagates(self) -> None:
        client = MockLLMClient(responses=[])  # will raise on generate
        strategy = RuleBasedRouting(rule=lambda ctx: "main")

        router = RoutingLLMClient(clients={"main": client}, strategy=strategy)
        with pytest.raises(ValueError, match="no more scripted responses"):
            await router.generate(system_prompt="sys", messages=_message())


# --- Event emission ---


class TestEventEmission:
    @pytest.mark.anyio
    async def test_emits_model_routing_event(self) -> None:
        client = MockLLMClient(responses=[_response()])
        strategy = RuleBasedRouting(rule=lambda ctx: "main")
        emitter = InMemoryEmitter(trace_id="t1")

        router = RoutingLLMClient(
            clients={"main": client},
            strategy=strategy,
            emitter=emitter,
        )
        await router.generate(system_prompt="sys", messages=_message())

        routing_events = [e for e in emitter.events if isinstance(e, ModelRoutingEvent)]
        assert len(routing_events) == 1
        event = routing_events[0]
        assert event.strategy_name == "RuleBasedRouting"
        assert event.selected_key == "main"
        assert event.available_keys == ["main"]
        assert event.trace_id == "t1"

    @pytest.mark.anyio
    async def test_no_event_without_emitter(self) -> None:
        client = MockLLMClient(responses=[_response()])
        strategy = RuleBasedRouting(rule=lambda ctx: "main")

        router = RoutingLLMClient(
            clients={"main": client},
            strategy=strategy,
        )
        # Should not raise — no emitter means no event
        await router.generate(system_prompt="sys", messages=_message())


# --- Constructor validation ---


class TestConstructorValidation:
    def test_rejects_empty_clients(self) -> None:
        strategy = RuleBasedRouting(rule=lambda ctx: "main")
        with pytest.raises(ValueError, match="non-empty"):
            RoutingLLMClient(clients={}, strategy=strategy)

    def test_rejects_invalid_default_key(self) -> None:
        client = MockLLMClient(responses=[])
        strategy = RuleBasedRouting(rule=lambda ctx: "main")
        with pytest.raises(ValueError, match="not in clients"):
            RoutingLLMClient(
                clients={"main": client},
                strategy=strategy,
                default="nonexistent",
            )


@pytest.mark.parametrize("value", [0, -1])
def test_cost_budget_routing_rejects_non_positive_budget(value: int) -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        CostBudgetRouting(budget=value, thresholds=[(0.5, "small")], default="main")


def test_routing_client_model_is_none() -> None:
    client = MockLLMClient(responses=[])
    strategy = RuleBasedRouting(rule=lambda ctx: "main")
    router = RoutingLLMClient(clients={"main": client}, strategy=strategy)
    assert router.model is None


# --- Cross-provider protocol guarantee ---


def _make_anthropic_text_response(text: str = "anthropic ok") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    response.usage = MagicMock()
    response.usage.input_tokens = 5
    response.usage.output_tokens = 10
    return response


def _make_anthropic_stream_cm(response: MagicMock) -> AsyncMock:
    stream_obj = AsyncMock()
    stream_obj.get_final_message = AsyncMock(return_value=response)

    async def _empty_text_stream() -> Any:
        return
        yield  # pragma: no cover  # unreachable, but marks function as async generator

    stream_obj.text_stream = _empty_text_stream()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=stream_obj)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_openai_text_response(text: str = "openai ok") -> MagicMock:
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 10
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestCrossProviderDispatch:
    """Lock in the cross-provider protocol guarantee: RoutingLLMClient dispatches
    to real OpenAI and Anthropic client instances through the same code path."""

    def test_openai_and_anthropic_both_satisfy_protocol(self) -> None:
        openai_client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        anthropic_client = AnthropicLLMClient(model="claude-test", api_key="test-key")
        assert isinstance(openai_client, LLMClient)
        assert isinstance(anthropic_client, LLMClient)

    @pytest.mark.anyio
    async def test_dispatches_to_openai_and_anthropic(self) -> None:
        openai_client = OpenAILLMClient(model="gpt-4o-mini", api_key="test-key")
        anthropic_client = AnthropicLLMClient(model="claude-test", api_key="test-key")

        strategy = RuleBasedRouting(rule=lambda ctx: "openai" if ctx.tools is None else "anthropic")
        router = RoutingLLMClient(
            clients={"openai": openai_client, "anthropic": anthropic_client},
            strategy=strategy,
        )

        # Route to OpenAI (no tools)
        with patch.object(
            openai_client._client.chat.completions,
            "create",
            new=AsyncMock(return_value=_make_openai_text_response("from-openai")),
        ):
            r1 = await router.generate(system_prompt="sys", messages=_message())
        assert r1.content == "from-openai"
        assert r1.model == "gpt-4o-mini"

        # Route to Anthropic (with tools)
        from nanitics.infrastructure.llm.protocol import ToolSchema

        with patch.object(
            anthropic_client._client.messages,
            "stream",
            return_value=_make_anthropic_stream_cm(_make_anthropic_text_response("from-anthropic")),
        ):
            r2 = await router.generate(
                system_prompt="sys",
                messages=_message(),
                tools=[ToolSchema(name="t", description="d", parameters={})],
            )
        assert r2.content == "from-anthropic"
        assert r2.model == "claude-test"


class TestLiteLLMProtocolConformance:
    """Lock in cross-client protocol conformance: a LiteLLMClient and a native
    client (AnthropicLLMClient) both satisfy the LLMClient protocol and dispatch
    correctly through RoutingLLMClient. If LiteLLMClient ever regresses on
    protocol conformance, this test fails loudly."""

    def test_litellm_client_satisfies_protocol(self) -> None:
        litellm_client = LiteLLMClient(model="openai/gpt-4o-mini", api_key="test-key")
        assert isinstance(litellm_client, LLMClient)

    @pytest.mark.anyio
    async def test_dispatches_to_litellm_and_native_client(self) -> None:
        litellm_client = LiteLLMClient(model="openai/gpt-4o-mini", api_key="test-key")
        anthropic_client = AnthropicLLMClient(model="claude-test", api_key="test-key")

        # Route to LiteLLM when there are no tools, Anthropic when tools are present.
        strategy = RuleBasedRouting(rule=lambda ctx: "anthropic" if ctx.tools else "litellm")
        router = RoutingLLMClient(
            clients={"litellm": litellm_client, "anthropic": anthropic_client},
            strategy=strategy,
        )

        # Route to LiteLLM (no tools) — patch the module-level litellm.acompletion.
        with patch(
            "nanitics.infrastructure.llm.litellm.litellm.acompletion",
            new=AsyncMock(return_value=_make_openai_text_response("from-litellm")),
        ):
            r1 = await router.generate(system_prompt="sys", messages=_message())
        assert r1.content == "from-litellm"
        assert r1.model == "openai/gpt-4o-mini"

        # Route to Anthropic (with tools).
        from nanitics.infrastructure.llm.protocol import ToolSchema

        with patch.object(
            anthropic_client._client.messages,
            "stream",
            return_value=_make_anthropic_stream_cm(_make_anthropic_text_response("from-anthropic")),
        ):
            r2 = await router.generate(
                system_prompt="sys",
                messages=_message(),
                tools=[ToolSchema(name="t", description="d", parameters={})],
            )
        assert r2.content == "from-anthropic"
        assert r2.model == "claude-test"
