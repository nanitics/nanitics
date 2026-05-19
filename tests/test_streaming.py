"""Tests for LLM token streaming (on_token callback and LLMTokenEvent)."""

from nanitics.infrastructure import (
    LLMTokenEvent,
    MockLLMClient,
    RoutingLLMClient,
    RuleBasedRouting,
)
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from tests.testing_helpers import make_emitter, make_response


@tool(name="noop", description="Does nothing")
async def noop_tool() -> str:
    return "ok"


# --- MockLLMClient on_token tests ---


async def test_mock_client_calls_on_token_with_word_chunks():
    """MockLLMClient should call on_token with word-level chunks when callback provided."""
    client = MockLLMClient(responses=[make_response(content="hello world foo")])
    tokens: list[str] = []

    await client.generate(
        system_prompt="test",
        messages=[],
        on_token=lambda t: tokens.append(t),
    )

    assert tokens == ["hello ", "world ", "foo "]


async def test_mock_client_no_on_token_backward_compatible():
    """MockLLMClient should work without on_token (backward compatibility)."""
    client = MockLLMClient(responses=[make_response(content="hello world")])

    response = await client.generate(
        system_prompt="test",
        messages=[],
    )

    assert response.content == "hello world"


async def test_mock_client_on_token_none_content():
    """MockLLMClient should not call on_token when content is None."""
    client = MockLLMClient(responses=[make_response(content=None)])
    tokens: list[str] = []

    await client.generate(
        system_prompt="test",
        messages=[],
        on_token=lambda t: tokens.append(t),
    )

    assert tokens == []


# --- RoutingLLMClient on_token passthrough test ---


async def test_routing_client_passes_on_token_through():
    """RoutingLLMClient should pass on_token to the selected underlying client."""
    inner = MockLLMClient(responses=[make_response(content="routed text")])
    routing = RoutingLLMClient(
        clients={"default": inner},
        strategy=RuleBasedRouting(rule=lambda ctx: "default"),
    )
    tokens: list[str] = []

    await routing.generate(
        system_prompt="test",
        messages=[],
        on_token=lambda t: tokens.append(t),
    )

    assert tokens == ["routed ", "text "]


# --- ReActAgent streaming tests ---


async def test_react_agent_streaming_emits_token_events():
    """ReActAgent with streaming=True should emit LLMTokenEvents."""
    client = MockLLMClient(responses=[make_response(content="streamed output")])
    emitter = make_emitter()

    agent = ReActAgent(
        name="test-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a test agent.",
        tools=[noop_tool],
        streaming=True,
    )

    result = await agent.run("do something")

    assert result.output == "streamed output"
    token_events = [e for e in emitter.events if isinstance(e, LLMTokenEvent)]
    assert len(token_events) > 0
    assert all(e.agent_name == "test-agent" for e in token_events)
    # Reconstructed tokens should match content
    reconstructed = "".join(e.token for e in token_events)
    assert reconstructed.strip() == "streamed output"


async def test_react_agent_no_streaming_emits_zero_token_events():
    """ReActAgent with streaming=False (default) should emit zero LLMTokenEvents."""
    client = MockLLMClient(responses=[make_response(content="no streaming")])
    emitter = make_emitter()

    agent = ReActAgent(
        name="test-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a test agent.",
        tools=[noop_tool],
    )

    result = await agent.run("do something")

    assert result.output == "no streaming"
    token_events = [e for e in emitter.events if isinstance(e, LLMTokenEvent)]
    assert len(token_events) == 0
