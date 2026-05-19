"""Real-provider validation for ``LiteLLMClient`` (Anthropic via LiteLLM).

Drives a real ``ReActAgent`` with a trivial echo tool through ``LiteLLMClient``
pointed at ``anthropic/claude-haiku-4-5-20251001`` — LiteLLM's provider-prefixed
naming convention. Gated on both the ``litellm`` extra being installed and
``ANTHROPIC_API_KEY`` (LiteLLM relays to Anthropic in this configuration).

Acceptance criteria:
  - ``AgentStartEvent`` observed — the agent loop started with ``echo``
    registered as an available tool.
  - Agent completes cleanly (``termination_reason == "complete"``) with
    ``total_steps >= 1`` and non-empty output.
  - Real round-trip: ``LLMResponseEvent`` emitted with positive input and
    output token counts (usage populated by the adapter from LiteLLM).
  - Tool round-trip: ``ToolInvokeEvent`` and successful ``ToolResultEvent``
    both emitted for the ``echo`` tool (confirms LiteLLM-relayed tool
    calling works end-to-end).
  - Structured output: a separate call with ``output_schema`` set on
    ``LiteLLMClient.generate()`` returns a ``parsed`` pydantic instance
    matching the schema (proves LiteLLM's forced-tool structured-output
    pathway).
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from nanitics.infrastructure import (
    AgentCompleteEvent,
    AgentStartEvent,
    LLMResponseEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import (
    InMemoryEmitter,
    Message,
)
from validation.helpers import (
    assert_trace_contains,
    run_with_retry,
)
from validation.helpers.skips import requires_litellm


@tool("echo", "Return the input message verbatim.")
async def echo(message: str) -> str:
    return message


def _make_litellm_client() -> object:
    from nanitics.infrastructure.llm.litellm import LiteLLMClient

    return LiteLLMClient(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )


@pytest.mark.quick
@requires_litellm
async def test_litellm_react_agent_real(traced_emitter: InMemoryEmitter) -> None:
    client = _make_litellm_client()
    agent = ReActAgent(
        name="litellm-agent",
        llm_client=client,  # type: ignore[arg-type]
        emitter=traced_emitter,
        system_prompt="You are a helpful assistant. Use the echo tool when asked to echo.",
        tools=[echo],
    )

    result = await run_with_retry(
        lambda: agent.run("Use the echo tool to repeat the word 'litellm'."),
        max_attempts=2,
    )

    assert_trace_contains(
        traced_emitter,
        AgentStartEvent,
        predicate=lambda e: "echo" in e.tools_available,
    )
    assert_trace_contains(
        traced_emitter,
        AgentCompleteEvent,
        predicate=lambda e: e.termination_reason == "complete",
    )
    assert_trace_contains(
        traced_emitter,
        LLMResponseEvent,
        predicate=lambda e: e.usage.input_tokens > 0 and e.usage.output_tokens > 0,
    )
    assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "echo",
    )
    assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == "echo" and e.success is True,
    )

    assert result.output, "Agent produced an empty output."
    assert result.total_steps >= 1, f"Expected total_steps >= 1, got {result.total_steps}"
    assert result.termination_reason == "complete"
    assert result.usage.total_tokens > 0


class _Verdict(BaseModel):
    subject: str
    sentiment: str


@pytest.mark.quick
@requires_litellm
async def test_litellm_structured_output_real(traced_emitter: InMemoryEmitter) -> None:
    """LiteLLM structured output: forced tool call returns a parsed pydantic instance."""
    del traced_emitter  # fixture triggers trace save on teardown
    client = _make_litellm_client()

    response = await run_with_retry(
        lambda: client.generate(  # type: ignore[attr-defined]
            system_prompt="Classify the sentiment of the given subject.",
            messages=[Message(role="user", content="Subject: a bright sunny day")],
            output_schema=_Verdict,
        ),
        max_attempts=2,
    )

    assert response.parsed is not None, "Expected parsed output from output_schema path."
    assert isinstance(response.parsed, _Verdict)
    assert response.parsed.subject, "Parsed subject must be non-empty."
    assert response.parsed.sentiment, "Parsed sentiment must be non-empty."
    assert response.usage.total_tokens > 0
