"""Real-provider validation for ``OpenAILLMClient`` (and ``MistralLLMClient`` when keyed).

Drives a real ``ReActAgent`` through ``OpenAILLMClient`` with a trivial echo
tool, and separately exercises the structured-output path via
``output_schema``. When ``MISTRAL_API_KEY`` is present a parallel ``ReActAgent``
run through ``MistralLLMClient`` asserts the same provider-neutral shape.

Acceptance criteria:
  - OpenAI agent run: ``AgentStartEvent`` with ``echo`` registered;
    ``AgentCompleteEvent`` with ``termination_reason == "complete"``;
    ``LLMResponseEvent`` with ``model_name`` starting with ``"gpt"``
    (proves provider-appropriate attribution) and positive input/output
    token counts; ``ToolInvokeEvent`` + successful ``ToolResultEvent``
    for ``echo`` (tool round-trip).
  - OpenAI structured output: a ``generate()`` call with ``output_schema``
    returns a ``parsed`` pydantic instance matching the schema.
  - Mistral parity (gated on ``MISTRAL_API_KEY``): analogous ``ReActAgent``
    run completes with ``termination_reason == "complete"`` and
    ``LLMResponseEvent.model_name`` starts with ``"mistral"``.
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
    make_llm_client,
    run_with_retry,
)
from validation.helpers.skips import requires_mistral, requires_openai


@tool("echo", "Return the input message verbatim.")
async def echo(message: str) -> str:
    return message


@pytest.mark.quick
@requires_openai
async def test_openai_react_agent_real(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("openai")
    agent = ReActAgent(
        name="openai-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt="You are a helpful assistant. Use the echo tool when asked to echo.",
        tools=[echo],
    )

    result = await run_with_retry(
        lambda: agent.run("Use the echo tool to repeat the word 'openai'."),
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

    # Model-name attribution — proves the trace identifies the provider.
    # We use the event's ``model_name`` to confirm provider attribution since
    # ``LLMRequestEvent``/``LLMResponseEvent`` don't carry a separate ``provider``
    # field at emission time — the model string is the attribution.
    llm_responses = [e for e in traced_emitter.events if isinstance(e, LLMResponseEvent)]
    assert llm_responses, "Expected at least one LLMResponseEvent."
    assert any(e.model_name.startswith("gpt") for e in llm_responses), (
        f"Expected an OpenAI model_name (starts with 'gpt'); got: {[e.model_name for e in llm_responses]}"
    )
    assert all(e.usage.input_tokens > 0 and e.usage.output_tokens > 0 for e in llm_responses), (
        "Every LLMResponseEvent must carry positive input and output token counts."
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
    assert result.total_steps >= 1
    assert result.termination_reason == "complete"
    assert result.usage.total_tokens > 0


class _Classification(BaseModel):
    topic: str
    confidence: str


@pytest.mark.quick
@requires_openai
async def test_openai_structured_output_real(traced_emitter: InMemoryEmitter) -> None:
    """OpenAI structured output: forced tool call returns a parsed pydantic instance."""
    del traced_emitter  # fixture saves trace on teardown
    client = make_llm_client("openai")

    response = await run_with_retry(
        lambda: client.generate(
            system_prompt="Classify the user's message into a topic.",
            messages=[Message(role="user", content="The Mars rover sent back new photos today.")],
            output_schema=_Classification,
        ),
        max_attempts=2,
    )

    assert response.parsed is not None, "Expected parsed output from output_schema path."
    assert isinstance(response.parsed, _Classification)
    assert response.parsed.topic, "Parsed topic must be non-empty."
    assert response.parsed.confidence, "Parsed confidence must be non-empty."
    assert response.usage.total_tokens > 0
    assert response.model.startswith("gpt"), f"Expected model to start with 'gpt'; got {response.model!r}"


@pytest.mark.quick
@requires_mistral
async def test_mistral_react_agent_real(traced_emitter: InMemoryEmitter) -> None:
    """Mistral parity: same ReActAgent shape, different provider."""
    client = make_llm_client("mistral")
    agent = ReActAgent(
        name="mistral-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt="You are a helpful assistant. Use the echo tool when asked to echo.",
        tools=[echo],
    )

    result = await run_with_retry(
        lambda: agent.run("Use the echo tool to repeat the word 'mistral'."),
        max_attempts=2,
    )

    assert_trace_contains(
        traced_emitter,
        AgentCompleteEvent,
        predicate=lambda e: e.termination_reason == "complete",
    )

    llm_responses = [e for e in traced_emitter.events if isinstance(e, LLMResponseEvent)]
    assert llm_responses, "Expected at least one LLMResponseEvent."
    assert any(e.model_name.startswith("mistral") for e in llm_responses), (
        f"Expected a Mistral model_name (starts with 'mistral'); got: {[e.model_name for e in llm_responses]}"
    )

    assert_trace_contains(
        traced_emitter,
        ToolResultEvent,
        predicate=lambda e: e.tool_name == "echo" and e.success is True,
    )

    assert result.termination_reason == "complete"
    assert result.usage.total_tokens > 0
    # Reference MISTRAL_API_KEY to make the gate coupling obvious to readers.
    assert "MISTRAL_API_KEY" in os.environ
