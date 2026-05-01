"""ReasoningAgent validation: single-call reasoning with structured output.

Acceptance criteria:
  - An ``LLMRequestEvent`` is emitted whose ``output_schema`` is non-null,
    whose ``title`` equals ``"Sentiment"``, and whose ``properties`` include
    both ``label`` and ``confidence`` (catches regressions that corrupt or
    drop fields from the forwarded JSON schema — stronger than a title-only
    check).
  - An ``AgentStepEvent`` with ``step_number == 1`` is emitted.
  - ``result.total_steps == 1`` (proves the reasoning agent did not
    inadvertently loop).
  - ``result.termination_reason == "complete"`` (distinguishes a clean exit
    from an evaluator-loop or iteration-limit exit).
  - ``result.parsed`` is a valid ``Sentiment`` instance (structured-output
    round-trip).
  - The parsed sentiment classifies the input as positive with confidence
    above 0.7.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from nanitics import InMemoryEmitter, ReasoningAgent
from nanitics.infrastructure import AgentStepEvent, LLMRequestEvent
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


class Sentiment(BaseModel):
    label: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence 0.0 to 1.0")


def _schema_is_sentiment(schema: dict | None) -> bool:
    """Distinguishing predicate: verifies title AND presence of both fields.

    A regression that stripped ``properties`` from the forwarded JSON schema
    would keep ``title`` intact (pydantic sets ``title`` from the class name
    before populating ``properties``), so the title-only check is insufficient.
    """
    if schema is None:
        return False
    if schema.get("title") != "Sentiment":
        return False
    properties = schema.get("properties", {})
    return "label" in properties and "confidence" in properties


@pytest.mark.quick
async def test_reasoning_structured_output(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    agent = ReasoningAgent(
        name="reasoning-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt="Classify the sentiment of the given text. Return JSON.",
        output_schema=Sentiment,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Classify the sentiment of: 'I absolutely loved this product — it exceeded my expectations.'"
        ),
        max_attempts=2,
    )

    # --- Trace-shape invariants ---
    # LLMRequestEvent.output_schema is the JSON schema dict (pydantic
    # ``model_json_schema()`` output). Verify title AND field presence to
    # catch partial-schema regressions.
    assert_trace_contains(
        traced_emitter,
        LLMRequestEvent,
        predicate=lambda e: _schema_is_sentiment(e.output_schema),
    )
    assert_trace_contains(traced_emitter, AgentStepEvent, predicate=lambda e: e.step_number == 1)

    # --- Exactly one step + clean termination ---
    assert result.total_steps == 1, f"Expected 1 step, got: {result.total_steps}"
    assert result.termination_reason == "complete", (
        f"Expected termination_reason == 'complete', got: {result.termination_reason!r}"
    )

    # --- Structured output present and valid ---
    assert result.parsed is not None, "Expected parsed output"
    assert isinstance(result.parsed, Sentiment), f"Expected Sentiment instance, got: {type(result.parsed)}"

    # --- Fuzzy check on classification ---
    await assert_result_satisfies(
        f"label={result.parsed.label}, confidence={result.parsed.confidence}",
        "The sentiment is classified as positive with confidence above 0.7.",
    )
