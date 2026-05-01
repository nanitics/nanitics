"""ReasoningAgent: single-call reasoning, structured output, and evaluation-driven revision.

Demonstrates ReasoningAgent — the simplest agent type. One LLM call produces the
output. With output_schema, the LLM produces structured JSON. With an evaluator,
the agent self-revises until the output passes quality checks.

Related guide: docs/guides/agent-types.md
"""

import asyncio

from pydantic import BaseModel, Field

from examples.helpers import make_emitter, make_response
from nanitics import (
    EvaluationCheck,
    MockLLMClient,
    ProgrammaticEvaluator,
    ReasoningAgent,
)
from nanitics.infrastructure import (
    AgentCompleteEvent,
    AgentStartEvent,
    AgentStepEvent,
    EvaluationEvent,
    EvaluationRevisionEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    SpanEndEvent,
    SpanStartEvent,
)


async def main() -> None:
    # --- Section 1: Basic Single-Call Reasoning ---
    print("--- Section 1: Basic Single-Call Reasoning ---")

    client = MockLLMClient(
        responses=[
            make_response("This is a positive product review expressing satisfaction."),
        ]
    )
    emitter = make_emitter("reasoning-s1")

    agent = ReasoningAgent(
        name="classifier",
        llm_client=client,
        emitter=emitter,
        system_prompt="Classify the sentiment of the given text.",
    )

    result = await agent.run("I love this product!")

    assert result.output == "This is a positive product review expressing satisfaction."
    assert result.total_steps == 1, f"Expected 1 step, got: {result.total_steps}"
    assert result.termination_reason == "complete"
    assert len(result.messages) == 2, f"Expected 2 messages, got: {len(result.messages)}"
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "I love this product!"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == result.output

    # Usage reflects a single LLM call
    assert result.usage.input_tokens == 10  # make_usage defaults
    assert result.usage.output_tokens == 5

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Messages: {len(result.messages)} (user → assistant)")
    print(f"  Usage: {result.usage.input_tokens} in / {result.usage.output_tokens} out")
    print("✓ Single LLM call — no tools, no loop")

    # --- Section 2: Structured Output with Pydantic Schema ---
    print("\n--- Section 2: Structured Output with Pydantic Schema ---")

    class Sentiment(BaseModel):
        label: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(description="Confidence score 0.0 to 1.0")

    client = MockLLMClient(
        responses=[
            make_response('{"label": "positive", "confidence": 0.95}'),
        ]
    )
    emitter = make_emitter("reasoning-s2")

    agent = ReasoningAgent(
        name="sentiment-classifier",
        llm_client=client,
        emitter=emitter,
        system_prompt="Classify the sentiment of the given text. Return JSON.",
        output_schema=Sentiment,
    )

    result = await agent.run("I love this product!")

    assert result.output == '{"label": "positive", "confidence": 0.95}'

    # Parsed structured output is available directly
    assert result.parsed is not None
    assert result.parsed.label == "positive"
    assert result.parsed.confidence == 0.95

    # Can also parse manually from the raw JSON string
    parsed = Sentiment.model_validate_json(result.output)
    assert parsed.label == "positive"
    assert parsed.confidence == 0.95

    # Verify output_schema was passed to the LLM
    assert len(client.calls) == 1
    assert client.calls[0]["output_schema"] is Sentiment

    print(f"  Raw output: {result.output}")
    print(f"  Parsed: label={parsed.label}, confidence={parsed.confidence}")
    print(f"  output_schema passed to LLM: {client.calls[0]['output_schema'].__name__}")
    print("✓ LLM constrained to produce JSON matching Pydantic model")

    # --- Section 3: Evaluation and Revision ---
    print("\n--- Section 3: Evaluation and Revision ---")

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="detailed_analysis",
                check=lambda output: "detailed" in output.lower(),
                feedback="Provide a more detailed analysis with specific examples.",
            ),
        ],
        max_revisions=2,
    )

    client = MockLLMClient(
        responses=[
            make_response("Positive sentiment."),  # First attempt — fails check
            make_response("Detailed analysis: strongly positive sentiment with enthusiasm."),  # Revised — passes
        ]
    )
    emitter = make_emitter("reasoning-s3")

    agent = ReasoningAgent(
        name="evaluating-classifier",
        llm_client=client,
        emitter=emitter,
        system_prompt="Classify the sentiment of the given text.",
        output_evaluator=evaluator,
    )

    result = await agent.run("I love this product!")

    assert result.output == "Detailed analysis: strongly positive sentiment with enthusiasm."
    assert result.total_steps == 2, f"Expected 2 steps, got: {result.total_steps}"
    assert result.termination_reason == "complete"

    # Inspect conversation: user → assistant (bad) → user (feedback) → assistant (good)
    assert len(result.messages) == 4, f"Expected 4 messages, got: {len(result.messages)}"
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == "Positive sentiment."
    assert result.messages[2].role == "user"
    assert "detailed" in result.messages[2].content.lower()  # Feedback from evaluator
    assert result.messages[3].role == "assistant"
    assert result.messages[3].content == result.output

    # Usage aggregated across 2 LLM calls
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10

    print(f"  First attempt: {result.messages[1].content}")
    print(f"  Feedback: {result.messages[2].content}")
    print(f"  Revised output: {result.output}")
    print(f"  Steps: {result.total_steps}, Termination: {result.termination_reason}")
    print("✓ Evaluator rejected first attempt, agent revised successfully")

    # --- Section 4: Evaluation Budget Exhaustion ---
    print("\n--- Section 4: Evaluation Budget Exhaustion ---")

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="impossible_check",
                check=lambda _: False,  # Always fails
                feedback="This check can never pass.",
            ),
        ],
        max_revisions=1,
    )

    client = MockLLMClient(
        responses=[
            make_response("First attempt."),
            make_response("Second attempt after revision."),
        ]
    )
    emitter = make_emitter("reasoning-s4")

    agent = ReasoningAgent(
        name="exhausted-classifier",
        llm_client=client,
        emitter=emitter,
        system_prompt="Classify the sentiment.",
        output_evaluator=evaluator,
    )

    result = await agent.run("Some text")

    assert result.termination_reason == "evaluation_failed", (
        f"Expected evaluation_failed, got: {result.termination_reason}"
    )
    assert result.total_steps == 2, f"Expected 2 steps, got: {result.total_steps}"
    assert result.output == "Second attempt after revision."

    print(f"  Termination: {result.termination_reason}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Output (last attempt): {result.output}")
    print("✓ Evaluation budget exhausted — returns last attempt with evaluation_failed")

    # --- Section 5: Event Trace ---
    print("\n--- Section 5: Event Trace ---")

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="quality_check",
                check=lambda output: "thorough" in output.lower(),
                feedback="Please provide a more thorough analysis.",
            ),
        ],
        max_revisions=2,
    )

    client = MockLLMClient(
        responses=[
            make_response("Brief answer."),  # Fails check
            make_response("Thorough and complete analysis."),  # Passes check
        ]
    )
    emitter = make_emitter("reasoning-s5")

    agent = ReasoningAgent(
        name="traced-classifier",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the text.",
        output_evaluator=evaluator,
    )

    await agent.run("Analyze this text.")

    # Verify event sequence
    event_types = [e.event_type for e in emitter.events]

    # Expected sequence:
    # span.start → agent.start → llm.request → llm.response → agent.step (first)
    # → evaluation.result (revise) → evaluation.revision
    # → llm.request → llm.response → agent.step (revision) → evaluation.result (accept)
    # → agent.complete → span.end
    #
    # ReasoningAgent emits one ``agent.step`` per LLM call — the initial draft
    # (step 1) and each revision (step 2, step 3, ...). The step event happens
    # right after each ``llm.response``, before the evaluator runs on that draft.
    assert event_types[0] == "span.start"
    assert event_types[1] == "agent.start"
    assert event_types[2] == "llm.request"
    assert event_types[3] == "llm.response"
    assert event_types[4] == "agent.step"  # Step 1 — initial draft
    assert event_types[5] == "evaluation.result"  # First evaluation — revise
    assert event_types[6] == "evaluation.revision"
    assert event_types[7] == "llm.request"  # Retry
    assert event_types[8] == "llm.response"
    assert event_types[9] == "agent.step"  # Step 2 — revised draft
    assert event_types[10] == "evaluation.result"  # Second evaluation — accept
    assert event_types[11] == "agent.complete"
    assert event_types[12] == "span.end"

    # Inspect specific event details
    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 2
    assert eval_events[0].verdict == "revise"
    assert eval_events[0].revision_attempt == 0
    assert eval_events[1].verdict == "accept"
    assert eval_events[1].revision_attempt == 1

    revision_events = [e for e in emitter.events if isinstance(e, EvaluationRevisionEvent)]
    assert len(revision_events) == 1
    assert "thorough" in revision_events[0].feedback.lower()
    assert revision_events[0].max_revisions == 2

    # Verify span and lifecycle events
    span_starts = [e for e in emitter.events if isinstance(e, SpanStartEvent)]
    span_ends = [e for e in emitter.events if isinstance(e, SpanEndEvent)]
    assert len(span_starts) == 1
    assert len(span_ends) == 1

    starts = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
    assert len(starts) == 1
    assert starts[0].agent_name == "traced-classifier"
    assert starts[0].tools_available == []  # ReasoningAgent has no tools

    completes = [e for e in emitter.events if isinstance(e, AgentCompleteEvent)]
    assert len(completes) == 1
    assert completes[0].termination_reason == "complete"
    assert completes[0].total_steps == 2

    print("  Event trace:")
    for event in emitter.events:
        et = event.event_type
        if isinstance(event, SpanStartEvent):
            print(f"    {et}: span={event.span_id}")
        elif isinstance(event, AgentStartEvent):
            print(f"    {et}: agent={event.agent_name}, tools={event.tools_available}")
        elif isinstance(event, LLMRequestEvent):
            print(f"    {et}: {len(event.messages)} messages")
        elif isinstance(event, LLMResponseEvent):
            content_preview = (event.content or "")[:40]
            print(f"    {et}: {content_preview!r}")
        elif isinstance(event, EvaluationEvent):
            print(f"    {et}: verdict={event.verdict}, attempt={event.revision_attempt}")
        elif isinstance(event, EvaluationRevisionEvent):
            print(f"    {et}: attempt={event.revision_attempt}/{event.max_revisions}")
        elif isinstance(event, AgentStepEvent):
            print(f"    {et}: step {event.step_number}")
        elif isinstance(event, AgentCompleteEvent):
            print(f"    {et}: reason={event.termination_reason}, steps={event.total_steps}")
        elif isinstance(event, SpanEndEvent):
            print(f"    {et}: span={event.span_id}")
        else:
            print(f"    {et}")
    print("✓ Complete event trace with evaluation and revision events")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
