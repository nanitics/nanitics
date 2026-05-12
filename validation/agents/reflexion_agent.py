"""ReflexionAgent validation: evaluate-reflect-retry against a real LLM.

Acceptance criteria:
  - Trace contains between 1 and ``max_attempts - 1`` ``EvaluationEvent``s
    with ``verdict == 'revise'`` (the first attempt is rejected; the
    model may legitimately need 1 or 2 reflection cycles to converge on
    both keyword constraints).
  - Trace contains at least one ``ReflectionGeneratedEvent`` whose
    ``reflection_text`` is non-empty and whose ``evaluation_feedback``
    surfaces at least one of the evaluator's feedback strings — proving
    the reflection LLM call actually ran against the failing evaluator
    output and produced content.
  - Trace contains at least one ``EpisodeRecallEvent`` with
    ``results_count >= 1`` — proving the reflection episode reached the
    inner agent's context in the retry attempt, which is the Reflexion-
    distinguishing mechanism (reflection → episode store → recall on
    the next attempt).
  - Trace contains an ``EvaluationEvent`` with ``verdict == 'accept'``
    whose ``revision_attempt`` is strictly greater than the rejecting
    event's — proving order. An accept is required (the loop cannot
    silently no-op into ``evaluation_failed``).
  - The verbatim evaluator-feedback strings (``jellyfish``,
    ``lighthouse``) appear under an ``Evaluator feedback:`` header in
    at least one ``LLMRequestEvent.messages`` payload. This is the
    SDK-fix observable: ``EpisodicMemoryProvider.provide()`` renders the
    verbatim feedback into the recalled ``[Past Experiences]`` block
    ahead of the LLM-narrativised reflection, so the inner agent treats
    it as a binding constraint rather than advisory text.
  - ``result.termination_reason == 'complete'`` (loop accepted within
    budget) and ``result.total_steps <= max_attempts``.

The user prompt deliberately omits the keyword constraints; the evaluator
surfaces them via feedback on the failing first attempt. Between
attempts, the ReflexionAgent's reflection LLM call converts that feedback
into a reflection stored in the shared episode store, and the inner
``ReActAgent``'s ``EpisodicMemoryProvider`` recalls both the verbatim
feedback (rendered as ``Evaluator feedback: ...``) and the LLM-narrative
reflection into attempt 2's context — that recall path is what this
script is meant to validate.

The evaluator itself is deterministic (literal substring checks), so the
final output's constraint-satisfaction is already guaranteed by
``termination_reason == 'complete'``; we therefore omit the LLM-judge
check that would only re-litigate the same deterministic fact.
"""

from __future__ import annotations

from nanitics import (
    EpisodicMemoryProvider,
    EvaluationCheck,
    InMemoryEmitter,
    InMemoryEpisodeStore,
    ProgrammaticEvaluator,
    ReActAgent,
)
from nanitics.experimental import ReflexionAgent
from nanitics.infrastructure import (
    EpisodeRecallEvent,
    EvaluationEvent,
    LLMRequestEvent,
    ReflectionGeneratedEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_embedding_client,
    make_llm_client,
    requires_voyage,
    run_with_retry,
)


@requires_voyage
async def test_reflexion_evaluate_reflect_retry(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    # Real embedding client: the ``EpisodicMemoryProvider`` path under
    # test exercises the store's ranking surface. Using
    # ``MockEmbeddingClient`` here would violate the validation rubric's
    # "no mocks" floor. ``@requires_voyage`` skips the test when the
    # VOYAGE_API_KEY env var is absent.
    embedding_client = make_embedding_client("voyage")
    episode_store = InMemoryEpisodeStore(embedding_client=embedding_client)
    inner_agent = ReActAgent(
        name="haiku-worker",
        llm_client=client,
        # Overwritten by ReflexionAgent.bind() at attempt start; trace_id here never reaches the trace file.
        emitter=InMemoryEmitter(trace_id="unused-inner-bound-by-reflexion"),
        # Bare prompt: the evaluator surfaces the keyword constraints via
        # reflection feedback. If the constraints were stated in the system
        # prompt, the first attempt would usually satisfy them, defeating the
        # reflect-retry test.
        system_prompt="You write three-line haikus. Be poetic.",
        tools=[],
        max_iterations=3,
        # Wire the episode store back to the inner agent so reflections from
        # rejected attempts surface in subsequent attempts as a context block.
        # ``emitter=traced_emitter`` ensures EpisodeRecallEvents land in the
        # persisted outer trace.
        context_providers=[EpisodicMemoryProvider(store=episode_store, emitter=traced_emitter, limit=10)],
    )
    jellyfish_feedback = "The haiku must include the literal word 'jellyfish'."
    lighthouse_feedback = "The haiku must include the literal word 'lighthouse'."
    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="mentions_jellyfish",
                check=lambda output: "jellyfish" in output.lower(),
                feedback=jellyfish_feedback,
            ),
            EvaluationCheck(
                name="mentions_lighthouse",
                check=lambda output: "lighthouse" in output.lower(),
                feedback=lighthouse_feedback,
            ),
        ],
    )
    reflexion_agent = ReflexionAgent(
        name="reflexion-haiku",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt="Write a haiku that satisfies the user's constraints.",
        inner_agent=inner_agent,
        evaluator=evaluator,
        episode_store=episode_store,
        max_attempts=3,
    )

    # The user prompt does NOT mention the keyword constraints — the evaluator
    # supplies them via reflection feedback after the first attempt.
    result = await run_with_retry(
        lambda: reflexion_agent.run("Write a short haiku about the sea."),
        max_attempts=2,
    )

    # --- Order: at least one revise then an accept within budget. ---
    # Models may legitimately need 1 or 2 reflection cycles to converge on
    # both keyword constraints; the loop must still produce an accept (so a
    # silent no-op cannot pass), but pinning to exactly one revise turns
    # legitimate model behaviour into a false negative.
    first_revise = assert_trace_contains(traced_emitter, EvaluationEvent, predicate=lambda e: e.verdict == "revise")
    revise_events = [e for e in traced_emitter.events if isinstance(e, EvaluationEvent) and e.verdict == "revise"]
    max_revises = reflexion_agent._max_attempts - 1
    assert 1 <= len(revise_events) <= max_revises, (
        f"Expected between 1 and {max_revises} 'revise' EvaluationEvents (the model may need 1 or 2 "
        f"reflection cycles, but must accept within budget); got {len(revise_events)}."
    )
    final_accept = assert_trace_contains(traced_emitter, EvaluationEvent, predicate=lambda e: e.verdict == "accept")
    assert final_accept.revision_attempt > first_revise.revision_attempt, (
        f"Accept (attempt {final_accept.revision_attempt}) must follow revise (attempt {first_revise.revision_attempt})"
    )

    # --- Reflection: non-empty text, evaluator feedback propagated through. ---
    reflection = assert_trace_contains(
        traced_emitter,
        ReflectionGeneratedEvent,
        predicate=lambda e: (
            bool(e.reflection_text.strip())
            and e.evaluation_feedback is not None
            and ("jellyfish" in e.evaluation_feedback.lower() or "lighthouse" in e.evaluation_feedback.lower())
        ),
    )
    assert reflection.episode_id, (
        f"ReflectionGeneratedEvent must carry a non-empty episode_id linking the reflection to a stored "
        f"episode; got {reflection.episode_id!r}."
    )

    # --- Distinguishing capability: reflection reached the retry via recall. ---
    # This is the Reflexion-specific claim. A no-op reflection, a
    # disconnected episode store, or a provider that didn't recall the
    # reflection would all miss this assertion — whereas the plain
    # "evaluator gave literal feedback that reached attempt 2" path (which
    # could succeed without any Reflexion machinery) would not emit any
    # EpisodeRecallEvent.
    assert_trace_contains(
        traced_emitter,
        EpisodeRecallEvent,
        predicate=lambda e: e.results_count >= 1,
    )

    # --- SDK-fix observable: verbatim evaluator feedback reaches the inner
    # agent's prompt on attempt >= 2. ``EpisodicMemoryProvider.provide()``
    # renders an ``Evaluator feedback: <verbatim>`` line into the recalled
    # ``[Past Experiences]`` block; the inner agent's next LLMRequestEvent
    # carries that block as a user-role message. Without this rendering
    # path, the inner agent only sees the LLM-narrative reflection — which
    # the trace evidence shows it tends to treat as advisory rather than
    # binding. Asserting against the message payload is the integration
    # boundary the SDK fix is meant to influence.
    inner_request_with_verbatim_feedback = [
        e
        for e in traced_emitter.events
        if isinstance(e, LLMRequestEvent)
        and any(
            isinstance(m.get("content"), str)
            and "Evaluator feedback:" in m["content"]
            and "jellyfish" in m["content"]
            and "lighthouse" in m["content"]
            for m in e.messages
        )
    ]
    assert inner_request_with_verbatim_feedback, (
        "Expected at least one LLMRequestEvent whose messages payload carries the verbatim "
        "evaluator feedback strings ('Evaluator feedback:' header followed by both 'jellyfish' "
        "and 'lighthouse'). The SDK rendering path EpisodicMemoryProvider.provide() is "
        "responsible for emitting this line into the recalled [Past Experiences] block "
        "ahead of the LLM-narrativised reflection."
    )

    # --- Result invariants: accepted within budget. ---
    assert result.termination_reason == "complete", f"Expected 'complete', got: {result.termination_reason}"
    assert result.total_steps <= reflexion_agent._max_attempts, (
        f"Expected the loop to accept within the max-attempts budget "
        f"({reflexion_agent._max_attempts}); got total_steps={result.total_steps}."
    )
