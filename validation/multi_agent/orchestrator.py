"""Orchestrator decomposes a task across two real specialist agents.

Two tests cover the two ``FinalOutputStrategy`` modes of
``create_orchestrator`` — each exercises the mode on the task shape it
is designed for:

``test_orchestrator_decomposes_task`` (``RELAY_LAST``)
    A research specialist feeds a writing specialist; the writer's
    article *is* the deliverable. ``RELAY_LAST`` returns the writer's
    output verbatim so the coordinator does not compress a long
    finished text into a meta-description.

``test_orchestrator_synthesizes_specialist_fragments`` (``SYNTHESIZE``)
    Two specialists with non-overlapping domains each produce
    complementary fragments. The coordinator must genuinely merge them
    into a coherent brief — no single specialist's output is the
    deliverable. ``SYNTHESIZE`` is the correct mode here and the judge
    rejects any output that draws on only one specialist.

Each ``AgentTool`` is constructed with ``caller_name="unset"`` so the
``create_orchestrator`` ``_caller_name`` rebind path is what actually
surfaces ``"coordinator"`` on the ``DelegationEvent``.
"""

from __future__ import annotations

from nanitics import (
    AgentTool,
    FinalOutputStrategy,
    InMemoryEmitter,
    ReActAgent,
    ReasoningAgent,
)
from nanitics.experimental.coordination import create_orchestrator
from nanitics.infrastructure import (
    AgentCompleteEvent,
    DelegationEvent,
)
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


async def test_orchestrator_decomposes_task(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    researcher = ReasoningAgent(
        name="researcher",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a research specialist. Given a topic, produce a short list "
            "of factual findings (2-4 bullet points). Do not write prose."
        ),
    )

    writer = ReActAgent(
        name="writer",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a technical writer. Given research findings, produce a "
            "polished one-paragraph article. Do not invent new facts."
        ),
        tools=[],
        max_iterations=2,
    )

    # caller_name="unset" is deliberate: create_orchestrator rebinds each
    # specialist's _caller_name to the orchestrator's name. If that rebind
    # regresses, the DelegationEvent.caller_agent assertions below will
    # observe "unset" and fail.
    specialists = [
        AgentTool(
            agent=researcher,
            emitter=traced_emitter,
            description="Delegate research tasks — finding facts and analyzing data.",
            caller_name="unset",
        ),
        AgentTool(
            agent=writer,
            emitter=traced_emitter,
            description="Delegate writing tasks — producing articles from findings.",
            caller_name="unset",
        ),
    ]

    # RELAY_LAST is the correct mode for this pipeline: the writer's
    # article *is* the deliverable, and the coordinator has nothing
    # useful to add by paraphrasing it. Structural relay also removes
    # the prompt-layer fragility that caused the original regression,
    # where Haiku read "synthesize" as "summarize" and replaced the
    # writer's article with a meta-description.
    orchestrator = create_orchestrator(
        name="coordinator",
        llm_client=client,
        emitter=traced_emitter,
        specialists=specialists,
        final_output_strategy=FinalOutputStrategy.RELAY_LAST,
    )

    result = await run_with_retry(
        lambda: orchestrator.run("Write a short article about Python 3.13's new free-threading feature."),
        max_attempts=2,
    )

    # --- Trace-shape invariants ---
    assert_trace_contains(
        traced_emitter,
        DelegationEvent,
        predicate=lambda e: e.delegate_agent == "researcher" and e.caller_agent == "coordinator",
    )
    assert_trace_contains(
        traced_emitter,
        DelegationEvent,
        predicate=lambda e: e.delegate_agent == "writer" and e.caller_agent == "coordinator",
    )
    assert_trace_contains(
        traced_emitter,
        AgentCompleteEvent,
        predicate=lambda e: e.agent_name == "researcher",
    )
    assert_trace_contains(
        traced_emitter,
        AgentCompleteEvent,
        predicate=lambda e: e.agent_name == "writer",
    )

    # --- Fuzzy output ---
    # The criterion requires concrete research facts so a "writer-only"
    # regression — where the coordinator skips the researcher entirely —
    # would fail even if the writer alone produced article-shaped prose.
    await assert_result_satisfies(
        result.output or "",
        (
            "The output is a short article about Python 3.13's free-threading "
            "feature that cites at least one concrete technical detail about "
            "the feature (for example the PEP number, the GIL-disabled build, "
            "the build flag, or a runtime-level mechanism). A generic article "
            "that only describes free-threading in vague terms without any "
            "specific technical fact should fail."
        ),
    )


async def test_orchestrator_synthesizes_specialist_fragments(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")

    historian = ReasoningAgent(
        name="historian",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a history-of-science specialist. Given a topic, produce "
            "a short list of 2-3 concrete historical facts (people, dates, "
            "events, publications). No molecular or structural content. No "
            "prose."
        ),
    )

    biologist = ReasoningAgent(
        name="biologist",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a molecular biology specialist. Given a topic, produce "
            "a short list of 2-3 concrete structural or mechanistic facts "
            "(geometry, bonds, base pairing, strand orientation). No "
            "historical content. No prose."
        ),
    )

    specialists = [
        AgentTool(
            agent=historian,
            emitter=traced_emitter,
            description=(
                "Delegate history-of-science questions — discoverers, dates, publications. No molecular detail."
            ),
            caller_name="unset",
        ),
        AgentTool(
            agent=biologist,
            emitter=traced_emitter,
            description="Delegate molecular biology questions — structure, bonds, mechanism. No historical detail.",
            caller_name="unset",
        ),
    ]

    # final_output_strategy is passed explicitly — SYNTHESIZE is the
    # default, but stating it makes the test's intent unambiguous and
    # keeps the test pinned to the mode it exercises.
    orchestrator = create_orchestrator(
        name="coordinator",
        llm_client=client,
        emitter=traced_emitter,
        specialists=specialists,
        final_output_strategy=FinalOutputStrategy.SYNTHESIZE,
    )

    result = await run_with_retry(
        lambda: orchestrator.run(
            "Produce a two-sentence brief about DNA that draws on BOTH the "
            "history of its discovery and its molecular structure."
        ),
        max_attempts=2,
    )

    # --- Trace-shape invariants ---
    assert_trace_contains(
        traced_emitter,
        DelegationEvent,
        predicate=lambda e: e.delegate_agent == "historian" and e.caller_agent == "coordinator",
    )
    assert_trace_contains(
        traced_emitter,
        DelegationEvent,
        predicate=lambda e: e.delegate_agent == "biologist" and e.caller_agent == "coordinator",
    )
    assert_trace_contains(
        traced_emitter,
        AgentCompleteEvent,
        predicate=lambda e: e.agent_name == "historian",
    )
    assert_trace_contains(
        traced_emitter,
        AgentCompleteEvent,
        predicate=lambda e: e.agent_name == "biologist",
    )

    # --- Fuzzy output ---
    # The criterion requires a concrete fact from EACH domain, so a
    # regression where the coordinator calls only one specialist — or
    # synthesises but drops one domain entirely — fails even if the
    # prose itself is coherent. This is the property SYNTHESIZE must
    # preserve that RELAY_LAST cannot provide.
    await assert_result_satisfies(
        result.output or "",
        (
            "The output is a short brief about DNA that cites BOTH (a) a "
            "concrete historical fact about its discovery (for example a "
            "specific year such as 1953, a named discoverer such as Watson, "
            "Crick, Franklin, or Wilkins, or a named publication) AND (b) a "
            "concrete molecular-structural fact (for example the double "
            "helix, complementary base pairing, A–T / G–C pairing, the "
            "sugar-phosphate backbone, or antiparallel strands). An output "
            "that contains only historical facts OR only structural facts "
            "should fail."
        ),
    )
