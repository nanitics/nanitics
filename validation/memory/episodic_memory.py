"""Episodic memory recall under real Voyage embeddings.

Validates that real embeddings produce qualitatively correct recall — the
semantically closest episode ranks above unrelated ones — and that the
:class:`EpisodicMemoryProvider` injects a ``[Past Experiences]`` context
block and emits :class:`EpisodeRecallEvent` through the standard path.

Exact scores are not asserted. Real embeddings drift between model versions;
ordering and loose score floors are the stable properties.

Acceptance criteria:
  - Direct recall: the financial-report episode ranks first for a financial query.
  - Top similarity score clears a degeneracy floor (rules out zero vectors, mock
    embeddings, wrong-query embedding).
  - Relevant episode scores measurably above irrelevant episodes (separation gap).
  - Agent section emits an ``EpisodeRecallEvent`` with ``results_count >= 1``.
  - Agent output references past experience with financial report analysis.
"""

from __future__ import annotations

from nanitics.infrastructure import EpisodeRecallEvent
from nanitics.memory import (
    Episode,
    EpisodicMemoryContributor,
    EpisodicMemoryProvider,
    InMemoryEpisodeStore,
    OutcomeType,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_embedding_client,
    make_llm_client,
    requires_voyage,
    run_with_retry,
)


@requires_voyage
async def test_episodic_recall_and_provider(traced_emitter: InMemoryEmitter) -> None:
    embedding_client = make_embedding_client("voyage")
    store = InMemoryEpisodeStore(embedding_client=embedding_client)

    await store.record(
        Episode(
            situation="Summarize a quarterly financial report",
            action="Extracted revenue, costs, and outlook into sections",
            outcome=OutcomeType.SUCCESS,
        )
    )
    await store.record(
        Episode(
            situation="Cook pasta al dente",
            action="Boiled for 8 minutes and tasted at 7",
            outcome=OutcomeType.SUCCESS,
        )
    )
    await store.record(
        Episode(
            situation="Debug a memory leak in a Python service",
            action="Attached tracemalloc and ran a long soak test",
            outcome=OutcomeType.FAILURE,
        )
    )

    # --- Direct recall ---
    results = await store.recall("Analyze the Q4 earnings release")
    assert len(results) >= 1, f"Expected at least one recall result, got: {len(results)}"
    assert results[0].episode.situation == "Summarize a quarterly financial report", (
        f"Expected financial-report episode on top, got: {results[0].episode.situation!r}"
    )
    assert results[0].similarity_score > 0.2, (
        f"Expected top similarity above degeneracy floor, got: {results[0].similarity_score}"
    )
    if len(results) > 1:
        separation = results[0].similarity_score - results[-1].similarity_score
        assert separation > 0.05, (
            f"Expected relevant episode to score measurably above irrelevant ones, "
            f"got scores: {[r.similarity_score for r in results]}"
        )

    # --- Agent-integrated: EpisodicMemoryContributor + EpisodicMemoryProvider ---
    provider = EpisodicMemoryProvider(store=store, emitter=traced_emitter, limit=3)
    agent = ReActAgent(
        name="episodic-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="You are an analyst who learns from past experience.",
        tools=[],
        prompt_contributors=[EpisodicMemoryContributor()],
        context_providers=[provider],
        max_iterations=3,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Recall any relevant past experience to help with analyzing this quarter's financial report."
        ),
        max_attempts=2,
    )

    assert_trace_contains(traced_emitter, EpisodeRecallEvent, predicate=lambda e: e.results_count >= 1)
    await assert_result_satisfies(
        result.output or "",
        "The output references past experience with financial report analysis.",
    )
