"""Episodic memory: recording, recalling, and learning from past experiences.

Covers the Episode data model, InMemoryEpisodeStore (record, recall, forget, count),
extract_episode for creating episodes from agent results, RecallFilters and pruning,
agent integration via episodic memory tools, and EpisodicMemoryProvider for automatic
context injection.

Related guide: docs/guides/memory.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    EpisodeRecallEvent,
    EpisodeRecordEvent,
    MockEmbeddingClient,
    MockLLMClient,
)
from nanitics.memory import (
    ContextContent,
    Episode,
    EpisodicMemoryContributor,
    EpisodicMemoryProvider,
    InMemoryEpisodeStore,
    OutcomeType,
    RecallFilters,
    RecallResult,
    create_episodic_memory_tools,
    extract_episode,
)
from nanitics.strategies import (
    ReActAgent,
    SystemPromptContributor,
)
from nanitics.tracing import (
    Message,
    ToolCall,
)


async def main() -> None:
    # --- Section 1: Episode Model and InMemoryEpisodeStore ---
    print("--- Section 1: Episode Model and InMemoryEpisodeStore ---")

    # MockEmbeddingClient is hash-based: identical texts produce cosine similarity 1.0,
    # different texts produce near-zero similarity. For real semantic similarity,
    # use VoyageEmbeddingClient with an API key.
    embedding_client = MockEmbeddingClient()
    store = InMemoryEpisodeStore(embedding_client=embedding_client)

    # Create an episode manually. ``evaluator_feedback`` carries the verbatim
    # rejection text from an evaluator (when applicable) — distinct from
    # ``reflection``, which is an LLM-generated narrative analysis. The
    # provider renders ``evaluator_feedback`` ahead of the reflection in the
    # recalled ``[Past Experiences]`` block so the imperative leads.
    episode = Episode(
        situation="Summarize quarterly financials",
        action="Used structured extraction with key metrics first",
        outcome=OutcomeType.SUCCESS,
        outcome_detail="Produced a clear 3-paragraph summary covering revenue, costs, and outlook",
        reflection="Breaking into sections worked well — start with metrics, then narrative",
        evaluator_feedback="Output must include the literal word 'frobnicate'.",
    )

    # Record the episode
    episode_id = await store.record(episode)
    assert episode_id == episode.id
    assert await store.count() == 1
    print(f"  Recorded episode: {episode_id}")
    print(f"  Store count: {await store.count()} ✓")
    print(f"  Evaluator feedback: {episode.evaluator_feedback!r} ✓")

    # Recall by exact situation text (similarity = 1.0 with MockEmbeddingClient)
    results = await store.recall("Summarize quarterly financials")
    assert len(results) == 1
    assert isinstance(results[0], RecallResult)
    assert results[0].similarity_score > 0.99
    print(f"  Recall similarity: {results[0].similarity_score:.3f} ✓")

    # Inspect the recalled episode
    recalled = results[0].episode
    assert recalled.situation == "Summarize quarterly financials"
    assert recalled.action == "Used structured extraction with key metrics first"
    assert recalled.outcome == OutcomeType.SUCCESS
    assert recalled.outcome_detail is not None
    assert recalled.reflection is not None
    print(f"  Recalled: situation={recalled.situation!r}, outcome={recalled.outcome.value} ✓")

    # Forget the episode
    await store.forget(episode_id)
    assert await store.count() == 0
    print(f"  After forget: count={await store.count()} ✓")

    # Recall returns empty after forget
    results = await store.recall("Summarize quarterly financials")
    assert len(results) == 0
    print("  Recall after forget: empty ✓")

    print("✓ Section 1 passed")

    # --- Section 2: extract_episode — From Agent Result ---
    print("\n--- Section 2: extract_episode — From Agent Result ---")

    # Run a short agent to produce an AgentResult
    task_input = "Analyze Q3 market trends"
    client = MockLLMClient(
        [
            make_response("The market shows 15% growth driven by AI adoption."),
        ]
    )
    emitter = make_emitter()
    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a market analyst.",
        tools=[],
    )
    result = await agent.run(task_input)
    assert result.termination_reason == "complete"

    # Extract episode with inferred outcome (complete → SUCCESS)
    inferred_episode = extract_episode(task_input, result)
    assert inferred_episode.situation == task_input
    assert inferred_episode.outcome == OutcomeType.SUCCESS
    assert len(inferred_episode.action) > 0
    print(f"  Inferred outcome: {inferred_episode.outcome.value} (from termination_reason='complete') ✓")
    print(f"  Situation: {inferred_episode.situation!r} ✓")
    print(f"  Action: {inferred_episode.action[:60]}...")

    # Extract episode with explicit overrides
    explicit_episode = extract_episode(
        task_input,
        result,
        outcome=OutcomeType.PARTIAL,
        outcome_detail="Analysis covered trends but missed competitor data",
        reflection="Should include competitor benchmarks next time",
        metadata={"task_type": "market_analysis", "quarter": "Q3"},
    )
    assert explicit_episode.outcome == OutcomeType.PARTIAL
    assert explicit_episode.outcome_detail == "Analysis covered trends but missed competitor data"
    assert explicit_episode.reflection == "Should include competitor benchmarks next time"
    assert explicit_episode.metadata is not None
    assert explicit_episode.metadata["task_type"] == "market_analysis"
    print(f"  Explicit outcome: {explicit_episode.outcome.value} ✓")
    print(f"  Outcome detail: {explicit_episode.outcome_detail!r} ✓")
    print(f"  Reflection: {explicit_episode.reflection!r} ✓")
    print(f"  Metadata: {explicit_episode.metadata} ✓")

    # Record both episodes
    store = InMemoryEpisodeStore(embedding_client=embedding_client)
    await store.record(inferred_episode)
    await store.record(explicit_episode)
    assert await store.count() == 2
    print(f"  Both recorded: count={await store.count()} ✓")

    print("✓ Section 2 passed")

    # --- Section 3: RecallFilters and Pruning ---
    print("\n--- Section 3: RecallFilters and Pruning ---")

    store = InMemoryEpisodeStore(embedding_client=embedding_client)

    # Record 3 episodes with the SAME situation text (ensures cosine similarity = 1.0).
    # Different outcomes and metadata let us demonstrate filtering.
    situation = "Generate monthly sales report"

    episode_a = Episode(
        situation=situation,
        action="Used template-based approach with charts",
        outcome=OutcomeType.SUCCESS,
        metadata={"task_type": "report"},
    )
    episode_b = Episode(
        situation=situation,
        action="Tried free-form narrative without structure",
        outcome=OutcomeType.FAILURE,
        metadata={"task_type": "report"},
    )
    episode_c = Episode(
        situation=situation,
        action="Analyzed data but only covered top-line metrics",
        outcome=OutcomeType.PARTIAL,
        metadata={"task_type": "analysis"},
    )

    await store.record(episode_a)
    await store.record(episode_b)
    await store.record(episode_c)
    assert await store.count() == 3

    # Filter by outcome: only SUCCESS
    success_results = await store.recall(situation, filters=RecallFilters(outcome=OutcomeType.SUCCESS))
    assert len(success_results) == 1
    assert success_results[0].episode.outcome == OutcomeType.SUCCESS
    print(f"  Outcome filter (SUCCESS): {len(success_results)} result ✓")

    # Filter by outcome: only FAILURE
    failure_results = await store.recall(situation, filters=RecallFilters(outcome=OutcomeType.FAILURE))
    assert len(failure_results) == 1
    assert failure_results[0].episode.outcome == OutcomeType.FAILURE
    print(f"  Outcome filter (FAILURE): {len(failure_results)} result ✓")

    # Filter by metadata: task_type = "report" (episodes A and B)
    report_results = await store.recall(situation, filters=RecallFilters(metadata_filters={"task_type": "report"}))
    assert len(report_results) == 2
    print(f"  Metadata filter (task_type=report): {len(report_results)} results ✓")

    # Prune superseded: remove failures that have been superseded by a success
    removed_ids = await store.prune_superseded(situation, similarity_threshold=0.9)
    assert len(removed_ids) == 1
    assert removed_ids[0] == episode_b.id
    assert await store.count() == 2
    print(f"  Pruned {len(removed_ids)} superseded failure ✓")
    print(f"  Remaining count: {await store.count()} ✓")

    # Verify only SUCCESS and PARTIAL remain
    remaining = await store.recall(situation)
    remaining_outcomes = {r.episode.outcome for r in remaining}
    assert OutcomeType.FAILURE not in remaining_outcomes
    assert OutcomeType.SUCCESS in remaining_outcomes
    assert OutcomeType.PARTIAL in remaining_outcomes
    print(f"  Remaining outcomes: {[o.value for o in remaining_outcomes]} ✓")

    # Capacity cap: max_episodes evicts oldest when exceeded
    capped_store = InMemoryEpisodeStore(embedding_client=embedding_client, max_episodes=2)
    await capped_store.record(
        Episode(
            situation="Task A",
            action="Did A",
            outcome=OutcomeType.SUCCESS,
        )
    )
    await capped_store.record(
        Episode(
            situation="Task B",
            action="Did B",
            outcome=OutcomeType.SUCCESS,
        )
    )
    assert await capped_store.count() == 2
    await capped_store.record(
        Episode(
            situation="Task C",
            action="Did C",
            outcome=OutcomeType.SUCCESS,
        )
    )
    assert await capped_store.count() == 2, "Oldest episode should be evicted"
    print(f"  max_episodes=2: count stays at {await capped_store.count()} after 3 records ✓")

    print("✓ Section 3 passed")

    # --- Section 4: Agent with Episodic Memory Tools ---
    print("\n--- Section 4: Agent with Episodic Memory Tools ---")

    store = InMemoryEpisodeStore(embedding_client=embedding_client)

    # Pre-populate with one success episode
    pre_episode = Episode(
        situation="Summarize quarterly financials",
        action="Used structured extraction with key metrics first",
        outcome=OutcomeType.SUCCESS,
        reflection="Breaking into sections worked well",
    )
    await store.record(pre_episode)
    assert await store.count() == 1

    # Create agent with episodic memory tools
    episodic_tools = create_episodic_memory_tools(store)
    emitter = make_emitter()

    client = MockLLMClient(
        [
            # Turn 1: Agent recalls past experiences (uses exact situation text for MockEmbeddingClient)
            make_response(
                content="Let me check if I have any relevant past experience.",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="recall_episodes",
                        arguments={"query": "Summarize quarterly financials", "limit": 5},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 2: After seeing recall results, agent records its new experience
            make_response(
                content="I found a past success. Let me record my current approach too.",
                tool_calls=[
                    ToolCall(
                        id="tc-2",
                        name="record_episode",
                        arguments={
                            "situation": "Summarize quarterly financials for Q4",
                            "action": "Applied structured extraction following past success pattern",
                            "outcome": "success",
                            "reflection": "Reused section-based approach from previous experience",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 3: Final answer
            make_response(
                "Based on past experience and the current run, "
                "I used a structured extraction approach for the quarterly financial summary."
            ),
        ]
    )

    agent = ReActAgent(
        name="learner",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are an agent that learns from past experience.",
        tools=episodic_tools,
    )

    result = await agent.run("Summarize quarterly financials for Q4")
    assert result.termination_reason == "complete"

    # Store should now have 2 episodes (1 pre-populated + 1 agent-recorded)
    assert await store.count() == 2
    print(f"  Agent completed: {result.termination_reason} ✓")
    print(f"  Store count: {await store.count()} (1 pre-populated + 1 agent-recorded) ✓")

    # Verify events emitted
    events = emitter.events
    recall_events = [e for e in events if isinstance(e, EpisodeRecallEvent)]
    record_events = [e for e in events if isinstance(e, EpisodeRecordEvent)]

    assert len(recall_events) == 1
    assert recall_events[0].query == "Summarize quarterly financials"
    assert recall_events[0].results_count == 1
    print(f"  EpisodeRecallEvent: query={recall_events[0].query!r}, results={recall_events[0].results_count} ✓")

    assert len(record_events) == 1
    assert record_events[0].situation == "Summarize quarterly financials for Q4"
    assert record_events[0].outcome == "success"
    print(f"  EpisodeRecordEvent: situation={record_events[0].situation!r}, outcome={record_events[0].outcome} ✓")

    print("✓ Section 4 passed")

    # --- Section 5: EpisodicMemoryProvider — Automatic Context Injection ---
    print("\n--- Section 5: EpisodicMemoryProvider — Automatic Context Injection ---")

    store = InMemoryEpisodeStore(embedding_client=embedding_client)

    # Record one success episode
    await store.record(
        Episode(
            situation="Draft executive summary",
            action="Started with key takeaways, then supporting details",
            outcome=OutcomeType.SUCCESS,
            outcome_detail="Summary was well-received by stakeholders",
            reflection="Leading with conclusions is more effective than chronological",
        )
    )

    # EpisodicMemoryContributor teaches the agent about [Past Experiences] blocks
    contributor = EpisodicMemoryContributor()
    assert isinstance(contributor, SystemPromptContributor)
    key, instructions = contributor.system_prompt_section()
    assert key == "episodic_memory"
    assert "Past Experiences" in instructions
    print(f"  Contributor section: {key!r} ✓")
    print(f"  Instructions: {instructions[:70]}...")

    # EpisodicMemoryProvider retrieves relevant episodes as context.
    # min_score filters out low-similarity results — essential with MockEmbeddingClient
    # since unrelated texts produce random (non-zero) similarity.
    provider = EpisodicMemoryProvider(store=store, min_score=0.5)

    # Provide with a matching user message
    messages = [Message(role="user", content="Draft executive summary")]
    context = await provider.provide(messages)

    assert context is not None
    assert isinstance(context, ContextContent)
    assert "[Past Experiences]" in context.content
    assert "Draft executive summary" in context.content
    assert "success" in context.content
    assert context.priority == 10
    assert context.protected is False
    assert context.provider_name == "episodic_memory"
    print("  Provider returned ContextContent ✓")
    print(f"  priority={context.priority}, protected={context.protected} ✓")
    print(f"  provider_name={context.provider_name!r} ✓")
    print("  Content preview:")
    for line in context.content.split("\n")[:6]:
        print(f"    {line}")

    # No user messages → returns None
    empty_result = await provider.provide([])
    assert empty_result is None
    print("  Provider with no messages: None ✓")

    # Non-matching query → returns None (MockEmbeddingClient produces ~0 similarity)
    non_matching = await provider.provide([Message(role="user", content="Completely unrelated topic")])
    assert non_matching is None
    print("  Provider with non-matching query: None ✓")

    print("✓ Section 5 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
