"""ReflexionAgent: evaluate → reflect → retry loop with episodic memory.

Demonstrates ReflexionAgent — a wrapper that adds an evaluate-reflect-retry loop
around any inner agent. After each attempt, a ProgrammaticEvaluator judges the output.
On failure, a reflection is generated analyzing what went wrong, stored as an episode,
and the inner agent retries. This enables learning from mistakes across attempts.

Related guide: docs/guides/agent-types.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    EvaluationCheck,
    InMemoryEpisodeStore,
    MockEmbeddingClient,
    MockLLMClient,
    OutcomeType,
    ProgrammaticEvaluator,
    ReActAgent,
    ReflexionAgent,
)
from nanitics.infrastructure import (
    EvaluationEvent,
    ReflectionGeneratedEvent,
)


async def main() -> None:
    # --- Section 1: Success on First Attempt ---
    print("--- Section 1: Success on First Attempt ---")

    # Inner agent answers correctly on the first try
    inner_client = MockLLMClient(
        responses=[
            make_response("The answer is 15"),
        ]
    )
    # Reflexion rebinds this emitter at attempt start — see Agent.bind().
    inner_emitter = make_emitter("reflexion-s1-inner")

    inner_agent = ReActAgent(
        name="math-worker",
        llm_client=inner_client,
        emitter=inner_emitter,
        system_prompt="Solve math problems.",
        tools=[],
    )

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="contains_15",
                check=lambda output: "15" in output,
                feedback="Output must contain '15'",
            )
        ],
    )

    episode_store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

    # ReflexionAgent's own LLM client — not called when the first attempt succeeds
    reflection_client = MockLLMClient(responses=[])

    reflexion_agent = ReflexionAgent(
        name="reflexion-math",
        llm_client=reflection_client,
        emitter=make_emitter("reflexion-s1"),
        system_prompt="Solve math problems with reflection.",
        inner_agent=inner_agent,
        evaluator=evaluator,
        episode_store=episode_store,
    )

    result = await reflexion_agent.run("What is 7 + 8?")

    assert result.output == "The answer is 15", f"Unexpected output: {result.output}"
    assert result.termination_reason == "complete"
    assert result.total_steps == 1

    # Episode stored as success
    episode_count = await episode_store.count()
    assert episode_count == 1, f"Expected 1 episode, got {episode_count}"

    episodes = await episode_store.recall("What is 7 + 8?", limit=10)
    assert episodes[0].episode.outcome == OutcomeType.SUCCESS

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps} (accepted on first attempt)")
    print(f"  Episodes stored: {episode_count}")
    print("✓ First attempt succeeds — no reflection needed")

    # --- Section 2: Retry After Reflection ---
    print("\n--- Section 2: Retry After Reflection ---")

    # Inner agent: first attempt wrong, second attempt correct
    inner_client = MockLLMClient(
        responses=[
            make_response("The answer is 13"),  # Attempt 1: wrong
            make_response("The answer is 15"),  # Attempt 2: correct
        ]
    )
    # Reflexion rebinds this emitter at attempt start — see Agent.bind().
    inner_emitter = make_emitter("reflexion-s2-inner")

    inner_agent = ReActAgent(
        name="math-worker",
        llm_client=inner_client,
        emitter=inner_emitter,
        system_prompt="Solve math problems.",
        tools=[],
    )

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="contains_15",
                check=lambda output: "15" in output,
                feedback="Output must contain '15'",
            )
        ],
    )

    episode_store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

    # ReflexionAgent's LLM generates one reflection (after the failed first attempt)
    reflection_client = MockLLMClient(
        responses=[
            make_response(
                "The previous attempt produced 13, which is incorrect. "
                "7 + 8 = 15. The arithmetic was wrong — need to compute the sum correctly."
            ),
        ]
    )

    reflexion_agent = ReflexionAgent(
        name="reflexion-math",
        llm_client=reflection_client,
        emitter=make_emitter("reflexion-s2"),
        system_prompt="Solve math problems with reflection.",
        inner_agent=inner_agent,
        evaluator=evaluator,
        episode_store=episode_store,
    )

    result = await reflexion_agent.run("What is 7 + 8?")

    assert result.output == "The answer is 15", f"Unexpected output: {result.output}"
    assert result.termination_reason == "complete"
    assert result.total_steps == 2

    # Two episodes: one failure with reflection, one success
    episode_count = await episode_store.count()
    assert episode_count == 2, f"Expected 2 episodes, got {episode_count}"

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps} (failed once, then succeeded)")
    print(f"  Episodes stored: {episode_count}")
    print("✓ Core reflexion cycle: fail → reflect → retry → succeed")

    # --- Section 3: Max Attempts Exhausted ---
    print("\n--- Section 3: Max Attempts Exhausted ---")

    # Inner agent: both attempts wrong
    inner_client = MockLLMClient(
        responses=[
            make_response("The answer is 13"),  # Attempt 1: wrong
            make_response("The answer is 14"),  # Attempt 2: still wrong
        ]
    )
    # Reflexion rebinds this emitter at attempt start — see Agent.bind().
    inner_emitter = make_emitter("reflexion-s3-inner")

    inner_agent = ReActAgent(
        name="math-worker",
        llm_client=inner_client,
        emitter=inner_emitter,
        system_prompt="Solve math problems.",
        tools=[],
    )

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="contains_15",
                check=lambda output: "15" in output,
                feedback="Output must contain '15'",
            )
        ],
    )

    episode_store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

    # One reflection generated (between attempt 1 and 2; no reflection after final attempt)
    reflection_client = MockLLMClient(
        responses=[
            make_response(
                "The previous attempt produced 13, which is incorrect. 7 + 8 = 15. Need to compute the sum correctly."
            ),
        ]
    )

    reflexion_agent = ReflexionAgent(
        name="reflexion-math",
        llm_client=reflection_client,
        emitter=make_emitter("reflexion-s3"),
        system_prompt="Solve math problems with reflection.",
        inner_agent=inner_agent,
        evaluator=evaluator,
        episode_store=episode_store,
        max_attempts=2,
    )

    result = await reflexion_agent.run("What is 7 + 8?")

    assert result.output == "The answer is 14", f"Unexpected output: {result.output}"
    assert result.termination_reason == "evaluation_failed"
    assert result.total_steps == 2

    print(f"  Output: {result.output} (last attempt's output)")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Steps: {result.total_steps}")
    print("✓ All attempts exhausted — returns last output with evaluation_failed")

    # --- Section 4: Inspecting Events and Episodes ---
    print("\n--- Section 4: Inspecting Events and Episodes ---")

    # Same setup as Section 2 (retry then succeed), with event inspection
    inner_client = MockLLMClient(
        responses=[
            make_response("The answer is 13"),  # Attempt 1: wrong
            make_response("The answer is 15"),  # Attempt 2: correct
        ]
    )
    # Reflexion rebinds this emitter at attempt start — see Agent.bind().
    inner_emitter = make_emitter("reflexion-s4-inner")

    inner_agent = ReActAgent(
        name="math-worker",
        llm_client=inner_client,
        emitter=inner_emitter,
        system_prompt="Solve math problems.",
        tools=[],
    )

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="contains_15",
                check=lambda output: "15" in output,
                feedback="Output must contain '15'",
            )
        ],
    )

    episode_store = InMemoryEpisodeStore(embedding_client=MockEmbeddingClient())

    reflection_client = MockLLMClient(
        responses=[
            make_response(
                "The previous attempt produced 13, which is incorrect. "
                "7 + 8 = 15. The arithmetic was wrong — need to compute the sum correctly."
            ),
        ]
    )

    emitter = make_emitter("reflexion-s4")
    reflexion_agent = ReflexionAgent(
        name="reflexion-math",
        llm_client=reflection_client,
        emitter=emitter,
        system_prompt="Solve math problems with reflection.",
        inner_agent=inner_agent,
        evaluator=evaluator,
        episode_store=episode_store,
    )

    result = await reflexion_agent.run("What is 7 + 8?")

    # Inspect EvaluationEvents — one per attempt
    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 2, f"Expected 2 EvaluationEvents, got {len(eval_events)}"

    assert eval_events[0].verdict == "revise"
    assert eval_events[0].revision_attempt == 1
    assert eval_events[0].evaluator_name == "programmatic"

    assert eval_events[1].verdict == "accept"
    assert eval_events[1].revision_attempt == 2
    assert eval_events[1].evaluator_name == "programmatic"

    print("  EvaluationEvents:")
    for ev in eval_events:
        print(f"    Attempt {ev.revision_attempt}: verdict={ev.verdict}, evaluator={ev.evaluator_name}")

    # Inspect ReflectionGeneratedEvent — one after the failed first attempt
    reflection_events = [e for e in emitter.events if isinstance(e, ReflectionGeneratedEvent)]
    assert len(reflection_events) == 1, f"Expected 1 ReflectionGeneratedEvent, got {len(reflection_events)}"

    ref_event = reflection_events[0]
    assert ref_event.attempt_number == 1
    assert "13" in ref_event.reflection_text
    assert ref_event.episode_id is not None

    print(f"  ReflectionGeneratedEvent: attempt={ref_event.attempt_number}, episode={ref_event.episode_id}")
    print(f"    Reflection: {ref_event.reflection_text[:80]}...")

    # Inspect stored episodes
    episode_count = await episode_store.count()
    assert episode_count == 2, f"Expected 2 episodes, got {episode_count}"

    # Recall all episodes for this task. The failed attempt's episode carries a
    # reflection; the successful one does not.
    recalled = await episode_store.recall("What is 7 + 8?", limit=10)
    reflections = [r.episode for r in recalled if r.episode.reflection is not None]
    no_reflections = [r.episode for r in recalled if r.episode.reflection is None]

    assert len(reflections) == 1, "Exactly one rejected episode should carry a reflection"
    assert len(no_reflections) == 1, "Exactly one accepted episode should have no reflection"

    print(f"  Episodes stored: {episode_count}")
    print("    Episode with reflection: 1 (evaluator rejected attempt 1)")
    print("    Episode without reflection: 1 (evaluator accepted attempt 2)")
    print("✓ Events and episodes capture the full reflexion cycle")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
