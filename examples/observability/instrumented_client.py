"""Instrumented LLM client: tracing non-agent LLM calls with InstrumentedLLMClient.

Wrapping any LLMClient with InstrumentedLLMClient causes every generate() call to emit
LLMRequestEvent and LLMResponseEvent through a shared EventEmitter. This closes the
observability gap for LLM calls made outside the agent loop — evaluators, context-transfer
summarizers, bid generators, orchestrator planners — which would otherwise leave holes in
the trace. The optional ``label`` argument partitions those non-agent calls in the trace
so downstream tooling can filter or group them cleanly.

Related guide: docs/guides/observability.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.evaluation import (
    EvaluationCheck,
    ProgrammaticEvaluator,
)
from nanitics.infrastructure import (
    LLMRequestEvent,
    LLMResponseEvent,
    MockEmbeddingClient,
    MockLLMClient,
)
from nanitics.memory import InMemoryEpisodeStore
from nanitics.specialized import ReflexionAgent
from nanitics.strategies import ReActAgent
from nanitics.tracing import InstrumentedLLMClient


async def main() -> None:
    # --- Section 1: Basic Instrumentation ---
    print("--- Section 1: Basic Instrumentation ---")

    # Wrap any LLMClient (here a MockLLMClient) with InstrumentedLLMClient.
    # Every generate() call emits LLMRequestEvent/LLMResponseEvent through the shared emitter,
    # carrying the label so downstream filters can find these non-agent calls.
    emitter = make_emitter("instrumented-s1")
    raw_client = MockLLMClient(responses=[make_response("Evaluation: pass.")])
    instrumented = InstrumentedLLMClient(
        client=raw_client,
        emitter=emitter,
        label="evaluator",
    )

    response = await instrumented.generate(system_prompt="You are an evaluator.", messages=[])

    assert response.content == "Evaluation: pass."

    llm_requests = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
    llm_responses = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
    assert len(llm_requests) == 1, f"Expected 1 LLMRequestEvent, got {len(llm_requests)}"
    assert len(llm_responses) == 1, f"Expected 1 LLMResponseEvent, got {len(llm_responses)}"
    assert llm_requests[0].label == "evaluator"
    assert llm_responses[0].label == "evaluator"

    print(f"  Wrapped client emitted {len(llm_requests)} request + {len(llm_responses)} response event")
    print(f"  Label on both events: '{llm_requests[0].label}'")
    print("  Non-agent LLM calls are now visible in the trace")

    # --- Section 2: Instrumented Evaluator Inside a ReflexionAgent ---
    print("\n--- Section 2: Instrumented Evaluator Inside an Agent ---")

    # One outer emitter owns the Reflexion trace. The inner agent receives a
    # distinct throwaway emitter — ReflexionAgent.bind() rebinds it to a child
    # of ``shared_emitter`` at each attempt, so inner LLM events (label=None)
    # are forwarded into ``shared_emitter.events`` alongside the reflection
    # LLM's events (label="reflection"). Label partitioning therefore works
    # through the child-emitter listener-copy + forwarding chain: every event
    # appears once in ``shared_emitter.events`` with its label intact.
    shared_emitter = make_emitter("instrumented-s2")

    inner_client = MockLLMClient(
        responses=[
            make_response("The answer is 13"),  # attempt 1: wrong
            make_response("The answer is 15"),  # attempt 2: correct
        ]
    )
    inner_agent = ReActAgent(
        name="math-worker",
        llm_client=inner_client,
        # Reflexion rebinds this emitter at attempt start — see Agent.bind().
        emitter=make_emitter("instrumented-s2-inner-unused"),
        system_prompt="Solve math problems.",
        tools=[],
    )

    # The evaluator here is programmatic (no LLM), so to prove the label-partitioning
    # we wrap the ReflexionAgent's own reflection LLM — another non-agent call site.
    # ReflexionAgent may call its reflection LLM multiple times per cycle; provide enough
    # scripted responses to cover all invocations.
    reflection_raw = MockLLMClient(
        responses=[
            make_response("Previous attempt was 13, which is wrong. Recompute: 7 + 8 = 15."),
            make_response("Second reflection: the arithmetic was off; 7 + 8 = 15."),
        ]
    )
    reflection_instrumented = InstrumentedLLMClient(
        client=reflection_raw,
        emitter=shared_emitter,
        label="reflection",
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

    reflexion_agent = ReflexionAgent(
        name="reflexion-math",
        llm_client=reflection_instrumented,
        emitter=shared_emitter,
        system_prompt="Solve math problems with reflection.",
        inner_agent=inner_agent,
        evaluator=evaluator,
        episode_store=InMemoryEpisodeStore(embedding_client=MockEmbeddingClient()),
    )

    result = await reflexion_agent.run("What is 7 + 8?")
    assert result.output == "The answer is 15"

    # Partition the trace by label: agent calls (None) vs. reflection calls ("reflection").
    all_llm_requests = [e for e in shared_emitter.events if isinstance(e, LLMRequestEvent)]
    agent_llm = [e for e in all_llm_requests if e.label is None]
    reflection_llm = [e for e in all_llm_requests if e.label == "reflection"]

    assert len(agent_llm) >= 2, "Inner agent made at least 2 LLM calls"
    assert len(reflection_llm) >= 1, "At least one reflection LLM call made"
    assert len(agent_llm) + len(reflection_llm) == len(all_llm_requests), (
        "Labels partition every LLMRequestEvent into exactly one bucket"
    )

    label_counts: dict[str, int] = {}
    for e in all_llm_requests:
        key = e.label if e.label is not None else "(agent)"
        label_counts[key] = label_counts.get(key, 0) + 1

    print(f"  Total LLMRequestEvents in trace: {len(all_llm_requests)}")
    print("  Partitioned by label:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")
    print("  Filtering by label cleanly separates agent calls from non-agent calls")

    # --- Section 3: Custom Label Semantics ---
    print("\n--- Section 3: Custom Label Semantics ---")

    # Two different non-agent call sites, one shared emitter, two labels.
    multi_emitter = make_emitter("instrumented-s3")

    summarizer = InstrumentedLLMClient(
        client=MockLLMClient(
            responses=[
                make_response("Q3 revenue was up 12%."),
                make_response("Year-end outlook positive."),
            ]
        ),
        emitter=multi_emitter,
        label="summarizer",
    )
    bid_generator = InstrumentedLLMClient(
        client=MockLLMClient(responses=[make_response("Bid: 42")]),
        emitter=multi_emitter,
        label="bid_generator",
    )

    await summarizer.generate(system_prompt="Summarize.", messages=[])
    await bid_generator.generate(system_prompt="Produce a bid.", messages=[])
    await summarizer.generate(system_prompt="Summarize again.", messages=[])

    requests = [e for e in multi_emitter.events if isinstance(e, LLMRequestEvent)]
    summarizer_requests = [e for e in requests if e.label == "summarizer"]
    bid_requests = [e for e in requests if e.label == "bid_generator"]

    assert len(summarizer_requests) == 2
    assert len(bid_requests) == 1
    assert len(summarizer_requests) + len(bid_requests) == len(requests)

    print(f"  Emitter captured {len(requests)} LLMRequestEvents total")
    print(f"    label='summarizer':    {len(summarizer_requests)}")
    print(f"    label='bid_generator': {len(bid_requests)}")
    print("  Each label isolates one caller's calls — the reason ``label`` exists")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
