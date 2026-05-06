"""Three-layer composition — ``researcher`` → ``grounded`` → ``coordinator``.

``researcher`` is a ``ReActAgent`` with a ``search`` tool. ``grounded`` wraps it in
a ``ReflexionAgent`` that self-evaluates with retry. ``coordinator`` is a
``ReActAgent`` that delegates to ``grounded`` via ``AgentTool``. Each layer maps to
one differentiated capability: tool-using agent, self-evaluation with retry,
agent-as-tool composition.

The visible portion of this file (delimited by the ``HOMEPAGE VISIBLE`` start/end
comment markers below) is fetched at build time by the website and rendered
verbatim on the homepage proof section. Imports, ``MockLLMClient`` setup, the
static mock corpus, the closing assertions, and the ``asyncio.run(main())`` wrapper
sit outside the markers so the snippet stays focused on composition.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    AgentResult,
    AgentTool,
    EvaluationCheck,
    InMemoryEmitter,
    InMemoryEpisodeStore,
    MockEmbeddingClient,
    MockLLMClient,
    ProgrammaticEvaluator,
    ReActAgent,
    ReflexionAgent,
    ToolCall,
    tool,
)
from nanitics.infrastructure import (
    DelegationEvent,
    EvaluationEvent,
    LLMResponseEvent,
    ReflectionGeneratedEvent,
)

# Static mock corpus served by the search tool. Stable IDs R-1, R-2, R-3 let the
# inner researcher cite hits as [R-1], [R-2], etc. — what the bracket-counting
# evaluator predicate inspects on the visible homepage snippet.
_SEARCH_CORPUS: dict[str, list[tuple[str, str]]] = {
    "retry policy quarter": [
        ("R-1", "Retry policy: exponential backoff, max 5 attempts (Q3 baseline)."),
    ],
    "retry policy changes Q4": [
        ("R-1", "Retry policy: exponential backoff, max 5 attempts (Q3 baseline)."),
        ("R-2", "Q4 change: jitter added to backoff; max attempts lowered to 3."),
        ("R-3", "Q4 change: idempotency-key requirement on retried POSTs."),
    ],
}


@tool("search", "Search the codebase for relevant snippets")
async def search(query: str) -> str:
    hits = _SEARCH_CORPUS.get(query, [])
    if not hits:
        return "No results."
    return "\n".join(f"[{rid}] {snippet}" for rid, snippet in hits)


# Scripted mocks: 4 researcher responses (search → cite [R-1] → refine → cite [R-1] [R-2]),
# 1 reflection response between attempts, 2 coordinator responses (delegate → final answer).
_RESEARCHER_RESPONSES = [
    make_response(
        "Looking up the retry policy.",
        tool_calls=[ToolCall(id="r-tc-1", name="search", arguments={"query": "retry policy quarter"})],
        stop_reason="tool_use",
    ),
    make_response("Retry uses exponential backoff [R-1]."),
    make_response(
        "Refining the search.",
        tool_calls=[ToolCall(id="r-tc-2", name="search", arguments={"query": "retry policy changes Q4"})],
        stop_reason="tool_use",
    ),
    make_response("Q4 added jitter and lowered max attempts to 3 [R-1] [R-2]."),
]
_REFLECTION_RESPONSES = [
    make_response(
        "Attempt 1 cited only [R-1]; the question asks what changed, "
        "so the answer must cover at least two sources from the refined search."
    ),
]
_COORDINATOR_RESPONSES = [
    make_response(
        "Delegating to the grounded researcher.",
        tool_calls=[
            ToolCall(
                id="c-tc-1",
                name="grounded",
                arguments={"task": "What changed in our retry policy last quarter?"},
            )
        ],
        stop_reason="tool_use",
    ),
    make_response("Last quarter we added backoff jitter and lowered max retry attempts to 3 [R-1] [R-2]."),
]


async def main(emitter: InMemoryEmitter | None = None) -> tuple[AgentResult, InMemoryEmitter]:
    if emitter is None:
        emitter = make_emitter("homepage")
    researcher_llm = MockLLMClient(responses=_RESEARCHER_RESPONSES)
    reflection_llm = MockLLMClient(responses=_REFLECTION_RESPONSES)
    coordinator_llm = MockLLMClient(responses=_COORDINATOR_RESPONSES)
    # Production evaluators would parse structured output rather than count brackets;
    # the bracket-count predicate keeps the homepage snippet legible at one glance.

    # --- HOMEPAGE VISIBLE START ---
    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="cites_two_sources",
                check=lambda out: out.count("[") >= 2,
                feedback="Cite at least two sources from search().",
            )
        ],
    )
    researcher = ReActAgent(
        name="researcher",
        llm_client=researcher_llm,
        emitter=emitter,
        system_prompt="Research the question using search() and cite results as [R-N].",
        tools=[search],
    )
    grounded = ReflexionAgent(
        name="grounded",
        llm_client=reflection_llm,
        emitter=emitter,
        system_prompt="Reflect on the previous attempt and prescribe a corrected approach.",
        inner_agent=researcher,
        evaluator=evaluator,
        episode_store=InMemoryEpisodeStore(embedding_client=MockEmbeddingClient()),
    )
    coordinator = ReActAgent(
        name="coordinator",
        llm_client=coordinator_llm,
        emitter=emitter,
        system_prompt="Delegate research to the grounded researcher, then synthesise the final answer.",
        tools=[
            AgentTool(
                agent=grounded,
                emitter=emitter,
                caller_name="coordinator",
                description="Delegate research questions to the grounded researcher.",
            )
        ],
    )
    result = await coordinator.run("What changed in our retry policy last quarter?")
    # --- HOMEPAGE VISIBLE END ---

    # --- Trace-shape invariants this Phase pins. CI-grade pinning is Phase 2.2. ---
    assert result.termination_reason == "complete"
    assert isinstance(result.output, str) and result.output, "coordinator final answer is non-empty"

    eval_events = [e for e in emitter.events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 2, f"Expected 2 EvaluationEvents, got {len(eval_events)}"
    assert (eval_events[0].verdict, eval_events[1].verdict) == ("revise", "accept")

    reflection_events = [e for e in emitter.events if isinstance(e, ReflectionGeneratedEvent)]
    assert len(reflection_events) == 1, f"Expected 1 ReflectionGeneratedEvent, got {len(reflection_events)}"

    delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 1
    assert delegation_events[0].caller_agent == "coordinator"
    assert delegation_events[0].delegate_agent == "grounded"
    assert delegation_events[0].task == "What changed in our retry policy last quarter?"

    llm_responses = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
    assert len(llm_responses) == 7, (
        f"Expected 7 LLMResponseEvents (4 researcher + 1 reflection + 2 coordinator), got {len(llm_responses)}"
    )

    print("--- Homepage example: composing ReActAgent + ReflexionAgent + AgentTool ---")
    print(f"  Coordinator final answer: {result.output}")
    print(f"  Evaluator verdicts: {[e.verdict for e in eval_events]}")
    print(f"  Reflections generated: {len(reflection_events)}")
    print(f"  Delegations recorded: {delegation_events[0].caller_agent} → {delegation_events[0].delegate_agent}")
    print(f"  LLM responses on shared timeline: {len(llm_responses)}")
    print("✓ Trace shape: attempt 1 (revise) → reflect → attempt 2 (accept) → coordinator final answer")

    return result, emitter


if __name__ == "__main__":
    asyncio.run(main())
