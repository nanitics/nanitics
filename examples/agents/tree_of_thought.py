"""Tree-of-Thought: exploring multiple reasoning paths via tree search.

Demonstrates TreeOfThoughtAgent — generating candidate continuations, evaluating
and pruning branches, and selecting the best solution across search strategies.

Related guide: docs/guides/agent-types.md
"""

import asyncio
import json

from examples.helpers import make_emitter, make_response
from nanitics import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    MockLLMClient,
)
from nanitics.experimental import (
    SearchStrategy,
    TreeOfThoughtAgent,
)
from nanitics.infrastructure import (
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodeEvaluatedEvent,
    TreeSearchNodePrunedEvent,
)

# --- Evaluators (teaching OutputEvaluator contract) ---


class ScoreAllEvaluator:
    """Accepts every node with a fixed score. Simplest evaluator."""

    def __init__(self, score: float = 0.8) -> None:
        self._score = score

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, score=self._score, evaluator_name="score-all")


class KeywordScorer:
    """Assigns scores based on content keywords. Used for BEST_FIRST demos."""

    def __init__(self, scores: dict[str, float], default: float = 0.5) -> None:
        self._scores = scores
        self._default = default

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        score = self._default
        for keyword, keyword_score in self._scores.items():
            if keyword in output:
                score = keyword_score
                break
        return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, score=score, evaluator_name="keyword-scorer")


class PruningEvaluator:
    """Rejects nodes containing a banned keyword. Used for pruning demos."""

    def __init__(self, reject_keyword: str, accept_score: float = 0.7) -> None:
        self._reject_keyword = reject_keyword
        self._accept_score = accept_score

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        if self._reject_keyword in output:
            return EvaluationResult(verdict=EvaluationVerdict.REJECT, score=0.0, evaluator_name="pruning")
        return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, score=self._accept_score, evaluator_name="pruning")


# --- Helper ---


def make_generation(candidates: list[tuple[str, bool]]) -> str:
    """Build GenerationResponse JSON from (reasoning, is_complete) tuples."""
    return json.dumps({"candidates": [{"reasoning": r, "is_complete": c} for r, c in candidates]})


async def main() -> None:
    # --- Section 1: Basic Tree Search (BFS) ---
    print("--- Section 1: Basic Tree Search (BFS) ---")

    client = MockLLMClient(
        responses=[
            # LLM call 1 (expand root): one terminal, one non-terminal
            make_response(
                make_generation(
                    [
                        ("analyzing the problem step by step", False),
                        ("the answer is 4", True),
                    ]
                )
            ),
            # LLM call 2 (expand non-terminal at depth 1): two terminal candidates
            make_response(
                make_generation(
                    [
                        ("refined analysis: answer is 4", True),
                        ("alternative: answer is 5", True),
                    ]
                )
            ),
        ]
    )
    emitter = make_emitter("tot-s1")

    agent = TreeOfThoughtAgent(
        name="bfs-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="Solve the given problem by exploring different approaches.",
        node_evaluator=ScoreAllEvaluator(score=0.8),
        search_strategy=SearchStrategy.BFS,
        branching_factor=2,
        max_depth=2,
        max_nodes=20,
    )

    result = await agent.run("What is 2 + 2?")

    assert result.output is not None, "Expected output from terminal node"
    assert result.termination_reason == "no_expandable_nodes"
    assert result.total_steps >= 1

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Termination: {result.termination_reason}")
    print("✓ BFS generates candidates, evaluates them, selects best terminal node")

    # --- Section 2: Depth-First Search ---
    print("\n--- Section 2: Depth-First Search ---")

    client = MockLLMClient(
        responses=[
            # LLM call 1 (expand root): two non-terminal candidates
            make_response(
                make_generation(
                    [
                        ("approach-alpha", False),
                        ("approach-beta", False),
                    ]
                )
            ),
            # LLM call 2 (DFS picks one of the depth-1 nodes): one terminal, one non-terminal
            make_response(
                make_generation(
                    [
                        ("alpha-conclusion: 42", True),
                        ("alpha-alternative", False),
                    ]
                )
            ),
            # LLM call 3 (DFS continues expanding remaining nodes)
            make_response(
                make_generation(
                    [
                        ("beta-conclusion", True),
                        ("beta-alt", True),
                    ]
                )
            ),
            # LLM call 4 (expand alpha-alternative at depth 2)
            make_response(
                make_generation(
                    [
                        ("alpha-alt-conclusion", True),
                        ("alpha-alt-2", True),
                    ]
                )
            ),
        ]
    )
    emitter = make_emitter("tot-s2")

    agent = TreeOfThoughtAgent(
        name="dfs-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="Solve the given problem by exploring different approaches.",
        node_evaluator=ScoreAllEvaluator(score=0.8),
        search_strategy=SearchStrategy.DFS,
        branching_factor=2,
        max_depth=3,
        max_nodes=20,
    )

    result = await agent.run("Find the answer to life.")

    assert result.output is not None, "Expected output from terminal node"
    assert result.termination_reason == "no_expandable_nodes"
    assert result.total_steps >= 1

    print(f"  Output: {result.output}")
    print(f"  Steps: {result.total_steps}")
    print("  Search strategy: DFS")
    print("✓ DFS dives deep — expands deepest node first")

    # --- Section 3: Max Nodes Budget ---
    print("\n--- Section 3: Max Nodes Budget ---")

    client = MockLLMClient(
        responses=[
            # LLM call 1 (expand root): 2 non-terminal children → total 3 nodes (root + 2)
            make_response(
                make_generation(
                    [
                        ("branch-1", False),
                        ("branch-2", False),
                    ]
                )
            ),
            # LLM call 2 (expand one child): 2 more children → total 5 nodes = max_nodes
            make_response(
                make_generation(
                    [
                        ("branch-1-a", False),
                        ("branch-1-b", False),
                    ]
                )
            ),
        ]
    )
    emitter = make_emitter("tot-s3")

    agent = TreeOfThoughtAgent(
        name="budget-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="Explore the problem space.",
        node_evaluator=ScoreAllEvaluator(score=0.5),
        search_strategy=SearchStrategy.BFS,
        branching_factor=2,
        max_depth=5,
        max_nodes=5,
    )

    result = await agent.run("Expand until budget exhausted.")

    assert result.termination_reason == "max_nodes", f"Expected max_nodes, got: {result.termination_reason}"

    # Verify via TreeSearchCompleteEvent
    complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].total_nodes <= 5
    assert complete_events[0].termination_reason == "max_nodes"

    print(f"  Total nodes: {complete_events[0].total_nodes}")
    print(f"  Termination: {result.termination_reason}")
    print("✓ max_nodes caps exploration — prevents runaway tree search")

    # --- Section 4: Best-First Search with Scoring ---
    print("\n--- Section 4: Best-First Search with Scoring ---")

    client = MockLLMClient(
        responses=[
            # LLM call 1 (root): two non-terminal candidates with different scores
            make_response(
                make_generation(
                    [
                        ("analyze-data", False),
                        ("guess-randomly", False),
                    ]
                )
            ),
            # LLM call 2 (BEST_FIRST picks "analyze-data" since score 0.9 > 0.3): one terminal
            make_response(
                make_generation(
                    [
                        ("data analysis complete: 42", True),
                        ("need more data", False),
                    ]
                )
            ),
            # LLM call 3 (expand remaining non-terminals)
            make_response(
                make_generation(
                    [
                        ("random guess: 7", True),
                        ("another random guess", True),
                    ]
                )
            ),
            # LLM call 4 (expand "need more data")
            make_response(
                make_generation(
                    [
                        ("gathered more data: 42", True),
                        ("still not enough", True),
                    ]
                )
            ),
        ]
    )
    emitter = make_emitter("tot-s4")

    agent = TreeOfThoughtAgent(
        name="best-first-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="Solve the given problem using the best approach.",
        node_evaluator=KeywordScorer(
            scores={
                "analyze-data": 0.9,
                "data analysis": 0.9,
                "guess-randomly": 0.3,
                "random guess": 0.3,
                "need more data": 0.7,
                "gathered more": 0.85,
            },
            default=0.5,
        ),
        search_strategy=SearchStrategy.BEST_FIRST,
        branching_factor=2,
        max_depth=3,
        max_nodes=20,
    )

    result = await agent.run("What is the answer?")

    assert result.output is not None
    # The output should be from the high-score branch (analyze-data → data analysis complete)
    eval_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeEvaluatedEvent)]
    assert len(eval_events) > 0, "Expected evaluation events"

    print(f"  Output: {result.output}")
    print(f"  Evaluation events: {len(eval_events)}")
    for ev in eval_events[:4]:
        print(f"    Node {ev.node_id[:8]}... score={ev.score}, terminal={ev.is_terminal}")
    print("✓ BEST_FIRST expands highest-scored node — evaluator guides search")

    # --- Section 5: Pruning Rejected Nodes ---
    print("\n--- Section 5: Pruning Rejected Nodes ---")

    client = MockLLMClient(
        responses=[
            # LLM call 1 (root): "bad-approach" will be rejected, "good-approach" accepted
            make_response(
                make_generation(
                    [
                        ("bad-approach", False),
                        ("good-approach", False),
                    ]
                )
            ),
            # LLM call 2 (expand "good-approach" only — "bad-approach" was pruned)
            make_response(
                make_generation(
                    [
                        ("good-conclusion: correct answer", True),
                        ("good-alternative", True),
                    ]
                )
            ),
        ]
    )
    emitter = make_emitter("tot-s5")

    agent = TreeOfThoughtAgent(
        name="pruning-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="Solve the problem, avoiding bad approaches.",
        node_evaluator=PruningEvaluator(reject_keyword="bad"),
        search_strategy=SearchStrategy.BFS,
        branching_factor=2,
        max_depth=3,
        max_nodes=20,
    )

    result = await agent.run("Find the correct answer.")

    assert result.output is not None
    assert "good" in result.output.lower(), f"Expected output from good branch, got: {result.output}"

    # Verify pruning event was emitted
    prune_events = [e for e in emitter.events if isinstance(e, TreeSearchNodePrunedEvent)]
    assert len(prune_events) >= 1, "Expected at least one pruned node"

    print(f"  Output: {result.output}")
    print(f"  Pruned nodes: {len(prune_events)}")
    for ev in prune_events:
        print(f"    Pruned node {ev.node_id[:8]}... reason={ev.reason}")
    print("✓ Evaluator rejects bad branches — pruned nodes are never expanded")

    # --- Section 6: Event Inspection ---
    print("\n--- Section 6: Event Inspection ---")

    client = MockLLMClient(
        responses=[
            # Same setup as Section 1
            make_response(
                make_generation(
                    [
                        ("analyzing the problem step by step", False),
                        ("the answer is 4", True),
                    ]
                )
            ),
            make_response(
                make_generation(
                    [
                        ("refined analysis: answer is 4", True),
                        ("alternative: answer is 5", True),
                    ]
                )
            ),
        ]
    )
    emitter = make_emitter("tot-s6")

    agent = TreeOfThoughtAgent(
        name="observable-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="Solve the given problem.",
        node_evaluator=ScoreAllEvaluator(score=0.8),
        search_strategy=SearchStrategy.BFS,
        branching_factor=2,
        max_depth=2,
        max_nodes=20,
    )

    result = await agent.run("What is 2 + 2?")

    # Inspect tree search events
    created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
    evaluated_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeEvaluatedEvent)]
    complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]

    # Root + 2 children from call 1 + 2 children from call 2 = 5 nodes total
    assert len(created_events) >= 4, f"Expected ≥4 created events (excl. root), got: {len(created_events)}"

    # All non-root nodes are evaluated
    assert len(evaluated_events) >= 4, f"Expected ≥4 evaluation events, got: {len(evaluated_events)}"

    # Exactly one completion event
    assert len(complete_events) == 1
    complete = complete_events[0]
    assert complete.total_nodes >= 5, f"Expected ≥5 total nodes, got: {complete.total_nodes}"
    assert complete.search_strategy == "bfs"
    assert complete.selected_node_id != ""

    print(f"  Nodes created: {len(created_events)}")
    print(f"  Nodes evaluated: {len(evaluated_events)}")
    print("  Search complete event:")
    print(f"    Total nodes: {complete.total_nodes}")
    print(f"    Max depth reached: {complete.max_depth_reached}")
    print(f"    Selected node: {complete.selected_node_id[:8]}...")
    print(f"    Termination: {complete.termination_reason}")
    print(f"    Strategy: {complete.search_strategy}")
    print("✓ Tree search is fully observable — events reconstruct the search tree")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
