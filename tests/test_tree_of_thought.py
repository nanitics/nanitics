import json

import pytest
from pydantic import ValidationError

from nanitics import (
    MockLLMClient,
)
from nanitics.infrastructure.observability.events import (
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodeEvaluatedEvent,
    TreeSearchNodePrunedEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.agents.tree_of_thought import (
    SearchStrategy,
    ThoughtNode,
    TreeOfThoughtAgent,
    _Candidate,
    _GenerationResponse,
)
from tests.testing_helpers import make_emitter, make_response


def make_generation_response(
    candidates: list[tuple[str, bool]],
) -> str:
    """Build a JSON string for _GenerationResponse from (reasoning, is_complete) tuples."""
    return _GenerationResponse(
        candidates=[_Candidate(reasoning=r, is_complete=c) for r, c in candidates]
    ).model_dump_json()


class _AcceptEvaluator:
    """Evaluator that accepts everything with a fixed score."""

    def __init__(self, score: float = 0.8) -> None:
        self._score = score

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=self._score,
            evaluator_name="test-evaluator",
        )


class _RejectEvaluator:
    """Evaluator that rejects everything."""

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.REJECT,
            score=0.1,
            feedback="Not acceptable",
            evaluator_name="test-evaluator",
        )


class _ScoringEvaluator:
    """Evaluator that assigns scores based on content keywords."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        score = self._scores.get(output, 0.5)
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=score,
            evaluator_name="test-evaluator",
        )


class _RejectThenAcceptEvaluator:
    """Evaluator that rejects the first N calls, then accepts."""

    def __init__(self, reject_count: int = 1, accept_score: float = 0.9) -> None:
        self._reject_count = reject_count
        self._accept_score = accept_score
        self._call_count = 0

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        self._call_count += 1
        if self._call_count <= self._reject_count:
            return EvaluationResult(
                verdict=EvaluationVerdict.REJECT,
                score=0.1,
                evaluator_name="test-evaluator",
            )
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=self._accept_score,
            evaluator_name="test-evaluator",
        )


# ──────────────────────────────────────────────────────────
# BFS Search Tests
# ──────────────────────────────────────────────────────────


class TestBFSSearch:
    async def test_bfs_expands_breadth_first(self) -> None:
        """BFS should expand all nodes at current depth before going deeper."""
        # Root generates 2 children (depth 1), one is terminal
        gen1 = make_generation_response(
            [
                ("approach-A", False),
                ("approach-B", True),
            ]
        )
        # Search continues: depth-1 non-terminal "approach-A" is expanded
        gen2 = make_generation_response(
            [
                ("approach-A-1", True),
                ("approach-A-2", True),
            ]
        )

        client = MockLLMClient([make_response(gen1), make_response(gen2)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think step by step",
            node_evaluator=_AcceptEvaluator(),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=3,
            max_nodes=50,
        )

        result = await agent.run("What is 2+2?")

        # All terminals found; best selected from accepted terminals
        assert result.output in ("approach-B", "approach-A-1", "approach-A-2")
        assert result.termination_reason == "no_expandable_nodes"


# ──────────────────────────────────────────────────────────
# DFS Search Tests
# ──────────────────────────────────────────────────────────


class TestDFSSearch:
    async def test_dfs_expands_deepest_first(self) -> None:
        """DFS should expand the deepest node first."""
        # Root generates 2 children at depth 1
        gen1 = make_generation_response(
            [
                ("deep-path", False),
                ("shallow-path", False),
            ]
        )
        # DFS picks the deepest (both at depth 1, picks first). Expand it:
        gen2 = make_generation_response(
            [
                ("deep-path-1", True),
            ]
        )
        # Search continues — "shallow-path" is still expandable
        gen3 = make_generation_response(
            [
                ("shallow-path-1", True),
            ]
        )

        client = MockLLMClient([make_response(gen1), make_response(gen2), make_response(gen3)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think step by step",
            node_evaluator=_AcceptEvaluator(),
            search_strategy=SearchStrategy.DFS,
            branching_factor=2,
            max_depth=3,
            max_nodes=50,
        )

        result = await agent.run("Solve this problem")

        # Best terminal selected from all accepted terminals
        assert result.output in ("deep-path-1", "shallow-path-1")
        assert result.termination_reason == "no_expandable_nodes"

    async def test_dfs_respects_max_depth(self) -> None:
        """Nodes at max_depth should not be expanded."""
        # max_depth=1, so root children (depth 1) are not expanded further
        gen1 = make_generation_response(
            [
                ("depth-1-thought", False),
            ]
        )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think step by step",
            node_evaluator=_AcceptEvaluator(score=0.7),
            search_strategy=SearchStrategy.DFS,
            branching_factor=1,
            max_depth=1,
            max_nodes=50,
        )

        result = await agent.run("Question")

        # Node at depth 1 is at max_depth, not expandable -> no_expandable_nodes
        assert result.termination_reason == "no_expandable_nodes"
        assert result.output == "depth-1-thought"


# ──────────────────────────────────────────────────────────
# Best-First Search Tests
# ──────────────────────────────────────────────────────────


class TestBestFirstSearch:
    async def test_best_first_expands_highest_score(self) -> None:
        """Best-first should expand the highest-scoring node next."""
        # Root generates 2 children
        gen1 = make_generation_response(
            [
                ("low-score", False),
                ("high-score", False),
            ]
        )
        # "high-score" should be expanded first (score 0.9 vs 0.3)
        gen2 = make_generation_response(
            [
                ("high-score-child", True),
            ]
        )
        # Search continues — "low-score" is still expandable
        gen3 = make_generation_response(
            [
                ("low-score-child", True),
            ]
        )

        scores = {"low-score": 0.3, "high-score": 0.9, "high-score-child": 0.95, "low-score-child": 0.4}
        client = MockLLMClient([make_response(gen1), make_response(gen2), make_response(gen3)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think step by step",
            node_evaluator=_ScoringEvaluator(scores),
            search_strategy=SearchStrategy.BEST_FIRST,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
        )

        result = await agent.run("Complex problem")

        # Best accepted terminal is "high-score-child" with score 0.95
        assert result.output == "high-score-child"
        assert result.termination_reason == "no_expandable_nodes"


# ──────────────────────────────────────────────────────────
# Terminal Detection
# ──────────────────────────────────────────────────────────


class TestTerminalDetection:
    async def test_terminal_candidate_collected(self) -> None:
        """When a candidate is marked is_complete=True, it's collected as accepted terminal."""
        gen1 = make_generation_response(
            [
                ("The answer is 42", True),
                ("Still thinking...", False),
            ]
        )
        # Search continues — "Still thinking..." is expandable
        gen2 = make_generation_response(
            [
                ("deeper thought", True),
            ]
        )

        scores = {"The answer is 42": 0.95, "Still thinking...": 0.5, "deeper thought": 0.7}
        client = MockLLMClient([make_response(gen1), make_response(gen2)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Answer questions",
            node_evaluator=_ScoringEvaluator(scores),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
        )

        result = await agent.run("What is the meaning of life?")

        # Best accepted terminal is "The answer is 42" with score 0.95
        assert result.output == "The answer is 42"
        assert result.termination_reason == "no_expandable_nodes"

        # Verify accepted_count in complete event
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert complete_events[0].accepted_count == 2

    async def test_single_node_immediate_terminal(self) -> None:
        """Root's only child is terminal -> search stops with no expandable nodes."""
        gen1 = make_generation_response(
            [
                ("immediate-answer", True),
            ]
        )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Answer questions",
            node_evaluator=_AcceptEvaluator(),
            search_strategy=SearchStrategy.BFS,
            branching_factor=1,
            max_depth=5,
            max_nodes=50,
        )

        result = await agent.run("Simple question")

        assert result.output == "immediate-answer"
        assert result.termination_reason == "no_expandable_nodes"
        assert result.total_steps == 1


# ──────────────────────────────────────────────────────────
# Budget Limits
# ──────────────────────────────────────────────────────────


class TestBudgetLimits:
    async def test_max_nodes_stops_search(self) -> None:
        """Search stops when node count reaches max_nodes."""
        # Branching factor 3, max_nodes 4 (root + 3 children = 4)
        gen1 = make_generation_response(
            [
                ("child-1", False),
                ("child-2", False),
                ("child-3", False),
            ]
        )

        scores = {"child-1": 0.5, "child-2": 0.8, "child-3": 0.3}
        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_ScoringEvaluator(scores),
            search_strategy=SearchStrategy.BFS,
            branching_factor=3,
            max_depth=10,
            max_nodes=4,  # root + 3 children = 4
        )

        result = await agent.run("Question")

        assert result.termination_reason == "max_nodes"
        assert result.output == "child-2"  # highest scoring

    async def test_max_depth_not_expanded(self) -> None:
        """Nodes at max_depth are not expanded."""
        gen1 = make_generation_response(
            [
                ("depth-1-a", False),
            ]
        )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_AcceptEvaluator(),
            search_strategy=SearchStrategy.BFS,
            branching_factor=1,
            max_depth=1,
            max_nodes=50,
        )

        result = await agent.run("Question")

        # depth-1 node can't be expanded further
        assert result.termination_reason == "no_expandable_nodes"


# ──────────────────────────────────────────────────────────
# Pruning
# ──────────────────────────────────────────────────────────


class TestPruning:
    async def test_rejected_nodes_are_pruned(self) -> None:
        """Nodes with REJECT verdict are pruned and not expanded."""
        gen1 = make_generation_response(
            [
                ("bad-idea", False),
                ("good-idea", True),
            ]
        )

        class _RejectBadEvaluator:
            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                if "bad" in output:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REJECT,
                        score=0.1,
                        evaluator_name="test-evaluator",
                    )
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=0.9,
                    evaluator_name="test-evaluator",
                )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_RejectBadEvaluator(),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
        )

        result = await agent.run("Question")

        assert result.output == "good-idea"
        assert result.termination_reason == "no_expandable_nodes"

        # Verify pruning event was emitted
        prune_events = [e for e in emitter.events if isinstance(e, TreeSearchNodePrunedEvent)]
        assert len(prune_events) == 1
        assert prune_events[0].reason == "evaluation_rejected"

    async def test_all_nodes_pruned(self) -> None:
        """When all nodes are pruned, search terminates."""
        gen1 = make_generation_response(
            [
                ("bad-1", False),
                ("bad-2", False),
            ]
        )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_RejectEvaluator(),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
        )

        result = await agent.run("Question")

        assert result.termination_reason == "no_expandable_nodes"


# ──────────────────────────────────────────────────────────
# Cancellation
# ──────────────────────────────────────────────────────────


class TestCancellation:
    async def test_cancellation_stops_search(self) -> None:
        """Cancelled token stops search between iterations."""
        token = CancellationToken()
        token.cancel()

        client = MockLLMClient([])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_AcceptEvaluator(),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
            cancellation_token=token,
        )

        result = await agent.run("Question")

        assert result.termination_reason == "cancelled"


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestEventEmission:
    async def test_all_expected_events_emitted(self) -> None:
        """Verify all tree search events and standard agent events are emitted."""
        gen1 = make_generation_response(
            [
                ("solution", True),
            ]
        )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_AcceptEvaluator(score=0.9),
            search_strategy=SearchStrategy.BFS,
            branching_factor=1,
            max_depth=5,
            max_nodes=50,
        )

        await agent.run("Question")

        event_types = [e.event_type for e in emitter.events]

        assert "agent.start" in event_types
        assert "agent.complete" in event_types
        assert "tree_search.node.created" in event_types
        assert "tree_search.node.evaluated" in event_types
        assert "tree_search.complete" in event_types

        # Root creation + child creation
        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        assert len(created_events) >= 2  # root + at least 1 child

        # Evaluation of child
        eval_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeEvaluatedEvent)]
        assert len(eval_events) >= 1

        # Completion
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].search_strategy == "bfs"

    async def test_enriched_node_created_events(self) -> None:
        """TreeSearchNodeCreatedEvent should carry full content and is_terminal."""
        gen1 = make_generation_response(
            [
                ("detailed solution reasoning", True),
                ("continuation thought", False),
            ]
        )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_AcceptEvaluator(score=0.9),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=3,  # root + 2 children = 3, stops before expanding further
        )

        await agent.run("Question")

        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        assert len(created_events) >= 3  # root + 2 children

        # Root event should have full content
        root_event = created_events[0]
        assert root_event.content == "Question"
        assert root_event.content == "Question"
        assert root_event.is_terminal is False
        assert root_event.is_failed is False
        assert root_event.action is None

        # Child events should have full content and correct is_terminal
        child_events = [e for e in created_events if e.parent_id is not None]
        terminal_children = [e for e in child_events if e.is_terminal]
        non_terminal_children = [e for e in child_events if not e.is_terminal]
        assert len(terminal_children) >= 1
        assert len(non_terminal_children) >= 1

        # All child events should carry full content
        for child_event in child_events:
            assert child_event.content is not None
            assert len(child_event.content) > 0

    async def test_enriched_events_serialize_correctly(self) -> None:
        """Enriched events should serialize/deserialize without errors."""
        event = TreeSearchNodeCreatedEvent(
            trace_id="t",
            span_id="s",
            node_id="n1",
            parent_id=None,
            depth=0,
            node_type="thought",
            content="full content here",
            is_terminal=True,
        )
        data = event.model_dump()
        assert data["content"] == "full content here"
        assert data["is_terminal"] is True
        assert data["is_failed"] is False
        assert data["action"] is None
        assert data["observation"] is None

        # Round-trip through JSON
        json_str = event.model_dump_json()
        restored = TreeSearchNodeCreatedEvent.model_validate_json(json_str)
        assert restored.content == "full content here"
        assert restored.is_terminal is True

    async def test_pruning_events_emitted(self) -> None:
        """Verify pruning events are emitted for rejected nodes."""
        gen1 = make_generation_response(
            [
                ("pruned-node", False),
                ("good-node", True),
            ]
        )

        class _RejectFirstEvaluator:
            def __init__(self) -> None:
                self._call_count = 0

            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                self._call_count += 1
                if self._call_count == 1:
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REJECT,
                        score=0.1,
                        evaluator_name="test-evaluator",
                    )
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=0.9,
                    evaluator_name="test-evaluator",
                )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_RejectFirstEvaluator(),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
        )

        await agent.run("Question")

        prune_events = [e for e in emitter.events if isinstance(e, TreeSearchNodePrunedEvent)]
        assert len(prune_events) == 1


# ──────────────────────────────────────────────────────────
# ThoughtNode Model Tests
# ──────────────────────────────────────────────────────────


class TestThoughtNode:
    def test_default_values(self) -> None:
        node = ThoughtNode(content="test")
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.depth == 0
        assert node.score is None
        assert node.is_terminal is False
        assert node.is_pruned is False
        assert len(node.id) > 0

    def test_frozen(self) -> None:
        node = ThoughtNode(content="test")
        with pytest.raises(ValidationError):
            node.content = "modified"


# ──────────────────────────────────────────────────────────
# GenerationResponse Model Tests
# ──────────────────────────────────────────────────────────


class TestGenerationResponse:
    def test_parse_json(self) -> None:
        data = {
            "candidates": [
                {"reasoning": "path A", "is_complete": False},
                {"reasoning": "path B", "is_complete": True},
            ]
        }
        resp = _GenerationResponse.model_validate_json(json.dumps(data))
        assert len(resp.candidates) == 2
        assert resp.candidates[0].reasoning == "path A"
        assert resp.candidates[0].is_complete is False
        assert resp.candidates[1].is_complete is True


# ──────────────────────────────────────────────────────────
# min_depth Tests
# ──────────────────────────────────────────────────────────


class TestMinDepth:
    async def test_min_depth_suppresses_terminal_at_depth_1(self) -> None:
        """With min_depth=2, terminal candidates at depth 1 are suppressed."""
        # Root generates children at depth 1 — one marked complete
        gen1 = make_generation_response(
            [
                ("shallow-answer", True),
                ("keep-going", False),
            ]
        )
        # Depth-1 nodes are not terminal, so both can be expanded.
        # Expand the non-pruned nodes at depth 1 -> produce depth 2 children
        gen2 = make_generation_response(
            [
                ("deeper-answer", True),
            ]
        )
        gen3 = make_generation_response(
            [
                ("another-deeper", True),
            ]
        )

        client = MockLLMClient(
            [
                make_response(gen1),
                make_response(gen2),
                make_response(gen3),
            ]
        )
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think step by step",
            node_evaluator=_AcceptEvaluator(score=0.9),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
            min_depth=2,
        )

        result = await agent.run("Question")

        # Should NOT return "shallow-answer" (depth 1, suppressed)
        assert result.output != "shallow-answer"
        assert result.termination_reason == "no_expandable_nodes"

        # Verify terminal_suppressed event was emitted
        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        suppressed_events = [e for e in created_events if e.terminal_suppressed is True]
        assert len(suppressed_events) >= 1
        # The suppressed node should have is_terminal=False
        for ev in suppressed_events:
            assert ev.is_terminal is False

    async def test_min_depth_allows_terminal_at_threshold(self) -> None:
        """With min_depth=2, terminal candidates at depth 2 are allowed."""
        # Root -> depth 1 (non-terminal)
        gen1 = make_generation_response(
            [
                ("intermediate", False),
            ]
        )
        # depth 1 -> depth 2 (terminal allowed)
        gen2 = make_generation_response(
            [
                ("final-answer", True),
            ]
        )

        client = MockLLMClient([make_response(gen1), make_response(gen2)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think step by step",
            node_evaluator=_AcceptEvaluator(score=0.9),
            search_strategy=SearchStrategy.BFS,
            branching_factor=1,
            max_depth=5,
            max_nodes=50,
            min_depth=2,
        )

        result = await agent.run("Question")

        assert result.output == "final-answer"
        assert result.termination_reason == "no_expandable_nodes"

    def test_min_depth_greater_than_max_depth_raises(self) -> None:
        """min_depth > max_depth should raise ValueError."""
        with pytest.raises(ValueError, match=r"min_depth.*must be <= max_depth"):
            TreeOfThoughtAgent(
                name="test-tot",
                llm_client=MockLLMClient([]),
                emitter=make_emitter(),
                system_prompt="Think",
                node_evaluator=_AcceptEvaluator(),
                min_depth=6,
                max_depth=5,
            )

    async def test_min_depth_zero_preserves_existing_behavior(self) -> None:
        """min_depth=0 (default) should not suppress any terminals."""
        gen1 = make_generation_response(
            [
                ("immediate-answer", True),
            ]
        )

        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_AcceptEvaluator(score=0.9),
            search_strategy=SearchStrategy.BFS,
            branching_factor=1,
            max_depth=5,
            max_nodes=50,
            min_depth=0,
        )

        result = await agent.run("Question")

        assert result.output == "immediate-answer"
        assert result.termination_reason == "no_expandable_nodes"

        # No suppression events
        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        suppressed_events = [e for e in created_events if e.terminal_suppressed is True]
        assert len(suppressed_events) == 0

    async def test_generation_prompt_includes_depth_context(self) -> None:
        """Generation prompt should include depth information and min_depth guidance."""
        gen1 = make_generation_response(
            [
                ("thought", False),
            ]
        )
        gen2 = make_generation_response(
            [
                ("deeper-thought", True),
            ]
        )

        client = MockLLMClient([make_response(gen1), make_response(gen2)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_AcceptEvaluator(score=0.9),
            search_strategy=SearchStrategy.BFS,
            branching_factor=1,
            max_depth=5,
            max_nodes=50,
            min_depth=2,
        )

        await agent.run("Question")

        # Inspect the LLM calls to verify depth context was included
        # First call generates depth-1 children (below min_depth)
        first_call_messages = client.calls[0]["messages"]
        last_user_msg = [m for m in first_call_messages if m.role == "user"][-1]
        assert "step 1" in last_user_msg.content
        assert "Do not mark any continuation as complete" in last_user_msg.content

        # Second call generates depth-2 children (at min_depth)
        second_call_messages = client.calls[1]["messages"]
        last_user_msg_2 = [m for m in second_call_messages if m.role == "user"][-1]
        assert "step 2" in last_user_msg_2.content
        assert "Mark a continuation as complete" in last_user_msg_2.content


# ──────────────────────────────────────────────────────────
# Continue-After-Accept Tests
# ──────────────────────────────────────────────────────────


class TestContinueAfterAccept:
    async def test_multiple_accepted_terminals_best_selected(self) -> None:
        """When multiple terminals are found, the highest-scoring one is selected."""
        # Root generates 2 non-terminal children
        gen1 = make_generation_response(
            [
                ("path-A", False),
                ("path-B", False),
            ]
        )
        # Expand path-A → terminal with lower score
        gen2 = make_generation_response(
            [
                ("answer-A", True),
            ]
        )
        # Expand path-B → terminal with higher score
        gen3 = make_generation_response(
            [
                ("answer-B", True),
            ]
        )

        scores = {"path-A": 0.5, "path-B": 0.6, "answer-A": 0.7, "answer-B": 0.95}
        client = MockLLMClient([make_response(gen1), make_response(gen2), make_response(gen3)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_ScoringEvaluator(scores),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
        )

        result = await agent.run("Question")

        # Best accepted terminal should be "answer-B" (score 0.95)
        assert result.output == "answer-B"
        assert result.termination_reason == "no_expandable_nodes"

        # Verify accepted_count
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert complete_events[0].accepted_count == 2

    async def test_no_accepted_terminals_falls_back(self) -> None:
        """When no terminals are found, falls back to _select_best_node()."""
        gen1 = make_generation_response(
            [
                ("thought-A", False),
            ]
        )

        scores = {"thought-A": 0.6}
        client = MockLLMClient([make_response(gen1)])
        emitter = make_emitter()

        agent = TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_ScoringEvaluator(scores),
            search_strategy=SearchStrategy.BFS,
            branching_factor=1,
            max_depth=1,
            max_nodes=50,
        )

        result = await agent.run("Question")

        # No terminals, falls back to best non-root node
        assert result.output == "thought-A"
        assert result.termination_reason == "no_expandable_nodes"

        # Verify accepted_count is 0
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert complete_events[0].accepted_count == 0


# ──────────────────────────────────────────────────────────
# Internal Method Unit Tests
# ──────────────────────────────────────────────────────────


class TestToTInternals:
    def _make_agent(self) -> TreeOfThoughtAgent:
        client = MockLLMClient([])
        emitter = make_emitter()
        return TreeOfThoughtAgent(
            name="test-tot",
            llm_client=client,
            emitter=emitter,
            system_prompt="Think",
            node_evaluator=_ScoringEvaluator({}),
            search_strategy=SearchStrategy.BFS,
            branching_factor=2,
            max_depth=5,
            max_nodes=50,
        )

    def test_select_best_node_empty_nodes(self) -> None:
        """_select_best_node returns None when _nodes is empty."""
        agent = self._make_agent()
        agent._nodes = {}
        assert agent._select_best_node() is None

    def test_select_best_node_with_candidate_ids(self) -> None:
        """_select_best_node with candidate_ids selects from that set."""
        agent = self._make_agent()
        root = ThoughtNode(id="root", content="root", depth=0, score=0.1)
        t1 = ThoughtNode(id="t1", content="thought A", parent_id="root", depth=1, is_terminal=True, score=0.6)
        t2 = ThoughtNode(id="t2", content="thought B", parent_id="root", depth=1, is_terminal=True, score=0.9)
        agent._nodes = {"root": root, "t1": t1, "t2": t2}

        result = agent._select_best_node(candidate_ids={"t1", "t2"})
        assert result is not None
        assert result.id == "t2"

    def test_select_best_node_fallback_to_candidates(self) -> None:
        """_select_best_node falls back to non-root, non-pruned nodes when no terminals exist."""
        agent = self._make_agent()
        root = ThoughtNode(id="root", content="root", depth=0, score=0.1)
        n1 = ThoughtNode(id="n1", content="thought A", parent_id="root", depth=1, score=0.7)
        n2 = ThoughtNode(id="n2", content="thought B", parent_id="root", depth=1, score=0.4)
        agent._nodes = {"root": root, "n1": n1, "n2": n2}

        result = agent._select_best_node()
        assert result is not None
        assert result.id == "n1"

    def test_select_best_node_terminals_without_candidate_ids(self) -> None:
        """_select_best_node picks best terminal when no candidate_ids given."""
        agent = self._make_agent()
        root = ThoughtNode(id="root", content="root", depth=0, score=0.1)
        t1 = ThoughtNode(id="t1", content="A", parent_id="root", depth=1, is_terminal=True, score=0.6)
        t2 = ThoughtNode(id="t2", content="B", parent_id="root", depth=1, is_terminal=True, score=0.9)
        agent._nodes = {"root": root, "t1": t1, "t2": t2}

        result = agent._select_best_node()
        assert result is not None
        assert result.id == "t2"

    def test_select_best_node_fallback_to_root(self) -> None:
        """_select_best_node falls back to root when all non-root nodes are pruned."""
        agent = self._make_agent()
        root = ThoughtNode(id="root", content="root", depth=0, score=0.1)
        n1 = ThoughtNode(id="n1", content="pruned A", parent_id="root", depth=1, score=0.7, is_pruned=True)
        n2 = ThoughtNode(id="n2", content="pruned B", parent_id="root", depth=1, score=0.9, is_pruned=True)
        agent._nodes = {"root": root, "n1": n1, "n2": n2}

        result = agent._select_best_node()
        assert result is not None
        assert result.id == "root"
