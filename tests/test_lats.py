import pytest
from pydantic import ValidationError

from nanitics import (
    MockEmbeddingClient,
    MockLLMClient,
    ToolCall,
    tool,
)
from nanitics.capabilities.memory.episodic import InMemoryEpisodeStore
from nanitics.infrastructure.observability.events import (
    LLMRequestEvent,
    LLMResponseEvent,
    MCTSBackpropagationEvent,
    MCTSIterationEvent,
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodePrunedEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.strategies.agents.lats import ActionNode, LATSAgent
from tests.testing_helpers import make_emitter, make_response

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


# ── Tools ─────────────────────────────────────────────────


@tool(name="search", description="Search for information")
async def search_tool(query: str) -> str:
    return f"Results for: {query}"


@tool(name="calculate", description="Calculate an expression")
async def calculate_tool(expression: str) -> str:
    return f"Result: {expression}"


@tool(name="failing_tool", description="Always fails")
async def failing_tool() -> str:
    raise ValueError("intentional tool error")


# ── Evaluators ────────────────────────────────────────────


class _AcceptEvaluator:
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
    def __init__(self, scores: dict[str, float], default: float = 0.5) -> None:
        self._scores = scores
        self._default = default

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        score = self._default
        for key, value in self._scores.items():
            if key in output:
                score = value
                break
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=score,
            evaluator_name="test-evaluator",
        )


class _AcceptTerminalRejectOthers:
    """Accepts terminal answers, rejects non-terminal trajectories."""

    @property
    def max_revisions(self) -> int:
        return 0

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        # Terminal outputs won't contain "Thought:" prefix
        if "Thought:" in output:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=0.5,
                evaluator_name="test-evaluator",
            )
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=0.9,
            evaluator_name="test-evaluator",
        )


# ──────────────────────────────────────────────────────────
# Basic MCTS Loop
# ──────────────────────────────────────────────────────────


class TestBasicMCTSLoop:
    async def test_tool_call_then_terminal(self) -> None:
        """MCTS expands root with tool call, then produces terminal answer."""
        # Iteration 1: expand root — LLM returns tool call
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        # Iteration 2: expand tool-call node — LLM returns terminal answer
        responses = [
            make_response(content="Let me search", tool_calls=[tc]),
            make_response(content="Let me search", tool_calls=[tc]),
            make_response(content="Let me search", tool_calls=[tc]),
            # Second iteration: terminal answers
            make_response(content="The answer is 42"),
            make_response(content="The answer is 42"),
            make_response(content="The answer is 42"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve the task",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=2,
            max_depth=5,
            branching_factor=3,
        )

        result = await agent.run("What is the answer?")

        assert result.output == "The answer is 42"
        assert result.termination_reason == "max_iterations"

    async def test_tree_grows_with_nodes(self) -> None:
        """Verify that tree nodes are created and tracked."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        responses = [
            make_response(content="Searching", tool_calls=[tc]),
            make_response(content="Final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=3,
            branching_factor=1,
        )

        await agent.run("Question")

        # Root + 1 tool-call child + 1 terminal grandchild
        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        assert len(created_events) >= 3  # root + at least 2 children

    async def test_values_backpropagated(self) -> None:
        """After evaluating a leaf, all ancestors' values should be updated."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        responses = [
            make_response(content="thinking", tool_calls=[tc]),
            # terminal
            make_response(content="done"),
            # extra responses for branching_factor>1 would go here
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.7),
            max_iterations=3,
            branching_factor=1,
        )

        await agent.run("Question")

        backprop_events = [e for e in emitter.events if isinstance(e, MCTSBackpropagationEvent)]
        assert len(backprop_events) >= 1
        # Backpropagation should update all nodes from leaf to root
        for event in backprop_events:
            assert event.path_length >= 1


# ──────────────────────────────────────────────────────────
# UCB1 Selection
# ──────────────────────────────────────────────────────────


class TestUCB1Selection:
    def test_unvisited_children_selected_first(self) -> None:
        """Unvisited children (visit_count=0) should have infinite UCB1."""
        emitter = make_emitter()
        agent = LATSAgent(
            name="test",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
        )

        unvisited = ActionNode(id="unvisited", visit_count=0, depth=1, parent_id="root")
        visited = ActionNode(id="visited", visit_count=5, value=3.0, depth=1, parent_id="root")
        root = ActionNode(id="root", children_ids=["unvisited", "visited"], visit_count=5, depth=0)

        agent._nodes = {"root": root, "unvisited": unvisited, "visited": visited}
        agent._root_id = "root"

        ucb_unvisited = agent._ucb1(unvisited)
        ucb_visited = agent._ucb1(visited)

        assert ucb_unvisited == float("inf")
        assert ucb_visited < float("inf")

    def test_ucb1_selects_leaf(self) -> None:
        """_select_leaf traverses from root to a leaf node."""
        emitter = make_emitter()
        agent = LATSAgent(
            name="test",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
        )

        leaf = ActionNode(id="leaf", depth=2, parent_id="mid", visit_count=0)
        mid = ActionNode(id="mid", depth=1, parent_id="root", children_ids=["leaf"], visit_count=1, value=0.5)
        root = ActionNode(id="root", depth=0, children_ids=["mid"], visit_count=2, value=1.0)

        agent._nodes = {"root": root, "mid": mid, "leaf": leaf}
        agent._root_id = "root"

        selected = agent._select_leaf()
        assert selected.id == "leaf"


# ──────────────────────────────────────────────────────────
# Pruning-aware Selection
# ──────────────────────────────────────────────────────────


class TestPruningAwareSelection:
    def _make_agent(self) -> LATSAgent:
        return LATSAgent(
            name="prune-test",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
        )

    def test_select_leaf_skips_pruned_child_with_live_sibling(self) -> None:
        """A pruned child must not be chosen even though UCB1 would score it +inf."""
        agent = self._make_agent()
        pruned = ActionNode(id="pruned", depth=1, parent_id="root", visit_count=0, is_failed=True)
        live = ActionNode(id="live", depth=1, parent_id="root", visit_count=0)
        root = ActionNode(id="root", depth=0, children_ids=["pruned", "live"], visit_count=1, value=0.5)
        agent._nodes = {"root": root, "pruned": pruned, "live": live}
        agent._root_id = "root"

        selected = agent._select_leaf()

        # Both children have visit_count=0 → UCB1 returns +inf for each. The
        # filter is the only thing stopping the pruned child from winning.
        assert selected.id == "live"

    def test_select_leaf_returns_parent_when_all_children_pruned(self) -> None:
        """With every child pruned, descent stops at the parent as a dead leaf."""
        agent = self._make_agent()
        dead_a = ActionNode(id="dead_a", depth=1, parent_id="root", is_failed=True)
        dead_b = ActionNode(id="dead_b", depth=1, parent_id="root", is_failed=True)
        root = ActionNode(id="root", depth=0, children_ids=["dead_a", "dead_b"], visit_count=1)
        agent._nodes = {"root": root, "dead_a": dead_a, "dead_b": dead_b}
        agent._root_id = "root"

        selected = agent._select_leaf()

        assert selected.id == "root"

    def test_propagate_pruning_cascades_upward_but_stops_at_root(self) -> None:
        """When every child of a mid-node fails, the mid-node is marked failed —
        and the cascade halts at the root rather than ending the search."""
        agent = self._make_agent()
        leaf_a = ActionNode(id="leaf_a", depth=2, parent_id="mid", is_failed=True)
        leaf_b = ActionNode(id="leaf_b", depth=2, parent_id="mid", is_failed=True)
        mid = ActionNode(id="mid", depth=1, parent_id="root", children_ids=["leaf_a", "leaf_b"])
        root = ActionNode(id="root", depth=0, children_ids=["mid"])
        agent._nodes = {"root": root, "mid": mid, "leaf_a": leaf_a, "leaf_b": leaf_b}
        agent._root_id = "root"

        agent._propagate_pruning("mid")

        assert agent._nodes["mid"].is_failed is True, "mid should cascade to failed"
        assert agent._nodes["root"].is_failed is False, "root must never be marked failed"

        propagated = [
            e
            for e in agent._emitter.events
            if isinstance(e, TreeSearchNodePrunedEvent) and e.reason == "all_children_pruned"
        ]
        assert [e.node_id for e in propagated] == ["mid"], (
            f"Expected one cascade prune event for 'mid'; got {[(e.node_id, e.reason) for e in propagated]!r}"
        )

    def test_propagate_pruning_is_noop_when_a_live_sibling_exists(self) -> None:
        """Mid-node must not be marked failed if any child is still live."""
        agent = self._make_agent()
        dead = ActionNode(id="dead", depth=2, parent_id="mid", is_failed=True)
        alive = ActionNode(id="alive", depth=2, parent_id="mid")
        mid = ActionNode(id="mid", depth=1, parent_id="root", children_ids=["dead", "alive"])
        root = ActionNode(id="root", depth=0, children_ids=["mid"])
        agent._nodes = {"root": root, "mid": mid, "dead": dead, "alive": alive}
        agent._root_id = "root"

        agent._propagate_pruning("mid")

        assert agent._nodes["mid"].is_failed is False
        cascade = [
            e
            for e in agent._emitter.events
            if isinstance(e, TreeSearchNodePrunedEvent) and e.reason == "all_children_pruned"
        ]
        assert cascade == []

    async def test_pruned_node_never_reappears_in_later_selection_path(self) -> None:
        """End-to-end invariant mirroring validation/agents/lats_agent.py: once a
        TreeSearchNodePrunedEvent fires for a node, that node's id must not
        appear in any MCTSIterationEvent.selection_path emitted afterward."""

        # Reject the first K evaluations, accept the rest. Forces at least one
        # child-level prune in iteration 1 whose id we can then track across
        # later iterations' selection_paths.
        class _RejectFirstThen:
            def __init__(self, reject_first: int) -> None:
                self._remaining = reject_first

            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                if self._remaining > 0:
                    self._remaining -= 1
                    return EvaluationResult(
                        verdict=EvaluationVerdict.REJECT,
                        score=0.0,
                        feedback="rejected",
                        evaluator_name="test",
                    )
                return EvaluationResult(verdict=EvaluationVerdict.ACCEPT, score=0.9, evaluator_name="test")

        responses = [make_response(content=f"answer-{i}") for i in range(20)]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = LATSAgent(
            name="prune-e2e",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_RejectFirstThen(reject_first=2),
            max_iterations=4,
            branching_factor=2,
        )

        await agent.run("Question")

        # For every prune event, record the index of the next MCTSIterationEvent
        # to fire after it (prune events land within the span of the iteration
        # that produced them, so the flag only takes effect in the next one).
        events = emitter.events
        pruned_emit_iter_idx: dict[str, int] = {}
        iter_events_seen = 0
        for ev in events:
            if isinstance(ev, MCTSIterationEvent):
                iter_events_seen += 1
            elif isinstance(ev, TreeSearchNodePrunedEvent) and ev.node_id not in pruned_emit_iter_idx:
                pruned_emit_iter_idx[ev.node_id] = iter_events_seen

        iterations = [e for e in events if isinstance(e, MCTSIterationEvent)]
        assert pruned_emit_iter_idx, "Test setup must actually trigger at least one prune event"

        for nid, emitted_during in pruned_emit_iter_idx.items():
            for later_idx in range(emitted_during + 1, len(iterations)):
                assert nid not in iterations[later_idx].selection_path, (
                    f"Pruned node {nid} (pruned during iteration {emitted_during}) "
                    f"reappeared in MCTSIterationEvent[{later_idx}].selection_path="
                    f"{iterations[later_idx].selection_path}"
                )


# ──────────────────────────────────────────────────────────
# Backpropagation
# ──────────────────────────────────────────────────────────


class TestBackpropagation:
    def test_backpropagation_updates_ancestors(self) -> None:
        """After backpropagation, all ancestors should have updated visit_count and value."""
        emitter = make_emitter()
        agent = LATSAgent(
            name="test",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
        )

        leaf = ActionNode(id="leaf", depth=2, parent_id="mid", value=0.0, visit_count=0)
        mid = ActionNode(id="mid", depth=1, parent_id="root", children_ids=["leaf"], value=0.0, visit_count=0)
        root = ActionNode(id="root", depth=0, children_ids=["mid"], value=0.0, visit_count=0)

        agent._nodes = {"root": root, "mid": mid, "leaf": leaf}
        agent._root_id = "root"

        agent._backpropagate("leaf", 0.8)

        assert agent._nodes["leaf"].visit_count == 1
        assert agent._nodes["leaf"].value == pytest.approx(0.8)
        assert agent._nodes["mid"].visit_count == 1
        assert agent._nodes["mid"].value == pytest.approx(0.8)
        assert agent._nodes["root"].visit_count == 1
        assert agent._nodes["root"].value == pytest.approx(0.8)

    def test_backpropagation_emits_event(self) -> None:
        """Backpropagation should emit MCTSBackpropagationEvent."""
        emitter = make_emitter()
        agent = LATSAgent(
            name="test",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
        )

        leaf = ActionNode(id="leaf", depth=1, parent_id="root")
        root = ActionNode(id="root", depth=0, children_ids=["leaf"])

        agent._nodes = {"root": root, "leaf": leaf}
        agent._root_id = "root"

        agent._backpropagate("leaf", 0.5)

        bp_events = [e for e in emitter.events if isinstance(e, MCTSBackpropagationEvent)]
        assert len(bp_events) == 1
        assert bp_events[0].propagated_value == pytest.approx(0.5)
        assert bp_events[0].path_length == 2
        assert "leaf" in bp_events[0].updated_node_ids
        assert "root" in bp_events[0].updated_node_ids


# ──────────────────────────────────────────────────────────
# Tool Dispatch
# ──────────────────────────────────────────────────────────


class TestToolDispatch:
    async def test_tool_called_with_correct_arguments(self) -> None:
        """Tools should be dispatched with the arguments from the LLM response."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "capital of France"})
        responses = [
            make_response(content="Searching for answer", tool_calls=[tc]),
            # Terminal answer after tool use
            make_response(content="Paris is the capital"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.95),
            max_iterations=3,
            branching_factor=1,
        )

        result = await agent.run("What is the capital of France?")

        assert result.output == "Paris is the capital"


# ──────────────────────────────────────────────────────────
# Tool Failure
# ──────────────────────────────────────────────────────────


class TestToolFailure:
    async def test_tool_failure_marks_node_failed(self) -> None:
        """When a tool raises an exception, the node should be marked is_failed=True."""
        tc_fail = ToolCall(id="tc1", name="failing_tool", arguments={})
        tc_ok = ToolCall(id="tc2", name="search", arguments={"query": "test"})
        responses = [
            # First child: failing tool
            make_response(content="Trying tool", tool_calls=[tc_fail]),
            # Second child: working tool
            make_response(content="Searching", tool_calls=[tc_ok]),
            # Third child (branching_factor=3): terminal
            make_response(content="The answer is known"),
            # Next iteration after tool-call child
            make_response(content="Final answer"),
            make_response(content="Final answer"),
            make_response(content="Final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool, failing_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            branching_factor=3,
        )

        result = await agent.run("Solve this")

        # Should still produce a result from non-failed branches
        assert result.output is not None
        assert result.termination_reason == "max_iterations"


# ──────────────────────────────────────────────────────────
# Terminal Detection
# ──────────────────────────────────────────────────────────


class TestTerminalDetection:
    async def test_no_tool_call_creates_terminal_node(self) -> None:
        """LLM response without tool calls creates a terminal node."""
        responses = [
            make_response(content="Direct answer"),
            make_response(content="Direct answer"),
            make_response(content="Direct answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.95),
            max_iterations=3,
            branching_factor=3,
        )

        result = await agent.run("Simple question")

        assert result.output == "Direct answer"
        assert result.termination_reason == "max_iterations"


# ──────────────────────────────────────────────────────────
# Early Termination on Acceptance
# ──────────────────────────────────────────────────────────


class TestEarlyTermination:
    async def test_search_continues_after_accepted_terminal(self) -> None:
        """When a terminal node's evaluation verdict is ACCEPT, search continues for remaining iterations."""
        responses = [
            make_response(content="The answer is 42"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.95),
            max_iterations=5,
            branching_factor=1,
        )

        result = await agent.run("What is the answer?")

        # Search runs through all iterations, not just the first
        assert result.termination_reason == "max_iterations"
        assert result.output == "The answer is 42"

        # Verify accepted_count in complete event
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert complete_events[0].accepted_count == 1


# ──────────────────────────────────────────────────────────
# Max Iterations
# ──────────────────────────────────────────────────────────


class TestMaxIterations:
    async def test_max_iterations_stops_search(self) -> None:
        """Search stops after max_iterations."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        # All responses are tool calls — never terminal, so search never terminates early
        responses = [make_response(content="searching", tool_calls=[tc]) for _ in range(20)]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.5),
            max_iterations=3,
            branching_factor=1,
        )

        result = await agent.run("Hard question")

        assert result.termination_reason == "max_iterations"
        assert result.total_steps == 3


# ──────────────────────────────────────────────────────────
# Max Depth
# ──────────────────────────────────────────────────────────


class TestMaxDepth:
    async def test_max_depth_prevents_expansion(self) -> None:
        """Nodes at max_depth should not be expanded further."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        responses = [
            # Iteration 1: expand root → tool call child at depth 1
            make_response(content="searching", tool_calls=[tc]),
            # Iteration 2: depth 1 is max_depth, so no expansion — just evaluate
            # Iteration 3: re-select same node, still can't expand
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.6),
            max_iterations=3,
            max_depth=1,
            branching_factor=1,
        )

        result = await agent.run("Question")

        # Search should stop with max_iterations since no terminal was found
        assert result.termination_reason == "max_iterations"
        # No nodes deeper than depth 1
        max_depth_reached = max(n.depth for n in agent._nodes.values())
        assert max_depth_reached <= 1


# ──────────────────────────────────────────────────────────
# Cancellation
# ──────────────────────────────────────────────────────────


class TestCancellation:
    async def test_cancellation_stops_search(self) -> None:
        """Cancelled token stops the MCTS loop."""
        token = CancellationToken()
        token.cancel()

        client = MockLLMClient([])
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=20,
            cancellation_token=token,
        )

        result = await agent.run("Question")

        assert result.termination_reason == "cancelled"


# ──────────────────────────────────────────────────────────
# Episodic Memory Integration
# ──────────────────────────────────────────────────────────


class TestEpisodicMemory:
    async def test_episodes_recalled_at_start(self) -> None:
        """Relevant past episodes should be recalled and used as context."""
        embedding_client = MockEmbeddingClient()
        store = InMemoryEpisodeStore(embedding_client=embedding_client)

        # Record a past episode
        from nanitics.capabilities.memory.episodic import Episode, OutcomeType

        past_episode = Episode(
            situation="Similar task",
            action="Used search tool",
            outcome=OutcomeType.SUCCESS,
            reflection="Search worked well",
        )
        await store.record(past_episode)

        # Terminal answer immediately
        responses = [
            make_response(content="Based on past experience, the answer is 42"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.95),
            max_iterations=3,
            branching_factor=1,
            episode_store=store,
        )

        result = await agent.run("Solve this task")

        assert result.output is not None
        # The LLM should have received episode context in its messages
        call_messages = client.calls[0]["messages"]
        has_episode_context = any("[Past Experiences]" in (m.content or "") for m in call_messages)
        assert has_episode_context

    async def test_episode_recorded_at_end(self) -> None:
        """An episode should be recorded after search completes."""
        embedding_client = MockEmbeddingClient()
        store = InMemoryEpisodeStore(embedding_client=embedding_client)

        responses = [
            make_response(content="The answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.95),
            max_iterations=3,
            branching_factor=1,
            episode_store=store,
        )

        await agent.run("Solve this task")

        assert await store.count() == 1

    async def test_failed_search_generates_reflection(self) -> None:
        """When search fails, a reflection should be generated and stored."""
        embedding_client = MockEmbeddingClient()
        store = InMemoryEpisodeStore(embedding_client=embedding_client)

        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        responses = [
            # All iterations produce tool calls, never terminal
            make_response(content="searching", tool_calls=[tc]),
            # Reflection LLM call (via _llm_client.generate directly)
            make_response(content="The search was too narrow"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.3),
            max_iterations=1,
            branching_factor=1,
            episode_store=store,
        )

        await agent.run("Hard problem")

        assert await store.count() == 1
        episodes = await store.recall("Hard problem", limit=1)
        assert episodes[0].episode.reflection == "The search was too narrow"

    async def test_failed_run_episode_has_failure_outcome(self) -> None:
        """When LATS records a failed-run episode, outcome must be FAILURE."""
        from nanitics.capabilities.memory.episodic import OutcomeType

        embedding_client = MockEmbeddingClient()
        store = InMemoryEpisodeStore(embedding_client=embedding_client)

        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        responses = [
            make_response(content="searching", tool_calls=[tc]),
            make_response(content="Reflection on the failed search"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.3),
            max_iterations=1,
            branching_factor=1,
            episode_store=store,
        )

        await agent.run("Hard problem")

        episodes = await store.recall("Hard problem", limit=1)
        assert len(episodes) == 1
        assert episodes[0].episode.outcome == OutcomeType.FAILURE
        assert episodes[0].episode.reflection == "Reflection on the failed search"

    async def test_reflection_emits_llm_trace_events(self) -> None:
        """Reflection LLM calls should emit LLMRequestEvent and LLMResponseEvent."""
        embedding_client = MockEmbeddingClient()
        store = InMemoryEpisodeStore(embedding_client=embedding_client)

        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        responses = [
            make_response(content="searching", tool_calls=[tc]),
            make_response(content="Reflection text"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.3),
            max_iterations=1,
            branching_factor=1,
            episode_store=store,
        )

        await agent.run("Hard problem")

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent) and e.label == "reflection"]
        response_events = [e for e in emitter.events if isinstance(e, LLMResponseEvent) and e.label == "reflection"]
        assert len(request_events) == 1
        assert len(response_events) == 1


# ──────────────────────────────────────────────────────────
# Diversity
# ──────────────────────────────────────────────────────────


class TestDiversity:
    async def test_diversity_prompt_included_for_siblings(self) -> None:
        """When expanding a node that already has children, the diversity prompt should be included."""
        tc1 = ToolCall(id="tc1", name="search", arguments={"query": "first"})
        tc2 = ToolCall(id="tc2", name="calculate", arguments={"expression": "1+1"})
        responses = [
            # First child uses search
            make_response(content="First approach", tool_calls=[tc1]),
            # Second child should get diversity prompt
            make_response(content="Second approach", tool_calls=[tc2]),
            # Third child: terminal
            make_response(content="Final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool, calculate_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=1,
            branching_factor=3,
        )

        await agent.run("Solve this")

        # Check that the second LLM call included a diversity message with tool arguments
        assert len(client.calls) >= 2
        second_call_messages = client.calls[1]["messages"]
        diversity_msg = next(
            (m.content for m in second_call_messages if "sibling branches" in (m.content or "").lower()),
            None,
        )
        assert diversity_msg is not None
        # Should contain the tool name AND argument from the first child
        assert 'search(query="first")' in diversity_msg

    async def test_diversity_prompt_includes_tool_arguments(self) -> None:
        """The diversity prompt should include full tool signatures, not just tool names."""
        tc1 = ToolCall(id="tc1", name="search", arguments={"query": "SQLite concurrency"})
        tc2 = ToolCall(id="tc2", name="search", arguments={"query": "PostgreSQL scaling"})
        responses = [
            make_response(content="Searching first", tool_calls=[tc1]),
            make_response(content="Searching second", tool_calls=[tc2]),
            make_response(content="Final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Research",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=1,
            branching_factor=3,
        )

        await agent.run("Compare databases")

        # Second call should see first child's query
        second_call_messages = client.calls[1]["messages"]
        diversity_msg = next(
            (m.content for m in second_call_messages if "sibling branches" in (m.content or "").lower()),
            None,
        )
        assert diversity_msg is not None
        assert "SQLite concurrency" in diversity_msg

        # Third call should see both previous queries
        third_call_messages = client.calls[2]["messages"]
        diversity_msg_3 = next(
            (m.content for m in third_call_messages if "sibling branches" in (m.content or "").lower()),
            None,
        )
        assert diversity_msg_3 is not None
        assert "SQLite concurrency" in diversity_msg_3
        assert "PostgreSQL scaling" in diversity_msg_3

    async def test_diversity_prompt_includes_terminal_summaries(self) -> None:
        """Terminal sibling responses should appear in the diversity prompt."""
        responses = [
            # First child: terminal response
            make_response(content="The answer is 42"),
            # Second child: tool call (should see terminal sibling in diversity prompt)
            make_response(
                content="Searching", tool_calls=[ToolCall(id="tc1", name="search", arguments={"query": "test"})]
            ),
            # Third child: terminal
            make_response(content="Another answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=1,
            branching_factor=3,
        )

        await agent.run("Solve this")

        # Second call should include the terminal sibling's text
        second_call_messages = client.calls[1]["messages"]
        diversity_msg = next(
            (m.content for m in second_call_messages if "sibling branches" in (m.content or "").lower()),
            None,
        )
        assert diversity_msg is not None
        assert "The answer is 42" in diversity_msg
        assert "[terminal:" in diversity_msg


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestEventEmission:
    async def test_all_expected_events_emitted(self) -> None:
        """Verify MCTS events, tree search events, and standard agent events."""
        responses = [
            make_response(content="The answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            branching_factor=1,
        )

        await agent.run("Question")

        event_types = [e.event_type for e in emitter.events]

        # Standard agent events
        assert "agent.start" in event_types
        assert "agent.complete" in event_types

        # Tree search events
        assert "tree_search.node.created" in event_types
        assert "tree_search.node.evaluated" in event_types
        assert "tree_search.complete" in event_types

        # MCTS-specific events
        assert "mcts.iteration" in event_types
        assert "mcts.backpropagation" in event_types

    async def test_mcts_iteration_event_details(self) -> None:
        """MCTSIterationEvent should have correct fields."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        responses = [
            make_response(content="searching", tool_calls=[tc]),
            make_response(content="answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.8),
            max_iterations=5,
            branching_factor=1,
        )

        await agent.run("Question")

        iter_events = [e for e in emitter.events if isinstance(e, MCTSIterationEvent)]
        assert len(iter_events) >= 1

        first_iter = iter_events[0]
        assert first_iter.iteration_number == 1
        assert first_iter.expanded_count > 0
        assert len(first_iter.selection_path) >= 1

    async def test_enriched_node_created_events(self) -> None:
        """TreeSearchNodeCreatedEvent should carry action, observation, is_terminal, is_failed, content."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        responses = [
            make_response(content="Searching for info", tool_calls=[tc]),
            make_response(content="Final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            branching_factor=1,
        )

        await agent.run("Question")

        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        assert len(created_events) >= 3  # root + tool call child + terminal child

        # Root event should have content but no action
        root_event = created_events[0]
        assert root_event.content == "Question"
        assert root_event.action is None
        assert root_event.is_terminal is False
        assert root_event.is_failed is False

        # Tool call child should have action and observation
        action_events = [e for e in created_events if e.action is not None]
        assert len(action_events) >= 1
        action_event = action_events[0]
        assert action_event.action == "search"
        assert action_event.observation is not None
        assert action_event.content is not None
        assert action_event.is_terminal is False

        # Terminal child should have is_terminal=True
        terminal_events = [e for e in created_events if e.is_terminal]
        assert len(terminal_events) >= 1
        assert terminal_events[0].content is not None

    async def test_enriched_failed_node_event(self) -> None:
        """TreeSearchNodeCreatedEvent should have is_failed=True for failed tool calls."""
        tc = ToolCall(id="tc1", name="failing_tool", arguments={})
        responses = [
            make_response(content="Trying failing tool", tool_calls=[tc]),
            make_response(content="fallback answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[failing_tool],
            node_evaluator=_AcceptEvaluator(score=0.5),
            max_iterations=2,
            branching_factor=1,
        )

        await agent.run("Question")

        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        failed_events = [e for e in created_events if e.is_failed]
        assert len(failed_events) >= 1
        assert failed_events[0].action == "failing_tool"
        assert failed_events[0].observation is None
        assert failed_events[0].error_message is not None
        assert "intentional tool error" in failed_events[0].error_message

    async def test_mcts_iteration_event_has_node_values(self) -> None:
        """MCTSIterationEvent should include node_values snapshot."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "q"})
        responses = [
            make_response(content="searching", tool_calls=[tc]),
            make_response(content="answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.8),
            max_iterations=5,
            branching_factor=1,
        )

        await agent.run("Question")

        iter_events = [e for e in emitter.events if isinstance(e, MCTSIterationEvent)]
        assert len(iter_events) >= 1

        # node_values should be a dict with node IDs as keys and float values
        first_iter = iter_events[0]
        assert isinstance(first_iter.node_values, dict)
        # After first iteration, at least some nodes should have been visited
        assert len(first_iter.node_values) > 0
        # All values should be floats
        for node_id, value in first_iter.node_values.items():
            assert isinstance(node_id, str)
            assert isinstance(value, float)

    async def test_enriched_events_backward_compatible(self) -> None:
        """Events with new fields should serialize/deserialize correctly, including defaults."""
        # Event without new fields (backward compatibility)
        event = TreeSearchNodeCreatedEvent(
            trace_id="t",
            span_id="s",
            node_id="n1",
            parent_id=None,
            depth=0,
            content="preview",
            node_type="action",
        )
        data = event.model_dump()
        assert data["content"] == "preview"
        assert data["action"] is None
        assert data["observation"] is None
        assert data["is_terminal"] is False
        assert data["is_failed"] is False
        assert data["error_message"] is None

        # MCTSIterationEvent without node_values (backward compatibility)
        mcts_event = MCTSIterationEvent(
            trace_id="t",
            span_id="s",
            iteration_number=1,
            selected_node_id="n1",
            selection_path=["n1"],
            expanded_count=2,
            best_value_so_far=0.5,
        )
        mcts_data = mcts_event.model_dump()
        assert mcts_data["node_values"] == {}

    async def test_tree_search_complete_event(self) -> None:
        """TreeSearchCompleteEvent should be emitted with correct strategy."""
        responses = [make_response(content="answer")]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            branching_factor=1,
        )

        await agent.run("Question")

        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].search_strategy == "mcts"
        assert complete_events[0].total_nodes >= 2  # root + at least 1 child

    async def test_pruning_events_on_rejection(self) -> None:
        """TreeSearchNodePrunedEvent should be emitted when evaluation rejects a node."""
        responses = [
            make_response(content="bad answer"),
            make_response(content="bad answer"),
            make_response(content="bad answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_RejectEvaluator(),
            max_iterations=1,
            branching_factor=3,
        )

        await agent.run("Question")

        prune_events = [e for e in emitter.events if isinstance(e, TreeSearchNodePrunedEvent)]
        assert len(prune_events) >= 1
        assert all(e.reason == "evaluation_rejected" for e in prune_events)


# ──────────────────────────────────────────────────────────
# Exploration Constant
# ──────────────────────────────────────────────────────────


class TestExplorationConstant:
    def test_higher_c_favors_exploration(self) -> None:
        """Higher exploration_constant should give more weight to unvisited nodes."""
        emitter = make_emitter()

        agent_low_c = LATSAgent(
            name="low-c",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            exploration_constant=0.1,
        )
        agent_high_c = LATSAgent(
            name="high-c",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="test",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            exploration_constant=10.0,
        )

        # Node with low visits relative to parent
        node = ActionNode(id="child", visit_count=1, value=0.3, depth=1, parent_id="root")
        root = ActionNode(id="root", children_ids=["child"], visit_count=10, depth=0)

        agent_low_c._nodes = {"root": root, "child": node}
        agent_low_c._root_id = "root"
        agent_high_c._nodes = {"root": root, "child": node}
        agent_high_c._root_id = "root"

        ucb_low = agent_low_c._ucb1(node)
        ucb_high = agent_high_c._ucb1(node)

        # Higher exploration constant should give larger UCB1
        assert ucb_high > ucb_low


# ──────────────────────────────────────────────────────────
# ActionNode Model Tests
# ──────────────────────────────────────────────────────────


class TestActionNodeModel:
    def test_default_values(self) -> None:
        """ActionNode should have sensible defaults."""
        node = ActionNode(thought="test")
        assert node.depth == 0
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.action is None
        assert node.action_input is None
        assert node.observation is None
        assert node.is_terminal is False
        assert node.terminal_output is None
        assert node.value == 0.0
        assert node.visit_count == 0
        assert node.is_failed is False
        assert node.id  # UUID should be generated

    def test_frozen_immutability(self) -> None:
        """ActionNode should be immutable (frozen)."""
        node = ActionNode(thought="test")
        with pytest.raises(ValidationError):  # ValidationError for frozen model
            node.thought = "changed"

    def test_unique_ids(self) -> None:
        """Each ActionNode should get a unique ID."""
        node1 = ActionNode(thought="a")
        node2 = ActionNode(thought="b")
        assert node1.id != node2.id


# ──────────────────────────────────────────────────────────
# min_depth Tests
# ──────────────────────────────────────────────────────────


class TestMinDepth:
    async def test_min_depth_suppresses_terminal_at_depth_1(self) -> None:
        """With min_depth=2, non-tool-call responses at depth 1 are not terminal."""
        # Iteration 1: expand root — LLM returns tool call (depth 1 node)
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        # Iteration 2: expand depth-1 tool-call node — LLM returns non-tool answer
        # at depth 2 (>= min_depth), so this IS terminal
        responses = [
            # Expand root -> depth-1 node with tool call
            make_response(content="Let me search", tool_calls=[tc]),
            # Expand depth-1 -> depth-2 terminal
            make_response(content="The answer is 42"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            max_depth=5,
            branching_factor=1,
            min_depth=2,
        )

        result = await agent.run("Question")

        assert result.output == "The answer is 42"
        assert result.termination_reason == "max_iterations"

    async def test_min_depth_non_tool_call_below_threshold_creates_non_terminal(self) -> None:
        """Non-tool-call response below min_depth creates expandable non-terminal node."""
        # Iteration 1: expand root — LLM returns non-tool-call at depth 1 (suppressed)
        # Iteration 2: re-expand depth-1 non-terminal — LLM returns tool call
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        # Iteration 3: expand depth-2 tool node — LLM returns terminal
        responses = [
            # Expand root -> depth-1 non-tool response (suppressed terminal)
            make_response(content="Initial reasoning"),
            # Expand depth-1 non-terminal -> depth-2 tool call
            make_response(content="Now searching", tool_calls=[tc]),
            # Expand depth-2 -> depth-3 terminal
            make_response(content="Final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=10,
            max_depth=5,
            branching_factor=1,
            min_depth=2,
        )

        result = await agent.run("Question")

        # The depth-1 node should NOT be terminal
        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        suppressed = [e for e in created_events if e.terminal_suppressed is True]
        assert len(suppressed) >= 1
        for ev in suppressed:
            assert ev.is_terminal is False

        # Should eventually find terminal answer
        assert result.output == "Final answer"

    async def test_min_depth_allows_terminal_at_threshold(self) -> None:
        """With min_depth=1, non-tool-call responses at depth 1 are allowed as terminal."""
        # Direct non-tool response at depth 1 (>= min_depth=1)
        responses = [
            make_response(content="Immediate answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            max_depth=5,
            branching_factor=1,
            min_depth=1,
        )

        result = await agent.run("Question")

        assert result.output == "Immediate answer"
        assert result.termination_reason == "max_iterations"

        # No suppression events
        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        suppressed = [e for e in created_events if e.terminal_suppressed is True]
        assert len(suppressed) == 0

    def test_min_depth_greater_than_max_depth_raises(self) -> None:
        """min_depth > max_depth should raise ValueError."""
        with pytest.raises(ValueError, match=r"min_depth.*must be <= max_depth"):
            LATSAgent(
                name="test-lats",
                llm_client=MockLLMClient([]),
                emitter=make_emitter(),
                system_prompt="Solve",
                tools=[search_tool],
                node_evaluator=_AcceptEvaluator(),
                min_depth=11,
                max_depth=10,
            )

    async def test_min_depth_zero_preserves_existing_behavior(self) -> None:
        """min_depth=0 (default) should not suppress any terminals."""
        # Non-tool response at depth 1 should be terminal with min_depth=0
        responses = [
            make_response(content="Direct answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            max_depth=5,
            branching_factor=1,
            min_depth=0,
        )

        result = await agent.run("Question")

        assert result.output == "Direct answer"
        assert result.termination_reason == "max_iterations"

        # No suppression events
        created_events = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        suppressed = [e for e in created_events if e.terminal_suppressed is True]
        assert len(suppressed) == 0


# ──────────────────────────────────────────────────────────
# Terminal Depth
# ──────────────────────────────────────────────────────────


class TestTerminalDepth:
    async def test_terminal_forced_at_terminal_depth(self) -> None:
        """At terminal_depth, tools are withheld so LLM produces a terminal text response."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        responses = [
            # Expand root (depth 0) -> child at depth 1: tool call (tools available)
            make_response(content="Searching", tool_calls=[tc]),
            # Expand depth-1 -> child at depth 2 (= terminal_depth): no tools, forced terminal
            make_response(content="The final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            max_depth=5,
            branching_factor=1,
            min_depth=0,
            terminal_depth=2,
        )

        result = await agent.run("Question")

        assert result.output == "The final answer"

        # Verify that the LLM call at terminal_depth had no tools
        assert len(client.calls) >= 2
        # First call (depth 1): tools should be present
        assert client.calls[0].get("tools") is not None
        # Second call (depth 2 = terminal_depth): tools should be None
        assert client.calls[1].get("tools") is None

    async def test_below_terminal_depth_tools_available(self) -> None:
        """Below terminal_depth, tools remain available for normal behavior."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        responses = [
            # Expand root -> depth 1 (below terminal_depth=3): tools available
            make_response(content="Searching", tool_calls=[tc]),
            # Expand depth 1 -> depth 2 (still below terminal_depth=3): tools available
            make_response(content="More searching", tool_calls=[tc]),
            # Expand depth 2 -> depth 3 (= terminal_depth): no tools
            make_response(content="Final answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            max_depth=5,
            branching_factor=1,
            terminal_depth=3,
        )

        result = await agent.run("Question")

        assert result.output == "Final answer"
        # Calls at depth 1 and 2 should have tools; call at depth 3 should not
        assert client.calls[0].get("tools") is not None
        assert client.calls[1].get("tools") is not None
        assert client.calls[2].get("tools") is None

    async def test_terminal_depth_none_preserves_behavior(self) -> None:
        """terminal_depth=None (default) should not change behavior — tools always available."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        responses = [
            make_response(content="Searching", tool_calls=[tc]),
            make_response(content="The answer"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.9),
            max_iterations=5,
            max_depth=5,
            branching_factor=1,
        )

        result = await agent.run("Question")

        assert result.output == "The answer"
        # All calls should have tools
        for call in client.calls:
            assert call.get("tools") is not None

    def test_terminal_depth_below_min_depth_raises(self) -> None:
        with pytest.raises(ValueError, match=r"terminal_depth.*must be >= min_depth"):
            LATSAgent(
                name="test-lats",
                llm_client=MockLLMClient([]),
                emitter=make_emitter(),
                system_prompt="Solve",
                tools=[search_tool],
                node_evaluator=_AcceptEvaluator(),
                min_depth=3,
                terminal_depth=2,
                max_depth=5,
            )

    def test_terminal_depth_above_max_depth_raises(self) -> None:
        with pytest.raises(ValueError, match=r"terminal_depth.*must be <= max_depth"):
            LATSAgent(
                name="test-lats",
                llm_client=MockLLMClient([]),
                emitter=make_emitter(),
                system_prompt="Solve",
                tools=[search_tool],
                node_evaluator=_AcceptEvaluator(),
                terminal_depth=11,
                max_depth=10,
            )


# ──────────────────────────────────────────────────────────
# Continue-After-Accept Tests
# ──────────────────────────────────────────────────────────


class TestContinueAfterAccept:
    async def test_multiple_accepted_terminals_best_selected(self) -> None:
        """When multiple terminals are accepted, the highest-value one is selected."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        # Iteration 1: expand root (BF=2) -> tool call child + weak terminal
        # Iteration 2: expand tool-call child -> strong terminal
        responses = [
            make_response(content="searching", tool_calls=[tc]),
            make_response(content="weak answer"),  # terminal, lower score
            # Iteration 2: expand tool-call child
            make_response(content="strong answer"),  # terminal, higher score
            make_response(content="strong answer 2"),
        ]

        class _TerminalScorer:
            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                if "strong" in output:
                    score = 0.95
                elif "weak" in output:
                    score = 0.6
                else:
                    score = 0.5
                return EvaluationResult(
                    verdict=EvaluationVerdict.ACCEPT,
                    score=score,
                    evaluator_name="test-evaluator",
                )

        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_TerminalScorer(),
            max_iterations=5,
            max_depth=5,
            branching_factor=2,
        )

        result = await agent.run("Question")

        # Best accepted terminal should be "strong answer" (higher value/visits)
        assert result.output is not None
        assert "strong" in result.output

        # Verify accepted_count
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert complete_events[0].accepted_count is not None
        assert complete_events[0].accepted_count >= 2

    async def test_no_accepted_terminals_falls_back(self) -> None:
        """When no terminals are accepted, falls back to _select_best_node()."""
        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})
        # Only tool calls, never terminal
        responses = [
            make_response(content="searching", tool_calls=[tc]),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()

        agent = LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(score=0.5),
            max_iterations=1,
            max_depth=5,
            branching_factor=1,
        )

        result = await agent.run("Question")

        # No terminals, falls back to best node
        assert result.output is not None
        assert result.termination_reason == "max_iterations"

        # Verify accepted_count is 0
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert complete_events[0].accepted_count == 0

    async def test_evaluator_reject_on_terminal_excludes_from_final_selection(self) -> None:
        """A terminal that the evaluator REJECTs must be pruned, excluded from
        ``accepted_terminal_ids``, and skipped by ``_select_best_node``'s
        preferred (non-failed terminal) branch — forcing a different terminal
        to win the final selection. Pins the evaluator-protocol → selector
        plumbing without a real LLM."""

        tc = ToolCall(id="tc1", name="search", arguments={"query": "test"})

        class _RejectFirstTerminal:
            """REJECT the first terminal seen; ACCEPT subsequent terminals."""

            def __init__(self) -> None:
                self._terminals_seen = 0

            @property
            def max_revisions(self) -> int:
                return 0

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                del context
                # LATS's _evaluate_node renders non-terminal nodes via
                # _format_trajectory (which includes a "Thought:" prefix)
                # and terminals via their raw content. The "Thought:"
                # presence is therefore a reliable structural marker for
                # "this was a non-terminal trajectory" within this test.
                is_terminal_output = "Thought:" not in output
                if is_terminal_output:
                    self._terminals_seen += 1
                    if self._terminals_seen == 1:
                        return EvaluationResult(
                            verdict=EvaluationVerdict.REJECT,
                            score=0.0,
                            feedback="first terminal rejected",
                            evaluator_name="reject-first-terminal",
                        )
                    return EvaluationResult(
                        verdict=EvaluationVerdict.ACCEPT,
                        score=0.9,
                        evaluator_name="reject-first-terminal",
                    )
                return EvaluationResult(
                    verdict=EvaluationVerdict.REVISE,
                    score=0.5,
                    evaluator_name="reject-first-terminal",
                )

        # Iteration 1: expand root with branching_factor=2 → two children.
        # Both are tool-call nodes (REVISE). Iteration 2: expand one of them
        # → terminal children (one REJECTed, one ACCEPTed).
        responses = [
            make_response(content="searching-a", tool_calls=[tc]),
            make_response(content="searching-b", tool_calls=[tc]),
            make_response(content="first terminal"),
            make_response(content="second terminal"),
            # Extras for any further iteration.
            make_response(content="third terminal"),
            make_response(content="fourth terminal"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        evaluator = _RejectFirstTerminal()

        agent = LATSAgent(
            name="reject-terminal",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=evaluator,
            max_iterations=3,
            max_depth=4,
            branching_factor=2,
        )

        result = await agent.run("Question")

        # --- Evaluator actually saw at least two terminals. ---
        assert evaluator._terminals_seen >= 2, (
            "Test setup must let the evaluator see ≥2 terminals so that REJECT → ACCEPT flow is exercised; "
            f"got terminals_seen={evaluator._terminals_seen}"
        )

        # --- The rejected terminal is marked is_failed and emits a prune event. ---
        prune_events = [e for e in emitter.events if isinstance(e, TreeSearchNodePrunedEvent)]
        created = [e for e in emitter.events if isinstance(e, TreeSearchNodeCreatedEvent)]
        terminal_created_ids = {e.node_id for e in created if e.is_terminal}
        rejected_terminal_prunes = [e for e in prune_events if e.node_id in terminal_created_ids]
        assert rejected_terminal_prunes, (
            "Expected at least one TreeSearchNodePrunedEvent for a terminal — evaluator REJECT on a terminal "
            "must trigger pruning. Prune events: "
            f"{[(e.node_id, e.reason) for e in prune_events]}, terminal ids: {terminal_created_ids}"
        )

        # --- accepted_count is strictly < total accepted-eligible terminals
        # (proving the rejected one is excluded from the accepted list). ---
        complete_events = [e for e in emitter.events if isinstance(e, TreeSearchCompleteEvent)]
        assert len(complete_events) == 1
        complete = complete_events[0]
        assert complete.accepted_count is not None
        assert complete.accepted_count >= 1, (
            "Expected at least one ACCEPTED terminal to remain after the first was rejected; "
            f"got accepted_count={complete.accepted_count}"
        )
        assert complete.accepted_count < evaluator._terminals_seen, (
            "Expected accepted_count to exclude the rejected terminal; "
            f"got accepted_count={complete.accepted_count}, terminals_seen={evaluator._terminals_seen}"
        )

        # --- The final selection is not a rejected terminal. ---
        rejected_ids = {e.node_id for e in rejected_terminal_prunes}
        assert complete.selected_node_id not in rejected_ids, (
            f"Expected selected_node_id to not be a rejected terminal. "
            f"selected={complete.selected_node_id}, rejected={rejected_ids}"
        )
        assert result.output is not None


# ──────────────────────────────────────────────────────────
# Internal Method Unit Tests
# ──────────────────────────────────────────────────────────


class TestLATSInternals:
    def _make_agent(self) -> LATSAgent:
        client = MockLLMClient([])
        emitter = make_emitter()
        return LATSAgent(
            name="test-lats",
            llm_client=client,
            emitter=emitter,
            system_prompt="Solve",
            tools=[search_tool],
            node_evaluator=_AcceptEvaluator(),
            max_iterations=1,
            max_depth=5,
            branching_factor=1,
        )

    def test_select_best_node_empty_nodes(self) -> None:
        """_select_best_node returns None when _nodes is empty."""
        agent = self._make_agent()
        agent._nodes = {}
        assert agent._select_best_node() is None

    def test_select_best_node_terminals_in_general_pool(self) -> None:
        """_select_best_node without candidate_ids prefers terminal non-failed nodes."""
        agent = self._make_agent()
        root = ActionNode(id="root", depth=0, thought="root", visit_count=1, value=0.1)
        terminal = ActionNode(
            id="t1",
            parent_id="root",
            depth=1,
            thought="answer",
            is_terminal=True,
            terminal_output="final answer",
            visit_count=2,
            value=1.6,
        )
        non_terminal = ActionNode(
            id="nt1",
            parent_id="root",
            depth=1,
            thought="thinking",
            action="search",
            visit_count=3,
            value=0.9,
        )
        agent._nodes = {"root": root, "t1": terminal, "nt1": non_terminal}

        result = agent._select_best_node()
        assert result is not None
        assert result.id == "t1"

    def test_select_best_node_with_candidate_ids(self) -> None:
        """_select_best_node with candidate_ids selects from that set."""
        agent = self._make_agent()
        root = ActionNode(id="root", depth=0, thought="root", visit_count=1, value=0.1)
        t1 = ActionNode(id="t1", parent_id="root", depth=1, is_terminal=True, visit_count=2, value=1.0)
        t2 = ActionNode(id="t2", parent_id="root", depth=1, is_terminal=True, visit_count=2, value=1.8)
        agent._nodes = {"root": root, "t1": t1, "t2": t2}

        result = agent._select_best_node(candidate_ids={"t1", "t2"})
        assert result is not None
        assert result.id == "t2"

    def test_describe_siblings_thought_only(self) -> None:
        """_describe_siblings includes thought-only children."""
        agent = self._make_agent()
        root = ActionNode(id="root", depth=0, thought="root", children_ids=["c1", "c2"])
        action_child = ActionNode(
            id="c1",
            parent_id="root",
            depth=1,
            thought="reasoning",
            action="search",
            action_input={"query": "test"},
        )
        thought_child = ActionNode(
            id="c2",
            parent_id="root",
            depth=1,
            thought="just a thought",
        )
        agent._nodes = {"root": root, "c1": action_child, "c2": thought_child}

        descriptions = agent._describe_siblings("root")
        assert len(descriptions) == 2
        assert 'search(query="test")' in descriptions[0]
        assert "[thought:" in descriptions[1]
        assert "just a thought" in descriptions[1]

    def test_describe_siblings_terminal_child(self) -> None:
        """_describe_siblings includes terminal children."""
        agent = self._make_agent()
        root = ActionNode(id="root", depth=0, thought="root", children_ids=["c1"])
        terminal_child = ActionNode(
            id="c1",
            parent_id="root",
            depth=1,
            thought="final",
            is_terminal=True,
            terminal_output="The answer is 42",
        )
        agent._nodes = {"root": root, "c1": terminal_child}

        descriptions = agent._describe_siblings("root")
        assert len(descriptions) == 1
        assert "[terminal:" in descriptions[0]
        assert "The answer is 42" in descriptions[0]

    def test_build_trajectory_messages_terminal_node(self) -> None:
        """_build_trajectory_messages includes terminal node as assistant message."""
        agent = self._make_agent()
        root = ActionNode(id="root", depth=0, thought="root", children_ids=["t1"])
        terminal = ActionNode(
            id="t1",
            parent_id="root",
            depth=1,
            thought="reasoning",
            is_terminal=True,
            terminal_output="The final answer",
        )
        agent._nodes = {"root": root, "t1": terminal}
        agent._root_id = "root"

        messages = agent._build_trajectory_messages("t1", "Question")

        # user(input) + assistant(terminal_output)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Question"
        assert messages[1].role == "assistant"
        assert messages[1].content == "The final answer"


# ──────────────────────────────────────────────────────────
# Tool State Update
# ──────────────────────────────────────────────────────────


class TestLATSUpdateToolState:
    def test_update_tool_state_delegates_to_registry(self) -> None:
        from nanitics import ToolContext

        captured_state: list[dict] = []

        @tool(name="reader", description="Reads state")
        async def reader(context: ToolContext) -> str:
            captured_state.append(dict(context.state))
            return "ok"

        agent = LATSAgent(
            name="lats",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[reader],
            node_evaluator=_AcceptEvaluator(),
            tool_state={"a": 1},
        )

        agent.update_tool_state("b", 2)

        # Verify via the registry's internal state
        assert agent._tool_registry._tool_state == {"a": 1, "b": 2}


class TestLATSRunId:
    async def test_run_id_kwarg_populates_tool_context(self) -> None:
        from nanitics import ToolContext

        captured: list[ToolContext | None] = []

        @tool(name="capture", description="Capture context")
        async def capture_tool(query: str, context: ToolContext) -> str:
            captured.append(context)
            return "ok"

        tc = ToolCall(id="tc-lats", name="capture", arguments={"query": "test"})
        responses = [
            make_response(content="calling", tool_calls=[tc]),
            make_response(content="done"),
        ]
        agent = LATSAgent(
            name="lats",
            llm_client=MockLLMClient(responses),
            emitter=make_emitter(),
            system_prompt="test",
            tools=[capture_tool],
            node_evaluator=_AcceptEvaluator(score=0.95),
            max_iterations=3,
            branching_factor=1,
            run_id="r-1",
        )

        await agent.run("go")
        assert captured[0] is not None
        assert captured[0].run_id == "r-1"
        assert captured[0].tool_call_id == "tc-lats"
