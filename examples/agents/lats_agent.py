"""LATSAgent: MCTS-based tree search with evaluation-guided pruning.

Demonstrates LATSAgent — the most powerful agent type, using Monte Carlo Tree Search
to explore multiple solution paths. Shows the complete MCTS cycle (select → expand →
evaluate → backpropagate), tree inspection, evaluation-guided pruning that rejects
dead-end branches, and rich observability through MCTS-specific events.

Contrast with ReActAgent (single path) and TreeOfThoughtAgent (breadth-first):
  ReAct:         Single linear path — one thought, one action at a time
  TreeOfThought: Breadth-first parallel exploration — evaluate and select best thoughts
  LATS:          UCB1-guided tree search — balance exploration vs exploitation with backpropagation

Related guide: docs/guides/agent-types.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    MockLLMClient,
    ToolCall,
    tool,
)
from nanitics.infrastructure import (
    MCTSBackpropagationEvent,
    MCTSIterationEvent,
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodeEvaluatedEvent,
    TreeSearchNodePrunedEvent,
)
from nanitics.specialized import LATSAgent

# --- Shared evaluator ---


class SearchEvaluator:
    """Scores LATS nodes: accepts solutions, rejects dead ends, scores progress.

    This evaluator is the heart of the example — it controls which branches
    survive, which get pruned, and what the search considers a valid solution.
    Implements the OutputEvaluator protocol directly (rather than using
    ProgrammaticEvaluator) because LATS depends on all three verdicts
    (ACCEPT, REVISE, REJECT) and score granularity.
    """

    max_revisions = 0  # LATS doesn't use the revision loop

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        if "SOLUTION:" in output:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=1.0,
                evaluator_name="search",
            )
        if "dead_end" in output:
            return EvaluationResult(
                verdict=EvaluationVerdict.REJECT,
                score=0.0,
                evaluator_name="search",
            )
        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            score=0.6,
            evaluator_name="search",
        )


# --- Shared tools ---


@tool("search", "Search for information on a topic")
async def search_tool(query: str) -> str:
    return f"Results for: {query}"


@tool("analyze", "Analyze the given data")
async def analyze_tool(data: str) -> str:
    return f"Analysis of: {data}"


@tool("research", "Research a topic in depth")
async def research_tool(topic: str) -> str:
    if "wrong" in topic:
        return f"dead_end: no useful information found for '{topic}'"
    return f"Findings on '{topic}': useful data discovered"


async def main() -> None:
    # --- Section 1: Basic LATS Search ---
    print("--- Section 1: Basic LATS Search ---")

    # 4 LLM responses for 2 iterations × branching_factor=2
    # Iteration 1: expand root → 2 children (both tool calls)
    # Iteration 2: expand child 1 → 2 grandchildren (one is a solution)
    client = MockLLMClient(
        responses=[
            # Iteration 1: expand root
            make_response(
                "Let me search for the best approach.",
                tool_calls=[ToolCall(id="tc-1", name="search", arguments={"query": "best approach"})],
                stop_reason="tool_use",
            ),
            make_response(
                "I'll analyze the initial data.",
                tool_calls=[ToolCall(id="tc-2", name="analyze", arguments={"data": "initial data"})],
                stop_reason="tool_use",
            ),
            # Iteration 2: expand child 1 (UCB1 selects first child on tied scores)
            make_response("SOLUTION: The answer is 42"),
            make_response(
                "Let me search deeper.",
                tool_calls=[ToolCall(id="tc-4", name="search", arguments={"query": "deeper search"})],
                stop_reason="tool_use",
            ),
        ]
    )
    emitter = make_emitter("lats-s1")

    agent = LATSAgent(
        name="search-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research agent. Find the answer to the user's question.",
        tools=[search_tool, analyze_tool],
        node_evaluator=SearchEvaluator(),
        max_iterations=2,
        max_depth=3,
        branching_factor=2,
    )

    result = await agent.run("Find the answer to the ultimate question")

    assert "SOLUTION: The answer is 42" in result.output, f"Unexpected output: {result.output}"
    assert result.termination_reason == "max_iterations"
    assert result.total_steps == 2

    print(f"  Output: {result.output}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Steps (iterations): {result.total_steps}")
    print("✓ MCTS found the solution through UCB1-guided tree search")

    # --- Section 2: Inspecting the Search Tree ---
    print("\n--- Section 2: Inspecting the Search Tree ---")

    # Use the tree from Section 1 (agent._nodes is internal, but educational)
    # In production, use MCTS events for observability instead.
    nodes = agent._nodes

    # Tree shape: root + 2 children + 2 grandchildren = 5
    assert len(nodes) == 5, f"Expected 5 nodes, got {len(nodes)}"

    # Count nodes by depth
    depth_counts: dict[int, int] = {}
    for node in nodes.values():
        depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
    assert depth_counts == {0: 1, 1: 2, 2: 2}, f"Unexpected tree shape: {depth_counts}"

    # Find terminal nodes
    terminal_nodes = [n for n in nodes.values() if n.is_terminal]
    assert len(terminal_nodes) == 1, f"Expected 1 terminal node, got {len(terminal_nodes)}"
    accepted_terminal = terminal_nodes[0]
    assert accepted_terminal.terminal_output == "SOLUTION: The answer is 42"

    # Backpropagation: root has highest visit_count (visited on every backprop)
    root = nodes[agent._root_id]
    assert root.visit_count == max(n.visit_count for n in nodes.values()), "Root should have the highest visit count"

    # Accepted terminal has perfect average value
    avg_value = accepted_terminal.value / accepted_terminal.visit_count
    assert avg_value == 1.0, f"Expected accepted terminal avg value 1.0, got {avg_value}"

    # Print tree structure
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Tree shape by depth: {depth_counts}")
    print(f"  Root visit count: {root.visit_count}")

    # Path from root to accepted terminal
    path = agent._get_path_to_root(accepted_terminal.id)
    path_labels = []
    for n in path:
        if n.depth == 0:
            path_labels.append("root")
        elif n.action:
            path_labels.append(f"{n.action}({', '.join(f'{k}={v!r}' for k, v in (n.action_input or {}).items())})")
        elif n.is_terminal:
            path_labels.append(f"[terminal: {n.terminal_output!r}]")
    print(f"  Solution path: {' → '.join(path_labels)}")
    print("✓ Tree has expected shape with backpropagated values")

    # --- Section 3: Evaluation-Guided Pruning ---
    print("\n--- Section 3: Evaluation-Guided Pruning ---")

    # 4 LLM responses for 3 iterations (iteration 3 re-selects terminal, no expansion)
    # Iteration 1: expand root → child 1 (dead end) + child 2 (promising)
    # Iteration 2: expand child 2 → grandchild 1 (solution) + grandchild 2 (tool call)
    # Iteration 3: re-select terminal → backpropagate only, 0 LLM calls
    client2 = MockLLMClient(
        responses=[
            # Iteration 1: expand root
            make_response(
                "Let me research in the wrong direction.",
                tool_calls=[ToolCall(id="tc-p1", name="research", arguments={"topic": "wrong direction"})],
                stop_reason="tool_use",
            ),
            make_response(
                "Let me follow a promising lead.",
                tool_calls=[ToolCall(id="tc-p2", name="research", arguments={"topic": "promising lead"})],
                stop_reason="tool_use",
            ),
            # Iteration 2: expand child 2 (child 1 is pruned)
            make_response("SOLUTION: Found the answer"),
            make_response(
                "Let me research more data.",
                tool_calls=[ToolCall(id="tc-p4", name="research", arguments={"topic": "more data"})],
                stop_reason="tool_use",
            ),
        ]
    )
    emitter2 = make_emitter("lats-s3")

    agent2 = LATSAgent(
        name="pruning-agent",
        llm_client=client2,
        emitter=emitter2,
        system_prompt="You are a research agent. Find the answer.",
        tools=[research_tool],
        node_evaluator=SearchEvaluator(),
        max_iterations=3,
        max_depth=3,
        branching_factor=2,
    )

    result2 = await agent2.run("Find the answer through research")

    assert "SOLUTION:" in result2.output, f"Unexpected output: {result2.output}"

    # One node should be pruned (the dead-end branch)
    failed_nodes = [n for n in agent2._nodes.values() if n.is_failed]
    assert len(failed_nodes) == 1, f"Expected 1 failed node, got {len(failed_nodes)}"
    pruned = failed_nodes[0]
    assert pruned.observation is not None and "dead_end" in pruned.observation

    # Verify a pruning event was emitted
    prune_events = [e for e in emitter2.events if isinstance(e, TreeSearchNodePrunedEvent)]
    assert len(prune_events) >= 1, "Expected at least one TreeSearchNodePrunedEvent"

    # Solution came through child 2 (not through the pruned child 1)
    solution_nodes = [n for n in agent2._nodes.values() if n.is_terminal and not n.is_failed]
    assert len(solution_nodes) >= 1
    solution = solution_nodes[0]
    solution_path = agent2._get_path_to_root(solution.id)
    # Solution path should not go through the pruned node
    solution_ids = {n.id for n in solution_path}
    # They share the root but diverge after that
    assert pruned.id not in solution_ids, "Solution should not go through the pruned branch"

    print(f"  Output: {result2.output}")
    print("  Pruned branch: research(topic='wrong direction') → dead_end")
    print("  Solution branch: research(topic='promising lead') → SOLUTION")
    print(f"  Failed nodes: {len(failed_nodes)}")
    print(f"  Prune events: {len(prune_events)}")
    print("✓ Evaluator's REJECT verdict pruned the dead-end branch; search backtracked to find solution")

    # --- Section 4: MCTS Event Stream ---
    print("\n--- Section 4: MCTS Event Stream ---")

    # Use the emitter from Section 3 — no additional setup needed
    node_created = [e for e in emitter2.events if isinstance(e, TreeSearchNodeCreatedEvent)]
    node_evaluated = [e for e in emitter2.events if isinstance(e, TreeSearchNodeEvaluatedEvent)]
    node_pruned = [e for e in emitter2.events if isinstance(e, TreeSearchNodePrunedEvent)]
    backprop_events = [e for e in emitter2.events if isinstance(e, MCTSBackpropagationEvent)]
    iteration_events = [e for e in emitter2.events if isinstance(e, MCTSIterationEvent)]
    complete_events = [e for e in emitter2.events if isinstance(e, TreeSearchCompleteEvent)]

    # Node creation: root + 2 children + 2 grandchildren = 5
    assert len(node_created) == 5, f"Expected 5 TreeSearchNodeCreatedEvent, got {len(node_created)}"

    # Evaluations: all non-root nodes that aren't pre-failed get evaluated
    # child 1 (dead_end → REJECT), child 2 (REVISE), grandchild 1 (ACCEPT), grandchild 2 (REVISE)
    assert len(node_evaluated) == 4, f"Expected 4 TreeSearchNodeEvaluatedEvent, got {len(node_evaluated)}"

    # Pruning: 1 node rejected
    assert len(node_pruned) == 1, f"Expected 1 TreeSearchNodePrunedEvent, got {len(node_pruned)}"

    # Backpropagation: one per child evaluation + re-select in iteration 3
    assert len(backprop_events) >= 5, f"Expected ≥5 MCTSBackpropagationEvent, got {len(backprop_events)}"

    # Iterations: 3 (one per MCTS iteration)
    assert len(iteration_events) == 3, f"Expected 3 MCTSIterationEvent, got {len(iteration_events)}"

    # Completion: 1 summary event
    assert len(complete_events) == 1, f"Expected 1 TreeSearchCompleteEvent, got {len(complete_events)}"
    complete = complete_events[0]
    assert complete.accepted_count >= 1, f"Expected ≥1 accepted, got {complete.accepted_count}"
    assert complete.total_nodes == 5, f"Expected 5 total nodes, got {complete.total_nodes}"

    # Print event summary
    print(f"  TreeSearchNodeCreatedEvent:   {len(node_created)}")
    for e in node_created:
        label = f"depth={e.depth}"
        if e.action:
            label += f", action={e.action}"
        if e.is_terminal:
            label += ", terminal"
        if e.is_failed:
            label += ", failed"
        print(f"    node {e.node_id[:8]}... ({label})")

    print(f"  TreeSearchNodeEvaluatedEvent: {len(node_evaluated)}")
    for e in node_evaluated:
        print(f"    node {e.node_id[:8]}... score={e.score}, terminal={e.is_terminal}")

    print(f"  TreeSearchNodePrunedEvent:    {len(node_pruned)}")
    for e in node_pruned:
        print(f"    node {e.node_id[:8]}... reason={e.reason}")

    print(f"  MCTSBackpropagationEvent:     {len(backprop_events)}")
    for e in backprop_events:
        print(f"    value={e.propagated_value}, path_length={e.path_length}")

    print(f"  MCTSIterationEvent:           {len(iteration_events)}")
    for e in iteration_events:
        print(f"    iteration={e.iteration_number}, expanded={e.expanded_count}, best_value={e.best_value_so_far:.2f}")

    print(f"  TreeSearchCompleteEvent:      {len(complete_events)}")
    print(
        f"    total_nodes={complete.total_nodes}, accepted={complete.accepted_count}, "
        f"strategy={complete.search_strategy}"
    )

    print("✓ Complete MCTS event stream provides full observability into the search process")


if __name__ == "__main__":
    asyncio.run(main())
