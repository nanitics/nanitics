"""TreeOfThoughtAgent validation: termination-branch coverage.

Three non-parametrized tests cover execution shapes that
``validation/agents/tree_of_thought.py`` does not exercise:

1. ``test_max_nodes_termination`` — the ``len(self._nodes) >= max_nodes``
   guard at ``tree_of_thought.py`` around line 360 / line 373. Drives
   ``termination_reason == "max_nodes"`` on the ``TreeSearchCompleteEvent``.
2. ``test_cancelled_termination`` — the cancellation check at the top of
   the loop (``tree_of_thought.py`` around line 355) and its
   ``SafetyCancellationEvent`` emission via ``Agent._emit_safety_cancellation``
   (``base.py`` around line 570). Drives ``termination_reason == "cancelled"``.
3. ``test_no_terminal_fallback`` — the no-accepted-terminal fallback
   branch in ``TreeOfThoughtAgent._select_best_node`` (``tree_of_thought.py``
   around lines 434-437). Every depth-``accept_depth`` candidate is
   REJECTed, so ``accepted_terminal_ids`` stays empty and
   ``_select_best_node`` must fall back to the highest-scoring non-root,
   non-pruned node.

Each test is a distinct execution shape; parametrization would force
conditional assertions and hide per-path invariants.
"""

from __future__ import annotations

from nanitics.evaluation import (
    EvaluationContext,
    EvaluationResult,
)
from nanitics.infrastructure import (
    SafetyCancellationEvent,
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodeEvaluatedEvent,
    TreeSearchNodePrunedEvent,
)
from nanitics.safety import CancellationToken
from nanitics.specialized import (
    SearchStrategy,
    TreeOfThoughtAgent,
)
from nanitics.tracing import InMemoryEmitter
from validation.helpers import assert_trace_contains, make_llm_client, run_with_retry
from validation.helpers.search_evaluators import (
    DepthPrioritizedEvaluator,
    RejectTerminalEvaluator,
)

_PARAGRAPH_PROMPT = (
    "Propose three distinct architectural approaches for reducing "
    "tail latency in a high-traffic web service — each approach "
    "should rely on a different primary mechanism (e.g., caching "
    "layer, query-path optimisation, horizontal scaling). After "
    "proposing the three, compare them on two or three trade-offs, "
    "then recommend the best approach for a latency-sensitive "
    "read-heavy workload and justify the choice."
)


async def test_max_nodes_termination(traced_emitter: InMemoryEmitter) -> None:
    """The ``max_nodes`` loop guard terminates the search and surfaces on the complete event.

    Hyperparameters: ``branching_factor=3``, ``max_depth=3``,
    ``min_depth=2``, ``max_nodes=5``. Root + one depth-1 batch of 3
    children = 4 nodes; the next batch pushes past 5 and triggers the
    ``max_nodes`` break. ``tree_of_thought.py``'s outer guard (around
    line 360) and the mid-batch guard (around line 373) jointly cap the
    overshoot at one batch — ``total_nodes <= max_nodes + branching_factor``.
    """
    client = make_llm_client("anthropic")
    evaluator = DepthPrioritizedEvaluator(accept_depth=2)

    max_nodes = 5
    branching_factor = 3

    agent = TreeOfThoughtAgent(
        name="tot-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a systems engineer. Explore distinct strategies for "
            "the user's question. Prefer concrete, technical reasoning."
        ),
        node_evaluator=evaluator,
        search_strategy=SearchStrategy.BFS,
        branching_factor=branching_factor,
        max_depth=3,
        min_depth=2,
        max_nodes=max_nodes,
    )

    await run_with_retry(lambda: agent.run(_PARAGRAPH_PROMPT), max_attempts=2)

    events = traced_emitter.events

    # Exactly one TreeSearchCompleteEvent with the max_nodes termination.
    complete = [e for e in events if isinstance(e, TreeSearchCompleteEvent)]
    assert len(complete) == 1, f"Expected exactly one TreeSearchCompleteEvent; got {len(complete)}."
    complete_event = complete[0]
    assert complete_event.termination_reason == "max_nodes", (
        f"Expected termination_reason='max_nodes'; got {complete_event.termination_reason!r}. "
        "Verify the branching/max_nodes hyperparameters still force the node budget to "
        "exhaust before the tree fully expands."
    )
    assert complete_event.search_strategy == SearchStrategy.BFS.value, (
        f"Expected search_strategy='{SearchStrategy.BFS.value}'; got {complete_event.search_strategy!r}."
    )
    assert complete_event.selected_node_id, (
        "TreeSearchCompleteEvent.selected_node_id must be non-empty: the depth-1 batch "
        "finished before max_nodes fired, so at least one scored non-root node exists as "
        "a fallback pick."
    )

    # max_nodes branch actually taken: total_nodes >= max_nodes. Upper bound
    # `max_nodes + branching_factor` reflects the mid-batch guard capping
    # overshoot at one batch (one parent expanded before the outer-loop
    # check refires).
    assert complete_event.total_nodes >= max_nodes, (
        f"Expected total_nodes >= max_nodes ({max_nodes}) so the max_nodes branch was actually "
        f"reached (vs. an early no_expandable_nodes exit); got total_nodes={complete_event.total_nodes}."
    )
    assert complete_event.total_nodes <= max_nodes + branching_factor, (
        f"Expected total_nodes <= max_nodes + branching_factor ({max_nodes + branching_factor}): "
        f"the inner mid-batch guard should cap overshoot at one batch; got "
        f"total_nodes={complete_event.total_nodes}."
    )

    # At least one depth-1 creation (branching actually explored before
    # the node budget exhausted).
    assert_trace_contains(
        traced_emitter,
        TreeSearchNodeCreatedEvent,
        predicate=lambda e: e.depth == 1,
    )


class _CancellingEvaluator(DepthPrioritizedEvaluator):
    """Wraps ``DepthPrioritizedEvaluator`` and cancels a token on first evaluate.

    Rationale: cancelling via the evaluator is the natural per-node hook
    — it is already injected, runs synchronously inside ``_evaluate_node``,
    and cancels
    on the first call so the loop's top-of-iteration
    ``_is_cancelled`` check (``tree_of_thought.py`` around line 355)
    fires on the **next** iteration. Wrapping the emitter would bypass
    ``InMemoryEmitter``'s trace-id wiring and is not equivalent.
    """

    def __init__(self, *, accept_depth: int, token: CancellationToken) -> None:
        super().__init__(accept_depth=accept_depth)
        self._token = token
        self._cancelled_once = False

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        if not self._cancelled_once:
            self._token.cancel()
            self._cancelled_once = True
        return await super().evaluate(output, context)


async def test_cancelled_termination(traced_emitter: InMemoryEmitter) -> None:
    """A cancellation token flipped mid-run terminates the loop and emits SafetyCancellationEvent.

    The evaluator cancels on its first ``evaluate`` call; the current
    batch finishes, then the loop's top-of-iteration cancellation check
    fires and emits ``SafetyCancellationEvent`` before
    ``TreeSearchCompleteEvent`` is emitted with
    ``termination_reason == "cancelled"``.
    """
    client = make_llm_client("anthropic")
    token = CancellationToken()
    evaluator = _CancellingEvaluator(accept_depth=2, token=token)

    agent = TreeOfThoughtAgent(
        name="tot-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a systems engineer. Explore distinct strategies for "
            "the user's question. Prefer concrete, technical reasoning."
        ),
        node_evaluator=evaluator,
        search_strategy=SearchStrategy.BFS,
        branching_factor=2,
        max_depth=3,
        min_depth=2,
        max_nodes=50,
        cancellation_token=token,
    )

    await run_with_retry(lambda: agent.run(_PARAGRAPH_PROMPT), max_attempts=2)

    events = traced_emitter.events

    complete = [e for e in events if isinstance(e, TreeSearchCompleteEvent)]
    assert len(complete) == 1, f"Expected exactly one TreeSearchCompleteEvent; got {len(complete)}."
    complete_event = complete[0]
    assert complete_event.termination_reason == "cancelled", (
        f"Expected termination_reason='cancelled'; got {complete_event.termination_reason!r}. "
        "Verify the evaluator fired cancel() before the search otherwise terminated."
    )

    # Exactly one SafetyCancellationEvent, and it precedes the complete event.
    cancellations = [e for e in events if isinstance(e, SafetyCancellationEvent)]
    assert len(cancellations) == 1, f"Expected exactly one SafetyCancellationEvent; got {len(cancellations)}."
    cancellation_idx = events.index(cancellations[0])
    complete_idx = events.index(complete_event)
    assert cancellation_idx < complete_idx, (
        "SafetyCancellationEvent must precede TreeSearchCompleteEvent "
        f"(cancellation at index {cancellation_idx}, complete at index {complete_idx})."
    )

    # Root is always created, even if cancellation fires mid-batch.
    assert complete_event.total_nodes >= 1, (
        f"Expected total_nodes >= 1 (root always created); got {complete_event.total_nodes}."
    )

    # selected_node_id may be empty when no non-root node was evaluated
    # before cancellation. Must be a string; if non-empty, it must match
    # a created node id.
    selected = complete_event.selected_node_id
    assert isinstance(selected, str), f"selected_node_id must be a string; got {type(selected).__name__}."
    if selected:
        created_ids = {e.node_id for e in events if isinstance(e, TreeSearchNodeCreatedEvent)}
        assert selected in created_ids, f"selected_node_id={selected!r} is not among created node ids {created_ids}."


async def test_no_terminal_fallback(traced_emitter: InMemoryEmitter) -> None:
    """No node ever becomes an accepted terminal — ``_select_best_node`` falls back.

    ``RejectTerminalEvaluator(accept_depth=2)`` REJECTs every depth-2
    candidate; depth-1 candidates REVISE with score ``0.5``. ``min_depth=2``
    forces depth-1 non-terminal (otherwise the LLM can self-complete at
    depth 1, producing accepted terminals before the REJECT path fires).
    The loop exits via ``no_expandable_nodes`` (all depth-2 nodes are
    pruned, no expandables left). ``accepted_terminal_ids`` stays empty,
    so ``_select_best_node`` is called with ``candidate_ids=None`` and
    exercises the fallback branch at ``tree_of_thought.py`` around
    lines 434-437 (highest-scoring non-root, non-pruned node — a
    depth-1 node with score ``0.5``).
    """
    client = make_llm_client("anthropic")
    evaluator = RejectTerminalEvaluator(accept_depth=2)

    agent = TreeOfThoughtAgent(
        name="tot-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a systems engineer. Explore distinct strategies for "
            "the user's question. Prefer concrete, technical reasoning."
        ),
        node_evaluator=evaluator,
        search_strategy=SearchStrategy.BFS,
        branching_factor=2,
        max_depth=2,
        min_depth=2,
        max_nodes=50,
    )

    await run_with_retry(lambda: agent.run(_PARAGRAPH_PROMPT), max_attempts=2)

    events = traced_emitter.events

    complete = [e for e in events if isinstance(e, TreeSearchCompleteEvent)]
    assert len(complete) == 1, f"Expected exactly one TreeSearchCompleteEvent; got {len(complete)}."
    complete_event = complete[0]
    assert complete_event.accepted_count == 0, (
        f"Expected accepted_count=0 (evaluator REJECTs every terminal candidate); got {complete_event.accepted_count}."
    )
    assert complete_event.termination_reason == "no_expandable_nodes", (
        f"Expected termination_reason='no_expandable_nodes'; got {complete_event.termination_reason!r}."
    )

    # At least one depth-2 TreeSearchNodePrunedEvent proves the REJECT path fired.
    created = [e for e in events if isinstance(e, TreeSearchNodeCreatedEvent)]
    pruned = [e for e in events if isinstance(e, TreeSearchNodePrunedEvent)]
    depth_by_node = {e.node_id: e.depth for e in created}
    depth_2_pruned = [p for p in pruned if depth_by_node.get(p.node_id) == 2]
    assert depth_2_pruned, (
        f"Expected >= 1 TreeSearchNodePrunedEvent at depth 2; got pruned={ {p.node_id for p in pruned} } "
        f"with depths={ {p.node_id: depth_by_node.get(p.node_id) for p in pruned} }."
    )

    # Sanity: no surviving terminal node exists (the REJECT beat any
    # LLM-marked is_complete=True before the node could be accepted).
    pruned_ids = {p.node_id for p in pruned}
    surviving_terminals = [e for e in created if e.is_terminal and e.node_id not in pruned_ids]
    assert not surviving_terminals, (
        "Expected no surviving terminal nodes (evaluator REJECTs every depth-2 "
        f"candidate before acceptance); got {[e.node_id for e in surviving_terminals]}."
    )

    # Fallback branch exercised: selected_node_id is non-empty and names
    # a depth-1 non-pruned node (score 0.5 via REVISE).
    selected = complete_event.selected_node_id
    assert selected, (
        "selected_node_id must be non-empty: the fallback branch picks the highest-scoring non-root, non-pruned node."
    )
    evaluated = [e for e in events if isinstance(e, TreeSearchNodeEvaluatedEvent)]
    scores = {e.node_id: e.score for e in evaluated}
    assert selected in depth_by_node, f"selected_node_id={selected!r} was not emitted as a created node."
    assert depth_by_node[selected] == 1, (
        f"Expected selected_node_id to be at depth 1 (only surviving expandable depth); "
        f"got depth={depth_by_node[selected]} for node {selected!r}."
    )
    assert selected not in pruned_ids, (
        f"selected_node_id={selected!r} was pruned — the fallback must pick a non-pruned node."
    )
    # Score sanity: every non-pruned depth-1 node scores 0.5 under
    # RejectTerminalEvaluator, so the selected node's score is 0.5.
    assert scores.get(selected) == 0.5, (
        f"Expected selected node score == 0.5 (REVISE from RejectTerminalEvaluator); got {scores.get(selected)!r}."
    )
