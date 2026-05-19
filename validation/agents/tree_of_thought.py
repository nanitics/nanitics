"""TreeOfThoughtAgent validation: strategy-distinguishing branching search.

Parametrized across all three ``SearchStrategy`` values. Proves that under
a real provider, each strategy produces a measurably different expansion
order — an assertion that passes under one strategy must fail under the
other two — and that the structural evaluator's REJECT path reaches the
emitter end-to-end.

The SDK emits children as a batch: ``_generate_candidates`` emits
``branching_factor`` ``TreeSearchNodeCreatedEvent``s back-to-back for a
single selected parent, evaluates each, then the next selection happens.
All strategy-specific predicates below respect that batch-emission
contract.

Acceptance criteria (strategy-invariant, asserted for every parametrization):
  - Branching at depth 1: >= 2 ``TreeSearchNodeCreatedEvent`` at depth 1
    (multiple siblings actually expanded).
  - Pruning: >= 1 ``TreeSearchNodePrunedEvent`` (guaranteed by the
    structural evaluator's first-child-per-parent REJECT rule).
  - ``min_depth`` contract: every depth-1 creation has
    ``is_terminal == False`` (the SDK forces it per
    ``_build_completion_guidance`` since ``child_depth < min_depth``).
  - Exactly one ``TreeSearchCompleteEvent`` with
    ``termination_reason == "no_expandable_nodes"`` (the tree has been
    fully explored within the depth/node budget — neither cancelled nor
    ``max_nodes``-exhausted) and ``search_strategy`` echoing the
    parametrized strategy. ``accepted_count`` equals the validator's
    reconstructed accepted-terminal count.
  - Score-guided selection (only when scores strictly differentiate the
    accepted terminals — i.e. BEST_FIRST with ``LengthScoredEvaluator``):
    ``selected_node_id`` equals the argmax-scored accepted terminal.
    For tied-score evaluators (BFS/DFS under ``DepthPrioritizedEvaluator``),
    we only assert ``selected_node_id`` is *among* the score-max accepted
    terminals — the SDK's tie-break is insertion-order-driven and the
    validator cannot reconstruct ``self._nodes`` order without duplicating
    SDK bookkeeping.

Acceptance criteria (strategy-specific, one per parametrization):
  - BFS (id ``bfs``): creation-order depth sequence is monotonically
    non-decreasing — once depth increases, it does not return to a lower
    depth. (All depth-d nodes emit before any depth-(d+1) node.)
  - DFS (id ``dfs``): the first depth-3 creation precedes the last
    depth-2 creation — i.e., DFS dives into a depth-3 node before
    exhausting its depth-2 siblings. Under BFS this cannot hold (all
    depth-2 emit before any depth-3); under DFS it is guaranteed because
    after the depth-1 batch the next selection is the deepest expandable.
  - BEST_FIRST (id ``best_first``): for every *batch* (group of
    consecutive creations sharing a parent) where the parent's depth
    > 0, the batch's parent equals the argmax-scored expandable node at
    the time the first sibling of that batch was created. We evaluate
    the argmax-parent check *once per batch*, not once per sibling,
    because ``_generate_candidates`` emits all ``branching_factor``
    children for one parent in one pass.

Hyperparameters: ``branching_factor=3`` (forces genuine breadth),
``max_depth=3`` (enables depth-2 comparison after depth-1 proposals),
``min_depth=2`` (forces depth-2 reasoning before terminals are allowed).
``LengthScoredEvaluator`` is used for ``BEST_FIRST`` (its score
differentiates same-depth siblings, which is required by both the
strategy-specific and argmax-selection assertions); ``DepthPrioritizedEvaluator``
is used for BFS and DFS.
"""

from __future__ import annotations

import pytest

from nanitics import InMemoryEmitter
from nanitics.infrastructure import (
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodeEvaluatedEvent,
    TreeSearchNodePrunedEvent,
)
from nanitics.specialized import (
    SearchStrategy,
    TreeOfThoughtAgent,
)
from validation.helpers import assert_trace_contains, make_llm_client, run_with_retry
from validation.helpers.search_evaluators import (
    DepthPrioritizedEvaluator,
    LengthScoredEvaluator,
)


@pytest.mark.parametrize(
    "strategy",
    [SearchStrategy.BFS, SearchStrategy.DFS, SearchStrategy.BEST_FIRST],
    ids=["bfs", "dfs", "best_first"],
)
async def test_tree_of_thought_branching(traced_emitter: InMemoryEmitter, strategy: SearchStrategy) -> None:
    client = make_llm_client("anthropic")

    # Score-differentiated evaluator only for BEST_FIRST, whose selection
    # surface needs cross-sibling differentiation to be exercised.
    if strategy is SearchStrategy.BEST_FIRST:
        evaluator: DepthPrioritizedEvaluator | LengthScoredEvaluator = LengthScoredEvaluator(accept_depth=2)
    else:
        evaluator = DepthPrioritizedEvaluator(accept_depth=2)

    # BFS/DFS distinguishing predicates (monotonic depth for BFS; first
    # depth-3 creation preceding last depth-2 for DFS) require the search
    # to actually reach depth 3. With ``min_depth=2``, the LLM may
    # self-terminate at depth 2 — both strategies then produce identical
    # event streams and DFS cannot be distinguished from BFS. Forcing
    # ``min_depth=3`` routes through ``_build_completion_guidance`` to
    # suppress terminals below depth 3, guaranteeing depth-3 expansion.
    # BEST_FIRST keeps ``min_depth=2`` because its argmax-selection
    # invariant requires accepted terminals at depth 2 (differentiated by
    # ``LengthScoredEvaluator``), not depth-3 expansion order.
    min_depth = 3 if strategy in (SearchStrategy.BFS, SearchStrategy.DFS) else 2

    agent = TreeOfThoughtAgent(
        name="tot-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a systems engineer. Explore distinct strategies for "
            "the user's question. Prefer concrete, technical reasoning."
        ),
        node_evaluator=evaluator,
        search_strategy=strategy,
        branching_factor=3,
        max_depth=3,
        min_depth=min_depth,
    )

    await run_with_retry(
        lambda: agent.run(
            "Propose three distinct architectural approaches for reducing "
            "tail latency in a high-traffic web service — each approach "
            "should rely on a different primary mechanism (e.g., caching "
            "layer, query-path optimisation, horizontal scaling). After "
            "proposing the three, compare them on two or three trade-offs, "
            "then recommend the best approach for a latency-sensitive "
            "read-heavy workload and justify the choice."
        ),
        max_attempts=2,
    )

    events = traced_emitter.events

    # --- Strategy-invariant: branching at depth 1. ---
    assert_trace_contains(
        traced_emitter,
        TreeSearchNodeCreatedEvent,
        predicate=lambda e: e.depth == 1,
    )
    depth_1_created = [e for e in events if isinstance(e, TreeSearchNodeCreatedEvent) and e.depth == 1]
    assert len(depth_1_created) >= 2, (
        f"Expected >= 2 siblings at depth 1 (branching actually explored); "
        f"got {len(depth_1_created)} depth-1 creations."
    )

    # --- Strategy-invariant: pruning reached the emitter. ---
    assert_trace_contains(traced_emitter, TreeSearchNodePrunedEvent)
    pruned = [e for e in events if isinstance(e, TreeSearchNodePrunedEvent)]

    # --- Strategy-invariant: min_depth contract forces depth-1 non-terminal. ---
    # `_build_completion_guidance` with `child_depth=1 < min_depth` flips
    # `is_terminal=False` and sets `terminal_suppressed=True` iff the LLM had
    # proposed terminal. Both parametrizations use ``min_depth >= 2``, so the
    # SDK-side guarantee the validator can check is simply: no depth-1 node
    # is emitted as terminal.
    for e in depth_1_created:
        assert not e.is_terminal, (
            f"min_depth contract violated: depth-1 node {e.node_id} emitted with is_terminal=True "
            f"(min_depth={min_depth})."
        )

    # --- Strategy-invariant: exactly one complete event with matching metadata. ---
    complete_event = assert_trace_contains(traced_emitter, TreeSearchCompleteEvent)
    complete = [e for e in events if isinstance(e, TreeSearchCompleteEvent)]
    assert len(complete) == 1, f"Expected exactly one TreeSearchCompleteEvent; got {len(complete)}."
    assert complete_event.termination_reason == "no_expandable_nodes", (
        f"Expected termination_reason='no_expandable_nodes' (tree fully explored within "
        f"the depth/node budget); got {complete_event.termination_reason!r}. A 'max_nodes' "
        "reason would mean the node budget exhausted before the search settled; 'cancelled' "
        "would indicate the test harness cancelled mid-run."
    )
    assert complete_event.search_strategy == strategy.value, (
        f"TreeSearchCompleteEvent.search_strategy should echo the parametrized strategy "
        f"({strategy.value!r}); got {complete_event.search_strategy!r}."
    )
    selected = complete_event.selected_node_id
    assert selected, "TreeSearchCompleteEvent.selected_node_id must be non-empty."

    # --- Strategy-invariant: accepted_count matches validator reconstruction. ---
    created = [e for e in events if isinstance(e, TreeSearchNodeCreatedEvent)]
    evaluated = [e for e in events if isinstance(e, TreeSearchNodeEvaluatedEvent)]
    scores = {e.node_id: e.score for e in evaluated}
    terminal_ids = {e.node_id for e in created if e.is_terminal}
    pruned_ids = {e.node_id for e in pruned}
    accepted_terminals = {nid for nid in terminal_ids if nid in scores and nid not in pruned_ids}
    assert complete_event.accepted_count == len(accepted_terminals), (
        f"TreeSearchCompleteEvent.accepted_count ({complete_event.accepted_count}) disagrees with "
        f"the validator's reconstruction ({len(accepted_terminals)}): "
        f"terminal_ids={terminal_ids}, pruned_ids={pruned_ids}."
    )

    # --- Strategy-invariant: score-guided selection. ---
    # For tied-score evaluators (DepthPrioritizedEvaluator gives every
    # accepted terminal at a shared depth the same score) the SDK breaks
    # ties via insertion-order iteration over ``self._nodes``. The
    # validator only sees event-stream data, not ``self._nodes`` order,
    # so it can only assert ``selected`` is among the score-max pool.
    # For strictly-differentiated evaluators (LengthScoredEvaluator under
    # BEST_FIRST) we assert a unique argmax.
    if accepted_terminals:
        max_score = max(scores[nid] for nid in accepted_terminals)
        score_max_pool = {nid for nid in accepted_terminals if scores[nid] == max_score}
        if len(score_max_pool) == 1:
            (expected,) = score_max_pool
            assert selected == expected, (
                "Expected selected_node_id to be argmax-scored accepted terminal "
                f"({expected}, score={scores[expected]:.3f}); got {selected} "
                f"with score {scores.get(selected, 0.0):.3f}."
            )
        else:
            assert selected in score_max_pool, (
                "Expected selected_node_id to be among tied score-max accepted terminals "
                f"(pool={score_max_pool}, score={max_score:.3f}); got {selected} "
                f"with score {scores.get(selected, 0.0):.3f}."
            )

    # --- Strategy-specific predicates. ---
    if strategy is SearchStrategy.BFS:
        # Creation-order depth sequence is monotonically non-decreasing
        # after the root — once depth increases, it never returns.
        depths = [e.depth for e in created if e.depth > 0]
        seen_max = 0
        for d in depths:
            assert d >= seen_max, (
                f"BFS violated: creation returned to depth {d} after reaching {seen_max}. "
                f"Depth sequence (excl. root): {depths}"
            )
            seen_max = max(seen_max, d)

    elif strategy is SearchStrategy.DFS:
        # SDK emits ``branching_factor`` children as a batch per selection.
        # Under DFS, after the initial depth-1 batch, the next selection
        # is the deepest expandable (a depth-1 node), which emits a
        # depth-2 batch; then step 3 selects the deepest expandable (a
        # depth-2 node), which emits a depth-3 batch — all before the
        # remaining depth-1 siblings get their depth-2 children expanded.
        # The distinguishing property is therefore: the first depth-3
        # creation appears *before* the last depth-2 creation (DFS dives
        # before exhausting the depth-2 frontier).
        all_depths = [e.depth for e in created]
        depth_3_indices = [i for i, d in enumerate(all_depths) if d == 3]
        depth_2_indices = [i for i, d in enumerate(all_depths) if d == 2]
        assert depth_3_indices, (
            f"DFS: expected at least one depth-3 creation (max_depth=3). Depth sequence: {all_depths}"
        )
        assert depth_2_indices, f"DFS: expected at least one depth-2 creation. Depth sequence: {all_depths}"
        assert depth_3_indices[0] < depth_2_indices[-1], (
            "DFS violated: first depth-3 creation did not precede the last depth-2 creation "
            "(under DFS the deepest expandable is selected first, so depth-3 should appear "
            "before the depth-2 frontier is fully expanded). "
            f"first_depth_3_idx={depth_3_indices[0]}, last_depth_2_idx={depth_2_indices[-1]}. "
            f"Depth sequence: {all_depths}"
        )

    else:  # SearchStrategy.BEST_FIRST
        # Group consecutive creations by parent (a "batch" is the
        # ``branching_factor`` siblings emitted under one selection).
        # For each depth>1 batch, assert the parent equals the
        # argmax-scored expandable node at the time the batch's *first*
        # sibling was emitted. Subsequent siblings of the same batch
        # share the same parent by construction, so the per-sibling
        # re-check is ill-posed under the SDK's batch emission.
        # Mirror the SDK's ``_expandable_nodes`` filter exactly: a node is
        # expandable iff it is scored, not pruned, not terminal, below
        # ``max_depth``, and has not yet had children emitted under it.
        # Missing the terminal / max-depth filters makes the validator
        # hallucinate an argmax pool the SDK never considered — e.g. a
        # depth-2 ACCEPT node the LLM marked ``is_complete=True`` becomes
        # terminal with a high length-based score and is excluded from the
        # SDK's expandable pool but would otherwise dominate the
        # reconstruction.
        scored_timeline: dict[str, float] = {}
        pruned_set: set[str] = set()
        has_children: set[str] = set()
        terminal_set: set[str] = set()
        depth_by_node: dict[str, int] = {}
        violations: list[str] = []
        last_parent: str | None | object = object()  # sentinel distinct from any parent_id

        for ev in events:
            if isinstance(ev, TreeSearchNodeEvaluatedEvent):
                scored_timeline[ev.node_id] = ev.score
            elif isinstance(ev, TreeSearchNodePrunedEvent):
                pruned_set.add(ev.node_id)
            elif isinstance(ev, TreeSearchNodeCreatedEvent):
                depth_by_node[ev.node_id] = ev.depth
                if ev.is_terminal:
                    terminal_set.add(ev.node_id)
                is_batch_start = ev.parent_id != last_parent
                last_parent = ev.parent_id
                if is_batch_start and ev.parent_id is not None and ev.depth > 1:
                    # Depth > 1 means the parent is itself depth >= 1 and
                    # has been scored. (Depth-1 batches have parent=root
                    # which has no score, so BEST_FIRST trivially picks
                    # it — not distinguishing.)
                    expandable = {
                        nid: s
                        for nid, s in scored_timeline.items()
                        if nid not in pruned_set
                        and nid not in has_children
                        and nid not in terminal_set
                        and depth_by_node.get(nid, 0) < 3  # max_depth hyperparameter
                    }
                    if expandable:
                        max_score = max(expandable.values())
                        argmax_pool = {nid for nid, s in expandable.items() if s == max_score}
                        if ev.parent_id not in argmax_pool:
                            violations.append(
                                f"depth-{ev.depth} batch parent={ev.parent_id} "
                                f"(score={scored_timeline.get(ev.parent_id, 'none')}); "
                                f"argmax-expandable pool was {argmax_pool} "
                                f"(score={max_score})"
                            )
                if is_batch_start and ev.parent_id is not None:
                    # After the batch's first sibling is emitted, the
                    # parent is no longer expandable (has_children).
                    has_children.add(ev.parent_id)

        assert not violations, "BEST_FIRST violated argmax-score expansion on these batches: " + "; ".join(violations)
