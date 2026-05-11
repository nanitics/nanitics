"""LATSAgent validation: distinguishing MCTS capabilities with a real LLM.

Proves that under a real provider, LATS exhibits the four capabilities that
distinguish a tree-search agent from a linear chain, plus two integration
properties that only a real LLM can falsify:

Acceptance criteria:
  - Re-selection: at least one ``MCTSIterationEvent.selected_node_id``
    appears as the ``selected_node_id`` of a *later*
    ``MCTSIterationEvent``, or at least one non-root node id is shared
    across two distinct iterations' ``selection_path`` lists. This is
    the SDK's own record of which leaf UCB1 picked each iteration; a
    repeat proves a subtree was re-selected (the distinguishing MCTS
    property), which would not happen under a "pick each leaf once"
    regression.
  - Backpropagation value mutation: the final
    ``MCTSIterationEvent.node_values`` map contains the root id with a
    value strictly greater than 0. A regression where
    ``_backpropagate`` emitted the event but called ``_update_node``
    with ``value=0`` (or never ran) would leave the root unvisited in
    ``node_values`` or at 0.0 — both caught by this assertion.
  - Pruning is respected, not just emitted: at least one
    ``TreeSearchNodePrunedEvent`` fires (guaranteed by
    ``DepthPrioritizedEvaluator``'s first-child-per-parent REJECT rule
    under ``branching_factor >= 2``), and no pruned node id appears in
    any later ``MCTSIterationEvent.selection_path`` — proving the
    ``is_failed`` flag actually influences subsequent leaf selection,
    not only that the emit line was reached.
  - Evaluator-guided selection: the ``TreeSearchCompleteEvent.selected_node_id``
    equals the argmax-valued accepted terminal (or is among the tied
    score-max pool when ties exist). If no accepted terminals exist —
    which would contradict the hyperparameters tuned for ``accept_depth=2``
    with ``max_iterations=4`` — the test fails loudly rather than
    accepting the fallback as a pass.

Hyperparameters: ``max_iterations=4`` (budget for at least one
re-selection after the first accepted terminal is found), ``max_depth=3``,
``branching_factor=2`` (first-child-REJECT needs at least one survivor),
and ``DepthPrioritizedEvaluator(accept_depth=2)``.

The structural evaluator replaces the former ``SearchEvaluator`` (which
gated on LLM-emitted strings ``SOLUTION:`` / ``dead_end``); see
``validation/helpers/search_evaluators.py`` for the deterministic
gating rule. The LLM still drives node body generation and tool
sequencing through the real prompt and output schema.

The canned-term LLM-judge assertion from earlier revisions was removed:
it verifies tool-output incorporation (already exercised in
``validation/agents/react_agent.py`` and ``validation/tools/tool_execution.py``)
and is not LATS-distinguishing.

``test_lats_tool_failure_recovery`` (real LLM):
  - A poisoned tool that always raises is paired with a healthy one.
  - At least one ``TreeSearchNodeCreatedEvent`` carries ``is_failed=True``
    and an ``error_message`` for the poisoned tool's action — pins that
    tool-exception wiring fires under a real provider.
  - The ``MCTSBackpropagationEvent`` for the failed node propagates
    ``value=0.0`` — pins the zero-reward backprop contract.
  - The winning trajectory (path from ``TreeSearchCompleteEvent.selected_node_id``
    back to the root) contains no invocation of the poisoned tool — the
    real-LLM behavior mocks cannot surface: the agent pivots to the
    healthy tool after seeing the failure in sibling trajectories.

``test_lats_episodic_memory_cross_run`` (real LLM + real embeddings, gated on Voyage):
  - Two sequential runs share an ``InMemoryEpisodeStore`` under real
    Voyage embeddings. Run 1 stores one episode (``count()==1``), run 2
    adds a second (``count()==2``) — pins recording via the SDK path.
  - At least one ``LLMRequestEvent`` during run 2 carries messages
    containing the ``[Past Experiences]`` marker from the LATS-specific
    ``_recall_episodes`` rendering — pins recall → trajectory-message
    injection end-to-end under real embeddings. No fuzzy-judge on output:
    how the LLM consumes recalled context is emergent and not the
    machinery invariant this test pins.
"""

from __future__ import annotations

from nanitics import (
    InMemoryEmitter,
    InMemoryEpisodeStore,
    tool,
)
from nanitics.experimental.strategies import LATSAgent
from nanitics.infrastructure import (
    LLMRequestEvent,
    MCTSBackpropagationEvent,
    MCTSIterationEvent,
    TreeSearchCompleteEvent,
    TreeSearchNodeCreatedEvent,
    TreeSearchNodePrunedEvent,
)
from validation.helpers import (
    assert_trace_contains,
    make_embedding_client,
    make_llm_client,
    requires_voyage,
    run_with_retry,
)
from validation.helpers.search_evaluators import DepthPrioritizedEvaluator
from validation.helpers.tool_stubs import make_failing_tool


@tool("search", "Search for information on a topic")
async def search(query: str) -> str:
    del query  # unused — stub returns canned facts unconditional on query
    return (
        "Database-query latency is most commonly reduced by adding a B-tree "
        "index on the predicate column, which converts a sequential scan "
        "into a logarithmic lookup. Other proven techniques include using "
        "a connection pool to amortise TCP and authentication overhead "
        "across requests, and routing analytical reads to a read replica "
        "to offload the primary."
    )


@tool("analyze", "Analyze the given data")
async def analyze(data: str) -> str:
    # Compress the observation while preserving the specific canned technique
    # names so the solver sees them propagated through both plan steps.
    del data  # unused — stub returns canned analysis preserving key terms
    return (
        "Analysis: a B-tree index on the filter column gives a logarithmic "
        "lookup; a connection pool amortises TCP and authentication cost; "
        "a read replica offloads analytical reads from the primary. Choose "
        "based on the dominant bottleneck."
    )


async def test_lats_tree_search(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    agent = LATSAgent(
        name="lats-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a research agent. Explore approaches to the user's task. "
            "If a path seems unproductive, write 'dead_end'. When you have a "
            "concrete solution, start the final line with 'SOLUTION:'."
        ),
        tools=[search, analyze],
        node_evaluator=DepthPrioritizedEvaluator(accept_depth=2),
        max_iterations=4,
        max_depth=3,
        branching_factor=2,
        # Force terminal emergence at depth 3: otherwise a well-behaved
        # correct-pruning run (one pruned + one live child per depth)
        # can exhaust its iteration budget without the LLM ever choosing
        # to stop calling tools, leaving the terminal-argmax assertion
        # below with nothing to rank. Withholding tools at depth 3 makes
        # the depth-3 children text-only and therefore terminal by
        # construction — so the invariant being asserted is "evaluator
        # argmax picks the right terminal", not "the LLM happened to
        # stop using tools in time".
        terminal_depth=3,
    )

    await run_with_retry(
        lambda: agent.run(
            "Investigate techniques for reducing database-query latency in a "
            "read-heavy web service. First call `search` to retrieve candidate "
            "techniques, then call `analyze` on the most promising technique's "
            "description, then produce a final recommendation that names one of "
            "the canned techniques (a `B-tree index`, a `connection pool`, or "
            "a `read replica`) and justifies the choice in one or two sentences. "
            "If a path seems unproductive, write 'dead_end'. When you have the "
            "final recommendation, start the final line with 'SOLUTION:' "
            "followed by the technique name."
        ),
        max_attempts=2,
    )

    events = traced_emitter.events

    # --- Tree reconstruction ---
    created = [e for e in events if isinstance(e, TreeSearchNodeCreatedEvent)]
    root_id = next(e.node_id for e in created if e.parent_id is None)

    iterations = [e for e in events if isinstance(e, MCTSIterationEvent)]
    assert iterations, "Expected at least one MCTSIterationEvent."

    # --- Re-selection: either the same leaf was selected twice across
    # iterations, or two iterations share a non-root node in their
    # selection_path. Either is direct evidence UCB1 re-visited a subtree
    # — something that cannot happen under a one-shot-leaf regression. ---
    assert_trace_contains(traced_emitter, MCTSIterationEvent)
    selected_leaves = [e.selected_node_id for e in iterations]
    repeated_leaves = [nid for nid in selected_leaves if selected_leaves.count(nid) > 1]
    shared_path_ids: set[str] = set()
    for i in range(len(iterations)):
        for j in range(i + 1, len(iterations)):
            shared = set(iterations[i].selection_path) & set(iterations[j].selection_path)
            shared.discard(root_id)
            shared_path_ids.update(shared)
    assert repeated_leaves or shared_path_ids, (
        "Expected re-selection evidence: either a repeated MCTSIterationEvent.selected_node_id "
        "or a non-root node shared between two iterations' selection_path. "
        f"Selected leaves per iteration: {selected_leaves}. "
        f"Selection paths: {[e.selection_path for e in iterations]}."
    )

    # --- Backpropagation: event fires, reaches root, and mutates root value. ---
    assert_trace_contains(traced_emitter, MCTSBackpropagationEvent)
    backprop_events = [e for e in events if isinstance(e, MCTSBackpropagationEvent)]
    assert any(root_id in e.updated_node_ids for e in backprop_events), (
        f"Expected root node id {root_id} in at least one "
        "MCTSBackpropagationEvent.updated_node_ids; got paths: "
        f"{[e.updated_node_ids for e in backprop_events]}"
    )
    final_node_values = iterations[-1].node_values
    assert root_id in final_node_values, (
        f"Root id {root_id} missing from final MCTSIterationEvent.node_values "
        f"(keys={list(final_node_values.keys())}). Root should have visit_count >= 1 after backprop."
    )
    assert final_node_values[root_id] > 0.0, (
        f"Final root average value is {final_node_values[root_id]:.3f}; expected > 0.0. "
        "Backprop may have emitted the event but failed to mutate ancestor values."
    )

    # --- Pruning: emits AND is respected by later leaf selection. ---
    assert_trace_contains(traced_emitter, TreeSearchNodePrunedEvent)
    # Determine, per iteration, which pruned ids were already pruned at
    # that point (a node can only be "respected as pruned" from the
    # iteration AFTER it was pruned). Assert no such pruned id appears in
    # that later iteration's selection_path. The pruned event appears
    # inside the span of the iteration that produced the rejected child,
    # so by the *next* MCTSIterationEvent the flag is in effect.
    # For each pruned node, record which iteration-in-progress produced it
    # (0-indexed: equals the index of the next MCTSIterationEvent to fire).
    # Selection_paths for iterations strictly AFTER this index must not
    # include the pruned id. (Within the same iteration, the selection
    # path was captured before the prune, so the prune cannot influence
    # that iteration's selection.)
    pruned_emit_iter_idx: dict[str, int] = {}
    iter_events_seen = 0
    for ev in events:
        if isinstance(ev, MCTSIterationEvent):
            iter_events_seen += 1
        elif isinstance(ev, TreeSearchNodePrunedEvent) and ev.node_id not in pruned_emit_iter_idx:
            pruned_emit_iter_idx[ev.node_id] = iter_events_seen
    for nid, emitted_during in pruned_emit_iter_idx.items():
        # Iterations with index > emitted_during are strictly later.
        for later_idx in range(emitted_during + 1, len(iterations)):
            assert nid not in iterations[later_idx].selection_path, (
                f"Pruned node {nid} (pruned during iteration index {emitted_during}) appeared in "
                f"MCTSIterationEvent[{later_idx}].selection_path={iterations[later_idx].selection_path} "
                "— pruning is emitted but not respected by later leaf selection."
            )

    # --- Evaluator-guided selection: argmax-valued accepted terminal. ---
    complete_event = assert_trace_contains(traced_emitter, TreeSearchCompleteEvent)
    complete = [e for e in events if isinstance(e, TreeSearchCompleteEvent)]
    assert len(complete) == 1, f"Expected exactly one TreeSearchCompleteEvent; got {len(complete)}."
    selected = complete_event.selected_node_id
    assert selected, "TreeSearchCompleteEvent.selected_node_id must be non-empty."

    terminal_created = {e.node_id for e in created if e.is_terminal and not e.is_failed}
    candidates = {nid: v for nid, v in final_node_values.items() if nid in terminal_created}

    assert candidates, (
        "Expected at least one accepted terminal with a recorded value (hyperparameters are tuned "
        f"for accept_depth=2 under DepthPrioritizedEvaluator). accepted_count={complete_event.accepted_count}, "
        f"terminal_created={terminal_created}, final_node_values keys={list(final_node_values.keys())}."
    )
    max_value = max(candidates.values())
    argmax_pool = {nid for nid, v in candidates.items() if v == max_value}
    if len(argmax_pool) == 1:
        (expected,) = argmax_pool
        assert selected == expected, (
            "Expected selected_node_id to be argmax-valued accepted terminal "
            f"({expected}, value={candidates[expected]:.3f}); got {selected} "
            f"with value {final_node_values.get(selected, 0.0):.3f}."
        )
    else:
        # Under DepthPrioritizedEvaluator, same-depth accepted terminals
        # share the same score, so the argmax tie-break comes from the
        # SDK's insertion-ordered iteration over ``self._nodes``. The
        # validator only sees event-stream data and cannot reconstruct
        # that ordering; assert membership in the tied pool instead.
        assert selected in argmax_pool, (
            "Expected selected_node_id to be among tied argmax accepted terminals "
            f"(pool={argmax_pool}, value={max_value:.3f}); got {selected} "
            f"with value {final_node_values.get(selected, 0.0):.3f}."
        )


async def test_lats_tool_failure_recovery(traced_emitter: InMemoryEmitter) -> None:
    """Under a real LLM, when a prominently-described tool fails every
    call, the agent pivots to the healthy alternative and the winning
    trajectory avoids the poisoned tool.

    Mocks already prove the bookkeeping (tests/test_lats.py::TestToolFailure):
    ``is_failed=True`` plus zero-reward backprop. The real-LLM layer
    falsifies something mocks cannot: that the way LATS renders failed
    branches in the trajectory (plus the diversity prompt in
    ``_describe_siblings``) actually steers the LLM toward the healthy
    tool on later expansions. A regression where failure-surfacing text
    were silently dropped would leave the LLM re-picking the poisoned
    tool and the winning trajectory would include it.
    """
    failing_tool = make_failing_tool(
        name="lookup_record",
        description=(
            "Look up a specific record in the internal record store by id. "
            "Use this tool first to consult company-approved records before "
            "falling back to web search."
        ),
        message="record store unavailable",
    )
    agent = LATSAgent(
        name="lats-tool-failure",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a research agent. Prefer company-approved records when "
            "available, and fall back to web search when they are not. If "
            "a tool call fails, try a different tool or approach."
        ),
        tools=[search, analyze, failing_tool],
        node_evaluator=DepthPrioritizedEvaluator(accept_depth=2),
        max_iterations=5,
        max_depth=3,
        branching_factor=2,
        terminal_depth=3,
    )

    await run_with_retry(
        lambda: agent.run(
            "Investigate techniques for reducing database-query latency in a "
            "read-heavy web service. First call `lookup_record` with "
            "record_id='db-latency-best-practices' to consult the internal "
            "record store for company-approved recommendations. If that "
            "tool is unavailable, use `search` instead and proceed from "
            "there. Produce a final recommendation naming one of the "
            "canned techniques."
        ),
        max_attempts=2,
    )

    events = traced_emitter.events
    created = [e for e in events if isinstance(e, TreeSearchNodeCreatedEvent)]

    # --- Failure wiring: at least one failed node for the poisoned tool. ---
    failed_poisoned = [
        e
        for e in created
        if e.is_failed and e.action == "lookup_record" and "record store unavailable" in (e.error_message or "")
    ]
    assert failed_poisoned, (
        "Expected at least one TreeSearchNodeCreatedEvent with is_failed=True and "
        "action='lookup_record' — the LLM should have attempted the poisoned tool "
        "as instructed. Created events summary: "
        f"{[(e.action, e.is_failed, (e.error_message or '')[:40]) for e in created]}"
    )

    # --- Zero-reward backprop for the failed node. ---
    failed_node_ids = {e.node_id for e in failed_poisoned}
    backprops = [e for e in events if isinstance(e, MCTSBackpropagationEvent)]
    zero_backprops_for_failed = [
        e
        for e in backprops
        if e.updated_node_ids and e.updated_node_ids[0] in failed_node_ids and e.propagated_value == 0.0
    ]
    assert zero_backprops_for_failed, (
        "Expected a MCTSBackpropagationEvent with propagated_value=0.0 for at least "
        "one failed-tool node. Backprop summary: "
        f"{[(e.propagated_value, e.updated_node_ids[:1]) for e in backprops]}"
    )

    # --- Pivot: winning trajectory avoids the poisoned tool. ---
    complete_event = assert_trace_contains(traced_emitter, TreeSearchCompleteEvent)
    selected_id = complete_event.selected_node_id
    assert selected_id, "TreeSearchCompleteEvent.selected_node_id must be non-empty."

    # Reconstruct parent chain from the created-event stream.
    parent_of = {e.node_id: e.parent_id for e in created}
    action_of = {e.node_id: e.action for e in created}
    winning_path: list[str] = []
    current: str | None = selected_id
    while current is not None:
        winning_path.append(current)
        current = parent_of.get(current)
    winning_actions = [action_of.get(nid) for nid in winning_path]
    assert "lookup_record" not in winning_actions, (
        "Expected the winning trajectory to have pivoted away from the "
        f"poisoned tool; got actions along winning path: {winning_actions!r}"
    )

    # --- Alternative tool was actually used somewhere in the tree. ---
    healthy_use = [e for e in created if e.action == "search" and not e.is_failed]
    assert healthy_use, (
        "Expected at least one successful `search` invocation somewhere in the "
        f"tree, proving the LLM exercised the alternative. Created actions: "
        f"{[e.action for e in created]}"
    )


@requires_voyage
async def test_lats_episodic_memory_cross_run(traced_emitter: InMemoryEmitter) -> None:
    """Run 2 retrieves run 1's episode under real Voyage embeddings, and
    the recalled content reaches the LLM via the trajectory messages.

    Pins the LATS-specific episodic-memory integration (``_recall_episodes``
    + ``_record_episode`` on the agent itself, distinct from the
    ``EpisodicMemoryProvider`` path validated in
    ``validation/memory/episodic_memory.py``). Real embeddings are
    load-bearing: a mocked embedding client cannot falsify production
    recall semantics.
    """
    embedding_client = make_embedding_client("voyage")
    store = InMemoryEpisodeStore(embedding_client=embedding_client)

    # --- Run 1: populate the episode store. ---
    run1_emitter = InMemoryEmitter(trace_id=f"{traced_emitter.trace_id}-run1")
    run1_agent = LATSAgent(
        name="lats-episodic-run1",
        llm_client=make_llm_client("anthropic"),
        emitter=run1_emitter,
        system_prompt=(
            "You are a research agent. Explore approaches to the user's task, then produce a concise recommendation."
        ),
        tools=[search, analyze],
        node_evaluator=DepthPrioritizedEvaluator(accept_depth=2),
        max_iterations=4,
        max_depth=3,
        branching_factor=2,
        terminal_depth=3,
        episode_store=store,
    )
    await run_with_retry(
        lambda: run1_agent.run(
            "Investigate techniques for reducing database-query latency "
            "in a read-heavy web service and recommend one technique."
        ),
        max_attempts=2,
    )
    assert await store.count() == 1, f"Expected exactly one episode recorded after run 1; got {await store.count()}"

    # --- Run 2: paraphrased task, shared store. The agent should recall
    # run 1's episode and inject it into its trajectory messages. ---
    run2_agent = LATSAgent(
        name="lats-episodic-run2",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a research agent. Explore approaches to the user's task, then produce a concise recommendation."
        ),
        tools=[search, analyze],
        node_evaluator=DepthPrioritizedEvaluator(accept_depth=2),
        max_iterations=4,
        max_depth=3,
        branching_factor=2,
        terminal_depth=3,
        episode_store=store,
    )
    await run_with_retry(
        lambda: run2_agent.run(
            "A colleague's analytics-heavy service is suffering from slow "
            "lookups against its primary datastore. What should they try "
            "first to make reads faster?"
        ),
        max_attempts=2,
    )
    assert await store.count() == 2, f"Expected two episodes after run 2; got {await store.count()}"

    # --- Recall reached the LLM: [Past Experiences] marker in run 2's messages. ---
    # Intentionally no fuzzy-judge on run-2 output quality: how the LLM *uses*
    # recalled context is emergent and not what this test pins. The invariant
    # being pinned is "the machinery delivers the recalled episode to the LLM
    # call site under real embeddings" — the deterministic marker check
    # below plus the count assertions above are sufficient evidence.
    llm_requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent)]
    assert llm_requests, "Expected at least one LLMRequestEvent during run 2."
    recall_hits = [
        e for e in llm_requests if any("[Past Experiences]" in str(msg.get("content", "")) for msg in e.messages)
    ]
    assert recall_hits, (
        "Expected at least one run-2 LLMRequestEvent to carry a '[Past Experiences]' "
        "marker in its messages — proves _recall_episodes injected the recalled "
        "episode into the trajectory under real embeddings. Observed message roles "
        f"across {len(llm_requests)} request events: "
        f"{[{m.get('role') for m in e.messages} for e in llm_requests[:3]]}"
    )
