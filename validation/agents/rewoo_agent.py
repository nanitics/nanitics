"""ReWOOAgent validation: plan-first, execute, then synthesize.

Acceptance criteria:
  - Planner emits exactly one ``PlanCreatedEvent`` whose plan has exactly
    two steps, and at least one step's ``metadata["args"]`` carries a
    ``#N`` variable reference (or the step's ``metadata["depends_on"]``
    is non-empty) — proving the planner produced a ReWOO-shaped plan
    with inter-step dependencies, not two independent tool calls.
  - Exactly two ``PlanStepUpdatedEvent`` with ``new_status == 'completed'``
    (one per step, each completing successfully).
  - The first ``LLMRequestEvent`` carries the ``ReWOOPlan`` schema
    (planner), the final ``LLMRequestEvent`` carries no schema (solver).
  - Phase ordering: the ``PlanCreatedEvent`` precedes the first
    ``ToolInvokeEvent``, and the last ``ToolInvokeEvent`` precedes the
    final (solver) ``LLMRequestEvent`` — anchoring the
    plan → execute → synthesize phase boundaries.
  - Tool execution order reflects the dependency graph: the
    ``ToolInvokeEvent`` for ``search`` precedes the one for
    ``summarize``. Given the plan's ``depends_on`` edge, a
    topologically-incorrect execution would manifest here.
  - Variable substitution is load-bearing: the ``ToolInvokeEvent`` for
    ``summarize`` carries a resolved ``text`` argument containing
    fragments from the ``search`` observation — proving
    ``_substitute_variables`` replaced the planner's ``#1`` token with
    the actual observation before dispatch. A no-op substitution would
    leave a literal ``#1`` in the argument and fail this check.
  - Final output references the specific canned facts propagated through
    the search-then-summarize flow — namely ``qubits`` and the vendor
    ``IBM`` — proving the solver incorporated the tool observations
    rather than synthesising from training data alone.

The ``summarize`` tool is designed to echo a deterministic transform of
its input argument so that a broken substitution path would produce a
measurably different ``ToolResultEvent``. A constant-return stub would
mask substitution regressions.
"""

from __future__ import annotations

import pytest

from nanitics.infrastructure import (
    LLMRequestEvent,
    PlanCreatedEvent,
    PlanStepUpdatedEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.planning import InMemoryPlanStore
from nanitics.specialized import (
    ReWOOAgent,
    ReWOOPlan,
)
from nanitics.strategies import tool
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# A deterministic, recognisable fragment that appears in the search stub's
# output. The ``summarize`` stub echoes a prefix of its ``text`` argument so
# that a broken variable-substitution path (which would pass the literal
# ``"#1"`` into ``text``) would produce an observably different result.
_SEARCH_SIGNATURE = "Quantum computing uses qubits"


@tool("search", "Search for information on a topic")
async def search(query: str) -> str:
    del query  # unused — stub returns canned facts unconditional on query
    return (
        f"{_SEARCH_SIGNATURE}, which can exist in superposition "
        "of 0 and 1 states, unlike classical bits. IBM operates a fleet of "
        "superconducting quantum processors accessible via its Quantum "
        "Network, and has demonstrated error-mitigation techniques on "
        "devices exceeding 100 qubits."
    )


@tool("summarize", "Summarize the given text")
async def summarize(text: str) -> str:
    # Echo a deterministic transform of the input so that
    # ``_substitute_variables`` can be verified end-to-end: the
    # ``ToolResultEvent`` must contain fragments of the search
    # observation, which only happens if ``#1`` was resolved before
    # dispatch.
    return (
        f"Summary (from text[:80]={text[:80]!r}): qubits exploit superposition "
        "to represent multiple states simultaneously; IBM provides cloud access "
        "to superconducting quantum processors with over 100 qubits."
    )


@pytest.mark.quick
async def test_rewoo_plan_then_execute(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    agent = ReWOOAgent(
        name="rewoo-agent",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a research assistant. Plan your work using the available "
            "tools: first search, then summarize the search result. The "
            "summarize step's `text` argument must reference the search "
            "step's output via the ReWOO variable syntax (e.g. `#1`)."
        ),
        tools=[search, summarize],
        plan_store=InMemoryPlanStore(),
    )

    result = await run_with_retry(
        lambda: agent.run("Research quantum computing and summarize the findings."),
        max_attempts=2,
    )

    events = traced_emitter.events

    # --- Plan shape: exactly two steps; inter-step dependency visible. ---
    plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
    assert len(plan_events) == 1, f"Expected exactly one PlanCreatedEvent; got {len(plan_events)}."
    plan_event = assert_trace_contains(
        traced_emitter,
        PlanCreatedEvent,
        predicate=lambda e: e.step_count == 2,
    )
    has_variable_ref = any("#" in str(s.metadata.get("args", {})) for s in plan_event.steps)
    has_dependency = any(s.metadata.get("depends_on") for s in plan_event.steps)
    assert has_variable_ref or has_dependency, (
        "Plan must encode inter-step dependency via either a `#N` variable reference in some step's "
        "args or a non-empty `depends_on` list. Two independent tool calls with no dependency would "
        "pass the step-count check but not be a real ReWOO plan. "
        f"Steps: {[(s.metadata.get('args'), s.metadata.get('depends_on')) for s in plan_event.steps]}"
    )

    # --- Step completion: exactly two completed, one per planned step. ---
    completed_updates = [e for e in events if isinstance(e, PlanStepUpdatedEvent) and e.new_status == "completed"]
    assert len(completed_updates) == 2, (
        f"Expected exactly 2 completed-step updates (one per planned step); got: {len(completed_updates)}."
    )

    # --- Planner/solver LLM-call shape. ---
    llm_requests = [e for e in events if isinstance(e, LLMRequestEvent)]
    assert len(llm_requests) >= 2, f"Expected >=2 LLM requests (planner + solver), got: {len(llm_requests)}"
    first_schema = llm_requests[0].output_schema
    assert first_schema is not None, "First LLM call (planner) must carry an output_schema"
    assert first_schema.get("title") == ReWOOPlan.__name__, (
        f"First LLM call must be the planner with ReWOOPlan schema, got: {first_schema}"
    )
    assert llm_requests[-1].output_schema is None, (
        f"Last LLM call must be the solver without a schema, got: {llm_requests[-1].output_schema}"
    )

    # --- Phase ordering: plan → tools → solver. ---
    plan_idx = next(i for i, e in enumerate(events) if isinstance(e, PlanCreatedEvent))
    tool_invoke_indices = [i for i, e in enumerate(events) if isinstance(e, ToolInvokeEvent)]
    assert tool_invoke_indices, "Expected at least one ToolInvokeEvent."
    assert plan_idx < tool_invoke_indices[0], (
        f"PlanCreatedEvent (idx={plan_idx}) must precede the first ToolInvokeEvent "
        f"(idx={tool_invoke_indices[0]}) — plan-first phase boundary."
    )
    llm_req_indices = [i for i, e in enumerate(events) if isinstance(e, LLMRequestEvent)]
    solver_llm_idx = llm_req_indices[-1]
    assert tool_invoke_indices[-1] < solver_llm_idx, (
        f"Last ToolInvokeEvent (idx={tool_invoke_indices[-1]}) must precede the solver "
        f"LLMRequestEvent (idx={solver_llm_idx}) — execute-before-synthesize boundary."
    )

    # --- Tool execution order reflects the dependency graph. ---
    search_invokes = [
        (i, e) for i, e in enumerate(events) if isinstance(e, ToolInvokeEvent) and e.tool_name == "search"
    ]
    summarize_invokes = [
        (i, e) for i, e in enumerate(events) if isinstance(e, ToolInvokeEvent) and e.tool_name == "summarize"
    ]
    assert search_invokes, "Expected at least one ToolInvokeEvent for `search`."
    assert summarize_invokes, "Expected at least one ToolInvokeEvent for `summarize`."
    assert search_invokes[0][0] < summarize_invokes[0][0], (
        f"`search` must be invoked before `summarize` (dependency order). "
        f"First search idx={search_invokes[0][0]}, first summarize idx={summarize_invokes[0][0]}."
    )

    # --- Variable substitution: summarize received the resolved observation. ---
    # ``_substitute_variables`` replaces each ``#N`` token in the step's
    # string arguments with ``variable_map[N]`` before dispatch. If
    # substitution were a no-op, the literal ``#1`` would reach the tool
    # and appear in the ToolInvokeEvent's parameters — and the
    # ToolResultEvent would echo back ``#1`` in the prefix-of-text it
    # renders. Both checks are derived from SDK-emitted events.
    summarize_invoke_event = summarize_invokes[0][1]
    invoke_params_str = str(summarize_invoke_event.parameters)
    assert _SEARCH_SIGNATURE in invoke_params_str, (
        "Variable substitution (`#1` -> search observation) did not resolve before summarize dispatch. "
        f"Expected {_SEARCH_SIGNATURE!r} in summarize parameters; got parameters={invoke_params_str!r}."
    )
    # The ``#N`` token should be gone from the resolved argument.
    assert "#1" not in invoke_params_str, (
        f"Literal `#1` reached the summarize tool — variable substitution did not run. "
        f"summarize parameters={invoke_params_str!r}."
    )

    summarize_results = [e for e in events if isinstance(e, ToolResultEvent) and e.tool_name == "summarize"]
    assert summarize_results, "Expected at least one ToolResultEvent for `summarize`."
    summarize_result = summarize_results[0]
    assert summarize_result.result is not None, (
        f"summarize ToolResultEvent must have a non-None result. Got result={summarize_result.result!r}."
    )
    assert _SEARCH_SIGNATURE in summarize_result.result, (
        "summarize ToolResultEvent must reflect the resolved search observation in its echoed "
        f"text-prefix (confirming end-to-end substitution). Got result={summarize_result.result!r}."
    )

    # --- Fuzzy output ---
    await assert_result_satisfies(
        result.output or "",
        "The output references both the term 'qubits' and the vendor 'IBM', "
        "demonstrating that the solver incorporated the canned search/summarize "
        "observations rather than answering from training data alone.",
    )
