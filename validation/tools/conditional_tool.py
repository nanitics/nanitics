"""ConditionalTool controls tool visibility through a predicate over tool_state.

A ``ReActAgent`` is given two tools: ``search_records`` (always visible)
and a conditional ``delete_record`` whose ``is_enabled`` predicate reads
``tool_state["approved"]``. The scenario is parametrized over two cases:

  (a) ``approved=True`` — the delete tool is present in the
      ``AgentStartEvent.tool_schemas`` list and the agent can invoke it;
      a ``ToolInvokeEvent`` for ``delete_record`` is emitted.

  (b) ``approved=False`` — the delete tool is absent from
      ``AgentStartEvent.tool_schemas`` and the trace contains no
      ``ToolInvokeEvent`` naming ``delete_record``.

Both cases use a real Anthropic client. The load-bearing invariants are
the ``tool_schemas`` membership pin (cheap, deterministic) and the
tool-invocation trace pin (proves the agent actually acted, or didn't,
on the visibility contract rather than just receiving the right schema).

Acceptance criteria (parametrized):
  - ``AgentStartEvent.tool_schemas`` membership matches the
    approved/not-approved state — ``search_records`` always present,
    ``delete_record`` present iff ``approved=True``.
  - When ``approved=True``, at least one ``ToolInvokeEvent`` with
    ``tool_name == "delete_record"`` fires.
  - When ``approved=False``, zero ``ToolInvokeEvent`` instances name
    ``delete_record`` — the model cannot call what it cannot see.
  - The agent terminates with ``complete`` in both cases (the task is
    not designed to be loop-prone).
"""

from __future__ import annotations

import pytest

from nanitics.infrastructure import AgentStartEvent, ToolInvokeEvent
from nanitics.specialized import ConditionalTool
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


@tool("search_records", "Search for records by keyword.")
async def search_records(keyword: str) -> str:
    return f"Found 2 records matching '{keyword}': rec-1, rec-2"


@tool("delete_record", "Delete a record by ID. Irreversible.")
async def delete_record(record_id: str) -> str:
    return f"Deleted record {record_id}"


@pytest.mark.parametrize(
    ("approved", "should_see_delete"),
    [
        pytest.param(True, True, id="approved_true_delete_visible"),
        pytest.param(False, False, id="approved_false_delete_hidden"),
    ],
)
async def test_conditional_tool_visibility(
    traced_emitter: InMemoryEmitter,
    approved: bool,
    should_see_delete: bool,
) -> None:
    client = make_llm_client("anthropic")

    conditional_delete = ConditionalTool(
        tool=delete_record,
        is_enabled=lambda state: bool(state.get("approved", False)),
    )

    # In the approved case the task asks for a delete after searching, so
    # the model has a reason to exercise the tool. In the unapproved case
    # the task only asks for a search — both to avoid prompt-vs-schema
    # contradiction and to make the "zero invocations" assertion clean.
    if approved:
        user_task = (
            "First use `search_records` with keyword 'active' to find "
            "candidates, then call `delete_record` on the first record ID "
            "returned. Confirm what you deleted."
        )
    else:
        user_task = (
            "Use `search_records` with keyword 'active' and report the "
            "matching record IDs. Do not attempt any other action."
        )

    agent = ReActAgent(
        name="conditional-ops",
        llm_client=client,
        emitter=traced_emitter,
        system_prompt=(
            "You are a records operations assistant. Only use tools that are "
            "listed as available to you. Do not mention or attempt tools "
            "that are not in your available toolset."
        ),
        tools=[search_records, conditional_delete],
        tool_state={"approved": approved},
        max_iterations=5,
    )

    result = await run_with_retry(
        lambda: agent.run(user_task),
        max_attempts=2,
    )

    # --- Result-shape invariant ---
    assert result.termination_reason == "complete", (
        f"Expected termination_reason='complete', got: {result.termination_reason!r}"
    )

    # --- Schema-visibility invariant (the load-bearing pin) ---
    start_event = assert_trace_contains(traced_emitter, AgentStartEvent)
    schema_names = [s.name for s in start_event.tool_schemas]

    assert "search_records" in schema_names, (
        f"Expected 'search_records' to always be visible; schemas were: {schema_names}"
    )
    if should_see_delete:
        assert "delete_record" in schema_names, (
            f"Expected 'delete_record' to be visible when approved=True; schemas were: {schema_names}"
        )
    else:
        assert "delete_record" not in schema_names, (
            f"Expected 'delete_record' to be HIDDEN when approved=False; schemas were: {schema_names}"
        )

    # --- Invocation invariant ---
    # The model can only invoke what it can see. This turns the schema
    # check into a behavioural guarantee, not just a configuration echo.
    delete_invocations = [
        e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent) and e.tool_name == "delete_record"
    ]
    if should_see_delete:
        assert len(delete_invocations) >= 1, (
            "Expected at least one ToolInvokeEvent for 'delete_record' when approved=True "
            f"(the predicate should make it invocable); got {len(delete_invocations)}."
        )
    else:
        assert len(delete_invocations) == 0, (
            "Expected zero ToolInvokeEvent for 'delete_record' when approved=False "
            f"(it must not be reachable); got {len(delete_invocations)}."
        )
