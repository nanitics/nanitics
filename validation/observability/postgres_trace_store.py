"""Persistent trace store backed by PostgreSQL, end-to-end.

Exercises ``PostgresTraceStore`` (``nanitics.infrastructure.observability
.postgres_store``) against a live PostgreSQL database. Drives a real
``ReActAgent`` through a ``TraceCollector`` so the store receives the
full event shape (``llm.request``, ``llm.response``, ``agent.start``,
``agent.end``) and pins cross-table round-trip fidelity.

Each test uses unique table names (trace + runs) so parallel runs do
not collide, and drops them on teardown so reruns are deterministic.

Acceptance criteria:
  - ``ensure_schema()`` succeeds on fresh, uniquely-named tables and
    leaves them at the current migration version.
  - ``register_run``, ``update_run_status("completed")``, and
    ``get_run`` round-trip the run record.
  - Events flushed via ``TraceCollector`` are retrievable via
    ``query_events`` (no filters) with count == records flushed.
  - ``query_events(event_types=["llm.request"])`` returns every
    ``LLMRequestEvent`` payload, and each payload preserves the
    ``messages`` list — pins JSONB encode/decode of complex event
    shapes (list of dicts) end-to-end.
  - ``query_events(event_types=["llm.response"])`` preserves the
    nested ``usage`` object; ``input_tokens`` and ``output_tokens``
    survive JSONB round-trip as ints.
  - ``get_summary`` aggregates llm_calls and token totals that match a
    manual sum across the store's llm.response events.
  - ``get_span_tree(trace_id)`` returns every stored event for that
    trace, ordered by ``(sdk_timestamp, id)``.
  - ``get_event(id_that_does_not_exist)`` returns ``None`` (never
    fabricated defaults).

Teardown drops ``trace_events``, ``runs``, and the schema-version
tracking table whether the test passes or fails.
"""

from __future__ import annotations

import uuid

import pytest

from nanitics.strategies import ReActAgent
from nanitics.tracing import (
    InMemoryEmitter,
    PostgresTraceStore,
    RunResult,
    TraceCollector,
)
from validation.helpers import (
    make_llm_client,
    make_postgres_pool,
    requires_postgres,
    run_with_retry,
)


def _unique_names() -> tuple[str, str]:
    """Return unique (trace_table, runs_table) names for parallel-run safety."""
    tag = uuid.uuid4().hex[:10]
    return (f"val_trace_{tag}", f"val_runs_{tag}")


async def _drop_tables(pool: object, trace_table: str, runs_table: str) -> None:
    """Drop the trace + runs tables and their schema version companion."""
    async with pool.acquire() as conn:  # type: ignore[attr-defined]
        # Order matters only if runs had FK back to trace_events; they don't.
        await conn.execute(f"DROP TABLE IF EXISTS {trace_table}")
        await conn.execute(f"DROP TABLE IF EXISTS {runs_table}")
        await conn.execute(f"DROP TABLE IF EXISTS _{trace_table}_schema_version")


@pytest.mark.quick
@requires_postgres
async def test_postgres_trace_store_endtoend(traced_emitter: InMemoryEmitter) -> None:
    trace_table, runs_table = _unique_names()
    run_id = f"pg-trace-run-{uuid.uuid4().hex[:8]}"

    async with make_postgres_pool() as pool:
        store = PostgresTraceStore(pool, table_name=trace_table, runs_table=runs_table)
        try:
            await run_with_retry(lambda: store.ensure_schema(), max_attempts=2)

            # Register the run before emitting anything.
            await store.register_run(run_id, trace_id=traced_emitter.trace_id, metadata={"probe": "pg-trace"})

            # Wire a collector listener, drive a real Anthropic round-trip.
            collector = TraceCollector(store=store, parent_id=run_id)
            traced_emitter.add_listener(collector.handle)

            agent = ReActAgent(
                name="pg-trace-agent",
                llm_client=make_llm_client("anthropic"),
                emitter=traced_emitter,
                system_prompt="Reply with one short word.",
                tools=[],
                max_iterations=1,
            )
            await run_with_retry(lambda: agent.run("Say OK."), max_attempts=2)

            await store.update_run_status(run_id, "completed", result=RunResult(output="ok"))
            await collector.flush()
            await collector.close()

            # --- Run lifecycle round-trip ---
            run = await store.get_run(run_id)
            assert run is not None, "Registered run must be retrievable by id."
            assert run.status == "completed", f"Expected 'completed' status; got {run.status!r}"
            assert run.metadata == {"probe": "pg-trace"}, f"Metadata must round-trip JSONB intact; got {run.metadata!r}"
            assert run.result is not None
            assert run.result.output == "ok"

            # --- All events retrievable ---
            all_events = await store.query_events(run_id, limit=500)
            assert all_events, "Expected at least one stored event after the agent run."

            # --- Complex event round-trip: LLMRequestEvent messages list ---
            request_events = await store.query_events(run_id, event_types=["llm.request"], limit=500)
            assert request_events, "Real agent run must yield at least one llm.request event."
            for e in request_events:
                messages = e.payload.get("messages")
                assert isinstance(messages, list), (
                    f"llm.request payload must preserve messages as a list; got {messages!r}"
                )
                assert messages, f"llm.request payload must preserve a non-empty messages list; got {messages!r}"
                assert all(isinstance(m, dict) and "role" in m for m in messages), (
                    f"Each message entry must round-trip as a dict with a 'role' key; got {messages!r}"
                )

            # --- Complex event round-trip: LLMResponseEvent usage object ---
            response_events = await store.query_events(run_id, event_types=["llm.response"], limit=500)
            assert response_events, "Real agent run must yield at least one llm.response event."
            manual_input = 0
            manual_output = 0
            for e in response_events:
                usage = e.payload.get("usage")
                assert isinstance(usage, dict), (
                    f"Usage must survive as a nested dict, not a scalar or string; got {usage!r}"
                )
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]
                assert isinstance(input_tokens, int), f"input_tokens must round-trip as int; got {usage!r}"
                assert isinstance(output_tokens, int), f"output_tokens must round-trip as int; got {usage!r}"
                manual_input += input_tokens
                manual_output += output_tokens

            # --- Summary aggregation matches manual reduction ---
            summary = await store.get_summary(run_id)
            assert summary.llm_calls == len(response_events), (
                f"summary.llm_calls={summary.llm_calls} must equal stored llm.response count {len(response_events)}."
            )
            assert summary.total_input_tokens == manual_input, (
                f"summary.total_input_tokens={summary.total_input_tokens} != manual sum {manual_input}."
            )
            assert summary.total_output_tokens == manual_output, (
                f"summary.total_output_tokens={summary.total_output_tokens} != manual sum {manual_output}."
            )

            # --- Span tree: every event for the trace id shows up ---
            tree_events = await store.get_span_tree(traced_emitter.trace_id)
            assert len(tree_events) == len(all_events), (
                f"get_span_tree must return every event for the trace; "
                f"tree={len(tree_events)} parent-scoped={len(all_events)}."
            )
            timestamps = [(e.sdk_timestamp, e.id) for e in tree_events]
            assert timestamps == sorted(timestamps), (
                f"get_span_tree must return events ordered by (sdk_timestamp, id); got {timestamps!r}"
            )

            # --- Negative: unknown event id must surface as None ---
            missing = await store.get_event(event_id=-1)
            assert missing is None, f"get_event on a non-existent id must return None; got {missing!r}"
        finally:
            await _drop_tables(pool, trace_table, runs_table)


@pytest.mark.quick
@requires_postgres
async def test_postgres_trace_store_parent_child_runs() -> None:
    """Exercise the v3 parent_run_id schema: register parent + children, filter
    via the three-state ``parent_run_id`` predicate, delete-cascade the family.
    """
    trace_table, runs_table = _unique_names()
    parent_id = f"pg-parent-{uuid.uuid4().hex[:8]}"
    child_a_id = f"pg-child-a-{uuid.uuid4().hex[:8]}"
    child_b_id = f"pg-child-b-{uuid.uuid4().hex[:8]}"

    async with make_postgres_pool() as pool:
        store = PostgresTraceStore(pool, table_name=trace_table, runs_table=runs_table)
        try:
            await run_with_retry(lambda: store.ensure_schema(), max_attempts=2)

            await store.register_run(parent_id, trace_id="trace-parent", metadata={"role": "parent"})
            await store.register_run(
                child_a_id, trace_id="trace-parent", metadata={"role": "child"}, parent_run_id=parent_id
            )
            await store.register_run(
                child_b_id, trace_id="trace-child-b", metadata={"role": "child"}, parent_run_id=parent_id
            )

            # ``parent_run_id=None`` returns only top-level runs.
            top_level = await store.list_runs(parent_run_id=None)
            top_level_ids = {r.id for r in top_level}
            assert parent_id in top_level_ids
            assert child_a_id not in top_level_ids
            assert child_b_id not in top_level_ids
            assert await store.count_runs(parent_run_id=None) >= 1

            # ``parent_run_id=parent_id`` returns the two children.
            children = await store.list_runs(parent_run_id=parent_id)
            child_ids = {r.id for r in children}
            assert child_ids == {child_a_id, child_b_id}, f"unexpected children: {child_ids!r}"
            assert await store.count_runs(parent_run_id=parent_id) == 2

            # Default (sentinel) returns all runs (parent + 2 children).
            all_runs = await store.list_runs()
            all_ids = {r.id for r in all_runs}
            assert {parent_id, child_a_id, child_b_id}.issubset(all_ids)

            # Round-trip the parent_run_id field on the child rows.
            child_record = await store.get_run(child_a_id)
            assert child_record is not None
            assert child_record.parent_run_id == parent_id

            # Cascade delete: removing the parent drops the children too.
            assert await store.delete_run(parent_id) is True
            assert await store.get_run(parent_id) is None
            assert await store.get_run(child_a_id) is None
            assert await store.get_run(child_b_id) is None
        finally:
            await _drop_tables(pool, trace_table, runs_table)


@pytest.mark.quick
@requires_postgres
async def test_postgres_trace_store_sibling_conflict_raises() -> None:
    """Two stores sharing ``table_name`` but with different ``runs_table`` values:
    the second ``ensure_schema()`` must raise rather than silently re-run
    migrations under a fresh runs_table name.
    """
    trace_table, runs_table_a = _unique_names()
    _, runs_table_b = _unique_names()

    async with make_postgres_pool() as pool:
        store_a = PostgresTraceStore(pool, table_name=trace_table, runs_table=runs_table_a)
        store_b = PostgresTraceStore(pool, table_name=trace_table, runs_table=runs_table_b)
        try:
            await run_with_retry(lambda: store_a.ensure_schema(), max_attempts=2)
            with pytest.raises(RuntimeError) as excinfo:
                await store_b.ensure_schema()
            message = str(excinfo.value)
            assert runs_table_b in message
            assert trace_table in message
        finally:
            await _drop_tables(pool, trace_table, runs_table_a)
            await _drop_tables(pool, trace_table, runs_table_b)
