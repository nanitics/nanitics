"""Trace collection and storage: persisting, querying, and analyzing agent execution traces.

Covers the full observability pipeline — storing traces, collecting events in real time,
querying stored data, inspecting aggregate statistics, managing runs, and SSE streaming.
Builds on examples/tools/event_emitter.py which covers the emitter side (creation, emission, spans,
listeners, level classification). This example picks up where that one leaves off: what
happens to events after they're emitted.

Related guide: docs/guides/observability.md
"""

import asyncio
from datetime import UTC, datetime, timedelta

from nanitics.infrastructure import (
    AgentCompleteEvent,
    AgentStartEvent,
    LLMResponseEvent,
    SpanStartEvent,
    ToolInvokeEvent,
    ToolResultEvent,
    classify_level,
)
from nanitics.tracing import (
    InMemoryEmitter,
    InMemoryPersistentTraceStore,
    InMemoryTraceStore,
    RunRecord,
    StoredTraceEvent,
    Trace,
    TraceCollector,
    TracedExecutor,
    TraceEventRecord,
    TraceQuery,
    TraceSummary,
    TraceSummaryStats,
    Usage,
)


async def main() -> None:
    # --- Section 1: Simple Trace Storage ---
    print("--- Section 1: Simple Trace Storage ---")

    # InMemoryTraceStore persists full traces as a unit — a trace ID plus its events.
    # Good for local dev and testing. For production per-event storage, see Section 2.
    store = InMemoryTraceStore()

    # Build a trace from manually constructed events (no emitter needed).
    now = datetime.now(UTC)
    trace_id = "trace-simple-001"

    events = [
        AgentStartEvent(
            trace_id=trace_id,
            span_id="span-1",
            agent_name="analyzer",
            task_input="Analyze the dataset",
            tools_available=["search"],
            timestamp=now,
        ),
        LLMResponseEvent(
            trace_id=trace_id,
            span_id="span-1",
            model_name="claude-3",
            content="I'll analyze the dataset.",
            usage=Usage(input_tokens=100, output_tokens=50),
            duration_ms=200.0,
            timestamp=now + timedelta(milliseconds=100),
        ),
        LLMResponseEvent(
            trace_id=trace_id,
            span_id="span-1",
            model_name="claude-3",
            content="Here are my findings.",
            usage=Usage(input_tokens=200, output_tokens=80),
            duration_ms=300.0,
            timestamp=now + timedelta(milliseconds=300),
        ),
        AgentCompleteEvent(
            trace_id=trace_id,
            span_id="span-1",
            agent_name="analyzer",
            output="Analysis complete.",
            total_steps=2,
            termination_reason="complete",
            timestamp=now + timedelta(milliseconds=500),
        ),
    ]

    trace = Trace(trace_id=trace_id, events=events)
    await store.save_trace(trace)

    # Retrieve by ID
    loaded = await store.get_trace(trace_id)
    assert loaded is not None, "Trace should be retrievable"
    assert loaded.trace_id == trace_id
    assert len(loaded.events) == 4, "Should have 4 events"
    print(f"  Loaded trace '{loaded.trace_id}' with {len(loaded.events)} events")

    # Query by time range — matching range
    summaries = await store.query_traces(
        TraceQuery(start_time=now - timedelta(seconds=1), end_time=now + timedelta(seconds=1))
    )
    assert len(summaries) == 1, "Should find one trace in time range"
    summary: TraceSummary = summaries[0]
    assert summary.trace_id == trace_id
    assert summary.event_count == 4
    # Token totals are summed from LLMResponseEvent usage
    assert summary.total_input_tokens == 300, "100 + 200 input tokens"
    assert summary.total_output_tokens == 130, "50 + 80 output tokens"
    print(
        f"  Summary: {summary.event_count} events, "
        f"{summary.total_input_tokens} in / {summary.total_output_tokens} out tokens"
    )

    # Query with non-matching time range
    empty = await store.query_traces(TraceQuery(start_time=now + timedelta(hours=1)))
    assert len(empty) == 0, "No traces should match future time range"
    print("  Non-matching time range: 0 results (correct)")

    # Pagination: save a second trace and verify offset/limit
    trace2 = Trace(
        trace_id="trace-simple-002",
        events=[
            AgentStartEvent(
                trace_id="trace-simple-002",
                span_id="span-2",
                agent_name="summarizer",
                task_input="Summarize results",
                tools_available=[],
                timestamp=now + timedelta(seconds=1),
            ),
        ],
    )
    await store.save_trace(trace2)

    page1 = await store.query_traces(TraceQuery(limit=1, offset=0))
    assert len(page1) == 1, "Page 1 should have 1 trace"
    page2 = await store.query_traces(TraceQuery(limit=1, offset=1))
    assert len(page2) == 1, "Page 2 should have 1 trace"
    assert page1[0].trace_id != page2[0].trace_id, "Pages should have different traces"
    all_traces = await store.query_traces(TraceQuery(limit=10))
    assert len(all_traces) == 2, "Should have 2 total traces"
    print(f"  Pagination: {len(all_traces)} total traces, pages work correctly")

    # --- Section 2: Persistent Event-Level Storage ---
    print("\n--- Section 2: Persistent Event-Level Storage ---")

    # InMemoryPersistentTraceStore stores events individually with per-event IDs,
    # filtered queries, and aggregate statistics. This is the storage tier
    # applications use in production (via PostgresTraceStore).
    persistent_store = InMemoryPersistentTraceStore()
    parent_id = "run-001"

    # Construct TraceEventRecord objects manually
    base_time = datetime.now(UTC)
    records = [
        TraceEventRecord(
            event_type="agent.start",
            level="info",
            trace_id="trace-persist-001",
            span_id="span-a",
            parent_span_id=None,
            payload={"agent_name": "researcher", "task_input": "Research topic X"},
            sdk_timestamp=base_time,
        ),
        TraceEventRecord(
            event_type="llm.response",
            level="debug",
            trace_id="trace-persist-001",
            span_id="span-a",
            parent_span_id=None,
            payload={
                "model_name": "claude-3",
                "content": "Here are my findings...",
                "usage": {"input_tokens": 150, "output_tokens": 60, "total_tokens": 210},
                "duration_ms": 250.0,
            },
            sdk_timestamp=base_time + timedelta(milliseconds=100),
        ),
        TraceEventRecord(
            event_type="tool.invoke",
            level="debug",
            trace_id="trace-persist-001",
            span_id="span-a",
            parent_span_id=None,
            payload={"tool_name": "search", "tool_call_id": "tc-1", "parameters": {"query": "topic X"}},
            sdk_timestamp=base_time + timedelta(milliseconds=200),
        ),
        TraceEventRecord(
            event_type="agent.complete",
            level="info",
            trace_id="trace-persist-001",
            span_id="span-a",
            parent_span_id=None,
            payload={"agent_name": "researcher", "output": "Done", "total_steps": 1, "termination_reason": "complete"},
            sdk_timestamp=base_time + timedelta(milliseconds=500),
        ),
    ]

    await persistent_store.save_events_batch(parent_id, records)

    # Query all events
    all_events = await persistent_store.query_events(parent_id)
    assert len(all_events) == 4, "Should have 4 stored events"
    # StoredTraceEvent extends TraceEventRecord with a database-assigned ID
    assert all(isinstance(e, StoredTraceEvent) for e in all_events)
    assert all(e.id > 0 for e in all_events), "Each event gets a positive ID"
    print(f"  Stored {len(all_events)} events, IDs: {[e.id for e in all_events]}")

    # Level filtering — only info events
    info_events = await persistent_store.query_events(parent_id, levels=["info"])
    assert len(info_events) == 2, "Should have 2 info events (agent.start, agent.complete)"
    assert all(e.level == "info" for e in info_events)
    print(f"  Info-only filter: {[e.event_type for e in info_events]}")

    # Event type filtering
    llm_events = await persistent_store.query_events(parent_id, event_types=["llm.response"])
    assert len(llm_events) == 1, "Should have 1 llm.response event"
    assert llm_events[0].event_type == "llm.response"
    print(f"  Type filter (llm.response): {len(llm_events)} event")

    # Cursor pagination with after_id
    first_two = await persistent_store.query_events(parent_id, limit=2)
    assert len(first_two) == 2
    cursor = first_two[-1].id
    remaining = await persistent_store.query_events(parent_id, after_id=cursor)
    assert len(remaining) == 2, "Should have 2 remaining events after cursor"
    assert remaining[0].id > cursor, "Events after cursor should have higher IDs"
    print(f"  Cursor pagination: first 2 IDs {[e.id for e in first_two]}, next 2 IDs {[e.id for e in remaining]}")

    # Single event retrieval
    single = await persistent_store.get_event(all_events[0].id)
    assert single is not None
    assert single.event_type == "agent.start"
    print(f"  Single event retrieval: ID {single.id} → {single.event_type}")

    # Aggregate statistics
    stats: TraceSummaryStats = await persistent_store.get_summary(parent_id)
    assert stats.total_events == 4
    assert stats.events_by_level["info"] == 2
    assert stats.events_by_level["debug"] == 2
    assert stats.llm_calls == 1
    assert stats.tool_calls == 1
    assert stats.total_input_tokens == 150
    assert stats.total_output_tokens == 60
    assert "researcher" in stats.agent_names
    print(
        f"  Summary stats: {stats.total_events} events, {stats.llm_calls} LLM calls, "
        f"{stats.tool_calls} tool calls, {stats.total_input_tokens} in / {stats.total_output_tokens} out tokens"
    )

    # --- Section 3: TraceCollector Pipeline ---
    print("\n--- Section 3: TraceCollector Pipeline ---")

    # TraceCollector bridges EventEmitter → PersistentTraceStore.
    # It classifies events by level, buffers them, and flushes to the store.
    collector_store = InMemoryPersistentTraceStore()
    emitter = InMemoryEmitter(trace_id="trace-collector-001")
    collector_parent_id = "run-collector"

    collector = TraceCollector(store=collector_store, parent_id=collector_parent_id)

    # Register the collector as a listener — it receives every emitted event
    emitter.add_listener(collector.handle)

    # Emit events through the emitter (as you would during an agent run)
    emitter.emit(
        AgentStartEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            agent_name="collector-demo",
            task_input="Demonstrate the collector pipeline",
            tools_available=["search", "write"],
        )
    )
    emitter.emit(
        LLMResponseEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="claude-3",
            content="I'll search for the information.",
            usage=Usage(input_tokens=80, output_tokens=30),
            duration_ms=150.0,
        )
    )
    emitter.emit(
        ToolInvokeEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            tool_call_id="tc-demo-1",
            tool_name="search",
            parameters={"query": "collector pipeline"},
        )
    )
    emitter.emit(
        ToolResultEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            tool_call_id="tc-demo-1",
            tool_name="search",
            result="Found 3 results.",
            success=True,
            duration_ms=50.0,
        )
    )
    emitter.emit(
        AgentCompleteEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            agent_name="collector-demo",
            output="Pipeline demonstration complete.",
            total_steps=1,
            termination_reason="complete",
        )
    )

    # Flush buffered events to the store
    await collector.flush()

    # Verify events were persisted
    stored = await collector_store.query_events(collector_parent_id)
    assert len(stored) == 5, f"Expected 5 events, got {len(stored)}"
    stored_types = [e.event_type for e in stored]
    assert "agent.start" in stored_types
    assert "llm.response" in stored_types
    assert "tool.invoke" in stored_types
    assert "tool.result" in stored_types
    assert "agent.complete" in stored_types
    print(f"  Persisted {len(stored)} events: {stored_types}")

    # Verify levels were classified correctly by the collector
    for e in stored:
        expected_level = classify_level(e.event_type)
        assert e.level == expected_level, f"{e.event_type} should be {expected_level}, got {e.level}"
    print("  Level classification verified for all events")

    # Verify trace context is preserved
    assert all(e.trace_id == "trace-collector-001" for e in stored)
    assert all(e.span_id == emitter.span_id for e in stored)
    print(f"  Trace context preserved: trace_id='trace-collector-001', span_id='{emitter.span_id}'")

    # Verify payloads contain serialized event data
    start_event = next(e for e in stored if e.event_type == "agent.start")
    assert start_event.payload["agent_name"] == "collector-demo"
    assert start_event.payload["task_input"] == "Demonstrate the collector pipeline"
    print(f"  Payload verified: agent_name='{start_event.payload['agent_name']}'")

    # Close the collector (flushes remaining events and stops the background flush loop)
    await collector.close()
    print("  Collector closed")

    # --- Section 4: Querying and Hierarchy ---
    print("\n--- Section 4: Querying and Hierarchy ---")

    # Build a richer trace with nested spans to demonstrate hierarchy queries.
    # Reuse the collector_store from Section 3 and add more events.
    hierarchy_store = InMemoryPersistentTraceStore()
    hierarchy_emitter = InMemoryEmitter(trace_id="trace-hierarchy-001")
    hierarchy_parent = "run-hierarchy"

    hierarchy_collector = TraceCollector(store=hierarchy_store, parent_id=hierarchy_parent)
    hierarchy_emitter.add_listener(hierarchy_collector.handle)

    # Root-level agent start
    root_span_id = hierarchy_emitter.span_id
    hierarchy_emitter.emit(
        AgentStartEvent(
            trace_id=hierarchy_emitter.trace_id,
            span_id=root_span_id,
            parent_span_id=hierarchy_emitter.parent_span_id,
            agent_name="orchestrator",
            task_input="Coordinate the research",
            tools_available=["delegate"],
        )
    )

    # Enter a nested span (simulates a sub-task)
    with hierarchy_emitter.span("research-phase"):
        inner_span_id = hierarchy_emitter.span_id
        hierarchy_emitter.emit(
            LLMResponseEvent(
                trace_id=hierarchy_emitter.trace_id,
                span_id=inner_span_id,
                parent_span_id=hierarchy_emitter.parent_span_id,
                model_name="claude-3",
                content="Researching...",
                usage=Usage(input_tokens=50, output_tokens=20),
                duration_ms=100.0,
            )
        )
        hierarchy_emitter.emit(
            ToolInvokeEvent(
                trace_id=hierarchy_emitter.trace_id,
                span_id=inner_span_id,
                parent_span_id=hierarchy_emitter.parent_span_id,
                tool_call_id="tc-h-1",
                tool_name="delegate",
                parameters={"task": "deep dive"},
            )
        )

    # Back at root span — agent complete
    hierarchy_emitter.emit(
        AgentCompleteEvent(
            trace_id=hierarchy_emitter.trace_id,
            span_id=root_span_id,
            parent_span_id=None,
            agent_name="orchestrator",
            output="Research coordinated.",
            total_steps=2,
            termination_reason="complete",
        )
    )

    await hierarchy_collector.flush()

    # get_span_tree returns all events for a trace, ordered for tree reconstruction
    tree = await hierarchy_store.get_span_tree("trace-hierarchy-001")
    assert len(tree) > 0, "Span tree should have events"
    # Events are ordered by timestamp
    for i in range(1, len(tree)):
        assert tree[i].sdk_timestamp >= tree[i - 1].sdk_timestamp, "Events should be time-ordered"
    print(f"  Span tree: {len(tree)} events, time-ordered")

    # Identify span IDs in the tree
    span_ids_in_tree = {e.span_id for e in tree}
    assert len(span_ids_in_tree) >= 2, "Should have at least 2 distinct spans (root + nested)"
    print(f"  Distinct spans: {len(span_ids_in_tree)}")

    # get_events_by_span returns only events within a specific span
    inner_span_events = await hierarchy_store.get_events_by_span("trace-hierarchy-001", inner_span_id)
    assert len(inner_span_events) > 0, "Inner span should have events"
    assert all(e.span_id == inner_span_id for e in inner_span_events)
    inner_types = [e.event_type for e in inner_span_events]
    assert "llm.response" in inner_types
    assert "tool.invoke" in inner_types
    print(f"  Inner span events: {inner_types}")

    root_span_events = await hierarchy_store.get_events_by_span("trace-hierarchy-001", root_span_id)
    root_types = [e.event_type for e in root_span_events]
    assert "agent.start" in root_types
    assert "agent.complete" in root_types
    print(f"  Root span events: {root_types}")

    # Get summary with meaningful values from the richer trace
    hierarchy_stats = await hierarchy_store.get_summary(hierarchy_parent)
    assert hierarchy_stats.llm_calls == 1
    assert hierarchy_stats.tool_calls == 1
    assert "orchestrator" in hierarchy_stats.agent_names
    assert hierarchy_stats.total_input_tokens == 50
    assert hierarchy_stats.total_output_tokens == 20
    assert hierarchy_stats.total_duration_ms is not None, "Should have duration from multiple timestamps"
    print(
        f"  Stats: {hierarchy_stats.llm_calls} LLM calls, {hierarchy_stats.tool_calls} tool calls, "
        f"agents={hierarchy_stats.agent_names}, duration={hierarchy_stats.total_duration_ms}ms"
    )

    await hierarchy_collector.close()

    # --- Section 5: Run Management ---
    print("\n--- Section 5: Run Management ---")

    # PersistentTraceStore includes run management — register, track, query, and delete runs.
    run_store = InMemoryPersistentTraceStore()

    # Register a run
    await run_store.register_run(
        run_id="run-alpha",
        trace_id="trace-alpha",
        metadata={"task": "Analyze quarterly report", "user": "analyst-1"},
    )

    # Retrieve the run
    run: RunRecord | None = await run_store.get_run("run-alpha")
    assert run is not None
    assert run.id == "run-alpha"
    assert run.trace_id == "trace-alpha"
    assert run.status == "running", "Initial status should be 'running'"
    assert run.metadata["task"] == "Analyze quarterly report"
    assert run.started_at is not None
    assert run.completed_at is None, "Not completed yet"
    print(f"  Registered run '{run.id}': status={run.status}, task='{run.metadata['task']}'")

    # Update status to completed
    await run_store.update_run_status("run-alpha", "completed")
    run = await run_store.get_run("run-alpha")
    assert run is not None
    assert run.status == "completed"
    assert run.completed_at is not None, "Should have completed_at timestamp"
    print(f"  Updated status: {run.status}, completed_at={run.completed_at}")

    # Register more runs for filtering
    await run_store.register_run("run-beta", "trace-beta", {"task": "Draft summary"})
    await run_store.register_run("run-gamma", "trace-gamma", {"task": "Review findings"})
    await run_store.update_run_status("run-beta", "failed", error="LLM timeout")

    # List all runs
    all_runs = await run_store.list_runs()
    assert len(all_runs) == 3
    print(f"  Total runs: {len(all_runs)}")

    # Filter by status
    completed_runs = await run_store.list_runs(status="completed")
    assert len(completed_runs) == 1
    assert completed_runs[0].id == "run-alpha"
    print(f"  Completed runs: {[r.id for r in completed_runs]}")

    running_runs = await run_store.list_runs(status="running")
    assert len(running_runs) == 1
    assert running_runs[0].id == "run-gamma"
    print(f"  Running runs: {[r.id for r in running_runs]}")

    # Count runs
    total_count = await run_store.count_runs()
    assert total_count == 3
    completed_count = await run_store.count_runs(status="completed")
    assert completed_count == 1
    print(f"  Count: {total_count} total, {completed_count} completed")

    # Delete a run (also deletes associated events)
    # First, save some events under run-beta to verify cascade delete
    await run_store.save_events_batch(
        "run-beta",
        [
            TraceEventRecord(
                event_type="agent.start",
                level="info",
                trace_id="trace-beta",
                span_id="span-beta",
                parent_span_id=None,
                payload={"agent_name": "beta-agent"},
                sdk_timestamp=datetime.now(UTC),
            ),
        ],
    )
    events_before = await run_store.query_events("run-beta")
    assert len(events_before) == 1, "Should have 1 event before delete"

    deleted = await run_store.delete_run("run-beta")
    assert deleted is True
    assert await run_store.get_run("run-beta") is None, "Run should be gone"
    events_after = await run_store.query_events("run-beta")
    assert len(events_after) == 0, "Events should be deleted with the run"
    assert await run_store.count_runs() == 2, "Should have 2 remaining runs"
    print("  Deleted run-beta: run and events removed")

    # Delete non-existent run returns False
    assert await run_store.delete_run("non-existent") is False
    print("  Delete non-existent run: returns False (correct)")

    # --- Section 6: SSE Streaming Queue ---
    print("\n--- Section 6: SSE Streaming Queue ---")

    # TraceCollector can push qualifying events to an asyncio.Queue for SSE streaming.
    # The queue parameter + min_level control which events are streamed to clients.
    sse_store = InMemoryPersistentTraceStore()
    sse_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    sse_emitter = InMemoryEmitter(trace_id="trace-sse-001")

    sse_collector = TraceCollector(
        store=sse_store,
        parent_id="run-sse",
        queue=sse_queue,
        min_level="info",  # Only info-level events go to the queue
    )
    sse_emitter.add_listener(sse_collector.handle)

    # Emit events at different levels:
    # - agent.start → info (should appear in queue)
    # - llm.response → debug (should NOT appear in queue)
    # - span.start → verbose (should NOT appear in queue)
    # - agent.complete → info (should appear in queue)

    sse_emitter.emit(
        AgentStartEvent(
            trace_id=sse_emitter.trace_id,
            span_id=sse_emitter.span_id,
            parent_span_id=sse_emitter.parent_span_id,
            agent_name="sse-agent",
            task_input="Test SSE filtering",
            tools_available=[],
        )
    )
    sse_emitter.emit(
        LLMResponseEvent(
            trace_id=sse_emitter.trace_id,
            span_id=sse_emitter.span_id,
            parent_span_id=sse_emitter.parent_span_id,
            model_name="claude-3",
            content="This is a debug-level event.",
            usage=Usage(input_tokens=10, output_tokens=5),
            duration_ms=50.0,
        )
    )
    sse_emitter.emit(
        SpanStartEvent(
            trace_id=sse_emitter.trace_id,
            span_id=sse_emitter.span_id,
            parent_span_id=sse_emitter.parent_span_id,
            name="verbose-span",
        )
    )
    sse_emitter.emit(
        AgentCompleteEvent(
            trace_id=sse_emitter.trace_id,
            span_id=sse_emitter.span_id,
            parent_span_id=sse_emitter.parent_span_id,
            agent_name="sse-agent",
            output="SSE test done.",
            total_steps=1,
            termination_reason="complete",
        )
    )

    # All 4 events are buffered in the collector for storage, but only info events
    # were pushed to the queue.
    assert sse_queue.qsize() == 2, f"Expected 2 info events in queue, got {sse_queue.qsize()}"
    print(f"  Queue size after 4 events (min_level='info'): {sse_queue.qsize()}")

    # Inspect queue item format — this is the SSE payload format
    item1 = sse_queue.get_nowait()
    assert item1["event_type"] == "trace", "Queue items have event_type='trace'"
    payload = item1["payload"]
    assert isinstance(payload, dict)
    assert payload["sdk_event_type"] == "agent.start"
    assert payload["level"] == "info"
    assert payload["trace_id"] == "trace-sse-001"
    assert "span_id" in payload
    assert "timestamp" in payload
    print(
        f"  Queue item format: event_type='{item1['event_type']}', "
        f"sdk_event_type='{payload['sdk_event_type']}', level='{payload['level']}'"
    )

    item2 = sse_queue.get_nowait()
    payload2 = item2["payload"]
    assert payload2["sdk_event_type"] == "agent.complete"
    assert payload2["level"] == "info"
    print(f"  Second queue item: sdk_event_type='{payload2['sdk_event_type']}'")

    # Queue is now empty — debug and verbose events were not pushed
    assert sse_queue.empty(), "Queue should be empty after reading 2 info events"
    print("  Queue empty: debug/verbose events correctly filtered out")

    # Flush and close — all 4 events still get persisted to the store regardless of level
    await sse_collector.flush()
    all_stored = await sse_store.query_events("run-sse")
    assert len(all_stored) == 4, "All 4 events should be persisted (level filtering only affects the queue)"
    print(f"  Store has all {len(all_stored)} events (queue filtering doesn't affect persistence)")

    await sse_collector.close()
    print("  SSE collector closed")

    # --- Section 7: TracedExecutor — run lifecycle in one call ---
    print("\n--- Section 7: TracedExecutor ---")

    # TracedExecutor composes InMemoryEmitter, TraceCollector, and PersistentTraceStore
    # into a single entry point. It handles: generate IDs → register run → create emitter
    # → wire collector → execute → finalize status. Events are persisted in real-time
    # (not batched after completion), so even failed runs retain their trace data.

    executor_store = InMemoryPersistentTraceStore()
    executor = TracedExecutor(executor_store)

    # The callback receives an EventEmitter and returns any result.
    # TracedExecutor manages the full run lifecycle around it.

    async def my_work(emitter: InMemoryEmitter, run_id: str) -> str:
        del run_id  # unused in this factory
        with emitter.span("processing"):
            emitter.emit(
                AgentStartEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    agent_name="worker",
                    task_input="process data",
                    tools_available=[],
                )
            )
            emitter.emit(
                AgentCompleteEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    agent_name="worker",
                    output="done",
                    total_steps=1,
                    termination_reason="complete",
                )
            )
        return "all good"

    run_id, result = await executor.execute(my_work, metadata={"task": "example"})
    assert result == "all good"
    print(f"  Run completed: run_id={run_id}, result='{result}'")

    # Run is registered and marked completed
    run = await executor_store.get_run(run_id)
    assert run is not None
    assert run.status == "completed"
    assert run.metadata["task"] == "example"
    print(f"  Run status: {run.status}, metadata: {run.metadata}")

    # Events were persisted in real-time via the internal TraceCollector
    events = await executor_store.query_events(run_id)
    assert len(events) > 0
    print(f"  {len(events)} events persisted")

    # Optional SSE queue — same pattern as Section 6, but wired automatically
    sse_queue_exec: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    run_id2, _ = await executor.execute(my_work, queue=sse_queue_exec)
    assert not sse_queue_exec.empty()
    print(f"  SSE queue received events for run {run_id2}")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
