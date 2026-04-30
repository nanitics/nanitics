"""Event emission and tracing: emitters, events, spans, listeners, child emitters, and filtering.

Covers InMemoryEmitter for event collection, the BaseEvent/TraceEvent type hierarchy, hierarchical
spans for structured traces, real-time listeners, child emitters for linked multi-agent traces,
memory capping with max_events, and event level classification for filtering.

Related guide: docs/guides/observability.md
"""

import asyncio

from pydantic import TypeAdapter

from nanitics import (
    InMemoryEmitter,
    TraceEvent,
    Usage,
)
from nanitics.infrastructure import (
    AgentCompleteEvent,
    AgentStartEvent,
    AgentStepEvent,
    BaseEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    SpanEndEvent,
    SpanStartEvent,
    ToolInvokeEvent,
    ToolResultEvent,
    classify_level,
    is_level_included,
)


async def main() -> None:
    # --- Section 1: Creating an Emitter and Emitting Events ---
    print("--- Section 1: Creating an Emitter and Emitting Events ---")

    emitter = InMemoryEmitter(trace_id="demo-trace")

    # Emitter provides trace context for constructing events
    assert emitter.trace_id == "demo-trace", "trace_id should match constructor arg"
    assert emitter.span_id is not None, "span_id is auto-generated at root"
    assert emitter.parent_span_id is None, "parent_span_id is None at root level"
    print(f"  trace_id: {emitter.trace_id}")
    print(f"  span_id: {emitter.span_id}")
    print(f"  parent_span_id: {emitter.parent_span_id}")

    # Emit events by constructing them with trace context from the emitter
    emitter.emit(
        AgentStartEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            agent_name="demo-agent",
            task_input="Summarize the report",
            tools_available=["search", "read_file"],
        )
    )
    emitter.emit(
        AgentCompleteEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            agent_name="demo-agent",
            output="Report summary complete.",
            total_steps=3,
            termination_reason="complete",
        )
    )

    assert len(emitter.events) == 2, "Should have exactly 2 events"
    assert emitter.events[0].trace_id == emitter.events[1].trace_id, "Events share trace_id"
    assert emitter.events[0].span_id == emitter.events[1].span_id, "Events share span_id"
    assert emitter.events[0].event_id != emitter.events[1].event_id, "Each event has unique event_id"
    assert emitter.events[0].timestamp is not None, "Timestamp auto-generated"
    assert emitter.events[0].event_type == "agent.start"
    assert emitter.events[1].event_type == "agent.complete"
    print(f"  Events emitted: {[e.event_type for e in emitter.events]}")
    print(f"  event_id (first): {emitter.events[0].event_id}")

    # --- Section 2: Event Type Hierarchy ---
    print("\n--- Section 2: Event Type Hierarchy ---")

    emitter = InMemoryEmitter(trace_id="type-hierarchy")

    emitter.emit(
        LLMRequestEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
        )
    )
    emitter.emit(
        LLMResponseEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="claude-3",
            content="Hi there!",
            usage=Usage(input_tokens=10, output_tokens=5),
            duration_ms=120.0,
        )
    )
    emitter.emit(
        ToolInvokeEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            tool_call_id="call-1",
            tool_name="search",
            parameters={"query": "test"},
        )
    )
    emitter.emit(
        ToolResultEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            tool_call_id="call-1",
            tool_name="search",
            result="Found 3 results",
            success=True,
            duration_ms=45.0,
        )
    )

    # All events inherit from BaseEvent
    for event in emitter.events:
        assert isinstance(event, BaseEvent), f"{event.event_type} should be a BaseEvent"
    print(f"  All {len(emitter.events)} events are BaseEvent instances ✓")

    # Filter by isinstance — type-safe Python approach
    llm_events = [e for e in emitter.events if isinstance(e, (LLMRequestEvent, LLMResponseEvent))]
    tool_events = [e for e in emitter.events if isinstance(e, (ToolInvokeEvent, ToolResultEvent))]
    assert len(llm_events) == 2, "Should have 2 LLM events"
    assert len(tool_events) == 2, "Should have 2 tool events"
    print(f"  LLM events (isinstance): {[e.event_type for e in llm_events]}")
    print(f"  Tool events (isinstance): {[e.event_type for e in tool_events]}")

    # Filter by event_type string — useful for serialized/query contexts
    llm_by_string = [e for e in emitter.events if e.event_type.startswith("llm.")]
    tool_by_string = [e for e in emitter.events if e.event_type.startswith("tool.")]
    assert llm_events == llm_by_string, "Both filtering approaches yield same results"
    assert tool_events == tool_by_string, "Both filtering approaches yield same results"
    print("  String-based filtering matches isinstance filtering ✓")

    # TraceEvent discriminated union: deserialize back to correct type
    adapter = TypeAdapter(TraceEvent)
    for event in emitter.events:
        roundtripped = adapter.validate_python(event.model_dump())
        assert type(roundtripped) is type(event), f"Roundtrip preserves type for {event.event_type}"
        assert roundtripped.event_type == event.event_type
    print("  TraceEvent roundtrip deserialization preserves types ✓")

    # --- Section 3: Spans — Hierarchical Structure ---
    print("\n--- Section 3: Spans — Hierarchical Structure ---")

    emitter = InMemoryEmitter(trace_id="span-demo")
    root_span = emitter.span_id

    with emitter.span("data-processing"):
        processing_span = emitter.span_id
        assert processing_span != root_span, "span_id changes inside span"

        with emitter.span("fetch"):
            fetch_span = emitter.span_id
            assert fetch_span != processing_span, "Nested span gets new span_id"

            # Events inside a span inherit its trace context
            emitter.emit(
                ToolInvokeEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    tool_call_id="fetch-1",
                    tool_name="http_get",
                    parameters={"url": "https://api.example.com/data"},
                )
            )
            # The event's span_id matches the inner span
            tool_event = next(e for e in emitter.events if isinstance(e, ToolInvokeEvent))
            assert tool_event.span_id == fetch_span, "Event span_id matches inner span"
            assert tool_event.parent_span_id == processing_span, "Event parent_span_id matches outer span"

        with emitter.span("transform"):
            transform_span = emitter.span_id
            assert transform_span != fetch_span, "Sibling spans have different span_ids"

            emitter.emit(
                ToolInvokeEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    tool_call_id="transform-1",
                    tool_name="parse_json",
                    parameters={"format": "csv"},
                )
            )

    # After exiting all spans, span_id returns to root
    assert emitter.span_id == root_span, "span_id returns to root after exiting all spans"
    print("  Root span restored after nesting ✓")

    # Inspect span events
    span_starts = [e for e in emitter.events if isinstance(e, SpanStartEvent)]
    span_ends = [e for e in emitter.events if isinstance(e, SpanEndEvent)]
    assert len(span_starts) == 3, f"Expected 3 span starts, got {len(span_starts)}"
    assert len(span_ends) == 3, f"Expected 3 span ends, got {len(span_ends)}"

    span_names = [e.name for e in span_starts]
    assert span_names == ["data-processing", "fetch", "transform"], f"Span names: {span_names}"
    print(f"  Span names: {span_names}")

    for end in span_ends:
        assert end.duration_ms >= 0, f"Span '{end.name}' has non-negative duration"
    print("  All span durations non-negative ✓")

    # --- Section 4: Listeners — Real-Time Event Processing ---
    print("\n--- Section 4: Listeners — Real-Time Event Processing ---")

    emitter = InMemoryEmitter(trace_id="listener-demo")

    # Register a listener that collects all events
    collected: list[TraceEvent] = []
    emitter.add_listener(lambda e: collected.append(e))

    # Register a filtered listener that only collects tool events
    tool_only: list[TraceEvent] = []
    emitter.add_listener(lambda e: tool_only.append(e) if e.event_type.startswith("tool.") else None)

    # Emit a mix of event types
    emitter.emit(
        LLMRequestEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="claude-3",
            messages=[{"role": "user", "content": "test"}],
        )
    )
    emitter.emit(
        ToolInvokeEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            tool_call_id="call-1",
            tool_name="search",
            parameters={"q": "test"},
        )
    )
    emitter.emit(
        LLMResponseEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="claude-3",
            content="response",
            usage=Usage(input_tokens=10, output_tokens=5),
            duration_ms=100.0,
        )
    )
    emitter.emit(
        ToolResultEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            tool_call_id="call-1",
            tool_name="search",
            result="results",
            success=True,
            duration_ms=30.0,
        )
    )

    assert len(collected) == 4, f"All-events listener received {len(collected)}, expected 4"
    assert len(tool_only) == 2, f"Tool-only listener received {len(tool_only)}, expected 2"
    assert all(e.event_type.startswith("tool.") for e in tool_only)
    print(f"  All-events listener: {len(collected)} events")
    print(f"  Tool-only listener: {len(tool_only)} events")

    # Streaming pattern: push to an asyncio.Queue
    queue: asyncio.Queue[TraceEvent] = asyncio.Queue()
    emitter2 = InMemoryEmitter(trace_id="queue-demo")
    emitter2.add_listener(lambda e: queue.put_nowait(e))

    emitter2.emit(
        AgentStartEvent(
            trace_id=emitter2.trace_id,
            span_id=emitter2.span_id,
            parent_span_id=emitter2.parent_span_id,
            agent_name="streamed",
            task_input="stream test",
            tools_available=[],
        )
    )
    emitter2.emit(
        AgentCompleteEvent(
            trace_id=emitter2.trace_id,
            span_id=emitter2.span_id,
            parent_span_id=emitter2.parent_span_id,
            agent_name="streamed",
            output="done",
            total_steps=1,
            termination_reason="complete",
        )
    )

    queued_events = []
    while not queue.empty():
        queued_events.append(queue.get_nowait())
    assert len(queued_events) == 2, "Queue received all events"
    assert queued_events[0].event_type == "agent.start"
    assert queued_events[1].event_type == "agent.complete"
    print(f"  Queue streaming pattern: {len(queued_events)} events drained ✓")

    # --- Section 5: Child Emitters — Linked Traces ---
    print("\n--- Section 5: Child Emitters — Linked Traces ---")

    parent = InMemoryEmitter(trace_id="parent-trace")

    # Register a listener on the parent before creating the child
    parent_listener_events: list[TraceEvent] = []
    parent.add_listener(lambda e: parent_listener_events.append(e))

    with parent.span("orchestration"):
        parent_current_span = parent.span_id

        # Create a child emitter linked to the parent's current span
        child = parent.create_child()

        # Child shares the same trace_id
        assert child.trace_id == parent.trace_id, "Child shares parent's trace_id"
        # Child's parent_span_id links to parent's current span
        assert child.parent_span_id == parent_current_span, "Child links to parent's span"
        # Child has its own span_id
        assert child.span_id != parent.span_id, "Child has independent span_id"
        print(f"  Child trace_id matches parent: {child.trace_id}")
        print(f"  Child parent_span_id: {child.parent_span_id}")

    # Emit events via child
    child.emit(
        AgentStartEvent(
            trace_id=child.trace_id,
            span_id=child.span_id,
            parent_span_id=child.parent_span_id,
            agent_name="child-agent",
            task_input="subtask",
            tools_available=["tool_a"],
        )
    )

    # Child has its own events list.
    assert len(child.events) == 1, "Child has its own events"
    # Child-emitter events are forwarded into the parent's ``events`` list so
    # composite-agent inner events surface in the outer trace. The parent is
    # the authoritative source for trace consumers (e.g., ``save_trace``).
    parent_agent_events = [e for e in parent.events if isinstance(e, AgentStartEvent)]
    assert len(parent_agent_events) == 1, "Parent receives child's events via forwarding"
    assert parent_agent_events[0] is child.events[0], "Parent and child reference the same event"
    print(f"  Child events: {len(child.events)}, Parent agent events: {len(parent_agent_events)} ✓")

    # Listeners registered on parent are copied to child
    assert len(parent_listener_events) > 0, "Parent listener received child's event"
    child_emitted = [e for e in parent_listener_events if isinstance(e, AgentStartEvent)]
    assert len(child_emitted) == 1, "Parent's listener captured child's AgentStartEvent"
    assert child_emitted[0].agent_name == "child-agent"
    print(f"  Parent listener received child event: {child_emitted[0].event_type} ✓")

    # --- Section 6: Memory Capping with max_events ---
    print("\n--- Section 6: Memory Capping with max_events ---")

    emitter = InMemoryEmitter(trace_id="bounded", max_events=5)

    # Emit 10 events
    for i in range(10):
        emitter.emit(
            AgentStepEvent(
                trace_id=emitter.trace_id,
                span_id=emitter.span_id,
                parent_span_id=emitter.parent_span_id,
                agent_name="capped-agent",
                step_number=i,
            )
        )

    assert len(emitter.events) == 5, f"Expected 5 events, got {len(emitter.events)}"
    # Only the last 5 events remain (steps 5–9)
    step_numbers = [e.step_number for e in emitter.events]
    assert step_numbers == [5, 6, 7, 8, 9], f"Expected steps 5-9, got {step_numbers}"
    print(f"  {len(emitter.events)} events retained (max_events=5)")
    print(f"  Step numbers: {step_numbers} (last 5 of 10) ✓")

    # --- Section 7: Event Levels — Classification ---
    print("\n--- Section 7: Event Levels — Classification ---")

    # Classify representative event types
    assert classify_level("agent.start") == "info", "agent.start is info level"
    assert classify_level("llm.response") == "debug", "llm.response is debug level"
    assert classify_level("span.start") == "verbose", "span.start is verbose level"
    print(f"  agent.start → {classify_level('agent.start')}")
    print(f"  llm.response → {classify_level('llm.response')}")
    print(f"  span.start → {classify_level('span.start')}")

    # Level inclusion: threshold determines what's visible
    assert is_level_included("info", "info") is True, "info included at info"
    assert is_level_included("debug", "info") is False, "debug excluded at info"
    assert is_level_included("info", "debug") is True, "info included at debug"
    assert is_level_included("debug", "debug") is True, "debug included at debug"
    assert is_level_included("verbose", "debug") is False, "verbose excluded at debug"
    assert is_level_included("verbose", "verbose") is True, "verbose included at verbose"
    print("  info threshold: includes info only")
    print("  debug threshold: includes info + debug")
    print("  verbose threshold: includes all")

    # Practical pattern: filter emitter events by level
    emitter = InMemoryEmitter(trace_id="level-filter")
    emitter.emit(
        AgentStartEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            agent_name="test",
            task_input="test",
            tools_available=[],
        )
    )
    emitter.emit(
        LLMResponseEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            model_name="claude-3",
            usage=Usage(input_tokens=10, output_tokens=5),
            duration_ms=100.0,
        )
    )
    with emitter.span("inner"):
        pass  # Emits span.start (verbose) and span.end (verbose)

    info_events = [e for e in emitter.events if is_level_included(classify_level(e.event_type), "info")]
    debug_events = [e for e in emitter.events if is_level_included(classify_level(e.event_type), "debug")]
    all_events = [e for e in emitter.events if is_level_included(classify_level(e.event_type), "verbose")]

    assert len(info_events) == 1, f"Info level: {len(info_events)} events (agent.start only)"
    assert len(debug_events) == 2, f"Debug level: {len(debug_events)} events (agent.start + llm.response)"
    assert len(all_events) == len(emitter.events), "Verbose level includes everything"
    print(f"  Filtered at info: {len(info_events)} event(s)")
    print(f"  Filtered at debug: {len(debug_events)} event(s)")
    print(f"  Filtered at verbose: {len(all_events)} event(s)")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
