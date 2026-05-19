"""Redaction hooks: scrubbing adopter-surface content before persistence and streaming.

Shows how a ``RedactionHook`` fires inside ``TraceCollector.handle()`` — before events
are persisted to the store and before they land on the SSE streaming queue, while the
in-process emitter continues to see the un-redacted event. Covers both wire-in points
(``TraceCollector`` constructor and per-run ``TracedExecutor.execute(redaction_hook=...)``)
and demonstrates the fail-closed exception semantics the SDK guarantees.

Related guide: docs/guides/observability.md#trace-surface-hygiene
"""

import asyncio
import re

from nanitics.infrastructure.observability import (
    AgentCompleteEvent,
    AgentStartEvent,
    LLMRequestEvent,
)
from nanitics.tracing import (
    InMemoryEmitter,
    InMemoryPersistentTraceStore,
    RedactionHook,
    TraceCollector,
    TracedExecutor,
    TraceEvent,
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class ScrubEmails:
    """Replace email addresses in ``LLMRequestEvent.system_prompt`` with ``[email]``.

    Runtime-checkable against ``RedactionHook`` via ``isinstance(obj, RedactionHook)``.
    A real hook would enumerate every event type and field your threat model covers
    (prompts, tool inputs and outputs, custom event fields, tool exception messages).
    """

    def redact(self, event: TraceEvent) -> TraceEvent:
        if isinstance(event, LLMRequestEvent) and event.system_prompt:
            scrubbed = _EMAIL.sub("[email]", event.system_prompt)
            if scrubbed != event.system_prompt:
                return event.model_copy(update={"system_prompt": scrubbed})
        return event


def _llm_request(emitter: InMemoryEmitter) -> LLMRequestEvent:
    return LLMRequestEvent(
        trace_id=emitter.trace_id,
        span_id=emitter.span_id,
        parent_span_id=emitter.parent_span_id,
        model_name="mock",
        system_prompt="Contact the admin at ops@example.com for access.",
        messages=[],
        tools=[],
    )


async def main() -> None:
    # --- Section 1: Collector-level hook — store and SSE queue both redacted ---
    print("--- Section 1: TraceCollector(redaction_hook=...) ---")

    # Wire the hook on the collector itself — applies to every run this collector
    # handles. Use this path when the policy is global (same rules for every run).
    store = InMemoryPersistentTraceStore()
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    emitter = InMemoryEmitter(trace_id="trace-redact-1")
    hook = ScrubEmails()
    assert isinstance(hook, RedactionHook), "runtime-checkable protocol"

    collector = TraceCollector(
        store=store,
        parent_id="run-redact-1",
        queue=queue,
        min_level="debug",  # llm.request is debug-level; push it to the SSE queue
        redaction_hook=hook,
    )
    emitter.add_listener(collector.handle)

    emitter.emit(_llm_request(emitter))
    await collector.flush()

    # In-process listeners see the un-redacted event — the hook does not run on
    # emit, only inside ``TraceCollector.handle()``.
    assert isinstance(emitter.events[0], LLMRequestEvent)
    assert "ops@example.com" in (emitter.events[0].system_prompt or "")
    print("  emitter.events (un-redacted): system_prompt contains 'ops@example.com'")

    # The store received the redacted copy — the hook ran before the persistence
    # record was built.
    stored = await store.query_events("run-redact-1", event_types=["llm.request"])
    assert len(stored) == 1
    assert "[email]" in stored[0].payload["system_prompt"]
    assert "ops@example.com" not in stored[0].payload["system_prompt"]
    print("  store (redacted):             system_prompt contains '[email]'")

    # The SSE queue item also reflects the redaction — the hook ran before the
    # queue push too.
    sse_item = queue.get_nowait()
    sse_payload = sse_item["payload"]
    assert isinstance(sse_payload, dict)
    assert sse_payload["sdk_event_type"] == "llm.request"
    assert "[email]" in sse_payload["system_prompt"]
    print("  SSE queue (redacted):         system_prompt contains '[email]'")

    await collector.close()

    # --- Section 2: Per-run hook via TracedExecutor.execute ---
    print("\n--- Section 2: TracedExecutor.execute(redaction_hook=...) ---")

    # Use this path when the policy varies per run — e.g. tenant-specific scrubbing
    # where the hook is constructed from request state. The executor wires the
    # hook into the internal TraceCollector for the duration of this run only.
    executor_store = InMemoryPersistentTraceStore()
    executor = TracedExecutor(executor_store)

    async def _work(em: InMemoryEmitter, run_id: str) -> str:
        del run_id  # unused in this factory
        em.emit(
            AgentStartEvent(
                trace_id=em.trace_id,
                span_id=em.span_id,
                parent_span_id=em.parent_span_id,
                agent_name="worker",
                task_input="demo",
                tools_available=[],
            )
        )
        em.emit(_llm_request(em))
        em.emit(
            AgentCompleteEvent(
                trace_id=em.trace_id,
                span_id=em.span_id,
                parent_span_id=em.parent_span_id,
                agent_name="worker",
                output="done",
                total_steps=1,
                termination_reason="complete",
            )
        )
        return "ok"

    run_id, _ = await executor.execute(_work, redaction_hook=ScrubEmails())

    llm_records = await executor_store.query_events(run_id, event_types=["llm.request"])
    assert len(llm_records) == 1
    assert "[email]" in llm_records[0].payload["system_prompt"]
    assert "ops@example.com" not in llm_records[0].payload["system_prompt"]
    print(f"  run {run_id[:8]}… persisted llm.request with '[email]'")

    # --- Section 3: Fail-closed exception semantics ---
    print("\n--- Section 3: fail-closed on hook exception ---")

    # If the hook raises, the event is neither persisted nor pushed to the queue.
    # Silently persisting the un-redacted event would defeat the property the
    # adopter wired the hook in for — so the exception propagates to the caller.

    class BrokenHook:
        def redact(self, event: TraceEvent) -> TraceEvent:
            raise RuntimeError("redaction policy misconfigured")

    fc_store = InMemoryPersistentTraceStore()
    fc_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    fc_emitter = InMemoryEmitter(trace_id="trace-redact-fc")
    fc_collector = TraceCollector(
        store=fc_store,
        parent_id="run-redact-fc",
        queue=fc_queue,
        min_level="debug",
        redaction_hook=BrokenHook(),
    )

    try:
        fc_collector.handle(_llm_request(fc_emitter))
    except RuntimeError as exc:
        print(f"  hook raised: {exc}")

    await fc_collector.flush()
    not_stored = await fc_store.query_events("run-redact-fc")
    assert not_stored == [], "failing hook must not persist the event"
    assert fc_queue.empty(), "failing hook must not enqueue the event"
    print(f"  store:     {len(not_stored)} events (correct — fail-closed)")
    print("  SSE queue: empty        (correct — fail-closed)")

    await fc_collector.close()

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
