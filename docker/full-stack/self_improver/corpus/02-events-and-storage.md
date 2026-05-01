# Events and Storage

## Events

Every runtime behaviour in Nanitics — agent steps, LLM requests and
responses, tool invocations and results, span lifecycle, run lifecycle,
working-memory updates, human-in-the-loop handoffs — is represented as
a frozen pydantic event. All events inherit from `BaseEvent` with a
shared skeleton (`event_id`, `trace_id`, `span_id`, `parent_span_id`,
`timestamp`, `event_type`) and a typed, variant-specific payload.

The full union is the `TraceEvent` discriminated-union type. Adopters
that reason about typed events should validate through the canonical
`TypeAdapter(TraceEvent)`; the discriminator is the `event_type` field.

## From event to row

When an agent runs under `TracedExecutor`, a `TraceCollector` listens
on the emitter and batches events. On each flush the collector calls
`PersistentTraceStore.save_events_batch(parent_id, records)` with a
list of `TraceEventRecord` values. Each record carries the tracing
skeleton plus a `payload: dict[str, Any]` — the result of
`event.model_dump(mode="json")` on the original `TraceEvent`.

On read, the store returns `StoredTraceEvent` rows — the same shape
plus a database-assigned `id: int`. The payload remains a dict; the
store does not re-hydrate it into a typed variant for the caller.

## Read-path protocol

`PersistentTraceStore` exposes several read methods:

- `get_span_tree(trace_id)` — every event for a trace, ordered for
  tree reconstruction. This is what a trace-reading tool wants when it
  needs the full run.
- `query_events(parent_id, ...)` — filtered, cursor-paginated retrieval
  for live streaming and targeted queries.
- `get_events_by_span(trace_id, span_id)` — events within one span.
- `get_summary(parent_id)` — aggregated statistics for a parent.

Both the serialise-to-dict path on write and the dict-on-read contract
are stable; adopters composing their own tooling can rely on them.
