# Redaction Hooks

Nanitics runs redaction at event emission time, not on read. The
`RedactionHook` protocol lets adopters scrub sensitive fields from
event payloads before they leave the process.

## Protocol

A `RedactionHook` is a callable taking a `TraceEvent` and returning a
redacted `TraceEvent` (frozen models are replaced via `model_copy`,
not mutated). Hooks are registered on the `TraceCollector` and run
inside the collector's per-event pipeline before the event is placed
into the flush buffer.

## When the hook runs

- The emitter emits the original event to every listener first. In-memory
  observers see the unredacted event.
- When the `TraceCollector` receives the event through its listener
  registration, it applies every registered `RedactionHook` in
  declaration order.
- The redacted event is what ends up in the flush buffer and, via
  `save_events_batch`, in the `PersistentTraceStore`.

## What a hook can and cannot do

- **Can** replace scalar fields (API keys, user identifiers, email
  addresses) with placeholders.
- **Can** rewrite free-form content (`LLMResponseEvent.content`,
  tool-result payloads) to strip embedded secrets.
- **Cannot** re-type an event or add new required fields — the pydantic
  schema enforces the shape.
- **Cannot** retroactively redact events already flushed. Redaction is
  fail-fast: a hook that raises propagates the exception.

## Implication for trace consumers

By the time a `PersistentTraceStore` returns a `StoredTraceEvent`, the
payload has already been redacted. Consumers do not re-apply
redaction. They can safely display the stored payload verbatim as
long as the hook pipeline on the write side was correct.
