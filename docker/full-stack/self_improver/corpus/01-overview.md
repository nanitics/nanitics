# Observability in Nanitics — Overview

Nanitics ships a first-class observability story. Every agent emits
structured events through an `EventEmitter`; those events form a
**trace** — a typed, hierarchical record of one run.

The core building blocks an adopter composes are:

- `EventEmitter` — the protocol every emitter implements. Agents,
  workflows, tools, and evaluators write events through it.
- `TracedExecutor` — glues an in-memory emitter, a `TraceCollector`, and
  a `PersistentTraceStore` together so one `executor.execute(...)` call
  registers a run, collects the events, and finalises run status.
- `PersistentTraceStore` — the protocol that persists events
  individually. `PostgresTraceStore` is the production implementation;
  `InMemoryPersistentTraceStore` is for tests.
- **Observatory** — the `mount_observatory(app, store)` helper that
  attaches the JSON API and the embedded React bundle. The bundle ships
  inside the wheel; no frontend toolchain is needed at the call site.

This file is a high-level map. Details — how events are serialised,
how redaction interacts with storage, what `TracedExecutor` does
step-by-step, and how to consume traces programmatically — live in the
other files in this corpus. Reading any one file alone is not enough
to answer a non-trivial question about the observability surface.
