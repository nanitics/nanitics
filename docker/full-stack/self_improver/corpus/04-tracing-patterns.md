# Tracing Patterns

## `TracedExecutor`

`TracedExecutor` is the standard entry point for running an agent
under full observability. It wraps one factory callable — the caller
supplies a coroutine that builds and runs an agent given an
`EventEmitter` — and in one call:

1. Generates a fresh `run_id` and `trace_id`.
2. Registers the run with the `PersistentTraceStore`.
3. Builds an `InMemoryEmitter` and wires a `TraceCollector` that
   listens on it and flushes to the store.
4. Awaits the caller's factory, passing the emitter and the
   pre-generated `run_id` in.
5. Finalises run status (completed, failed with error, suspended) and
   closes the collector.

One line captures the common case:

```
run_id, result = await executor.execute(
    lambda emitter, run_id: build_agent(emitter).run(task),
    metadata={"runner": "self-improver"},
)
```

## Nesting runs

Two `executor.execute(...)` calls in the same process produce two
independent runs. The second run gets a fresh `trace_id`, so the
Observatory renders them as separate entries, but an application can
link them semantically — for example by threading the first run's
`trace_id` into the second run's `metadata`, or by reading the first
run's events and emitting them as context inside the second run.

## Consuming traces as data

The `PersistentTraceStore` read methods return `StoredTraceEvent` rows.
Tooling that wants typed `TraceEvent` values — `analyze()`, a bespoke
critic agent, or any post-hoc analyser — calls the SDK helper
`trace_events_from_stored(events)` to round-trip stored rows back
through the canonical `TraceEvent` adapter. Input order is preserved,
malformed rows raise `MalformedStoredEventError`, never silently skip.

## Embedded Observatory

`create_observatory_router(store, static_dir=...)` mounts a FastAPI
router that exposes the store over HTTP and serves the React viewer
bundle. The embedded-compose image uses it and adopter apps mount it
behind their own auth.
