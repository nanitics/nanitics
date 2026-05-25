# Observability

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Every agent requires an `EventEmitter`. The emitter captures structured events during execution — agent steps, LLM calls, tool invocations, memory operations, multi-agent coordination, and more. These events form a **trace**: a complete, typed record of everything the agent did, organized into hierarchical spans.

Nanitics emits 85+ event types covering the entire runtime surface. You can listen to events in real time (for streaming UIs, logging, or metrics) and persist traces for later analysis.

## When to Use

**Always provide an emitter.** It's a required dependency for every agent. Use `InMemoryEmitter` for local development and testing. For production, add listeners that forward events to your observability stack or stream them to clients.

**Use trace storage** when you need to persist and query execution history — debugging, analytics, audit trails.

## EventEmitter

> **See also:** [`examples/tools/event_emitter.py`](../../examples/tools/event_emitter.py) — runnable example covering emitters, events, spans, listeners, child emitters, memory capping, and event level classification.

`EventEmitter` is a runtime-checkable protocol. All emitters must implement `emit()`, `add_listener()`, `span()`, and expose `trace_id`, `span_id`, and `parent_span_id` properties.

`InMemoryEmitter` is the built-in implementation. It stores events in memory, supports hierarchical spans via context variables (async-safe), and optionally caps memory with `max_events`.

> **See also:** `EventEmitter` protocol and `InMemoryEmitter` docstrings for the full API surface.

For non-agent LLM calls (evaluators, context-transfer summarizers, bid generators, orchestrator planners), wrap the client in `InstrumentedLLMClient` to keep those calls in the trace — see [`examples/observability/instrumented_client.py`](../../examples/observability/instrumented_client.py).

## Event Hierarchy

All events inherit from `BaseEvent`, which provides common fields: `event_id`, `trace_id`, `span_id`, `parent_span_id`, `timestamp`, and `event_type` (the discriminator). Events are frozen Pydantic models organized into 25 categories covering the full runtime surface:

| Category | What it covers |
|----------|---------------|
| Agent Lifecycle | Start, step, complete, error |
| LLM Calls | Request and response with token usage |
| Tool Calls | Invocation and result with timing |
| Error Recovery | Retry, correction, degradation |
| Context Management | Truncation, summarization, assembly |
| Memory | Working, long-term, semantic, episodic, shared |
| Evaluation & Reflection | Evaluation results, revisions, self-reflection |
| Spans | Span start/end with duration |
| Planning | Plan creation, step updates, goal status, revisions |
| Code Execution | Code submission and execution results |
| Tree Search & MCTS | Node creation, evaluation, pruning, backpropagation |
| Workflow Orchestration | Workflow lifecycle, step completion, structure |
| Run Lifecycle | Run start, complete, failed, suspended |
| Multi-Agent (Basic) | Delegation, handoff, supervision |
| Broadcast | Task broadcast, responses, aggregation |
| Bidding | Auction start, bids, allocation |
| JudgeRouter | Routing start, rankings, allocation |
| Debate | Arguments, resolution, completion |
| Consensus | Voting, agreement measurement |
| Peer Network | Peer consultations |
| Message Bus | Publication, delivery, completion |
| Blackboard | Round-based contributions |
| HITL | Human input request and response |
| Revision | Revision workflow lifecycle |
| Durable Execution | Suspension, resumption, checkpoints |
| Model Routing | Backend selection |

`TraceEvent` is a discriminated union of all event types (using Pydantic's `Discriminator` on `event_type`), enabling type-safe deserialization.

### `AgentStepEvent` — reasoning vs. artifact

`AgentStepEvent` carries four semantically distinct fields, each with its own contract:

- `thought` — free-text reasoning from the model on this step. Sourced from `LLMResponse.reasoning_text`. Never populated with structured output, parsed JSON, or final content.
- `action` — what the agent did (tool name, concatenated code blocks). `None` for agents that did not act on this step.
- `observation` — what the agent observed (tool result, code execution output, or — on a terminal no-tool-calls step — the model's final content). `None` when nothing was observed and no final content was produced.
- `artifact` — structured per-step output as `dict[str, Any]`. Producers call `model.model_dump()` on the Pydantic model representing the step's structured output; consumers type-assert on the agent type and parse.

The field-level docstrings on `AgentStepEvent` in `nanitics/infrastructure/observability/events.py` are the authoritative contract.

> **See also:** Event class docstrings in `nanitics/infrastructure/observability/events.py` for fields and semantics of each event type.

## Level Classification

The SDK classifies its 85+ event types into three inclusive levels. This is the primary mechanism for controlling event granularity — use it to decide what to stream to clients, what to persist, and what to display.

| Level | Includes | What it captures |
|-------|----------|-----------------|
| `info` | info only | User-visible milestones: agent lifecycle, workflow lifecycle, multi-agent coordination, HITL, planning milestones, evaluation, durable execution |
| `debug` | info + debug | Operational detail: LLM calls, tool invocations, agent steps, context operations, error recovery, planning revisions, reflection, code execution |
| `verbose` | all | Everything: memory operations, spans, tree search/MCTS, all coordination primitives, model routing |

Levels are **inclusive** — `"debug"` includes all `"info"` events, `"verbose"` includes everything.

### Choosing a Level

The right level depends on the consumer:

- **SSE streaming to a UI**: Start with `"info"`. This gives the user meaningful progress updates (agent started, workflow step completed, evaluation result) without flooding the connection. Upgrade to `"debug"` for a developer-facing trace viewer.
- **Persistent storage**: Store everything (`"verbose"`), filter on read. Storage is cheap; missing events during debugging is expensive.
- **Cost tracking / analytics**: `"debug"` captures `LLMResponseEvent` (token usage) and `ToolResultEvent` (timing), which are the primary signals for cost and performance analysis.
- **TraceCollector queue**: Set `min_level` to control what reaches the SSE stream. The collector always persists all events to the store regardless of this setting — it only affects the queue.

### Level API

Use `classify_level()` to get the level of any event type, and `is_level_included()` to check containment:

```python
from nanitics.tracing import classify_level, is_level_included

classify_level("agent.start")           # "info"
classify_level("llm.response")          # "debug"
classify_level("memory.semantic.search") # "verbose"

is_level_included("info", "debug")      # True — info is included in debug
is_level_included("verbose", "debug")   # False — verbose is not in debug
```

### Level Reference (Condensed)

**Info events** — agent lifecycle (`agent.start`, `agent.complete`, `agent.error`), workflow lifecycle, run lifecycle, multi-agent coordination (delegation, handoff, supervision), HITL, evaluation results, durable execution (suspension/resumption), planning milestones (plan created, goal status), blackboard lifecycle.

**Debug events** — LLM calls (request/response), tool calls (invoke/result), agent steps, context operations (truncation, summarization, assembly), error recovery (retry, correction, degradation), planning updates (step updated, plan revised), revision workflow, reflection, code execution.

**Verbose events** — everything else: all memory events (working, long-term, semantic, episodic, shared), span events, tree search/MCTS, broadcast, bidding, debate, consensus, peer network, message bus, model routing, checkpoints.

> **See also:** `classify_level` and `LEVEL_CLASSIFICATIONS` in the source for the complete mapping.

## Spans

Spans create hierarchical structure within a trace. Use `emitter.span()` as a context manager to group related operations — events emitted inside a span are linked to it via `span_id` and `parent_span_id`. Spans automatically emit `SpanStartEvent` on entry and `SpanEndEvent` on exit (with duration). The span stack uses `ContextVar`, making it safe for concurrent async operations.

The agent loop creates spans for each step, and workflows create spans for each step in the workflow. This produces a tree structure where the root span contains the entire execution and child spans represent individual operations.

Spans are the foundation of the Observatory's tree visualization — the span hierarchy determines how the trace is rendered as a collapsible tree.

## Listeners

Register callbacks via `add_listener()` for real-time event processing. Listeners are called synchronously on `emit()` and receive every event — filter by type or level within the callback.

### Two tiers: external vs. internal

`add_listener(callback, *, internal=False)` has two failure modes, picked by the `internal` flag. The default `internal=False` is for adopter-supplied listeners (Slack alerters, metrics, SSE bridges) — if the listener raises, `emit()` catches the exception, issues a `warnings.warn`, and continues, so a buggy listener cannot crash the agent run. `internal=True` is for SDK-internal infrastructure (the trace collector, blackboard contribution forwarding) — exceptions propagate out of `emit()` and fail the run, so an observability-layer failure surfaces instead of silently truncating the trace.

Dispatch order in `emit()` is externals first, then internals. An SSE consumer observes the event before a failing internal listener short-circuits the dispatch.

Rationale: SDK-internal observability failures must surface as run failures; adopter listeners must not crash the run.

```python
# External (default): adopter code, soft-fail with warning.
emitter.add_listener(slack_alerter)

# Internal: SDK infrastructure, propagate on failure.
emitter.add_listener(collector.handle, internal=True)
```

Most application code uses the default. Reach for `internal=True` only when the listener is part of the SDK's own observability plumbing and you want its failures to abort the run.

Common listener patterns:
- **SSE streaming** — push events to an `asyncio.Queue`, consume from your SSE endpoint
- **Cost tracking** — filter for `LLMResponseEvent`, aggregate token usage
- **Logging** — write formatted events to structured logging
- **Metrics** — extract timing data from `ToolResultEvent` and `SpanEndEvent`

Listeners are the extension point for integrating with external systems. The SDK provides no built-in listeners beyond `TraceCollector` — you write callbacks tailored to your needs.

### Async sink listeners

Because `emit()` runs listeners synchronously, a listener cannot `await` — so any sink that requires async I/O (Postgres, HTTP, message brokers, SSE) must bridge from the sync listener to an async consumer. The pattern is the same in every case: the listener enqueues the event on an `asyncio.Queue`, and a background task drains the queue and performs the awaitable writes.

`TraceCollector` is the reference implementation — its `handle` method is a sync listener that buffers events and a background flush loop writes to a `PersistentTraceStore` (e.g. `PostgresTraceStore`). When writing your own async sink, follow the same shape rather than calling `asyncio.run_coroutine_threadsafe` or blocking the emit thread.

> **See also:** [Building Applications](building-applications.md) for the full SSE streaming pattern using listeners and async queues.

## Child Emitters and Binding

When agents are composed into workflows or multi-agent patterns, they need to share a single trace. `InMemoryEmitter.create_child()` creates a child emitter that shares the parent's `trace_id` and links to the parent's current span, producing a connected span tree. Child emitters have their own independent span stack (async-safe) and receive copies of the parent's listeners.

The `Agent.bind()` method replaces an agent's emitter with a child emitter linked to a parent. You typically don't call `create_child()` or `bind()` directly — all built-in composition patterns (workflows, `Broadcast`, `Debate`, `Consensus`, `Bidding`, `Blackboard`, `Supervisor`, `AgentTool`, `Handoff`, `PeerNetwork`, `MessageBus`, `ReflexionAgent`) bind agents automatically before running them.

The result is a single connected span tree for the entire execution, regardless of how many agents participate.

**Why this matters:** Without binding, each agent would have its own independent trace. Binding is what makes it possible to see a supervisor's delegation, the worker's execution, and a sub-workflow's steps all in one hierarchical view.

## Trace Storage

The SDK provides two storage protocols, serving different use cases. Choosing between them is a real architectural decision:

### TraceStore — Full-Trace Storage

`TraceStore` persists and retrieves complete traces as a unit. A `Trace` is a frozen model wrapping a `trace_id` and its `list[TraceEvent]`. Use `TraceQuery` to filter and paginate by time range. `InMemoryTraceStore` is the built-in implementation.

**Use when:** Local development, testing, simple applications where you save the full trace after the agent completes.

**Trade-off:** Simple to use, but you can only save/load entire traces. No querying during execution, no filtering by level or event type, no run lifecycle management.

### PersistentTraceStore — Per-Event Storage

`PersistentTraceStore` supports per-event persistence with level-based filtering, cursor pagination, span tree retrieval, and run lifecycle management. It's designed for production applications that need to query events while the agent is still running.

**Use when:** Production applications, live SSE streaming, retrospective trace analysis, run management dashboards.

**Trade-off:** More complex to implement and requires a database, but unlocks the full observatory stack (live streaming, span trees, run management, filtered queries).

Key capabilities beyond `TraceStore`:
- **Filtered queries** — retrieve events by level, event type, or span, with cursor-based pagination
- **Span tree retrieval** — get all events for a trace ordered for tree reconstruction
- **Run management** — register runs, update status, list/filter runs
- **Summary statistics** — aggregated event counts, token totals, error counts per run

`PostgresTraceStore` is the production-ready implementation backed by PostgreSQL (via `asyncpg`). Use `PostgresTraceStore.get_schema_sql()` to get the migration DDL.

> **See also:** `TraceStore`, `PersistentTraceStore`, and `PostgresTraceStore` docstrings for the full protocol surface and data models.

#### Parent/child runs

`RunRecord.parent_run_id` models a specialist (child) run dispatched on behalf of another run. Top-level runs leave the field as `None`; child runs set it to the parent's `id`. The runs table carries a self-referencing foreign key with `ON DELETE CASCADE`, so deleting a parent deletes its children (and, transitively, their descendants).

`list_runs` and `count_runs` take a three-state `parent_run_id` filter: omitted (the default) applies no filter and returns runs at every level; `None` restricts to top-level runs only; a string restricts to children of that parent.

<!-- verify: skip — illustrative fragment; `store` is caller-supplied and the `await` runs inside an async context -->
```python
await store.register_run("parent-1", "trace-1", {"task": "research"})
await store.register_run("child-a", "trace-1", {}, parent_run_id="parent-1")
await store.register_run("child-b", "trace-2", {}, parent_run_id="parent-1")

top_level = await store.list_runs(parent_run_id=None)        # → [parent-1]
children = await store.list_runs(parent_run_id="parent-1")   # → [child-a, child-b]

await store.delete_run("parent-1")                            # CASCADE removes both children
```

The SDK does not enforce `trace_id` parity between parent and child — the caller decides whether children share the parent's trace or each carry their own.

### Consuming traces from an external agent

`PersistentTraceStore.get_span_tree(trace_id)` returns `list[StoredTraceEvent]` — rows whose `payload` is a plain dict (produced on the write side by `event.model_dump(mode="json")`). Tools that reason over typed events — a bespoke critic, any post-hoc analyser — use the `trace_events_from_stored` helper to round-trip those rows back into the canonical `list[TraceEvent]`:

<!-- verify: skip — illustrative fragment; `trace_store` and `trace_id` are caller-supplied and the `await` runs inside an async context -->
```python
from nanitics.tracing import trace_events_from_stored

stored = await trace_store.get_span_tree(trace_id)
events = trace_events_from_stored(stored)
```

Input order is preserved; a row that fails validation raises `MalformedStoredEventError` rather than being silently skipped.

## TraceCollector

> **See also:** [`examples/observability/trace_collection.py`](../../examples/observability/trace_collection.py) — trace storage, collection pipeline, and SSE streaming queue.

`TraceCollector` bridges the `EventEmitter` and `PersistentTraceStore`. Register its `handle` method as an emitter listener — it classifies events by level, buffers them in memory, and flushes to the store on a configurable interval. Optionally, it pushes qualifying events (filtered by `min_level`) to an async queue for live SSE streaming.

The collector manages its own lifecycle: it starts a background flush loop on the first event and cancels it on `close()`. Always call `close()` in a `finally` block to flush remaining buffered events.

### Pipeline Architecture

The typical production pipeline is:

```
EventEmitter → [listener] → TraceCollector → PersistentTraceStore
                                   ↓
                            asyncio.Queue → SSE endpoint → client
```

1. Agent emits events through `EventEmitter`
2. `TraceCollector.handle` (registered as a listener) classifies, buffers, and optionally queues events
3. Background flush loop persists buffered events to `PersistentTraceStore`
4. Observatory SSE endpoint reads from the store (or directly from the queue) to stream to clients

You can wire this pipeline manually, or use `TracedExecutor` (see below) which composes these steps automatically.

## TracedExecutor

> **See also:** [`examples/observability/trace_collection.py`](../../examples/observability/trace_collection.py) Section 7 — `TracedExecutor` usage with and without SSE queue.

`TracedExecutor` composes `InMemoryEmitter`, `TraceCollector`, and `PersistentTraceStore` into a single entry point for run lifecycle management. It handles the full sequence: generate IDs → register run → create emitter → wire collector → execute → finalize status.

<!-- verify: skip — illustrative wiring; `asyncpg_pool`, `my_agent_factory`, `task`, `doc_id` are caller-supplied and the `await` runs inside an async context -->
```python
from nanitics.tracing import PostgresTraceStore, TracedExecutor

trace_store = PostgresTraceStore(pool=asyncpg_pool)
executor = TracedExecutor(trace_store)

run_id, result = await executor.execute(
    lambda emitter, run_id: my_agent_factory(emitter).run(task),
    metadata={"agent": "extraction", "document_id": doc_id},
)
```

The callback receives an `EventEmitter` and the pre-generated `run_id` and returns any result. The `run_id` passed into the callback is the same identifier the `execute` call returns — making it available inside the factory lets adopters key external durable state (HITL requests, waiter registries, resumable workflows) on the Observatory `run_id` before `execute` returns. `TracedExecutor` manages everything else — the application never touches emitter creation, collector wiring, event persistence, or status updates.

When the id needs to be known *before* the factory runs, pass `run_id="..."` as a keyword argument to `execute`. The canonical case is an HTTP route that returns `202 {"run_id": "..."}` before scheduling the executor on a background task, so the client can open an SSE connection keyed on that id before `fn` starts — no in-factory hook can run early enough for that sequence. When the deadline is "before `execute` returns" rather than "before `fn` starts", the in-factory `run_id` parameter is the simpler choice. Uniqueness is the caller's responsibility: the trace store's `runs.id` `PRIMARY KEY` surfaces collisions verbatim (Postgres raises a unique-violation `IntegrityError`; in-memory overwrites silently) — the SDK does not deduplicate.

### Why TracedExecutor over manual wiring

- **Events are persisted in real-time** via `TraceCollector`, not batched after completion. Failed and suspended runs retain their trace data.
- **SSE streaming** works by passing an `asyncio.Queue` — `TracedExecutor` wires it to the internal collector automatically.
- **Run status** is finalized correctly for all terminal states: completed, failed (with error message), and suspended.
- **No boilerplate.** Applications don't need to generate UUIDs, register runs, convert events to records, or manage collector lifecycle.

### When to use TracedExecutor vs manual wiring

Use `TracedExecutor` for **standalone agent runs** — the common case where your API endpoint receives a request, runs an agent, and persists the trace. This is what most application endpoints need.

Use manual `TraceCollector` wiring when you need control the pipeline doesn't provide — custom flush intervals, multiple collectors, non-standard emitter types, or integration with existing lifecycle management (e.g., workflows, which manage their own run lifecycle internally). The [Lifespan-Scoped Singleton-Emitter Listener](#lifespan-scoped-singleton-emitter-listener) section below names the canonical long-lived shape.

## Lifespan-Scoped Singleton-Emitter Listener

`TracedExecutor` and the **lifespan-scoped singleton-emitter listener** are co-equal patterns; neither is preferred. They answer different design questions: `TracedExecutor` reads as "wrap one async run with one trace"; the singleton pattern reads as "one emitter for the application; many agents emit through it." The singleton pattern is supported by existing primitives — no new SDK API is introduced. Construct one `InMemoryEmitter` at application startup, attach long-lived listeners (`TraceCollector`, metrics aggregators, SSE bridges) once, and hand each agent the singleton itself or a `create_child()` derivative; the child emitters inherit the listener list at child-creation time.

### Trade-offs

| Axis | `TracedExecutor` | Singleton-Emitter Listener |
|---|---|---|
| Scope | Per `execute()` call | Application lifespan |
| Run lifecycle | Auto: `register_run` + `update_run_status` for `completed` / `failed` / `suspended` | Application owns `register_run` and `update_run_status` calls |
| Status finalisation | Guaranteed on every termination path (success, exception, `SuspendExecution`) | Application owns finalisation; missing it leaves runs in an indeterminate state |
| Listener attachment | Listeners (collector, queue) wired before `execute()` runs `fn` | Listeners attached at startup; child emitters inherit the listener list **at child-creation time only** — listeners attached to the parent later are not retroactively propagated |
| Redaction policy | Per-run via `redaction_hook=` parameter on `execute()` | Fixed for the application lifespan; varying per-tenant requires application-level redaction at the listener boundary |
| `collector.close()` | Auto on every path | Application owns `close()` at shutdown; forgetting it loses buffered events |

### Wiring shape

```python
from nanitics.tracing import InMemoryEmitter, PostgresTraceStore, TraceCollector

# Module-level singletons live for the application lifespan.
_emitter: InMemoryEmitter | None = None
_collector: TraceCollector | None = None


async def lifespan_setup(trace_store: PostgresTraceStore) -> None:
    """Wire the singleton emitter and its listener once at startup."""
    global _emitter, _collector
    _emitter = InMemoryEmitter(trace_id="app-lifespan-trace")
    _collector = TraceCollector(store=trace_store, parent_id="app-lifespan")
    _emitter.add_listener(_collector.handle)


async def lifespan_teardown() -> None:
    """Final flush and listener cleanup at shutdown."""
    if _collector is not None:
        await _collector.close()


def emitter_for_run() -> InMemoryEmitter:
    """Hand an agent a child emitter that inherits the listener at create time."""
    assert _emitter is not None, "lifespan_setup must run before this is called."
    return _emitter.create_child()
```

The `assert` documents the load-bearing precondition: the listener inheritance happens *at* `create_child()`, so listeners must be attached to the singleton before any child is requested. Listeners added after child creation do not propagate to existing children.

### When to use which

Pick `TracedExecutor` when each request becomes one Observatory run, when redaction policy varies by tenant, or when suspended runs must be resumable (the auto status-finalisation is load-bearing). Pick the singleton when long-running daemons or workers emit across many runs and listener state (metrics aggregation, cross-run dashboards) should survive the per-run boundary, when a test harness captures all emitted events from many sequentially-run agents into one `events` list, or when the run lifecycle is owned upstream (e.g., a workflow primitive that already calls `register_run` itself).

The two patterns can coexist in one application — for example, a daemon that uses the singleton for cross-run metrics and runs each individual request through `TracedExecutor` to register that request as a distinct run. Each call to `TracedExecutor.execute` constructs its own emitter and collector regardless of any singletons in scope, so the two pipelines do not interfere.

## Observatory

The observatory layer provides both a backend API and a frontend component library for building trace viewer UIs. It requires `PersistentTraceStore` — the `TraceStore` protocol is insufficient because the observatory needs filtered queries, span trees, and run management.

### API Router

`mount_observatory(app, store, prefix="/observatory")` attaches both the JSON API and the embedded SPA to your FastAPI app in one line. Under the hood the API surface is `create_observatory_api_router(store)`, organized around five resource types:

- **Runs** — list, create, update status, get detail with summary statistics
- **Trace hierarchy** — span tree for tree visualization, events within a specific span
- **Agents** — agent metadata, stats, and span subtrees within a run
- **Workflow** — DAG structure with per-step status
- **Events** — flat event list with level/type filtering and cursor pagination
- **SSE streaming** — real-time event delivery with level filtering and reconnection support

Consumers that want different middleware on the API and UI surfaces — bearer-token auth on the data endpoints, session auth on the SPA — drop down to `create_observatory_api_router` and `create_observatory_ui_router` and wire them individually.

> **See also:** docstrings on `mount_observatory`, `create_observatory_api_router`, and `create_observatory_ui_router` in `nanitics/observatory/` for the full endpoint surface.

### Frontend Components

The `@nanitics/observatory` package (`observatory/`) provides React components for trace visualization: span tree rendering, event detail panels, run lists, and real-time SSE streaming. It uses a provider pattern (`ObservatoryProvider`) to inject the API client and an extensible event renderer registry.

The renderer registry supports priority-based pattern matching, allowing applications to register custom renderers for specific event types (e.g., rendering LLM events with token usage visualizations).

The package includes an embedded Vite dev shell for standalone development — run `npm run dev` in `observatory/` to iterate on components without a full application.

> **See also:** [Building Applications](building-applications.md) for integration patterns. See the `@nanitics/observatory` package README and component source for the full component API.

## Custom Emitters

Implement the `EventEmitter` protocol to build emitters that write to databases, message queues, or external systems. The protocol requires `trace_id`, `span_id`, `parent_span_id` properties, plus `emit()`, `add_listener()`, and `span()` methods.

For most applications, `InMemoryEmitter` with listeners is sufficient — custom emitters are useful when you need fundamentally different storage semantics (e.g., direct database writes without buffering, or integration with a proprietary observability platform).

> **See also:** The `EventEmitter` protocol definition for the exact interface contract.

## Trace Surface Hygiene

Trace events carry two very different kinds of content, and those two kinds have two different owners. Getting the split clear is the whole point of this section.

### The bifurcation

**SDK-surface content** is everything the SDK's own emission code writes into an event: the event type, span and trace identifiers, timing and usage numbers, response content the LLM returns, tool-result shapes the tool runtime produces. The SDK authors these fields. If a credential, auth header, or raw HTTP context ever appeared in one of them, that would be a defect in the SDK and would be fixed at the emission site.

**Adopter-surface content** is everything adopter code puts into an event via the inputs it hands the SDK: prompts, tool inputs and outputs, custom event fields, tool exception messages. The SDK never interprets this content — it emits it verbatim so you can see exactly what your agent saw. Scrubbing adopter content is the adopter's responsibility.

### The no-leakage guarantee

The SDK's emission code does not write provider credentials, auth headers, or raw HTTP context into any trace event. This holds for every LLM client the SDK ships (`AnthropicLLMClient`, `OpenAILLMClient`, `MistralLLMClient`, `LiteLLMClient`) and is locked in by a release-gate invariant test (`tests/test_no_leakage_invariant.py`) that exercises each client against a mock backend with a sentinel API key, collects every emitted event, and asserts the serialized payloads contain none of: the sentinel key, the literal strings `"Authorization"` or `"Bearer "`, the Anthropic header name `"x-api-key"`, or `"api_key"` as a dict key. A failing test names the offending event type and field path.

### The `RedactionHook` protocol

For adopter-surface scrubbing, implement the `RedactionHook` protocol:

```python
from nanitics.tracing import RedactionHook
```

The protocol has one method, `redact(event: TraceEvent) -> TraceEvent`. Wire a hook in at either of two points:

- `TraceCollector(..., redaction_hook=hook)` — the constructor takes a keyword-only `redaction_hook` argument.
- `TracedExecutor.execute(..., redaction_hook=hook)` — per-run hook forwarded to the internal collector. Use this path when redaction policy varies by tenant, user, or request.

The hook runs inside `TraceCollector.handle()` **before** the persistence record is built and **before** the SSE queue push, so whatever the hook returns is what downstream storage and live consumers see. If the hook raises, the exception propagates; the event is neither persisted nor enqueued. This is deliberate — silently persisting an un-redacted event on hook failure would defeat the property the adopter wired the hook in for.

Events are frozen Pydantic models. Implementations return a new copy via `model_copy(update=...)` rather than attempting to mutate in place. Adopters are expected to preserve the tracing skeleton — `event_id`, `trace_id`, `span_id`, `parent_span_id`, `timestamp`, `event_type` — so the Observatory UI and trace analyzer keep working.

### Example redaction hook (copy-paste)

The SDK intentionally ships no default redactor. Regex lists and PII shapes drift with provider evolution and your data is yours; a shipped default would be a promise the SDK cannot honour. The snippet below is an **example, not a shipped class** — adapt it to your threat model:

```python
import re

from nanitics.tracing import RedactionHook
from nanitics.infrastructure.observability import LLMRequestEvent, TraceEvent

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class ScrubEmails:
    """Example: replace email addresses in ``LLMRequestEvent.system_prompt``."""

    def redact(self, event: TraceEvent) -> TraceEvent:
        if isinstance(event, LLMRequestEvent) and event.system_prompt:
            scrubbed = _EMAIL.sub("[email]", event.system_prompt)
            if scrubbed != event.system_prompt:
                return event.model_copy(update={"system_prompt": scrubbed})
        return event


# Wire it in:
#   executor = TracedExecutor(trace_store)
#   async def fn(emitter, run_id): ...  # factory receives the run_id
#   run_id, result = await executor.execute(fn, redaction_hook=ScrubEmails())
```

This scrubs exactly one field for one event type — your real hook will enumerate the event types and fields your threat model covers (prompts, tool inputs, tool outputs, custom event fields, tool exception messages).

### What goes where

| SDK responsibility | Adopter responsibility |
|--------------------|------------------------|
| No-leakage invariant (release-gate test over every shipped LLM client) | Scrub content in **prompts** |
| Ship the `RedactionHook` protocol and the wire-in points (`TraceCollector`, `TracedExecutor.execute`) | Scrub content in **tool inputs and outputs** |
| Document the call ordering (hook runs before persistence and before SSE push) | Scrub content in **custom events** you emit from application code |
| Preserve the tracing skeleton fields needed by the UI and analyzer | Scrub content in **tool exception messages** (exceptions stringify freely) |
| | Terminate **auth at a reverse proxy** in front of the Observatory API |
| | Route to a **production-grade `PersistentTraceStore`** with appropriate retention, access control, and tenancy |

### Further reading

For the full trust model, production deployment checklist, and end-to-end guidance on sensitive data handling, see `docs/guides/security.md`.

## Pitfalls

- **External listener exceptions soft-fail.** Adopter-supplied listeners (the default) that throw are caught and converted to a `warnings.warn`; the listener stays registered. Keep listener logic simple and handle exceptions within the callback. SDK-internal listeners (`internal=True`) propagate instead — a failing internal listener fails the run.
- **`InMemoryEmitter` is not persistent.** Events are lost when the process ends. Use `TraceStore` or listeners for persistence.
- **`max_events` drops oldest events.** If you set a cap, you lose early events (agent start, initial context). Set it high enough to keep the full trace, or use a listener to persist events before they're dropped.
- **Span nesting relies on `ContextVar`.** If you create tasks with `asyncio.create_task()`, each task inherits the span stack at creation time but has its own copy. Spans created in child tasks don't affect the parent.
- **Always call `collector.close()`.** If you forget, buffered events may be lost. Use `try`/`finally` to guarantee cleanup.

## See Also

- [`examples/tools/event_emitter.py`](../../examples/tools/event_emitter.py) — emitters, events, spans, listeners, child emitters
- [`examples/observability/trace_collection.py`](../../examples/observability/trace_collection.py) — trace storage, persistent storage, `TraceCollector`, SSE streaming
- [`examples/observability/instrumented_client.py`](../../examples/observability/instrumented_client.py) — `InstrumentedLLMClient`, tracing non-agent LLM calls, `label` partitioning
- [Building Applications](building-applications.md) — full SSE streaming and observatory integration patterns
- Source: `nanitics/infrastructure/observability/` — event definitions, emitter, trace storage, collector, executor
- Source: `nanitics/observatory/` — API router, schemas
- Package: `observatory/` — React component library
