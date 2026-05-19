# Building Applications

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

The SDK provides the agent runtime — loops, tools, memory, coordination. To build a production application, you need an API layer that manages runs, streams events to clients, persists state, and handles human-in-the-loop flows. This guide covers the patterns for building that layer.

## Architecture Overview

```
┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│   Client     │◄────│   API Server     │◄────│   SDK    │
│              │ SSE │  (FastAPI)       │     │ (Agent)  │
└──────────────┘     │                  │     └──────────┘
                     │  • Run management│
                     │  • SSE streaming │
                     │  • HITL endpoints│
                     │  • Persistence   │
                     └──────────────────┘
                              │
                     ┌────────┴────────┐
                     │    Database     │
                     │  (PostgreSQL)   │
                     └─────────────────┘
```

**The SDK does not dictate your application architecture.** It provides protocols (`EventEmitter`, `CheckpointStore`, `HumanInputProvider`, `TraceStore`, memory stores) that you implement with your infrastructure. The patterns below are proven approaches, not requirements.

For the SDK component decision sequence — which agent type, which memory, which orchestration pattern — see [architecture-guide.md](architecture-guide.md). This guide covers the backend implementation patterns once those decisions are made.

## Choosing Between Agents and Plain Code

Not every feature in your application needs an agent. Agents add value when the task requires reasoning — deciding what to do, interpreting ambiguous input, or navigating multi-step processes with judgment. Deterministic operations don't benefit from agent overhead.

**Use agents for:** tasks that require LLM reasoning — analysis, research, content generation, complex decision-making, multi-step workflows with judgment calls.

**Use plain code for:** CRUD operations, data transformations, file serving, authentication, caching, straightforward API proxying, configuration management — anything where the logic is deterministic and well-defined.

A healthy application mixes both. An endpoint that analyzes a document benefits from an agent. An endpoint that registers a user, updates settings, or serves a file does not. Choose the simplest approach that solves the problem.

## Extending the SDK

The SDK is a foundation, not a fixed boundary. If your application needs a capability the SDK doesn't provide — a new agent type, a new orchestration pattern, a new memory strategy — add it to the SDK. Application needs drive SDK evolution: when you encounter a gap, extend the SDK so the capability is reusable rather than working around it in application code.

Tools especially: the SDK provides the tool protocol and creation utilities, but the actual tools are yours. You create whatever tools your application needs — database queries, API integrations, file operations, domain-specific actions. There is no predefined set of tools to choose from. See [Tools](tools.md) for the three creation methods.

## API Server Pattern

Use an async web framework (FastAPI, Starlette, aiohttp) that can run agent coroutines as background tasks and stream events via SSE.

### Run Lifecycle

A typical run goes through these states: `running → complete | error | suspended | cancelled`.

Use `TracedExecutor` to manage run lifecycle with persistent traces. It handles ID generation, run registration, emitter creation, real-time event persistence via `TraceCollector`, and status finalization — the application only provides a callback:

```python
from nanitics.infrastructure import AnthropicLLMClient
from nanitics.strategies import ReActAgent
from nanitics.tracing import PostgresTraceStore, TracedExecutor

trace_store = PostgresTraceStore(pool=asyncpg_pool)
executor = TracedExecutor(trace_store)

async def create_run(request: RunRequest) -> RunCreated:
    run_id, result = await executor.execute(
        lambda emitter, run_id: ReActAgent(
            name="my-agent",
            llm_client=AnthropicLLMClient(model="claude-haiku-4-5-20251001"),
            emitter=emitter,
            tools=[...],
            system_prompt="...",
        ).run(request.task),
        metadata={"task_type": "analysis"},
    )
    return RunCreated(run_id=run_id)
```

For background execution with SSE streaming, pass a queue:

```python
import asyncio

@app.post("/runs")
async def create_run(request: RunRequest) -> RunCreated:
    run_id = str(uuid4())
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def run_with_trace():
        await executor.execute(
            lambda emitter, run_id: build_agent(emitter).run(request.task),
            metadata={"task": request.task},
            queue=queue,
        )

    asyncio.create_task(run_with_trace())
    # Store queue for SSE endpoint to consume
    active_runs[run_id] = queue
    return RunCreated(run_id=run_id)
```

The key insight: **return the run ID immediately, execute the agent in the background, and stream results via SSE.** Agent runs can take seconds to minutes — you can't block the HTTP request.

### Run Context

Track active runs with their event queues and metadata:

```python
import asyncio
from dataclasses import dataclass, field
from nanitics.tracing import TraceEvent

@dataclass
class RunContext:
    run_id: str
    queue: asyncio.Queue[TraceEvent | None] = field(
        default_factory=asyncio.Queue
    )
    task: asyncio.Task | None = None
    completed: bool = False
```

Register an event listener on the emitter that forwards events to the queue:

```python
from nanitics.tracing import InMemoryEmitter

emitter = InMemoryEmitter(trace_id="...")
ctx = RunContext(run_id="...")

def forward_events(event: TraceEvent) -> None:
    ctx.queue.put_nowait(event)

emitter.add_listener(forward_events)
```

### Durable Workflow Execution

Multi-step workflows that support suspension (HITL approval, timeouts) must also go through `TracedExecutor` — otherwise their trace events are lost when the `InMemoryEmitter` is garbage-collected. Wrap the workflow creation and execution in a closure:

<!-- verify: skip — illustrative stub with `...` placeholders for caller-supplied args and metadata -->
```python
async def start_workflow(run_id: UUID, entity_id: UUID, pool: Pool, ...) -> UUID:
    async def _run_workflow(emitter: EventEmitter, _run_id: str) -> StepResult:
        workflow = await create_my_workflow(
            entity_id=str(entity_id),
            pool=pool,
            llm_client=llm_client,
            emitter=emitter,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        return await workflow.execute(task_input)

    try:
        run_id, result = await executor.execute(_run_workflow, metadata={...})
    except SuspendExecution as exc:
        # exc still carries suspension_info for HITL handling
        # TracedExecutor already marked the run as "suspended"
        await update_status(pool, run_id, "awaiting_approval",
                            hitl_request_id=exc.suspension_info.request_id)
        return run_id

    await update_status(pool, run_id, "completed")
    return run_id
```

The same pattern applies to resume — wrap `workflow.execute(input, resume_from=checkpoint)` in a closure passed to the executor.

> **Never create `InMemoryEmitter` directly in service code.** Let `TracedExecutor` create and wire the emitter — this guarantees trace persistence for all run outcomes (completed, failed, suspended). Direct emitter creation produces orphaned traces that are invisible to the Observatory.

## SSE Streaming

Server-Sent Events (SSE) provide a simple, unidirectional event stream from server to client. Every SDK event becomes an SSE message, giving the client real-time visibility into agent execution.

### Event Stream Endpoint

```python
import json
from collections.abc import AsyncGenerator
from fastapi import Request
from fastapi.responses import StreamingResponse

@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> StreamingResponse:
    ctx = active_runs.get(run_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return StreamingResponse(
        event_stream(ctx),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

async def event_stream(ctx: RunContext) -> AsyncGenerator[str]:
    sequence = 0
    while True:
        try:
            item = await asyncio.wait_for(ctx.queue.get(), timeout=15.0)
        except TimeoutError:
            # Send keepalive to prevent connection timeout
            yield ": keepalive\n\n"
            continue

        if item is None:
            # Run completed — send terminal event
            yield f"event: done\ndata: {{}}\n\n"
            break

        sequence += 1
        event_data = item.model_dump(mode="json")
        yield (
            f"id: {sequence}\n"
            f"event: {item.event_type}\n"
            f"data: {json.dumps(event_data)}\n\n"
        )
```

### SSE Format

Each SSE message has three parts:

```
id: 42
event: agent.step
data: {"agent_name": "my-agent", "step_number": 3, ...}

```

- **`id`**: Sequence number — enables reconnection (the client sends `Last-Event-ID` to resume)
- **`event`**: The `event_type` string — clients can listen for specific event types
- **`data`**: JSON-serialized event payload

### Reconnection

Clients can reconnect and resume from where they left off using the `Last-Event-ID` header:

```python
@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> StreamingResponse:
    last_event_id = request.headers.get("last-event-id")
    after_sequence = int(last_event_id) if last_event_id else 0

    ctx = active_runs.get(run_id)

    if ctx is not None and not ctx.completed:
        # Live stream — replay missed events from DB, then stream live
        return StreamingResponse(
            live_stream(ctx, after_sequence, run_id),
            media_type="text/event-stream",
        )
    else:
        # Run completed — replay all events from database
        return StreamingResponse(
            replay_stream(run_id, after_sequence),
            media_type="text/event-stream",
        )
```

This dual-mode approach (live queue + database replay) ensures clients can always get the full event stream, even if they connect after the run finishes or reconnect after a disconnect.

### Buffered Persistence

**For most applications, use `TracedExecutor`** (shown in [Run Lifecycle](#run-lifecycle)) — it composes `InMemoryEmitter`, `TraceCollector`, and `PersistentTraceStore` into a single call with correct lifecycle management.

If you need more control (custom emitter configuration, multiple collectors, non-agent workloads), wire `TraceCollector` directly:

<!-- verify: skip — illustrative wiring sketch; `asyncpg_pool`, `run_id`, `sse_queue`, `emitter` are caller-supplied and the trailing `await` runs inside an async context -->
```python
from nanitics.tracing import PostgresTraceStore, TraceCollector

trace_store = PostgresTraceStore(pool=asyncpg_pool)

collector = TraceCollector(
    store=trace_store,
    parent_id=run_id,
    queue=sse_queue,       # optional — for live SSE streaming
    min_level="info",      # only push info+ events to the queue
    flush_interval=0.5,    # seconds between automatic flushes
)

emitter.add_listener(collector.handle)

# ... run the agent ...

await collector.close()  # flush remaining events, stop background loop
```

The collector supports level-filtered SSE streaming — events at or above `min_level` are pushed to the optional `queue` for real-time delivery to clients.

See [Observability](observability.md) for the full `TracedExecutor`, `TraceCollector`, `PersistentTraceStore`, and `create_observatory_router()` API reference.

## Persistence

The SDK provides in-memory implementations of all stores. For production, implement the store protocols with your database.

### Store Protocols to Implement

| Protocol | SDK Default | Purpose |
|----------|-------------|---------|
| `TraceStore` | `InMemoryTraceStore` | Persist and query full traces |
| `PersistentTraceStore` | `PostgresTraceStore` | Per-event persistence with filtered queries and pagination |
| `CheckpointStore` | `InMemoryCheckpointStore` | Durable execution checkpoints |
| `LongTermStore` | `InMemoryLongTermStore` | Persistent key-value memory |
| `EpisodeStore` | `InMemoryEpisodeStore` | Past experience storage |
| `SemanticStore` | `InMemorySemanticStore` | Embedding-based retrieval |
| `HumanInputProvider` | `CallbackHumanInputProvider` | Human-in-the-loop integration |
| `HitlRequestStore` | `InMemoryHitlRequestStore` | Persist HITL requests/responses for durable suspension |

### Database-Backed CheckpointStore

Checkpoints enable durable execution — the agent can suspend (for HITL, timeouts, etc.) and resume later, even after process restart. Here's the pattern:

```python
from nanitics.composition import RunCheckpoint, SuspensionInfo

class PostgresCheckpointStore:
    def __init__(self, pool) -> None:
        self._pool = pool

    async def save(self, checkpoint: RunCheckpoint) -> None:
        await self._pool.execute(
            "INSERT INTO checkpoints (id, run_id, checkpoint_type, "
            "schema_version, state, suspension_info, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7)",
            checkpoint.checkpoint_id,
            checkpoint.run_id,
            checkpoint.checkpoint_type,
            checkpoint.schema_version,
            json.dumps(checkpoint.state),
            json.dumps(checkpoint.suspension_info.model_dump()),
            checkpoint.created_at,
        )

    async def load(self, run_id: str) -> RunCheckpoint | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM checkpoints WHERE run_id = $1 "
            "ORDER BY created_at DESC LIMIT 1",
            run_id,
        )
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    async def delete(self, checkpoint_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM checkpoints WHERE id = $1", checkpoint_id
        )

    async def delete_for_run(self, run_id: str) -> None:
        await self._pool.execute(
            "DELETE FROM checkpoints WHERE run_id = $1", run_id
        )
```

### Event Persistence

Store every event in the database for replay and analysis:

```python
# Schema
"""
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(id),
    event_type TEXT NOT NULL,
    event_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL,
    sequence INTEGER NOT NULL
);
"""

# Batch insert
async def insert_events_batch(pool, events: list[tuple]) -> None:
    await pool.executemany(
        "INSERT INTO events (run_id, event_type, event_id, trace_id, "
        "span_id, parent_span_id, timestamp, data, sequence) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)",
        events,
    )
```

## HITL API Pattern

Human-in-the-loop requires two API endpoints: one for the agent to request input, and one for the human to respond. The SDK provides the `HumanInputProvider` protocol — your application implements it to bridge the gap.

### In-Memory HITL (Standard Runs)

For runs where the process stays alive during human interaction:

```python
import asyncio
from nanitics.hitl import HumanInputRequest, HumanInputResponse

class ApiHumanInputProvider:
    """Suspends on an asyncio.Future until the API resolves it."""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[HumanInputResponse]] = {}

    async def request_input(
        self, request: HumanInputRequest
    ) -> HumanInputResponse:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[HumanInputResponse] = loop.create_future()
        self._pending[request.request_id] = future
        return await future

    def resolve(self, request_id: str, response: HumanInputResponse) -> bool:
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return False
        future.set_result(response)
        return True
```

### Durable HITL (Survives Restarts)

For runs that may suspend and resume across process restarts, use `DurableHumanInputProvider` with a database-backed request store:

```python
from nanitics.hitl import DurableHumanInputProvider

# The store persists HITL requests and responses to the database
hitl_store = PostgresHitlRequestStore(pool, run_id)
provider = DurableHumanInputProvider(hitl_store)
```

When the agent requests input, `DurableHumanInputProvider` saves the request to the store and raises `SuspendExecution`. The agent's state is checkpointed. On the resume side, use `ResumeService` — it owns the load-checkpoint / validate-response / save-response / re-execute cycle so your API layer doesn't have to:

```python
from nanitics.composition import DurableRun, ResumeContext, ResumeService
from nanitics.hitl import HumanInputResponse

def factory(ctx: ResumeContext) -> DurableRun:
    workflow = build_workflow(ctx.run_id, ctx.hitl_store, ctx.checkpoint_store)
    return DurableRun(
        workflow,
        hitl_store=ctx.hitl_store,
        checkpoint_store=ctx.checkpoint_store,
    )

resume_service = ResumeService(
    hitl_store=hitl_store,
    checkpoint_store=checkpoint_store,
    factory=factory,
)

async def resume_run(run_id: str, response: HumanInputResponse):
    # Loads the checkpoint, validates response.request_id against the
    # checkpoint's pending request, saves the response, reconstructs
    # the run via the factory, and drives it forward. Returns either
    # a ResumeResult (completed) or another SuspendedRun (nested
    # suspension — route through resume_run again).
    return await resume_service.resume(run_id, response)
```

`ResumeService.resume` never raises `SuspendExecution`; a nested suspension comes back as a `SuspendedRun` value your API layer can ship to the UI the same way it handled the first suspension. Mismatched `response.request_id`s raise `ValueError` rather than silently saving to the wrong slot.

> **Important:** `SuspendExecution` inherits from `BaseException`, not `Exception`. This is intentional — it must propagate through tool execution and agent loops without being caught by `except Exception` handlers. `TracedExecutor` handles this automatically (marking the run as `suspended`). If you write your own run lifecycle handler, catch `SuspendExecution` *before* `Exception`:
>
> ```python
> from nanitics.composition import SuspendExecution
>
> try:
>     result = await coro
> except SuspendExecution:
>     # Handle suspension (persist checkpoint, update status)
>     ...
> except Exception:
>     # Handle errors (this will NOT catch SuspendExecution)
>     ...
> ```

### HITL API Endpoints

```python
from fastapi import APIRouter
from nanitics.hitl import HumanDecision, HumanInputResponse
from datetime import datetime, UTC

router = APIRouter()

class HitlRespondRequest(BaseModel):
    decision: HumanDecision  # approve, reject, revise
    content: str | None = None

@router.post("/runs/{run_id}/hitl/{request_id}/respond")
async def respond_to_hitl(run_id: str, request_id: str, body: HitlRespondRequest):
    response = HumanInputResponse(
        request_id=request_id,
        decision=body.decision,
        content=body.content,
        responded_at=datetime.now(UTC),
    )

    # Check if run is active (in-memory) or suspended (durable)
    ctx = get_run_context(run_id)
    if ctx is not None:
        ctx.hitl_provider.resolve(request_id, response)
        return {"status": "resolved"}
    else:
        await resume_service.resume(run_id, response)
        return {"status": "resumed"}

@router.get("/runs/{run_id}/hitl/pending")
async def list_pending_hitl(run_id: str):
    """List pending HITL requests — check in-memory first, then database."""
    ctx = get_run_context(run_id)
    if ctx is not None:
        return ctx.hitl_provider.list_pending()

    # Fall back to database for suspended runs
    return await hitl_store.get_pending_requests(run_id)
```

## Client Integration

### Consuming SSE Events

```typescript
const eventSource = new EventSource(`/api/runs/${runId}/events`);

// Listen to specific event types
eventSource.addEventListener("agent.step", (e) => {
  const step = JSON.parse(e.data);
  console.log(`Step ${step.step_number}: ${step.thought}`);
});

eventSource.addEventListener("tool.result", (e) => {
  const result = JSON.parse(e.data);
  console.log(`Tool ${result.tool_name}: ${result.success}`);
});

eventSource.addEventListener("done", (e) => {
  const result = JSON.parse(e.data);
  console.log(`Finished: ${result.output}`);
  eventSource.close();
});

// Handle HITL requests
eventSource.addEventListener("hitl.request", (e) => {
  const request = JSON.parse(e.data);
  showApprovalDialog(request);
});
```

### State Management

Map event types to UI updates:

| Event Type | UI Update |
|------------|-----------|
| `agent.start` | Show "Running" status, display tools |
| `agent.step` | Add step to timeline, show thought/action |
| `llm.response` | Update token counter |
| `tool.invoke` / `tool.result` | Show tool execution in timeline |
| `hitl.request` | Show approval dialog or input form |
| `evaluation.result` | Show quality score |
| `agent.complete` | Show final output, mark as complete |
| `agent.error` | Show error state |

### Reconnection

The browser's `EventSource` automatically reconnects and sends `Last-Event-ID`. Your server must support resuming from a sequence number (see [Reconnection](#reconnection) above). If you need more control, use `fetch()` with `ReadableStream`:

```typescript
async function streamEvents(runId: string, lastEventId?: string) {
  const headers: Record<string, string> = {};
  if (lastEventId) {
    headers["Last-Event-ID"] = lastEventId;
  }

  const response = await fetch(`/api/runs/${runId}/events`, { headers });
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // Parse SSE format and dispatch events
  }
}
```

## Application Startup

Handle graceful startup and shutdown, including recovery from crashes:

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: initialize database, recover from crashes
    await db.connect()
    # Mark interrupted runs as errored
    await db.execute(
        "UPDATE runs SET status = 'error', "
        "error_message = 'Interrupted by server restart' "
        "WHERE status = 'running'"
    )
    yield
    # Shutdown: clean up
    await db.disconnect()

app = FastAPI(lifespan=lifespan)
```

Suspended runs (those waiting for HITL) survive restarts — they have checkpoints in the database and can be resumed when the human responds.

## Frontend Patterns

Application frontends use Next.js (App Router, TypeScript). All applications follow a consistent structure:

- `app/` — Next.js App Router pages. Server components by default, client components only when needed (interactivity, SSE consumption, state).
- `app/providers.tsx` — Client-side providers (e.g., `ObservatoryProvider`) wrapping the application.
- `components/` — Shared components. Co-locate route-specific components with their routes.
- `app/runs/` — Run management pages (list, detail) — typically built with observatory components.

### Observatory Integration

The `@nanitics/observatory` package (`/observatory/`) provides pre-built React components for trace visualization: `RunListPage`, `RunDetailPage`, `TraceTree`, `EventDetailPanel`, and an extensible `EventRendererRegistry`. All application frontends depend on it.

Setup: wrap your app in `ObservatoryProvider` with an `ObservatoryClient` pointing to your API server. The provider handles SSE streaming, event fetching, and state management. See the [Observability guide](observability.md#frontend-components) for component API details and custom event renderer registration.

### SSE Consumption

The "Client Integration" section above covers SSE event consumption patterns and state management. For observatory-based UIs, SSE is handled automatically by `ObservatoryProvider` — you only need manual SSE handling for custom (non-observatory) UI components.

## Pitfalls

- **Don't block the HTTP request.** Agent runs take seconds to minutes. Return a run ID immediately and execute in the background.
- **Buffer database writes.** Writing every event individually creates too many round-trips. Batch writes every 0.5–1 second.
- **Always flush on completion.** After the agent finishes, flush the event buffer before sending the terminal SSE event. Otherwise, the client may close before the last events are persisted.
- **Handle SSE disconnection gracefully.** Clients disconnect frequently (browser navigation, network issues). Use sequence numbers so they can reconnect and resume.
- **Clean up active runs.** Keep a grace period before removing completed run contexts — clients may still be consuming events from the queue after the run finishes.
- **Don't conflate in-memory and durable HITL.** In-memory HITL (asyncio.Future) is simpler but doesn't survive restarts. Durable HITL (checkpoint + database) is more complex but production-grade. Choose based on your reliability requirements.
