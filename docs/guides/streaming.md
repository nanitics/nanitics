# Streaming

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Streaming is a transport concern layered on top of observability. The SDK already emits structured events through `EventEmitter`; a streaming setup ships those events out of the process to a client — usually a browser — as Server-Sent Events. This guide is about that transport layer. [Observability](observability.md) is the guide for producing events; this one is for getting them onto the wire. WebSocket, gRPC, and other transports are out of scope — SSE covers the common case of agents-to-browser streams.

## Choosing a level for streaming

`TraceCollector` accepts a `min_level` that gates which events reach its SSE queue. The right level depends on the consumer and the bandwidth budget.

| Consumer | Level | Why |
|----------|-------|-----|
| End-user UI | `"info"` | Milestones only — agent start, step, complete, workflow progress. Low volume, high readability. |
| Developer trace viewer | `"debug"` | Adds LLM calls, tool invocations, error recovery. Manageable volume for a debug surface. |
| Deep debugging | `"verbose"` | Every span, memory op, routing decision. Expensive to render; use sparingly. |

See [Observability](observability.md) for the full classification — this table is the streaming-specific trade-off only. `min_level` on the collector controls what reaches the client; the store always persists everything regardless.

## Live queue vs DB replay

Every long-running agent needs both modes. When the client connects while the run is still executing, stream from an in-memory queue (fed by `TraceCollector`). When the client connects *after* the run is over, or reconnects mid-stream, replay from the trace store. One endpoint, two code paths.

- **Live queue**: `asyncio.Queue` attached to `TraceCollector` via the `queue=` parameter. Events arrive in real time.
- **DB replay**: `PostgresTraceStore.query_events(trace_id, after_sequence=...)` reads the persisted sequence.
- **Reconnection**: Honor the `Last-Event-ID` SSE header. Replay from the store up to the last sequence the client saw, then switch to the live queue if the run is still active.

See [Building Applications](building-applications.md#sse-streaming) for the full endpoint with live-then-replay logic and the reconnection branch. This guide does not duplicate that code.

## FastAPI shape

The canonical shape, stripped down to show how the primitives wire together:

<!-- verify: skip — illustrative, `collector_queue` is application-supplied -->
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from nanitics.tracing import TraceCollector

app = FastAPI()

@app.get("/runs/{run_id}/events")
async def stream(run_id: str) -> StreamingResponse:
    async def generator():
        async for event in collector_queue(run_id):
            yield f"event: {event.event_type}\ndata: {event.model_dump_json()}\n\n"
    return StreamingResponse(generator(), media_type="text/event-stream")
```

See [`examples/observability/trace_collection.py`](../../examples/observability/trace_collection.py) for a runnable version with full lifecycle wiring.

## Frontend consumption

SSE is framework-agnostic. The browser's `EventSource` API dispatches by the `event:` line, so each `event_type` becomes its own handler:

```js
const src = new EventSource("/runs/abc123/events");
src.addEventListener("agent.step", (e) => render(JSON.parse(e.data)));
src.addEventListener("tool.result", (e) => render(JSON.parse(e.data)));
```

`EventSource` handles reconnection automatically using `Last-Event-ID` — your server just needs to honor it.

## Non-agent workloads

`TraceCollector` is not agent-specific. Any code that owns an `EventEmitter` — a workflow runner, a multi-agent coordinator, a custom background job — can attach a collector the same way. The collector's only requirement is an emitter to listen to and (optionally) a store and queue to fan out to. This is useful when streaming workflow progress, orchestrator supervisions, or bus messages alongside agent events.

## What not to stream

At `"info"`, internal memory operations, span start/end, and model-routing decisions are filtered out — they are not meaningful to end users and inflate bandwidth. If you need them for a developer view, raise `min_level` to `"debug"` or `"verbose"` on that specific collector; don't override per-event. The level filter is the right knob.

## See also

- [Observability](observability.md) — event catalogue, `EventEmitter`, level classification
- [Building Applications](building-applications.md#sse-streaming) — full SSE endpoint with replay and reconnection
- [`examples/observability/trace_collection.py`](../../examples/observability/trace_collection.py) — runnable `TraceCollector` + queue + store setup
- [`examples/tools/event_emitter.py`](../../examples/tools/event_emitter.py) — emitter fundamentals
