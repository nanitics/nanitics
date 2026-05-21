# Observatory Integration

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Wire the Observatory — Nanitics' built-in trace viewer — into your own
application. This guide takes an adopter from install to a working
embedded Observatory served alongside a FastAPI app, covers custom view
and panel registration, and closes with what the Observatory is
deliberately not shipping today.

For the event model, trace levels, `TracedExecutor` fundamentals, and the
`RedactionHook` protocol, see [observability.md](observability.md). This
guide assumes that vocabulary.

## What the Observatory is (and is not)

The Observatory is a developer tool for **inspecting** the events your
agents emit — LLM calls, tool invocations, agent steps, coordination,
HITL, evaluation, memory, planning. It reads a `PersistentTraceStore`
and renders a run list, a span tree, agent-specific views (ReAct,
ReWOO, Reflexion, Tree-of-Thought, LATS, CodeAct), and capability
panels (LLM calls, tool analytics, memory, planning, HITL, evaluation,
error recovery, pattern detection). It ships as a FastAPI router
factory and an embedded React UI served by that router.

It is **not** a production observability platform. There is no
built-in auth, no multi-tenancy, no retention policy, and no default
credential scrubber. The "[For production](#for-production)" section
below names each of those seams and who owns them. Treat the
Observatory as a dev/demo tool that can be deployed for trusted teams
behind a reverse-proxy that handles auth — not as a public service you
expose to untrusted users.

## Quick start — the dev compose

The fastest way to see a live Observatory on your laptop is the compose
under `docker/observatory-dev/`.

### Prerequisites

- Docker with `docker compose` (v2).
- The Observatory embed bundle at `observatory/dist-embed/`. It is
  committed to the repo for convenience. If it is stale or you want to
  pick up frontend changes, rebuild it:

  ```sh
  just observatory-build
  ```

- No API keys needed. The app defaults to `MockLLMClient` so the
  first-run experience is key-free. If `ANTHROPIC_API_KEY` is present
  in your environment, the app uses `AnthropicLLMClient` instead and
  you will see real LLM traces.

### Bring it up

From the repo root:

```sh
just observatory-compose
```

Equivalent to:

```sh
cd docker/observatory-dev && docker compose up --build
```

The container binds to port 8001. The UI lives at
<http://localhost:8001/api/observatory/>.

### First run

Trigger a demo run against the app's `POST /run` endpoint:

```sh
curl -s -X POST http://localhost:8001/run \
  -H 'Content-Type: application/json' \
  -d '{"task": "Say hello to the world."}'
```

You should get back a JSON `{"run_id": "...", "result": "..."}`. Open
<http://localhost:8001/api/observatory/>, click the run, and the span
tree renders with at least one `agent.start` event visible.

### Shutting down

```sh
just observatory-compose-down
```

> For a full-stack compose with a real LLM and three showcase runners, see [Deployment](deployment.md).

## Mounting the backend in your own FastAPI app

The compose app is 60 lines and you probably want to copy it. Here is
the essential wiring — a `PersistentTraceStore`, a `TracedExecutor`
pointed at it, and the Observatory router mounted on your FastAPI app.

```python
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from nanitics.infrastructure import LLMResponse, MockLLMClient
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import InMemoryPersistentTraceStore, ToolCall, TracedExecutor, Usage
from nanitics.observatory import create_observatory_router

UI_DIR = Path("/srv/observatory-ui")  # where you copied observatory/dist-embed

store = InMemoryPersistentTraceStore()
executor = TracedExecutor(store)

app = FastAPI()
app.include_router(
    create_observatory_router(store, static_dir=UI_DIR),
    prefix="/api/observatory",
)


@tool("greet", "Greet someone by name.")
async def greet(name: str) -> str:
    return f"Hello, {name}!"


def _demo_client() -> MockLLMClient:
    """Two-turn ReAct script: call ``greet(name='world')``, then finish.

    Swap in ``AnthropicLLMClient(api_key=...)`` for real traces; see
    ``docker/observatory-dev/app.py`` for the key-aware variant.
    """
    usage = Usage(input_tokens=0, output_tokens=0)
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="greet", arguments={"name": "world"})],
                usage=usage, model="mock", stop_reason="tool_use",
            ),
            LLMResponse(
                content="Hello, world!",
                usage=usage, model="mock", stop_reason="end_turn",
            ),
        ]
    )


class RunRequest(BaseModel):
    task: str


@app.post("/run")
async def run(body: RunRequest) -> dict[str, str]:
    async def _work(emitter, run_id):
        del run_id  # unused in this factory
        agent = ReActAgent(
            name="demo",
            llm_client=_demo_client(),
            emitter=emitter,
            tools=[greet],
            system_prompt="You are a helpful assistant.",
        )
        return (await agent.run(body.task)).output

    run_id, result = await executor.execute(_work)
    return {"run_id": run_id, "result": str(result)}
```

Three things to understand:

1. **`PersistentTraceStore` is where events land.**
   `InMemoryPersistentTraceStore` is the right default for local dev —
   boots instantly, zero config. For durable storage across process
   restarts, swap in `PostgresTraceStore` (same protocol, different
   backend — see its docstring for constructor arguments and schema
   bootstrap). The full-stack compose under `docker/full-stack/` shows
   that path end-to-end (Nanitics + Postgres + embedded Observatory);
   the adopter-facing walkthrough is [`docs/guides/deployment.md`](deployment.md).

2. **`TracedExecutor` is the only supported way to produce runs the
   Observatory can show.** It owns the `run_id`, wires the
   `EventEmitter` into your work function, and writes events to the
   store. Do not construct `run_id`s yourself and do not instantiate
   `EventEmitter` outside `TracedExecutor` — see
   [observability.md](observability.md) for why.

3. **`create_observatory_router(store, *, static_dir=...)`** is the
   one factory you call. When `static_dir` points at a directory
   containing a built `index.html` and `assets/`, the router serves
   the embedded UI at its mount root. When `static_dir` is `None` the
   API endpoints still work and the UI root returns a helpful fallback
   page.

### SSE streaming

The router ships an SSE endpoint at `GET /runs/{run_id}/stream` that
streams events for a specific run as they are written to the store.
The embedded UI consumes this stream to animate the run detail page in
real time. You do not need to do anything to enable it — mounting the
router is enough. If you host the app behind a reverse proxy, ensure
the proxy passes through `Content-Type: text/event-stream` unbuffered
(nginx: `proxy_buffering off;`). See
[building-applications.md](building-applications.md) for the general
SSE patterns the Observatory builds on.

### Custom event renderers in Python?

Mostly a frontend concern — see [Wiring the frontend
ObservatoryProvider](#wiring-the-frontend-observatoryprovider) below.
The Python side is the raw event pipeline; every rendering decision
happens in the React UI.

## Wiring the frontend ObservatoryProvider

The compose path above uses the **embedded UI** — a pre-built React
bundle served by the Observatory router from `static_dir`. That is the
default and it requires no frontend toolchain.

When you want to embed Observatory components into your own React app
— your internal dashboard, a debugging console, an existing admin UI
— you install the `@nanitics/observatory` package from npm.

### Install

```bash
npm install @nanitics/observatory
```

Peer deps: `react@^19.2.6`, `react-dom@^19.2.6`.

Then wire an `ObservatoryClient`, the default registries, and the
provider tree. This mirrors `observatory/dev/app.tsx`:

```tsx
import {
  ObservatoryClient,
  ObservatoryProvider,
  RunListPage,
  createDefaultRegistries,
} from "@nanitics/observatory";
import "@nanitics/observatory/styles.css";

const client = new ObservatoryClient("/api/observatory");
const { registry, agentViewRegistry, panelRegistry } =
  createDefaultRegistries();

export function App() {
  return (
    <ObservatoryProvider
      client={client}
      registry={registry}
      agentViewRegistry={agentViewRegistry}
      panelRegistry={panelRegistry}
    >
      <RunListPage onSelectRun={(id) => {/* route to a run detail */}} />
    </ObservatoryProvider>
  );
}
```

`ObservatoryClient` takes the API base URL. `createDefaultRegistries`
returns three registries — one each for event renderers, agent views,
and capability panels — all populated with the defaults the embedded UI
uses. Pass them all to `ObservatoryProvider` and every built-in
component (`<RunListPage>`, `<RunDetailPage>`, `<AgentDetailPage>`,
`<WorkflowDetailPage>`) resolves renderers through those registries.

### Registering a custom agent view

Your custom agent type has its own UI story. Register an
`AgentViewRegistration` with the `agentViewRegistry` before passing it
to the provider:

```tsx
import type { AgentViewRegistration } from "@nanitics/observatory";

const MyAgentView: AgentViewRegistration["component"] = ({
  agent,
  events,
  spanTree,
}) => (
  <div>
    <h2>{agent.agent_name}</h2>
    <p>Events: {events.length}</p>
    <p>Root span: {spanTree.span_id}</p>
    {/* Your custom rendering here; spanTree is the rooted agent subtree. */}
  </div>
);

const registration: AgentViewRegistration = {
  agentType: "my-custom-agent",
  component: MyAgentView,
};

const { registry, agentViewRegistry, panelRegistry } =
  createDefaultRegistries();
agentViewRegistry.register(registration);
```

Capability panels follow the same shape — a `CapabilityPanelRegistration`
registered with `panelRegistry`. Event renderers register with
`registry` (an `EventRendererRegistry`). See `observatory/src/registry/`
for the full API and the defaults' source for live examples.

### Theming and dark mode

Every component styles through an oklch token palette; the `dark` class
on `document.documentElement` swaps the palette — no inline light/dark
branching anywhere. Mount `<ThemeToggle>` (imported from
`@nanitics/observatory`) in your app shell for a light ↔ dark switch;
it persists the choice at `localStorage['observatory-theme']` and falls
back to `prefers-color-scheme` when nothing is stored.

To avoid a flash of light theme for dark-preferring users, paste this
into your app shell's `<head>` before the React bundle `<script>`:

```html
<script>
  (function () {
    try {
      var stored = localStorage.getItem('observatory-theme');
      var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (stored === 'dark' || (stored !== 'light' && prefersDark)) {
        document.documentElement.classList.add('dark');
      }
    } catch (_) { /* no-op on storage errors */ }
  })();
</script>
```

Token definitions (oklch values for both themes) live in
`observatory/dev/globals.css`. Adopters embedding Observatory into a
custom bundle should import that stylesheet or copy its `.dark { … }`
block into their own CSS.

### Sharable filtered URLs

The Run List encodes its filter state in the hash query string so any
filtered view round-trips through reload, copy/paste, and bug-report
links. The convention is:

```
#/runs?status=<status>&sort=<sort>&search=<text>&started_after=<iso>&started_before=<iso>
```

All keys are optional; absent keys mean "use the default." Accepted
values:

| Key | Values |
|---|---|
| `status` | `running`, `completed`, `failed`, `suspended` |
| `sort` | `started_at_desc` (default), `started_at_asc`, `duration_desc`, `duration_asc` |
| `search` | any free-text string |
| `started_after` | ISO 8601 timestamp |
| `started_before` | ISO 8601 timestamp |

Adopters who embed the Run List in their own page can wire URL-state on
the same primitive the Observatory uses:

```ts
import { useUrlFilters } from "@nanitics/observatory";
```

`useUrlFilters` takes a per-page schema (one entry per filter, each with
`parse` and `stringify`) and returns the current values plus per-key
setters. Filter changes use `history.replaceState` so the back-stack
stays clean — the back button restores the previous *distinct* filter
URL, not a per-keystroke replay of the search box.

## For production

The Observatory is a dev tool today. Running it in production is
possible; we have not built the machinery a production deployment
needs. These are the four seams and who owns each. For the adopter-owned
security posture these seams sit inside — prompt-injection, redaction,
DockerSandbox limits, API-key handling — see the
[Security guide](security.md).

### Auth

Terminate auth at a reverse proxy in front of the app — nginx, Caddy,
a cloud load balancer. The Observatory endpoints are not
authentication-aware and the router does not accept an auth hook.
Protect the mount path (`/api/observatory`) and the UI path
(`/api/observatory/`) together. A future
`ObservatoryAuthProvider` protocol is post-launch and signal-driven
(see [What is not shipped](#what-is-not-shipped)).

### Multi-tenancy

The `PersistentTraceStore` protocol is the scoping seam. Implement a
tenant-scoped wrapper that filters runs and events by a tenant
identifier drawn from `TracedExecutor(metadata={...})`. Your custom
store reads the tenant id from request state (via a FastAPI
dependency) and partitions the query surface accordingly. The SDK does
not ship a multi-tenant store — the Observatory assumes a single
scope.

### Retention

`PostgresTraceStore` carries events indefinitely. Retention is your
policy: an adopter-owned periodic job that deletes old runs, a custom
store wrapper that applies a TTL, or external archival. The SDK does
not ship retention machinery.

### Content scrubbing

This is where the **`RedactionHook`** hook belongs. See
[Trace Surface Hygiene in observability.md](observability.md#trace-surface-hygiene)
for the protocol, the no-leakage invariant enforced for SDK-managed
fields, and the "four categories of adopter content" the hook covers.
No default scrubber is shipped — domain-appropriate redaction is
adopter-owned.

The Observatory is a dev tool today. Running it in production is
possible; the auth, multi-tenancy, retention, and scrubbing machinery
a real deployment needs ships when adopter signal shapes it.

## What is not shipped

Naming each gap so you can plan around it.

- **`ObservatoryAuthProvider` protocol.** A pluggable auth hook on
  `create_observatory_router`. Not currently shipped; a future, signal-
  driven addition.
- **Production deployment guide.** The adopter-facing "stand up
  Observatory in production with Postgres, Caddy, retention, and
  scrubbing" walkthrough lives at
  [`docs/guides/deployment.md`](deployment.md). The full-stack compose
  under `docker/full-stack/` (Nanitics + Postgres + embedded
  Observatory) is its substrate — deliberately separate from the
  narrow dev compose under `docker/observatory-dev/`, which is the
  key-free first-run path.
- **Default credential scrubber in the SDK.** A domain-neutral
  redaction implementation shipped as the default `RedactionHook`.
  Not shipped — redaction policy is domain-specific (credit-card
  numbers matter for payments; PII matters for user support; neither
  is universal) and a generic scrubber creates a false sense of
  safety. See [observability.md](observability.md#trace-surface-hygiene)
  for the reasoning. Adopters ship their own policy.

## Further reading

- [Observability](observability.md) — event levels, storage
  architecture, trace-surface hygiene, the `RedactionHook` protocol.
- [Building Applications](building-applications.md) — SSE streaming
  patterns, run lifecycle, HITL endpoints.
- Contributor workflow for the React UI: `just observatory-dev` starts
  the Vite dev server on port 5173 with a proxy to a backend on 8001.
  Useful when you are iterating on components in
  [`observatory/src/`](../../observatory/src/).
