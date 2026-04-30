# Observatory dev — demo

This demo shows what the Nanitics Observatory looks like in practice. A minimal
FastAPI app hosts a `ReActAgent` with a single tool — `greet(name)` — and
exposes a `/run` endpoint that executes the agent and records the full trace.
After each run you open the Observatory UI to inspect how the agent reasoned:
which tool it called, what arguments it chose, what the tool returned, and how
the agent formed its final response. This is the same observability surface that
adopters embed in their own applications.

No API key required — uses a scripted `MockLLMClient` by default that produces
a realistic two-turn ReAct trace. Set `ANTHROPIC_API_KEY` in your shell
environment to run against a real Haiku model instead.

---

## 1. Start the stack

```sh
# Build the Observatory UI bundle (only needed once, or after UI changes)
just observatory-build

# Start the compose
just observatory-compose
```

The app starts on port 8001. Check it's up:

```sh
curl -s http://localhost:8001/health
# → {"status":"ok"}
```

---

## 2. Trigger an agent run

The `/run` endpoint executes a demo `ReActAgent` that has one tool: `greet(name)`.

**Default task:**

```sh
curl -s -X POST http://localhost:8001/run \
  -H 'Content-Type: application/json' \
  -d '{"task": "Say hello to the world."}'
```

**Custom task (requires a real API key):**

```sh
curl -s -X POST http://localhost:8001/run \
  -H 'Content-Type: application/json' \
  -d '{"task": "Greet Alice and Bob separately."}'
```

Response shape:

```json
{"run_id": "01abc...", "result": "Hello, world!"}
```

---

## 3. Inspect the trace in the Observatory

Open <http://localhost:8001/api/observatory/> in your browser.

Find the run by its `run_id`. The Observatory renders every event in the run as
a nested span tree. For the default "Say hello to the world." task you will see:

1. **`AgentStartEvent`** — the agent starts, system prompt and initial task visible.
2. **`LLMInvokeEvent`** — the LLM receives the task and decides to call a tool.
3. **`ToolInvokeEvent`** — the agent calls `greet(name="world")`.
4. **`ToolResultEvent`** — the tool returns `"Hello, world!"`.
5. **`LLMInvokeEvent`** — the LLM receives the tool result and produces its final response.
6. **`AgentCompleteEvent`** — the agent finishes, final output visible.

With a real API key, "Greet Alice and Bob separately." will show two
tool-call/result pairs before the agent completes — the LLM decides to call
`greet` twice.

Each event carries its full payload: token counts on LLM events, arguments and
return values on tool events, elapsed time on all spans. This is the same
observability surface that adopters embed in their own applications by following
`docs/guides/observatory-integration.md`.

---

## 4. Stop

```sh
# Ctrl-C in the compose window, or from another terminal:
just observatory-compose-down
```

Traces are stored in-memory only. They are lost when the container stops.
