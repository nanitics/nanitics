# Retrospective self-improver runner

One of three showcase runners inside `docker/full-stack/`. Slug:
`self-improver`. Endpoint: `POST /runners/self-improver/run`.

## What this runner demonstrates

**Trace-as-data on the SDK's own live in-process trace store.**

Two runs per invocation:

1. A **task run** — a deliberately-imperfect single-shot
   [`ReActAgent`](../../../nanitics/core/agents/react.py) with two
   file-read tools (`list_bundled_docs`, `read_bundled_doc`) and a
   thin system prompt, answering a question about a bundled markdown
   corpus under `corpus/`. The agent is capped
   at six iterations. Its imperfections — no decomposition, thin tool
   descriptions, tight iteration cap — are the kinds of patterns the
   advisor's rubrics already fire on.
2. A **critic run** — the SDK reads the task's trace back via
   [`PersistentTraceStore.get_span_tree`](../../../nanitics/infrastructure/observability/storage.py),
   converts the stored rows to typed events via the SDK helper
   [`nanitics.trace_events_from_stored`](../../../nanitics/infrastructure/observability/storage.py),
   and passes the typed list to
   [`self_improver.advisor.analyze`](./advisor/analyze.py).
   The critic run is itself wrapped in `TracedExecutor.execute(...)`
   so its specialist fan-out appears alongside the task run in the
   Observatory UI — "trace of trace."

Both runs share the compose's `PostgresTraceStore`; no in-memory
state, no cross-run coupling.

## How to hit the endpoint

Two modes on one POST.

### Task mode (default)

Empty body runs the bundled task agent against the bundled corpus and
then critiques the fresh trace:

```
curl -s -X POST http://localhost:8000/runners/self-improver/run \
    -H 'content-type: application/json' -d '{}'
```

Pass a custom `task_input` to drive the task agent with a different
question:

```
curl -s -X POST http://localhost:8000/runners/self-improver/run \
    -H 'content-type: application/json' \
    -d '{"task_input": "Explain how redaction interacts with storage."}'
```

### Referenced-trace mode

Supply a `trace_id` to skip the task phase entirely and critique an
already-stored trace — from a prior invocation of this runner, from
either of the other showcase runners (`sql-analyst`,
`auction-routing`), or from any other `TracedExecutor`-driven run
whose events landed in the shared store:

```
curl -s -X POST http://localhost:8000/runners/self-improver/run \
    -H 'content-type: application/json' \
    -d '{"trace_id": "<a-prior-trace-id>"}'
```

When `trace_id` is unknown to the store, the endpoint returns
`404 {"error": "trace_not_found", "trace_id": "..."}`.

### Response shape

```json
{
  "task_run_id": "<uuid or null>",
  "critic_run_id": "<uuid>",
  "task_trace_id": "<uuid>",
  "report": { /* self_improver.advisor.AdvisorReport */ }
}
```

`task_run_id` is `null` in referenced-trace mode; `critic_run_id` is
always present. Both runs are visible in the Observatory under
`http://localhost:8000/api/observatory/`.

## How to swap the task or the corpus

The runner is intentionally small. To swap the task:

- Edit `TASK_SYSTEM_PROMPT` and `DEFAULT_TASK_INPUT` in
  [`runner.py`](./runner.py). Keep the prompt short — the demo's value
  depends on the advisor having something to critique.
- Edit `TASK_ITERATION_CAP` to loosen or tighten the cap.
- Swap the `@tool`-decorated functions for your own — the agent
  factory reads the module-level tool list.

To swap the corpus:

- Replace the markdown files under [`corpus/`](./corpus/).
  `list_bundled_docs` returns every `.md` file in the directory
  verbatim, so no manifest needs updating.
- Keep individual files small; the runner does not paginate tool
  outputs.

## Where the critic's proposals come from

The critic is not a fresh agent. It is
[`self_improver.advisor.analyze`](./advisor/analyze.py) — a pipeline of
three specialist agents (`prompts`, `tool-descriptions`,
`coordination-patterns`) reading the event list in parallel and
returning ranked `Proposal` values. The rubric corpus lives under
[`./advisor/rubrics/`](./advisor/rubrics/). Severity + ranking score
are handled by [`rank_proposals`](./advisor/ranking.py).

## Trace of trace

Open `http://localhost:8000/api/observatory/` after one invocation
and you see two run records: the **task** run (the `ReActAgent`'s
ReAct loop, tool calls, final answer) and the **critic** run (the
three specialist spans, each with its own LLM call). They are
separate entries linked in the response body by `task_run_id` and
`critic_run_id`. Clicking through either one opens its span tree.

## Caching note

Under `NANITICS_LLM_PROVIDER=anthropic`, the critic's call is made
against a reconstructed `AnthropicLLMClient` with `enable_caching=True`
so the shared trace prefix is written once (first specialist) and
read twice (parallel fan-out). Under `NANITICS_LLM_PROVIDER=openai`,
caching is not currently shipped — the critic uses the plain client
and pays the three-pass input cost. Expect roughly 2× the Haiku
invocation cost on OpenAI Haiku-equivalent models.
