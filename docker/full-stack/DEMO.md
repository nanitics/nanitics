# Full-stack compose — demo

A two-service compose (`app` + Postgres) that runs a production-like Nanitics
deployment with three showcase runners and a durable `PostgresTraceStore`. A
real LLM API key is required (no mock fallback).

---

## 1. Start the stack

```sh
# Build the Observatory UI bundle (only needed once, or after UI changes)
just observatory-build

# Copy the env template and fill in your API key
cp docker/full-stack/.env.example docker/full-stack/.env
# Edit docker/full-stack/.env:
#   Set ANTHROPIC_API_KEY (default provider), or switch
#   NANITICS_LLM_PROVIDER=openai and set OPENAI_API_KEY instead.

just full-stack-compose
```

Once both services are healthy, check them:

```sh
curl -s http://localhost:8000/healthz
# → {"status":"ok"}

curl -s http://localhost:8000/readyz
# → {"ready":true,"store":"ok"}

curl -s http://localhost:8000/runners
# → [{"slug":"sql-analyst",...},{"slug":"auction-routing",...},{"slug":"judge-routing",...},{"slug":"self-improver",...}]
```

Observatory UI: <http://localhost:8000/api/observatory/>

---

## 2. Runner: Self-healing SQL analyst (`/runners/sql-analyst/`)

The agent writes SQL against a five-table analytical schema, executes it, and
self-corrects through a `Supervisor` when the query errors, returns zero rows,
or fails a ground-truth evaluator. The interesting thing to observe is the
supervision loop — open the Observatory after each run and find the trace by
`run_id` to see `SupervisionEvent(action="retry")` and
`SupervisionEvent(action="accept")` interleaved with tool calls.

### Fetch the question catalog

```sh
curl -s http://localhost:8000/runners/sql-analyst/questions
```

### Run each canonical question

Each question ships with a hand-computed expected answer. The evaluator gates
the answer — a plausible-sounding but wrong response will trigger a rewrite.

**Total order count** (scalar, no tolerance):

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "total-orders-count"}'
```

**Total revenue** (scalar with ±0.01 tolerance):

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "revenue-total"}'
```

**Top 5 customers by revenue** (ordered row set):

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "top-5-customers-by-revenue"}'
```

**Orders per region** (unordered row set, three-table join):

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "orders-by-region"}'
```

**Top customer per region** (ordered row set, filtered aggregate):

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "top-customer-per-region"}'
```

**Cancelled orders by month** (ordered row set, date bucketing):

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "cancelled-orders-by-month"}'
```

### Response shape

```json
{
  "run_id": "01abc...",
  "accepted": true,
  "attempts": 1,
  "final_sql": "SELECT COUNT(*) FROM orders",
  "answer": "200",
  "rowcount": 1,
  "interventions": []
}
```

When `attempts > 1` the supervisor retried. `interventions` carries one entry
per retry with the `trigger_name` (`query_error_or_empty` or `quality`) and
the feedback the agent received. Open the Observatory and find the run by
`run_id` to see the full loop.

### Ad-hoc free-form question

No ground-truth gate — only the error-catch trigger fires, and the retry
budget is one:

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which product category generated the most revenue?"}'
```

---

## 3. Runner: Auction-routed request handling (`/runners/auction-routing/`)

Four specialist agents (`billing-specialist`, `technical-specialist`,
`account-specialist`, `policy-specialist`) each self-assess their fit for an
incoming request and submit a calibrated confidence bid plus a grounded
per-call cost (`base_rate * complexity`, where `complexity` is the LLM's 1–5
estimate). The auction selects the winner with
`HighestConfidence(tiebreaker=LowestCost())`: the highest-confidence bid
wins, and on a strict tie the bid with the lower grounded cost wins. There
is no HITL branch — every successful `/handle` returns
`"outcome": "specialist_answered"`.

### Submit a request

```sh
curl -s -X POST http://localhost:8000/runners/auction-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "My invoice shows the wrong amount."}'
```

```sh
curl -s -X POST http://localhost:8000/runners/auction-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "The app crashes when I try to export a report."}'
```

```sh
curl -s -X POST http://localhost:8000/runners/auction-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "I need to add a second user to my account."}'
```

```sh
curl -s -X POST http://localhost:8000/runners/auction-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "Does your terms of service permit reselling data I exported through your API?"}'
```

### Response shape

```json
{
  "run_id": "01abc...",
  "outcome": "specialist_answered",
  "winner": "billing-specialist",
  "bids": [
    {"agent_name": "billing-specialist", "confidence": 0.9, "capabilities": ["..."], "estimated_cost": 0.06, "reasoning": "..."},
    {"agent_name": "technical-specialist", "confidence": 0.4, "capabilities": ["..."], "estimated_cost": 0.09, "reasoning": "..."},
    ...
  ],
  "answer": "The discrepancy on your invoice is...",
  "trace_url": "/api/observatory/runs/01abc..."
}
```

Consumers that need a human-handoff gate can build one client-side around
`winner.confidence` from the response — the runner deliberately does not
embed one.

---

## 4. Runner: Judge-routed request handling (`/runners/judge-routing/`)

Four tool-using specialist agents (`billing-specialist`,
`technical-specialist`, `account-specialist`, `policy-specialist`) are
ranked by a single comparative-judgment LLM call (the *judge*); the
top-ranked specialist answers using its tool bundle against in-memory
fixtures. The tooled counterpart to `auction-routing`: comparative
judgment replaces self-assessed bidding, and each Observatory trace
shows the winning specialist's actual tool calls instead of a single
thin LLM step. There is no HITL branch — every successful `/handle`
returns the winning specialist's answer.

### Submit a request

```sh
curl -s -X POST http://localhost:8000/runners/judge-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "Why is invoice INV-1001 still marked unpaid?"}'
```

```sh
curl -s -X POST http://localhost:8000/runners/judge-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "Webhook deliveries to my endpoint started failing this morning."}'
```

```sh
curl -s -X POST http://localhost:8000/runners/judge-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "I lost access to ada@example.com — please reset the password."}'
```

```sh
curl -s -X POST http://localhost:8000/runners/judge-routing/handle \
  -H 'Content-Type: application/json' \
  -d '{"request_text": "What does your data-handling policy say about retaining exported data?"}'
```

### Response shape

```json
{
  "run_id": "01abc...",
  "winner": "billing-specialist",
  "ranking": [
    {"agent_name": "billing-specialist", "confidence": 0.9, "capabilities": ["..."], "estimated_cost": 0.06, "reasoning": "..."},
    {"agent_name": "account-specialist", "confidence": 0.4, "capabilities": ["..."], "estimated_cost": 0.03, "reasoning": "..."},
    ...
  ],
  "answer": "Invoice INV-1001 shows status `unpaid`; the most recent payment...",
  "trace_url": "/api/observatory/runs/01abc..."
}
```

Consumers that need a confidence gate can build one client-side from
`ranking[0].confidence` — the runner deliberately does not embed one.

---

## 5. Runner: Retrospective self-improver (`/runners/self-improver/`)

A deliberately-imperfect task agent runs a short research task against a
bundled markdown corpus under `self_improver/corpus/`, producing an Observatory
trace. A critic — `self_improver.advisor.analyze`'s three specialists — then reads
that trace back through the SDK's own trace API (not database queries, not
screen-scraping) and emits ranked, evidence-cited proposals. The critic run is
itself traced, so the Observatory shows a "trace of trace."

### Task mode — default

Empty body runs the bundled task agent and critiques the fresh trace:

```sh
curl -s -X POST http://localhost:8000/runners/self-improver/run \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Pass a custom `task_input` to steer the task agent:

```sh
curl -s -X POST http://localhost:8000/runners/self-improver/run \
  -H 'Content-Type: application/json' \
  -d '{"task_input": "Explain how redaction interacts with storage."}'
```

### Referenced-trace mode

Supply a `trace_id` from any prior `TracedExecutor`-driven run in this stack
(another `self-improver` call, or a `sql-analyst` / `auction-routing` trace
from earlier in this session) to skip the task phase and critique the
existing trace directly:

```sh
curl -s -X POST http://localhost:8000/runners/self-improver/run \
  -H 'Content-Type: application/json' \
  -d '{"trace_id": "<a-prior-trace-id>"}'
```

An unknown `trace_id` returns `404 {"error": "trace_not_found", ...}`.

### Response shape

```json
{
  "task_run_id": "<uuid or null>",
  "critic_run_id": "<uuid>",
  "task_trace_id": "<uuid>",
  "report": { /* self_improver.advisor.AdvisorReport */ }
}
```

`task_run_id` is `null` in referenced-trace mode; `critic_run_id` is always
present. Both runs appear in the Observatory — click either to see its span
tree.

---

## 6. Stop

```sh
just full-stack-compose-down
```

Traces persist in Postgres across restarts — the Observatory retains all prior
runs after `just full-stack-compose` brings the stack back up.
