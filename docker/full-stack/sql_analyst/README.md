# Self-healing SQL analyst runner

The first showcase runner registered on the full-stack compose. Mounted
at `/runners/sql-analyst/`, it writes SQL against a bundled analytical
schema, executes queries under a sandboxed Postgres role, and self-corrects
through a `Supervisor` driven by a programmatic ground-truth evaluator.

## 1. What this runner demonstrates

The load-bearing pattern is `Supervisor` plus a **programmatic**
ground-truth evaluator — **not** LLM-as-judge. Every canonical question
ships with a hand-computed expected value; the `GroundTruthEvaluator`
compares the agent's structured answer against that value and returns
`ACCEPT` or `REVISE` with concrete feedback (`expected COUNT(*) = 37,
got 42`). A second trigger — a `PredicateTrigger` on the last `run_sql`
result — drives a rewrite whenever the agent's query errors out or
comes back empty. That combination is what makes the showcase
differentiating: the answer path has a truth gate that a plausible-
sounding response cannot sneak past.

Open the Observatory UI at `/api/observatory/` and find the run that
matches a `POST /runners/sql-analyst/ask` response's `run_id`. The
self-healing loop reads off a `SupervisionEvent(action="retry")`
followed (on rewrite success) by a `SupervisionEvent(action="accept")`,
each interleaved with `ToolInvokeEvent(name="run_sql")` and
`ToolResultEvent` pairs. When the agent nails the answer on the first
attempt only the `accept` event appears — that is the trace signature
to look for.

## 2. Schema

Five tables. All primary keys are integer `SERIAL`; all foreign keys
are `ON DELETE RESTRICT`. Row counts below are the deterministic seed
shipped in `schema.sql`.

| Table         | Columns                                                                                              | Rows | Notes                                      |
| ------------- | ---------------------------------------------------------------------------------------------------- | ---: | ------------------------------------------ |
| `regions`     | `id`, `name`, `country_code`                                                                         |    5 | `country_code` is ISO 3166-1 alpha-2.      |
| `customers`   | `id`, `name`, `email`, `region_id` → `regions`, `signup_date`                                        |   50 | `signup_date` is deterministic.            |
| `products`    | `id`, `sku`, `name`, `category`, `unit_price NUMERIC(10,2)`                                          |   30 | Five categories.                           |
| `orders`      | `id`, `customer_id` → `customers`, `order_date`, `status`                                            |  200 | `status ∈ {pending, shipped, delivered, cancelled}`. |
| `order_items` | `id`, `order_id` → `orders`, `product_id` → `products`, `quantity`, `unit_price_at_order NUMERIC(10,2)` |  500 | `quantity > 0` CHECK constraint.           |

```text
regions ──< customers ──< orders ──< order_items >── products
```

The canonical source is [`schema.sql`](./schema.sql); this sketch is the
elevator version.

## 3. Sample questions

Six canonical questions ship with the runner, covering `COUNT`, `SUM`
with tolerance, `GROUP BY` with ordering, a multi-table join, a
filtered aggregate bucketed by month, and a per-group maximum using a
window function or correlated subquery.

| id                           | question                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| `total-orders-count`         | How many orders are there in total?                                                         |
| `revenue-total`              | Total revenue across all order items (`SUM(quantity * unit_price_at_order)`).               |
| `top-5-customers-by-revenue` | Top 5 customers by revenue, ordered descending.                                             |
| `orders-by-region`           | Order counts per region via a three-table join.                                             |
| `cancelled-orders-by-month`  | Cancelled orders per month, ordered by month ascending.                                     |
| `top-customer-per-region`    | For each region, the top-spending customer from non-cancelled orders only, ordered by region. |

The canonical answers are **not** published here — they live in
[`questions.py`](./questions.py) as the ground truth the evaluator
gates on. Exposing them in the README would defeat the point of the
pattern.

Fetch the live list (ids + natural-language prompts only) over HTTP:

```sh
curl -s http://localhost:8000/runners/sql-analyst/questions
```

## 4. How to run one question

Pick a question id from the catalog above and hit the ask endpoint:

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "total-orders-count"}'
```

Response shape:

```json
{
  "run_id": "...",
  "accepted": true,
  "attempts": 1,
  "final_sql": "SELECT COUNT(*) FROM orders",
  "answer": "<the value the question asked for>",
  "rowcount": 1,
  "interventions": []
}
```

On a rewrite path `attempts > 1` and `interventions` carries one entry
per retry with the `trigger_name` that fired (`query_error_or_empty`
or `quality`) and the feedback the agent received. Then open
[`/api/observatory/`](/api/observatory/) in a browser and locate the
run by `run_id` — the trace narrates the full supervise-rewrite-
evaluate loop.

The endpoint also accepts a free-form `question` for ad-hoc exploration:

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which region signed up the most customers in 2023?"}'
```

Ad-hoc runs install only the error-catch trigger — no value-level
gate — and the supervisor's retry budget is one.

## 5. Add a new sample question

Adding a question is a three-line workflow:

1. Append a new `SampleQuestion(id=..., question=..., expected=...)`
   entry to `QUESTIONS` in [`questions.py`](./questions.py). Compute
   the `expected` value by hand against the seed data (or run the
   canonical SQL manually once and copy the result).
2. Re-run the catalog invariant tests to confirm the new entry
   self-matches and the id is unique kebab-case:

   ```sh
   pytest tests/test_sql_analyst_runner.py::TestCatalogInvariants
   ```
3. Hit the new id with the `/ask` curl above.

No changes to `supervisor`, `evaluator`, `tool`, or `runner.py` are
required — that is the whole point of the catalog pattern.

## 6. Sandbox posture

The `run_sql` tool never connects as the app's privileged user. It
opens a fresh asyncpg connection per call against a dedicated Postgres
role (`sql_analyst_sandbox` by default) whose grants are `SELECT` only
on the five analyst tables. The role has no privileges on
`trace_events` or `runs`, no write privileges anywhere, and carries a
server-side `statement_timeout = '2s'` pinned by `ALTER ROLE`. On top
of that, the tool injects `LIMIT 200` into bare `SELECT`s and surfaces
asyncpg errors to the agent as `ToolResult` content starting with
`ERROR:` so rewrites are observable, not swallowed.

Override the sandbox password in production via
`NANITICS_SQL_ANALYST_SANDBOX_PASSWORD`; the `.env.example` default is
dev-only.

For the broader API-key and deployment security posture (secret
handling, `.env` hygiene, `.dockerignore` guards), see
[`../../../docs/guides/security.md`](../../../docs/guides/security.md).
This README deliberately does not duplicate that guidance.
