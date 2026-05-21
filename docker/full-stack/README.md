# Nanitics — full-stack compose

Docker compose that brings up a Nanitics FastAPI app backed by
Postgres, with the embedded Observatory mounted at
`/api/observatory/`. Designed to be the "one-minute bring-up" example
of a real-LLM Nanitics deployment — the full-stack successor to the
key-free local-dev compose under `docker/observatory-dev/`.

This compose is the substrate for the three showcase runners, each
mounted under `/runners/<slug>`.

## 1. One-minute bring-up

Prerequisite: the embedded Observatory SPA at
`nanitics/observatory/ui_assets/`. The directory is `.gitignore`d
(it's a build artifact that ships inside the wheel) — populate it with
`just observatory-build` before bringing the stack up.

From the repo root:

```sh
cp docker/full-stack/.env.example docker/full-stack/.env
# Edit docker/full-stack/.env and set ANTHROPIC_API_KEY (or switch
# NANITICS_LLM_PROVIDER=openai and set OPENAI_API_KEY).
just full-stack-compose
```

Three URLs once the stack is up:

- Liveness probe: <http://localhost:8000/healthz>
- Observatory UI: <http://localhost:8000/api/observatory/>
- Runner index: <http://localhost:8000/runners>

Shut it down cleanly with `just full-stack-compose-down` — SIGTERM
reaches uvicorn and the lifespan closes the asyncpg pool before the
container exits.

## 2. Services

| Service | Image / Build | Port | Purpose |
|---|---|---|---|
| `app` | Built from `Dockerfile` | 8000 | Nanitics FastAPI shell + embedded Observatory. All three showcase runners register here. |
| `postgres` | `postgres:16` | 5432 | Backs `PostgresTraceStore` for durable trace storage across compose restarts. The SQL analyst runner reuses this container. |

The Observatory is **not** a separate service — the router and UI
live inside `app`, matching the integration pattern adopters use in
their own FastAPI applications. See
[`docs/guides/observatory-integration.md`](../../docs/guides/observatory-integration.md).

## 3. Environment

Every variable is documented in [`.env.example`](./.env.example).

Provider-key constraint: exactly one of `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` must be set, matching `NANITICS_LLM_PROVIDER`. The
app's lifespan raises `RuntimeError` on a missing or mismatched key —
there is no `MockLLMClient` fallback (unlike the local-dev compose).
If you want a key-free first run, use `docker/observatory-dev/`.

## 4. Secret management

- Keys live in env vars only; the image never contains them.
- `.env` is `.gitignore`d so it never enters version control.
- `.dockerignore` excludes `.env` and `.env.example` from the build
  context, so even `docker build` cannot accidentally copy them into
  a layer.

For the full adopter-facing guidance on API-key handling, see
[`docs/guides/security.md`](../../docs/guides/security.md). This
README deliberately does not duplicate that guidance.

## 5. Operating notes

**Resource sizing.** For two concurrent agent runs against Anthropic,
plan on 2 CPU / 4 GB RAM for the `app` container as a starting point.
Scale from there against your actual workload; LLM latency dominates
CPU, so vertical scaling on `app` rarely helps — horizontal replicas
behind a load balancer do.

**Graceful shutdown.** `docker compose down` sends SIGTERM to both
containers. The uvicorn process invokes the FastAPI lifespan's
teardown, which closes the asyncpg pool before exiting.

**Deployment guide.** For the adopter-facing deployment walkthrough,
see [`docs/guides/deployment.md`](../../docs/guides/deployment.md).

## 6. Runners

Each runner mounts its routes under `/runners/<slug>/`. The runner's
own README is the authoritative reference for its pattern, endpoints,
and Observatory trace shape.

- **Self-healing SQL analyst** (`/runners/sql-analyst/*`) — writes SQL
  against a bundled analytical schema; a `Supervisor` retries with
  targeted feedback when the query errors, returns zero rows, or fails
  a ground-truth evaluator. See
  [`sql_analyst/README.md`](./sql_analyst/README.md).
- **Auction-routed request handling** (`/runners/auction-routing/*`) —
  four specialists bid on each incoming request with calibrated
  confidences and grounded per-call cost; the auction allocates to the
  highest-confidence specialist (cheaper bid wins on strict ties).
  See [`auction_routing/README.md`](./auction_routing/README.md).
- **Judge-routed request handling** (`/runners/judge-routing/*`) —
  four tool-using specialists are ranked by a single comparative-
  judgment LLM call; the winning specialist answers using in-memory
  billing/technical/account/policy fixtures. The tooled counterpart to
  `auction-routing`. See
  [`judge_routing/README.md`](./judge_routing/README.md).
- **Retrospective self-improver** (`/runners/self-improver/*`) — a
  deliberately-imperfect task agent runs end-to-end; a critic reads
  that trace back through the SDK's own trace API and emits ranked,
  evidence-cited improvement proposals. See
  [`self_improver/README.md`](./self_improver/README.md).
