# Deployment

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

> Looking for the embedded Observatory alone? See [Observatory Integration](observatory-integration.md). Looking for per-decision pre-launch guidance? See [Production](production.md).

This guide walks an adopter from a fresh clone of the Nanitics repo to a running full-stack deployment — the Nanitics FastAPI app, a durable Postgres-backed trace store, the embedded Observatory UI, and four showcase runners — in one bring-up. It then names the adopter-side decisions that sit between "this compose works on my laptop" and "this pattern runs in my own infrastructure."

## What this guide covers

This guide covers exactly one realistic deployment: the full-stack compose under [`docker/full-stack/`](../../docker/full-stack/). It treats that compose as the worked-out example an adopter can read, copy, and adapt — not as a production blueprint that ships turnkey. Operational substance for secrets, production posture, and Observatory auth lives in the guides that own those concerns; this guide cross-links rather than duplicates.

The guide deliberately does **not** cover:

- A framework matrix (FastAPI vs Django vs Flask). The SDK is framework-agnostic; the compose uses FastAPI because that is the minimal glue to mount `create_observatory_router` and host runner endpoints. The wiring pattern transfers.
- A serverless chapter. Long-running agent loops, durable HITL, and streaming traces have awkward shapes on request-per-function runtimes; the SDK does not ship a serverless adapter at v0.1.1.
- Reference Kubernetes manifests or Terraform. Adopter environments differ on image registry, secret source, Postgres provisioning, and reverse-proxy choice — a single reference manifest would be wrong for most readers. The "Take this to your own infrastructure" section below names the decisions in prose.

## Prerequisites

- Docker with `docker compose` v2.
- One LLM API key — `ANTHROPIC_API_KEY` (default, recommended) or `OPENAI_API_KEY`.
- The Observatory embed bundle at `observatory/dist-embed/`. It is committed to the repo for convenience; rebuild it with `just observatory-build` if it is missing or you want to pick up frontend changes.

Both `just` recipes named in this guide — `just full-stack-compose` and `just full-stack-compose-down` — are defined in the repo's [`justfile`](../../justfile).

## First run from a fresh clone

The full path from a fresh clone to running stack is five commands. Each one runs from the repo root.

Clone the repo and enter it:

```sh
git clone https://github.com/nanitics/nanitics
cd nanitics
```

Build the Observatory embed bundle. This only needs to run once per clone, or after a pull that includes frontend changes — the repo ships a pre-built bundle so the step is a no-op on an up-to-date tree:

```sh
just observatory-build
```

Copy the env template and set your API key. The template ships with `NANITICS_LLM_PROVIDER=anthropic` and `NANITICS_LLM_MODEL=claude-haiku-4-5-20251001` as the defaults — open the copy and uncomment `ANTHROPIC_API_KEY=` with your key (or set `NANITICS_LLM_PROVIDER=openai` and uncomment `OPENAI_API_KEY=` instead):

```sh
cp docker/full-stack/.env.example docker/full-stack/.env
# Edit docker/full-stack/.env and set ANTHROPIC_API_KEY.
```

Bring up the stack:

```sh
just full-stack-compose
```

Wait for both services to report healthy. The `postgres` service is the first to come up; the `app` container's healthcheck then hits `/readyz` every 10 seconds and starts returning `200 OK` once the asyncpg pool is live and the trace store has bootstrapped its schema. First boot takes around 60–90 seconds — most of it is the image build and the Python dependency install.

Three probes confirm the stack is up:

```sh
curl -s http://localhost:8000/healthz
# → {"status":"ok"}

curl -s http://localhost:8000/readyz
# → {"ready":true,"store":"ok"}

curl -s http://localhost:8000/runners
# → [{"slug":"sql-analyst",...},{"slug":"auction-routing",...},{"slug":"judge-routing",...},{"slug":"self-improver",...}]
```

The Observatory UI lives at <http://localhost:8000/api/observatory/>. It is empty until a runner emits the first trace — the run list page renders a "No runs yet" placeholder. Fire off the simplest runner invocation to populate it:

```sh
curl -s -X POST http://localhost:8000/runners/sql-analyst/ask \
  -H 'Content-Type: application/json' \
  -d '{"question_id": "total-orders-count"}'
```

The response includes a `run_id`. Refresh the Observatory UI and that run appears at the top of the list — click it to see the span tree, the LLM calls, and the `SupervisionEvent` / `ToolInvokeEvent` pairs the SQL-analyst runner emits.

For curl-by-curl walk-throughs of all four runners — the calibrated-bid auction for auction-routing, the comparative-judgment routing for judge-routing, the trace-of-trace pattern for self-improver, the trace signatures to look for in the Observatory — see the runner-specific READMEs under `docker/full-stack/`.

Shut down cleanly when you're done:

```sh
just full-stack-compose-down
```

`docker compose down` sends SIGTERM to both containers. The `app` container's uvicorn process invokes the FastAPI lifespan's teardown, which closes the asyncpg pool before the process exits. Traces persist in the `postgres_data` named volume across restarts — bring the stack back up with `just full-stack-compose` and the Observatory still shows every prior run.

## What the stack ships

Two services, one network, one persistent volume:

| Service | Image / Build | Port | Purpose |
|---|---|---|---|
| `app` | Built from `docker/full-stack/Dockerfile` | 8000 | Nanitics FastAPI shell. Mounts the embedded Observatory at `/api/observatory/` and registers the three showcase runners under `/runners/<slug>/`. |
| `postgres` | `postgres:16` | 5432 | Backs `PostgresTraceStore` for durable trace storage and `PostgresHitlRequestStore` for the auction-routing runner's HITL path. |

The `app` container exposes three kinds of endpoints in one process:

- **Application surface** — `GET /healthz` and `GET /readyz` for container orchestration, `GET /runners` for the runner registry.
- **Observatory surface** — `GET /api/observatory/` (UI root) and `GET /api/observatory/runs/...` (JSON API) and `GET /api/observatory/runs/{id}/stream` (SSE live-update).
- **Runner surface** — every route a `RunnerRegistration` installs under `/runners/<slug>/*`. The four showcase runners (`sql-analyst`, `auction-routing`, `judge-routing`, `self-improver`) each own their slug and their routes.

The Observatory is **not** a separate service — the router and the pre-built React bundle live inside `app`, matching the integration pattern adopters use in their own FastAPI applications (see [Observatory Integration](observatory-integration.md) for the in-process wiring). The compose's `app.py` is short on purpose: it reads the Postgres DSN, builds the `PostgresTraceStore` in `lifespan` setup, constructs a `TracedExecutor`, mounts `create_observatory_router` at `/api/observatory/`, and delegates every runner's endpoint installation to its `RunnerRegistration.register(app, ctx)` callable. Adopters replicating the pattern do the same five things in their own FastAPI shells.

For the authoritative stack reference — every env var documented, the full service table, the runner index — see [`docker/full-stack/README.md`](../../docker/full-stack/README.md). This guide does not duplicate it.

## Secrets and environment

The compose follows three env-var-only hygiene rules:

- Provider keys (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), the Postgres password, and the SQL-analyst sandbox password all live in env vars read at process startup. They never enter the image.
- [`docker/full-stack/.env`](../../docker/full-stack/) is `.gitignore`d so it cannot enter version control.
- The compose's [`.dockerignore`](../../docker/full-stack/.dockerignore) excludes `.env` and `.env.example` from the build context, so even `docker build` cannot accidentally copy them into a layer.

The lifespan raises `RuntimeError` on a missing or mismatched key — there is no `MockLLMClient` fallback. If you want a key-free first-run compose instead, use [`docker/observatory-dev/`](../../docker/observatory-dev/).

> For the full adopter-facing API-key handling guidance — rotation cadence, minimum-privilege scoping, the custom-event discipline that keeps the no-leakage invariant intact on your side of the trust boundary — see [Security](security.md#api-key-handling). The compose's pattern is a concrete application of that guidance; this section does not re-state it.

## Postgres provisioning

The default compose uses `postgres:16` with a named volume (`postgres_data`) that survives `docker compose down` and is removed by `docker compose down -v`. The `app` container waits on the postgres healthcheck via `depends_on: condition: service_healthy`, so the app never tries to open the pool against an uninitialised database.

Two roles live inside the database:

- The **application role** (`POSTGRES_USER`, default `nanitics`) owns `trace_events`, `runs`, the HITL request store, and the SQL-analyst's analytical tables. The FastAPI app authenticates as this role to read and write traces and to resolve HITL requests.
- The **sandbox role** (`NANITICS_SQL_ANALYST_SANDBOX_USER`, default `sql_analyst_sandbox`) carries `SELECT`-only grants on the five SQL-analyst tables and no access to `trace_events`, `runs`, or the HITL tables. It is pinned to `statement_timeout = '2s'` via `ALTER ROLE`. The SQL-analyst runner's `run_sql` tool authenticates as this role — the app's privileged user never executes LLM-generated SQL.

Production adopters replace this container with a managed Postgres. The compose reads the connection string from `POSTGRES_DSN`; override it to point at your managed endpoint and the app needs no further code changes:

```sh
POSTGRES_DSN=postgresql://nanitics:***@db.example.internal:5432/nanitics
```

The compose's role grants, the sandbox user's `statement_timeout = '2s'` pin, the `depends_on: condition: service_healthy` wiring, and the `pg_isready` healthcheck shape are the pieces worth copying; the `postgres:16` image itself is a local-dev convenience, not a production recommendation. Managed Postgres (RDS, Cloud SQL, AlloyDB, Crunchy Bridge, Supabase) substitutes for the container without any application-side work — connection pooling, failover, backup, and point-in-time recovery are the managed provider's job, not the SDK's.

> For the trace-store durability decision in the broader production context — `PostgresTraceStore` versus `InMemoryPersistentTraceStore`, when retention becomes an adopter-owned problem, and which connection-string and schema-bootstrap details live on the store's docstring — see [Production](production.md#persistence).

## Reverse-proxy auth

The compose exposes `app` on port 8000 with no authentication in front of it. The Observatory at v0.1.1 has no built-in auth hook on `create_observatory_router` — that seam is deliberate, and filling it is adopter-owned at v0.1.1.

Any deployment that reaches more than one trusted developer terminates auth at a reverse proxy — nginx, Caddy, a cloud load balancer — that sits in front of the compose. The proxy protects the mount path (`/api/observatory`) and the UI path (`/api/observatory/`) together. Two proxy-side settings are load-bearing:

- **SSE passthrough.** The Observatory's run-detail view consumes `GET /api/observatory/runs/{run_id}/stream` as an `EventSource`. The proxy must pass `Content-Type: text/event-stream` unbuffered — `proxy_buffering off;` in nginx, implicit in Caddy's `reverse_proxy` default, `buffering=false` on most cloud load balancers.
- **Auth policy scope.** Apply the same auth to `/api/observatory/` (the UI) and `/api/observatory/api/*` (the JSON API) — the UI calls the JSON API directly, so splitting the policy breaks the dashboard.

Applying auth at the proxy rather than inside the FastAPI app keeps Nanitics' surface unchanged across dev and production. The runner endpoints (`/runners/*`) live under the same proxy policy for most deployments; split them out if your runners are public endpoints served to untrusted users.

> For the full Observatory production posture — the four seams (auth, multi-tenancy, retention, content scrubbing), who owns each, and why no default `ObservatoryAuthProvider` ships at v0.1.1 — see [Observatory Integration](observatory-integration.md#for-production).

## Resource sizing and scaling

The `app` container carries one FastAPI process plus an asyncpg pool. For two concurrent agent runs against Anthropic, 2 CPU / 4 GB RAM is the starting point — LLM latency dominates CPU, so vertical scaling on `app` rarely helps. Horizontal replicas behind a load balancer do, because agents are stateless per run: the state lives in the trace store, the HITL store, and the in-flight emitter.

Three signals worth watching once the stack sees real traffic:

- **`app` CPU utilisation** — if it plateaus well below the container limit while latency rises, you are LLM-provider bound, not CPU bound. Horizontal replicas and prompt-caching are the right axes; bigger `app` containers are not.
- **`postgres` IOPS** — `trace_events` grows linearly with event emission rate. A single runner invocation on a Haiku-equivalent model writes on the order of 30–200 rows; a 50 GB volume is comfortable for weeks of demo traffic. Move to a managed Postgres with provisioned IOPS well before the volume fills.
- **`app` pool saturation** — asyncpg's pool exposes its size via the pool object. If connection wait time rises, the pool is undersized; bump the pool limit before reaching for more app replicas.

Agent composition (`Parallel`, `MapReduce`, `DAG`) is the right tool for within-run parallelism and carries trace context across the fan-out automatically. Do not reach for `asyncio.gather` directly inside an agent's work function — the composition paths emit lifecycle events the Observatory needs to render the span tree.

> For the full "agents are stateless per run" scaling model, the composition paths that carry trace context across replicas (`Parallel`, `MapReduce`, `DAG`), and the cost-and-rate-limit posture, see [Production](production.md#scaling).

## Graceful shutdown

`docker compose down` sends SIGTERM to both containers. Inside `app`, SIGTERM reaches the uvicorn process, uvicorn stops accepting new requests, the FastAPI lifespan's teardown runs, and the teardown closes the asyncpg pool before the process exits. No in-flight trace events are lost provided the agent loop is inside a `TracedExecutor.execute(...)` — that executor flushes events to the store on completion.

The load-bearing pattern for adopters is the FastAPI lifespan shape: open the Postgres pool and the trace store during `lifespan` setup, close the pool during `lifespan` teardown, and never leak connections past the process lifetime. See the [`app.py`](../../docker/full-stack/app.py) lifespan for the exact wiring. For the general async-lifespan pattern the stack replicates, see [Building Applications](building-applications.md).

## Take this to your own infrastructure

The compose is a worked-out local example, not a universal starting point. Taking this pattern to your own infrastructure means making five decisions your environment constrains — the compose makes each one in the simplest local-dev way, and a real deployment makes each one differently. None of them is binary, and none is well served by a reference Kubernetes manifest: every one is shaped by your existing infrastructure, not by the SDK.

**Image build.** Use [`docker/full-stack/Dockerfile`](../../docker/full-stack/Dockerfile) as a template, not as a ship-as-is artifact. The Dockerfile builds from `python:3.11-slim`, installs Nanitics with the `api,anthropic,openai,postgres` extras, and copies the pre-built Observatory embed bundle plus the compose's glue modules. In your environment you probably pin the SDK to a release tag (`pip install nanitics[api,anthropic,postgres]==0.1.1`) rather than installing from the source tree, you push the image to your own registry with your own tagging convention, and your CI builds the Observatory embed bundle as part of the image build rather than copying a committed artifact. The load-bearing pieces of the Dockerfile are the layering (pyproject+source first, bundle+glue last) and the `CMD ["uvicorn", "app:app", ...]` entrypoint — keep those, adjust everything else to your build system.

**Secrets source.** The compose reads provider keys from a `.env` file mounted via `docker compose`'s env-file mechanism. That is the cheapest-possible local-dev posture and it does not belong in production. Any adopter running in production replaces the `.env` with whatever secret store their environment already uses — cloud-provider KMS (AWS SSM/Secrets Manager, GCP Secret Manager, Azure Key Vault), a shared secrets tool (HashiCorp Vault), Kubernetes Secrets projected as env vars, or a CI-injected runtime env. The SDK never cares how the key gets into `os.environ` at the moment the LLM client is constructed; it cares that it is there and that it is never serialised into a trace event. The no-leakage invariant the SDK enforces on its side of the trust boundary (see [Security](security.md#api-key-handling)) is the SDK's half; getting the key into the process without leaving a copy on disk, in a log, or in an image layer is yours.

**Postgres provisioning.** The compose bundles `postgres:16` with a named volume. A managed Postgres (RDS, Cloud SQL, AlloyDB, Crunchy Bridge, Supabase) substitutes for that container without touching any application code — `POSTGRES_DSN` points at the managed endpoint, the application role is provisioned with the schema `PostgresTraceStore.bootstrap_schema()` expects, and the sandbox role and grants are recreated per the compose's SQL. Backup, failover, and point-in-time recovery are the managed provider's responsibility. Retention is yours: `PostgresTraceStore` carries events indefinitely, so once trace volume matters you need a periodic job that deletes old runs (matching your compliance policy) or a custom store wrapper that applies a TTL. The SDK ships no retention machinery — retention policy is domain-specific, and a one-size default would be wrong for most adopters.

**Reverse-proxy auth.** Who can reach `/api/observatory/` is your decision and not ours. Terminate auth at a proxy whose policy your organisation already operates — SSO via OIDC for the developer-facing UI, mTLS for service-to-service traffic, a VPN for the administrative surface, whatever your existing security boundary is. The Observatory router does not accept an auth hook at v0.1.1 and this is deliberate: an adopter's auth policy is almost never "the one this SDK ships" and a default hook would create a false sense of security if left at its defaults. A future `ObservatoryAuthProvider` protocol will fill the seam, and it will be signal-driven from adopter deployments like yours — deployments that name the specific policies they want the SDK to support shape the protocol.

**Resource sizing.** The 2 CPU / 4 GB starting point is a sighting shot, not a load profile. Your agents' latency and concurrency profile are yours to measure, and the measurement surfaces are already on the SDK: `AgentResult.usage.total_tokens` for the per-run token footprint, `TraceSummaryStats.cache_read_tokens` for cache-hit telemetry on Anthropic (see [Production](production.md#prompt-caching-anthropic)), and the per-event timing in every Observatory span for latency breakdown. Horizontal replicas of `app` behind a load balancer are the right scaling axis once one instance is saturated; vertical scaling on Postgres is the right axis once `trace_events` growth bottlenecks on I/O. Your actual numbers will diverge from the compose's defaults in both directions — a light-traffic internal tool runs comfortably on less than 2 CPU; a production support-agent fleet needs far more.

## Next steps

- Walk the three runners end-to-end. See the READMEs under `docker/full-stack/`.
- Work through the pre-launch operational decisions. See [Production](production.md).
- Understand the trace substrate the runners emit into. See [Observability](observability.md).
