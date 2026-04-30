# Production

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Production readiness is a sequence of decisions the SDK helps you make but doesn't make for you. This guide is an index: each section names the decision, points you at the guide that owns the details, and states the default posture. Use it as a pre-launch walk-through, not as an implementation tutorial.

## Persistence

Every store has an `InMemory*` implementation for dev and tests, and a durable implementation for production.

- **Traces**: `InMemoryTraceStore` → `PostgresTraceStore`. Required if you need to query, replay, or audit runs after they complete.
- **Checkpoints**: `InMemoryCheckpointStore` → implement `CheckpointStore` for your backend. Required if any workflow suspends (HITL, long-running tasks) and must resume after a restart.
- **Semantic memory**: `InMemorySemanticStore` → `PostgresSemanticStore`. Required if your agents rely on semantic recall across runs.
- **HITL requests**: `InMemoryHitlRequestStore` → `PostgresHitlRequestStore`. Required whenever human input survives a process lifetime.

Connection strings, schema DDL, and migration guidance live in each store's docstring — this guide does not reproduce them. See [Observability](observability.md) for trace storage, [Human-in-the-Loop](human-in-the-loop.md) for `DurableHumanInputProvider`.

> For a worked-out compose wiring Postgres to the trace store, see [Deployment](deployment.md#postgres-provisioning).

## Error handling posture

`ErrorHandler.default()` handles rate limits, provider errors, schema violations, and tool failures with sensible retry and correction defaults. It is the right choice for most services. Build a custom `ErrorHandler` when a specific error class needs domain-specific recovery (e.g., a vendor API that returns 200 with an error body). Never deploy `ErrorHandler.fail_fast()` to production — it is a development aid that surfaces bugs early, not a runtime posture. See [Error Handling](error-handling.md) for the full classifier-and-recovery flow.

## Rate limits and cost

`RetryPolicy` governs retry on rate-limit and transient errors — the defaults are safe starting points. For per-run cost telemetry, read `AgentResult.usage.total_tokens` and aggregate. For per-call detail, subscribe to `"llm.response"` events and extract `usage` from each payload. Model routing (`RoutingLLMClient`, `CostBudgetRouting`) caps spend by redirecting expensive requests to cheaper models.

## Prompt caching (Anthropic)

Anthropic prompt caching is **off by default** — opt in with `AnthropicLLMClient(enable_caching=True)` when the call pattern justifies it. Cache writes cost ~1.25× a baseline input token and cache reads cost ~0.1×, so caching is a net loss on a one-shot call and only breaks even once the cached prefix is reused ≥2 times within the 5-minute TTL. Multi-turn `ReActAgent` loops and repeated `.run(...)` calls that share a stable prefix are the wins; one-shot agents (e.g., a specialist invoked once per request) should leave it off.

When enabled, the agent threads structured system-prompt sections (authored via `SystemPromptBuilder.add_section`) through to the client, which marks the last cacheable section for ephemeral caching so repeated prompts read from the cache instead of re-billing the full prefix. Contributors that produce volatile per-turn content (working memory, current plan, per-turn context assembly) should pass `cacheable=False` on `add_section`; otherwise they rewrite the cache on every call and defeat the benefit.

Cache hits surface as `usage.cache_read_input_tokens` on each `LLMResponseEvent` and aggregate into `TraceSummaryStats.cache_read_tokens` — visible on the Observatory run card. See `nanitics/infrastructure/llm/anthropic.py` for the `enable_caching` argument and [Observability](observability.md) for the `Usage` and trace-summary surface.

OpenAI prompt caching is not yet wired — tracked in the post-release program.

## Scaling

Agents are stateless per run — the state lives in the stores (trace, checkpoint, memory) and the in-flight emitter. This means horizontal scaling is database-bound, not agent-bound. For parallelism within a single run, compose `Parallel`, `MapReduce`, or `DAG` steps rather than calling `asyncio.gather` directly — the composition paths carry trace context and emit lifecycle events. See [Orchestration](orchestration.md) for pattern selection. For background task lifecycle in a FastAPI backend, see [Building Applications](building-applications.md).

## Safety

Every agent should run with an `IterationLimiter` and a `ToolCallLimiter`. The defaults protect against infinite loops but are not budget caps — set them based on your worst-case expected work. Tool sandboxing is separate: `DockerSandbox` for production code execution, `MockSandbox` for tests. See [Safety](safety.md) for the full decision surface.

## HITL durability

If a human can be the critical path of a run, the run must be able to outlive the process. Use `DurableHumanInputProvider` with `PostgresHitlRequestStore`; pair with a `CheckpointStore` so the agent resumes where it left off. Approval gates and revision gates wrap individual tools or steps — choose the wrapper level based on what the human is approving. See [Human-in-the-Loop](human-in-the-loop.md).

## Pre-launch checklist

Each item is a one-liner — follow the link for the decision.

1. Real LLM credentials are injected as environment variables, never committed in code. See [Security](security.md).
2. A production [Error Handling](error-handling.md) posture is set — `ErrorHandler.default()` or a custom handler, never `fail_fast`.
3. Trace storage points at a durable backend (`PostgresTraceStore`) — see [Observability](observability.md).
4. Checkpointing is wired if any run can suspend — see [Human-in-the-Loop](human-in-the-loop.md).
5. `IterationLimiter` and `ToolCallLimiter` are configured per agent — see [Safety](safety.md).
6. Cost tracking is aggregating `AgentResult.usage` into your metrics pipeline.
7. HITL endpoints require authentication and authorization — the SDK does not provide these; see [Building Applications](building-applications.md).
8. Full quality gate (`just check`) is green on `main`, with coverage at 100%.
9. Integration tests have been run against the real LLM provider you ship with. The SDK does not ship an automated prompt-quality validation suite — pick a representative set of scenarios for your application and exercise them against your chosen provider before cutover. Use `MockLLMClient` for determinism in unit tests and keep provider-touching runs in a separate suite gated on credentials.

## See also

- [Safety](safety.md) — iteration limits, cancellation, sandboxing
- [Error Handling](error-handling.md) — classifier, retry, correction
- [Observability](observability.md) — event model, trace storage
- [Building Applications](building-applications.md) — API server, persistence wiring, SSE
- [Human-in-the-Loop](human-in-the-loop.md) — durable input providers and gates
