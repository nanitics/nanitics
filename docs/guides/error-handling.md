# Error Handling

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Agents interact with LLMs and tools that can fail — rate limits, invalid parameters, network errors, unexpected outputs. The error handling system classifies failures and applies the appropriate recovery strategy: retry, self-correct, degrade gracefully, or fail fast.

> **See also:** [examples/control/error_handling.py](../../examples/control/error_handling.py) — runnable example covering classification, self-correction, retry policies, and graceful degradation.

## When to Customize Error Handling

All agents use `ErrorHandler.default()` out of the box — retries with backoff, LLM self-correction for bad tool calls, and graceful degradation when correction fails. No configuration needed for standard resilience.

**Use `ErrorHandler.fail_fast()` when** you want immediate error propagation during development or testing.

**Build a custom `ErrorHandler` when** you need to change retry limits, correction budgets, or the classification logic itself (e.g., treating all tool errors as retryable instead of correctable). See the `ErrorHandler` and `RetryPolicy` docstrings for configuration options.

## Error Categories

Every error is classified into one of three categories that determine recovery behavior:

| Category | Meaning | Recovery | Examples |
|----------|---------|----------|----------|
| `RETRYABLE` | Transient infrastructure failure | Automatic retry with backoff | Rate limits, server errors (5xx), timeouts, provider overload (`LLMOverloadedError`) |
| `CORRECTABLE` | The agent made a mistake | Self-correction prompt | Bad tool parameters, wrong tool name, schema violations |
| `FATAL` | Unrecoverable | Raise or degrade | Context length exceeded, budget exhausted, 4xx client errors, auth failure (`LLMAuthenticationError`), quota exhaustion (`LLMQuotaExhaustedError`) |

> `LLMQuotaExhaustedError` is classified **FATAL** even though it surfaces from an upstream 429: quota exhaustion is a billing-state condition that retry cannot resolve within the budget window. Callers that previously substring-matched the message text for `"insufficient_quota"` or `"credit balance"` to detect this can now catch the typed subclass directly.

The built-in `classify_error` function maps every error in the SDK hierarchy to one of these categories. To override classification, implement the `ErrorClassifier` protocol — a callable that takes an `Exception` and returns an `ErrorCategory`.

## Recovery Flow

When an error occurs, the handler processes it through a pipeline: **classify → retry → correct → degrade → give up**. Each stage is a decision gate — if the error doesn't match or the budget is exhausted, it falls through to the next stage.

### Retry (retryable errors)

Retryable errors trigger automatic retry with exponential backoff and jitter. The `RetryPolicy` controls attempt limits, delay curves, and jitter. For `LLMRateLimitError` with a `retry_after` hint, the backoff respects the server's requested delay (using whichever is larger — the calculated backoff or the server hint).

Each retry emits an `ErrorRetryEvent` for observability. Retries are transparent to the LLM — it doesn't know a retry happened.

### Correction (correctable errors)

If the error is correctable and the correction budget isn't exhausted, the handler generates a structured correction prompt and injects it as the tool result. The LLM sees actionable feedback — which parameter failed, what schema was expected, which tools are available — and adjusts its next attempt.

The correction prompt is tailored per error type. `ToolParameterError` explains which parameter failed and why; `ToolNotFoundError` lists available tools; `LLMSchemaViolationError` shows the expected schema. This feedback gives the LLM the information it needs to self-correct.

Each correction emits an `ErrorCorrectionEvent`.

### Degradation (correction budget exhausted)

When correction attempts are exhausted (per-tool or per-run limit), the handler switches to graceful degradation: instead of crashing, it tells the LLM to complete the task with the information gathered so far and state what it could not accomplish. This produces a partial result rather than a hard failure.

Degradation emits an `ErrorDegradationEvent`.

### Fatal errors

Errors classified as `FATAL` bypass correction and degradation entirely — they propagate as exceptions. The calling code decides how to handle them.

## How Recovery Differs by Error Source

The three categories determine recovery **strategy**, but the mechanism differs depending on where the error originates:

**LLM errors:** `RETRYABLE` errors trigger automatic retry via `retry_with_backoff` — the agent re-calls the LLM transparently. `CORRECTABLE` and `FATAL` LLM errors propagate as exceptions.

**Tool errors:** Both `RETRYABLE` and `CORRECTABLE` tool errors generate a correction prompt — the LLM receives structured feedback and decides whether to retry with different parameters, switch tools, or abandon the approach. There is no automatic retry for tool errors. This is intentional: the LLM is better positioned than a blind retry loop to decide the right recovery action.

In practice, all tool errors follow the same code path in `handle_tool_error` — they produce correction prompts and consume the correction budget, regardless of whether they're classified as `RETRYABLE` or `CORRECTABLE`.

## Agent Type Differences

The correction and degradation flow applies to `ReActAgent` and agents that use structured tool calls. `CodeActAgent` handles errors differently:

- **Code execution errors** — tracebacks are fed directly back to the LLM as output, bypassing `ErrorHandler`. The LLM sees the full traceback and adjusts its code in the next iteration.
- **LLM errors** — handled identically to `ReActAgent` via `retry_with_backoff`.

The `ErrorHandler`'s tool-specific methods (`handle_tool_error`, `should_degrade`, `format_degradation_message`) are not used by `CodeActAgent`.

## The Error Hierarchy

All SDK errors extend `NaniticsError` with shared tracing fields (`trace_id`, `span_id`). The hierarchy determines default classification:

```
NaniticsError
├── LLMError
│   ├── LLMRateLimitError        — retryable
│   ├── LLMContextLengthError    — fatal
│   ├── LLMProviderError         — retryable (5xx) or fatal (4xx)
│   │   ├── LLMAuthenticationError — fatal (credentials rejected)
│   │   ├── LLMQuotaExhaustedError — fatal (billing state, not transient)
│   │   └── LLMOverloadedError     — retryable (transient capacity pressure)
│   └── LLMSchemaViolationError  — correctable
├── EmbeddingError
│   ├── EmbeddingRateLimitError  — retryable
│   └── EmbeddingProviderError   — retryable (5xx)
├── ToolError
│   ├── ToolNotFoundError        — correctable
│   ├── ToolParameterError       — correctable
│   ├── ToolExecutionError       — correctable
│   └── ToolTimeoutError         — retryable
└── AgentError
    ├── AgentIterationLimitError — fatal
    ├── AgentBudgetExceededError — fatal
    └── AgentEscalationError     — fatal
```

> **See also:** Error class docstrings for error-specific fields. [Observability guide](observability.md) for error event details (`ErrorRetryEvent`, `ErrorCorrectionEvent`, `ErrorDegradationEvent`, `AgentErrorEvent`).

## Putting It Together

A typical production agent uses `ErrorHandler.default()` or a lightly customized handler. The most common customizations are:

- **Increasing retry attempts** for agents that call rate-limited APIs heavily — adjust `RetryPolicy.max_attempts` and `max_delay`.
- **Raising correction budgets** for agents with many tools where trial-and-error is expected — adjust `max_corrections` and `max_total_corrections`.
- **Custom classification** for domain-specific tools where the default classification doesn't match — e.g., treating a specific `ToolExecutionError` as retryable because the underlying service has transient failures.

For context-length errors, error handling alone isn't sufficient — use [Context Management](context-management.md) to prevent them proactively.

## Pitfalls

**Correction without feedback.** The correction prompt is only useful if the LLM can understand what went wrong. Write clear error messages in your tools — `ToolParameterError("Invalid date format, expected YYYY-MM-DD")` is actionable; `ToolParameterError("bad input")` is not.

**Too many corrections.** If the LLM can't fix the issue in 2–3 attempts, it usually won't fix it in 10. The defaults (3 per tool, 5 per run) are a good balance.

**Swallowing fatal errors.** Don't catch `AgentIterationLimitError` or `AgentBudgetExceededError` inside tools — these are safety boundaries meant to stop execution. Let them propagate.

**Using `fail_fast()` in production.** `ErrorHandler.fail_fast()` disables all retry, correction, and degradation. Useful in tests, but in production it means any transient LLM error or minor tool-call mistake crashes the agent. Stick with the default unless you have a specific reason to disable resilience.
