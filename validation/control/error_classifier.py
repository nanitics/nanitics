"""ErrorClassifier categorizes SDK errors and RetryPolicy drives retry-with-backoff.

Two concerns, one file, both deterministic:

  1. ``classify_error`` must map the SDK error hierarchy to the right
     ``ErrorCategory``. This is a pure-function mapping — exercised as
     a table of (exception, expected category) cases. A novel error
     type falls through to ``FATAL``.

  2. ``retry_with_backoff`` must:
     - respect ``RetryPolicy.max_attempts`` (a counter-incrementing
       callable that raises ``N-1`` retryable errors then succeeds must
       succeed on the Nth attempt, and must fail if the policy allows
       fewer attempts than required);
     - propagate non-retryable (``FATAL`` / ``CORRECTABLE``) errors
       immediately, regardless of remaining attempts;
     - emit one ``ErrorRetryEvent`` per retry attempt;
     - follow exponential backoff — with jitter disabled, the
       observed delays form the sequence
       ``[base, base*exp_base, base*exp_base**2, ...]``, capped at
       ``max_delay``. We patch ``asyncio.sleep`` to record the delays
       without actually waiting; this keeps the test fast and pins
       the timing contract exactly rather than approximately.

Acceptance criteria (classification):
  - ``LLMRateLimitError`` → ``RETRYABLE``.
  - ``LLMProviderError(status_code=500)`` → ``RETRYABLE``.
  - ``LLMProviderError(status_code=400)`` → ``FATAL``.
  - ``LLMProviderError(status_code=None)`` → ``RETRYABLE``
    (unknown-status default).
  - ``LLMContextLengthError`` → ``FATAL``.
  - ``LLMSchemaViolationError`` → ``CORRECTABLE``.
  - ``ToolParameterError`` / ``ToolExecutionError`` → ``CORRECTABLE``.
  - A novel user-defined exception → ``FATAL`` (default fall-through).

Acceptance criteria (retry):
  - A function that raises ``max_attempts - 1`` retryable errors then
    returns succeeds and returns its value.
  - A function that always raises a retryable error exhausts the
    budget and re-raises the last error; number of calls equals
    ``max_attempts``.
  - A function that raises a FATAL error is called exactly once and
    the error propagates (no retry).
  - With ``jitter=False``, the recorded sleep delays match the
    exponential sequence exactly (within float tolerance) and respect
    ``max_delay``.
  - Exactly one ``ErrorRetryEvent`` fires per retry attempt, carrying
    the attempt number, ``max_attempts``, and ``delay_ms`` matching
    the recorded delay.
"""

from __future__ import annotations

import math

import pytest

from nanitics.capabilities.errors import retry as _retry_module
from nanitics.capabilities.errors.retry import retry_with_backoff
from nanitics.errors import (
    ErrorCategory,
    RetryPolicy,
    classify_error,
)
from nanitics.infrastructure import (
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaViolationError,
    ToolError,
    ToolExecutionError,
    ToolParameterError,
    ToolTimeoutError,
)
from nanitics.infrastructure.observability.events import ErrorRetryEvent
from nanitics.tracing import InMemoryEmitter

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class _NovelError(Exception):
    """An exception the classifier has never seen — must default to FATAL."""


class _AppDefinedToolError(ToolError):
    """An app-defined ToolError subclass — must classify as CORRECTABLE by default."""


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        pytest.param(
            LLMRateLimitError("rate limited"),
            ErrorCategory.RETRYABLE,
            id="rate_limit_retryable",
        ),
        pytest.param(
            LLMProviderError("server error", status_code=500),
            ErrorCategory.RETRYABLE,
            id="provider_5xx_retryable",
        ),
        pytest.param(
            LLMProviderError("bad request", status_code=400),
            ErrorCategory.FATAL,
            id="provider_4xx_fatal",
        ),
        pytest.param(
            LLMProviderError("unknown failure"),
            ErrorCategory.RETRYABLE,
            id="provider_unknown_status_retryable",
        ),
        pytest.param(
            LLMContextLengthError("too long"),
            ErrorCategory.FATAL,
            id="context_length_fatal",
        ),
        pytest.param(
            LLMSchemaViolationError("bad schema"),
            ErrorCategory.CORRECTABLE,
            id="schema_violation_correctable",
        ),
        pytest.param(
            ToolParameterError("bad param", tool_name="search"),
            ErrorCategory.CORRECTABLE,
            id="tool_parameter_correctable",
        ),
        pytest.param(
            ToolExecutionError("boom", tool_name="search"),
            ErrorCategory.CORRECTABLE,
            id="tool_execution_correctable",
        ),
        pytest.param(
            ToolTimeoutError("timed out", tool_name="search", timeout_seconds=5.0),
            ErrorCategory.RETRYABLE,
            id="tool_timeout_retryable",
        ),
        pytest.param(
            _AppDefinedToolError("app-specific tool failure"),
            ErrorCategory.CORRECTABLE,
            id="app_defined_tool_error_correctable",
        ),
        pytest.param(
            _NovelError("who knows"),
            ErrorCategory.FATAL,
            id="novel_error_defaults_to_fatal",
        ),
    ],
)
def test_classify_error_maps_exceptions_to_categories(error: Exception, expected_category: ErrorCategory) -> None:
    assert classify_error(error) == expected_category


# ---------------------------------------------------------------------------
# RetryPolicy + retry_with_backoff
# ---------------------------------------------------------------------------


class _Counter:
    """Mutable counter for a flaky callable. One instance per scenario."""

    def __init__(self) -> None:
        self.calls = 0


async def test_retry_succeeds_on_final_attempt(
    traced_emitter: InMemoryEmitter, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _Counter()
    policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter=False)

    async def flaky() -> str:
        counter.calls += 1
        if counter.calls < 3:
            raise LLMRateLimitError("slow down", retry_after=None)
        return "ok"

    # Scope the sleep patch to the retry module's local ``sleep`` seam so
    # other code using asyncio.sleep (e.g. the pytest-asyncio event loop)
    # is unaffected. Recording the delays lets us pin the backoff sequence
    # exactly rather than approximately.
    recorded_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_delays.append(delay)

    monkeypatch.setattr(_retry_module, "sleep", fake_sleep)
    result = await retry_with_backoff(flaky, policy=policy, classify=classify_error, emitter=traced_emitter)

    assert result == "ok"
    assert counter.calls == 3, f"Expected 3 calls (2 failures + 1 success), got {counter.calls}"

    # Two retries → two ErrorRetryEvents → two recorded delays.
    retry_events = [e for e in traced_emitter.events if isinstance(e, ErrorRetryEvent)]
    assert len(retry_events) == 2
    assert len(recorded_delays) == 2

    # Exponential backoff without jitter: delay_n = base * exp_base**n,
    # capped at max_delay. With base=0.01 and exp_base=2.0, delays are
    # [0.01, 0.02].
    expected = [0.01, 0.02]
    for observed, want in zip(recorded_delays, expected, strict=True):
        assert math.isclose(observed, want, rel_tol=1e-9), (
            f"Expected delay {want}, got {observed} (delays: {recorded_delays})"
        )

    # Each ErrorRetryEvent carries the same delay (in ms) and the correct
    # attempt number / budget.
    for event, observed_delay, expected_attempt in zip(retry_events, recorded_delays, (1, 2), strict=True):
        assert event.attempt == expected_attempt
        assert event.max_attempts == 3
        assert math.isclose(event.delay_ms, observed_delay * 1000, rel_tol=1e-9)
        assert event.category == ErrorCategory.RETRYABLE.value


async def test_retry_exhausts_budget_and_reraises(
    traced_emitter: InMemoryEmitter, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _Counter()
    policy = RetryPolicy(max_attempts=2, base_delay=0.01, jitter=False)

    async def always_fails() -> str:
        counter.calls += 1
        raise LLMRateLimitError("still slow")

    recorded_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_delays.append(delay)

    monkeypatch.setattr(_retry_module, "sleep", fake_sleep)
    with pytest.raises(LLMRateLimitError):
        await retry_with_backoff(always_fails, policy=policy, classify=classify_error, emitter=traced_emitter)

    assert counter.calls == 2, f"Expected exactly max_attempts=2 calls, got {counter.calls}"
    # One retry between two attempts → one retry event, one delay.
    retry_events = [e for e in traced_emitter.events if isinstance(e, ErrorRetryEvent)]
    assert len(retry_events) == 1
    assert len(recorded_delays) == 1


async def test_retry_propagates_fatal_immediately(traced_emitter: InMemoryEmitter) -> None:
    counter = _Counter()
    policy = RetryPolicy(max_attempts=5, base_delay=0.01, jitter=False)

    async def fatal() -> str:
        counter.calls += 1
        raise LLMContextLengthError("exceeded")

    with pytest.raises(LLMContextLengthError):
        await retry_with_backoff(fatal, policy=policy, classify=classify_error, emitter=traced_emitter)

    assert counter.calls == 1, f"Expected exactly 1 call for FATAL error, got {counter.calls}"
    # No retries → no retry events.
    retry_events = [e for e in traced_emitter.events if isinstance(e, ErrorRetryEvent)]
    assert retry_events == []


async def test_retry_respects_max_delay_cap(traced_emitter: InMemoryEmitter, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _Counter()
    # base=1.0, exp_base=10.0, max_delay=3.0 — the second retry would nominally
    # wait 10.0 but must be capped at 3.0.
    policy = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=3.0, exponential_base=10.0, jitter=False)

    async def flaky() -> str:
        counter.calls += 1
        if counter.calls < 4:
            raise LLMRateLimitError("slow")
        return "ok"

    recorded_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_delays.append(delay)

    monkeypatch.setattr(_retry_module, "sleep", fake_sleep)
    result = await retry_with_backoff(flaky, policy=policy, classify=classify_error, emitter=traced_emitter)

    assert result == "ok"
    # Nominal sequence: [1.0, 10.0, 100.0]. Capped: [1.0, 3.0, 3.0].
    assert recorded_delays == [1.0, 3.0, 3.0], f"Expected capped delays [1.0, 3.0, 3.0], got {recorded_delays}"
