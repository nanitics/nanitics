"""Unit tests for `validation.helpers.retry`.

Verifies the retry-set boundary, attempt counting, the ``on_retry`` callback,
and that non-retryable exceptions propagate immediately.
"""

from __future__ import annotations

import pytest

from nanitics.infrastructure.errors import (
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaViolationError,
    ToolExecutionError,
)
from validation.helpers.retry import run_with_retry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("validation.helpers.retry.asyncio.sleep", _instant)


async def test_returns_immediately_on_success() -> None:
    async def fn() -> str:
        return "ok"

    assert await run_with_retry(fn) == "ok"


async def test_retries_on_assertion_error_then_succeeds() -> None:
    attempts = []

    async def fn() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise AssertionError("judge failed")
        return "ok"

    assert await run_with_retry(fn, max_attempts=3) == "ok"
    assert len(attempts) == 3


async def test_retries_on_rate_limit_error() -> None:
    attempts = []

    async def fn() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise LLMRateLimitError("429")
        return "ok"

    assert await run_with_retry(fn, max_attempts=3) == "ok"
    assert len(attempts) == 2


async def test_retries_on_5xx_provider_error() -> None:
    attempts = []

    async def fn() -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise LLMProviderError("upstream down", status_code=503)
        return "ok"

    assert await run_with_retry(fn, max_attempts=3) == "ok"


async def test_does_not_retry_on_4xx_provider_error() -> None:
    async def fn() -> str:
        raise LLMProviderError("bad request", status_code=400)

    with pytest.raises(LLMProviderError):
        await run_with_retry(fn, max_attempts=3)


async def test_does_not_retry_on_context_length_error() -> None:
    async def fn() -> str:
        raise LLMContextLengthError("too long")

    with pytest.raises(LLMContextLengthError):
        await run_with_retry(fn, max_attempts=3)


async def test_does_not_retry_on_schema_violation_error() -> None:
    async def fn() -> str:
        raise LLMSchemaViolationError("bad schema")

    with pytest.raises(LLMSchemaViolationError):
        await run_with_retry(fn, max_attempts=3)


async def test_does_not_retry_on_tool_execution_error() -> None:
    async def fn() -> str:
        raise ToolExecutionError("boom", tool_name="t")

    with pytest.raises(ToolExecutionError):
        await run_with_retry(fn, max_attempts=3)


async def test_does_not_retry_on_arbitrary_exception() -> None:
    async def fn() -> str:
        raise RuntimeError("surprise")

    with pytest.raises(RuntimeError):
        await run_with_retry(fn, max_attempts=3)


async def test_raises_after_exhausting_attempts() -> None:
    async def fn() -> str:
        raise AssertionError("always fails")

    with pytest.raises(AssertionError):
        await run_with_retry(fn, max_attempts=3)


async def test_on_retry_callback_called_with_attempt_and_exception() -> None:
    attempts: list[int] = []
    captured: list[Exception] = []

    def on_retry(attempt: int, exc: Exception) -> None:
        attempts.append(attempt)
        captured.append(exc)

    async def fn() -> str:
        if len(attempts) < 2:
            raise AssertionError(f"attempt {len(attempts)}")
        return "ok"

    await run_with_retry(fn, max_attempts=5, on_retry=on_retry)
    assert attempts == [1, 2]
    assert all(isinstance(e, AssertionError) for e in captured)


async def test_rejects_invalid_max_attempts() -> None:
    async def fn() -> str:
        return "ok"

    with pytest.raises(ValueError, match="max_attempts"):
        await run_with_retry(fn, max_attempts=0)


async def test_no_status_code_provider_error_does_not_retry() -> None:
    async def fn() -> str:
        raise LLMProviderError("network blip", status_code=None)

    with pytest.raises(LLMProviderError):
        await run_with_retry(fn, max_attempts=3)
