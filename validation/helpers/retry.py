"""Retry wrapper for transient non-determinism in real-service calls.

Narrowly scoped: retries only on a fixed set of recognised transient
failures. All other exceptions propagate immediately so correctness bugs
are not masked.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from nanitics.errors import (
    LLMProviderError,
    LLMRateLimitError,
)

_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 10.0

T = TypeVar("T")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, AssertionError):
        return True
    if isinstance(exc, LLMRateLimitError):
        return True
    if isinstance(exc, LLMProviderError):
        return exc.status_code is not None and exc.status_code >= 500
    return False


def _compute_delay(attempt: int) -> float:
    """Exponential backoff with jitter. ``attempt`` is 1-indexed."""
    base: float = min(_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), _MAX_DELAY_SECONDS)
    jitter: float = random.uniform(0, base / 2)
    return base + jitter


async def run_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Retry ``fn`` on transient failures.

    Retries on:
        - :class:`AssertionError` (LLM-as-judge non-determinism)
        - :class:`~nanitics.LLMRateLimitError`
        - :class:`~nanitics.LLMProviderError` with
          ``status_code >= 500``.

    Does NOT retry on :class:`LLMContextLengthError`, :class:`LLMSchemaViolationError`,
    :class:`ToolExecutionError`, or arbitrary exceptions — those propagate
    immediately so correctness bugs are not masked by retry churn.

    Args:
        fn: An awaitable-returning callable. Called fresh on each attempt.
        max_attempts: Total attempts including the first. Default 3.
        on_retry: Optional callback ``(attempt_number, exception)`` invoked
            after each failure that will be retried.

    Returns:
        The result of the successful attempt.

    Raises:
        Exception: The last exception encountered if all attempts fail, or
            any non-retryable exception raised by ``fn``.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            await asyncio.sleep(_compute_delay(attempt))
    raise AssertionError("unreachable")  # pragma: no cover
