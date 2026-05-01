import random
from asyncio import sleep  # module-local seam — see ``await sleep(delay)`` below
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from nanitics.capabilities.errors.classification import ErrorCategory, ErrorClassifier
from nanitics.infrastructure.errors import LLMRateLimitError
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import ErrorRetryEvent

T = TypeVar("T")


class RetryPolicy(BaseModel):
    """Configuration for exponential backoff retry behavior.

    Controls how retryable errors (rate limits, server errors) are retried.
    Delay grows exponentially: ``base_delay * exponential_base^attempt``,
    capped at ``max_delay``. Jitter applies a random multiplier between
    0.5x and 1.0x to avoid thundering herd effects (delay is only ever
    reduced, never increased beyond the calculated value).

    Attributes:
        max_attempts: Maximum number of attempts including the initial call.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff growth.
        jitter: Whether to randomize delay (0.5x–1.0x multiplier).
    """

    model_config = ConfigDict(frozen=True)

    max_attempts: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    @field_validator("max_attempts")
    @classmethod
    def max_attempts_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_attempts must be at least 1")
        return v

    @field_validator("base_delay", "max_delay")
    @classmethod
    def delays_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("delay values must be non-negative")
        return v

    @field_validator("exponential_base")
    @classmethod
    def exponential_base_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("exponential_base must be positive")
        return v


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    classify: ErrorClassifier,
    emitter: EventEmitter | None = None,
) -> T:
    """Retry an async callable with exponential backoff.

    Calls ``fn`` up to ``policy.max_attempts`` times. On each failure,
    classifies the error: non-retryable errors propagate immediately,
    retryable errors trigger a delay before the next attempt. Respects
    ``retry_after`` hints from ``LLMRateLimitError``.

    Emits an ``ErrorRetryEvent`` before each retry delay.

    Args:
        fn: Async callable to retry.
        policy: Retry configuration (attempts, delays, backoff).
        classify: Error classifier to determine retryability.
        emitter: Optional event emitter for retry tracking.

    Returns:
        The result of a successful call.

    Raises:
        Exception: The last error if all attempts are exhausted or error is non-retryable.
    """
    last_error: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            return await fn()
        except Exception as error:
            last_error = error
            category = classify(error)

            if category != ErrorCategory.RETRYABLE:
                raise

            if attempt + 1 >= policy.max_attempts:
                raise

            delay = min(
                policy.base_delay * (policy.exponential_base**attempt),
                policy.max_delay,
            )

            if isinstance(error, LLMRateLimitError) and error.retry_after is not None:
                delay = max(error.retry_after, delay)

            if policy.jitter:
                delay *= random.uniform(0.5, 1.0)

            if emitter is not None:
                emitter.emit(
                    ErrorRetryEvent(
                        trace_id=emitter.trace_id,
                        span_id=emitter.span_id,
                        parent_span_id=emitter.parent_span_id,
                        error_type=type(error).__name__,
                        error_message=str(error),
                        attempt=attempt + 1,
                        max_attempts=policy.max_attempts,
                        delay_ms=delay * 1000,
                        category=category.value,
                    )
                )

            # Uses module-local ``sleep`` (patched by ``_patch_retry_sleep`` in
            # tests/conftest.py) — not ``asyncio.sleep``, which would patch the
            # asyncio module globally and break ``await asyncio.sleep(0)`` yields.
            await sleep(delay)

    # Unreachable: max_attempts >= 1 (validated), and every iteration either
    # returns on success or raises. Kept to satisfy the type checker.
    assert last_error is not None  # pragma: no cover
    raise last_error  # pragma: no cover
