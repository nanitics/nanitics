from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from nanitics.capabilities.errors.classification import classify_error
from nanitics.capabilities.errors.retry import (
    RetryPolicy,
    retry_with_backoff,
)
from nanitics.infrastructure.errors import (
    LLMProviderError,
    LLMRateLimitError,
)
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import ErrorRetryEvent

SLEEP_PATH = "nanitics.capabilities.errors.retry.sleep"


class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True

    def test_default_retry_sequence(self) -> None:
        """Default policy produces delays of ~2s, 4s, 8s, 16s (before jitter), capped at 60s."""
        policy = RetryPolicy(jitter=False)
        delays = [
            min(policy.base_delay * (policy.exponential_base**attempt), policy.max_delay)
            for attempt in range(policy.max_attempts - 1)
        ]
        assert delays == [2.0, 4.0, 8.0, 16.0]

    def test_frozen(self) -> None:
        policy = RetryPolicy()
        with pytest.raises(ValidationError):
            policy.max_attempts = 5

    def test_custom_values(self) -> None:
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert policy.max_attempts == 5
        assert policy.base_delay == 0.5
        assert policy.max_delay == 10.0
        assert policy.exponential_base == 3.0
        assert policy.jitter is False

    def test_max_attempts_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max_attempts must be at least 1"):
            RetryPolicy(max_attempts=0)

    def test_max_attempts_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="max_attempts must be at least 1"):
            RetryPolicy(max_attempts=-1)

    def test_negative_base_delay_rejected(self) -> None:
        with pytest.raises(ValidationError, match="delay values must be non-negative"):
            RetryPolicy(base_delay=-1.0)

    def test_negative_max_delay_rejected(self) -> None:
        with pytest.raises(ValidationError, match="delay values must be non-negative"):
            RetryPolicy(max_delay=-1.0)

    def test_zero_exponential_base_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exponential_base must be positive"):
            RetryPolicy(exponential_base=0.0)

    def test_negative_exponential_base_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exponential_base must be positive"):
            RetryPolicy(exponential_base=-1.0)


class TestRetryWithBackoff:
    async def test_successful_call_returns_immediately(self) -> None:
        fn = AsyncMock(return_value="result")
        policy = RetryPolicy(max_attempts=3)

        result = await retry_with_backoff(fn, policy, classify_error)

        assert result == "result"
        assert fn.call_count == 1

    async def test_retryable_error_succeeds_on_second_attempt(
        self,
    ) -> None:
        fn = AsyncMock(
            side_effect=[
                LLMProviderError("server error", status_code=500),
                "success",
            ]
        )
        policy = RetryPolicy(max_attempts=3, jitter=False)

        with patch(SLEEP_PATH, new_callable=AsyncMock):
            result = await retry_with_backoff(fn, policy, classify_error)

        assert result == "success"
        assert fn.call_count == 2

    async def test_retryable_error_exhausts_budget(self) -> None:
        error = LLMProviderError("server error", status_code=500)
        fn = AsyncMock(side_effect=error)
        policy = RetryPolicy(max_attempts=3, jitter=False)

        with (
            patch(SLEEP_PATH, new_callable=AsyncMock),
            pytest.raises(LLMProviderError, match="server error"),
        ):
            await retry_with_backoff(fn, policy, classify_error)

        assert fn.call_count == 3

    async def test_non_retryable_error_raises_immediately(
        self,
    ) -> None:
        error = LLMProviderError("unauthorized", status_code=401)
        fn = AsyncMock(side_effect=error)
        policy = RetryPolicy(max_attempts=3)

        with pytest.raises(LLMProviderError, match="unauthorized"):
            await retry_with_backoff(fn, policy, classify_error)

        assert fn.call_count == 1

    async def test_retry_after_from_rate_limit_is_respected(
        self,
    ) -> None:
        fn = AsyncMock(
            side_effect=[
                LLMRateLimitError("rate limited", retry_after=10.0),
                "ok",
            ]
        )
        policy = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False)

        with patch(SLEEP_PATH, new_callable=AsyncMock) as mock_sleep:
            result = await retry_with_backoff(fn, policy, classify_error)

        assert result == "ok"
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay >= 10.0

    async def test_jitter_produces_varied_delays(self) -> None:
        error = LLMProviderError("server error", status_code=500)
        fn = AsyncMock(side_effect=[error, error, "ok"])
        policy = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=True)

        delays: list[float] = []

        async def capture_delay(delay: float) -> None:
            delays.append(delay)

        with patch(SLEEP_PATH, side_effect=capture_delay):
            await retry_with_backoff(fn, policy, classify_error)

        assert len(delays) == 2
        # Attempt 0: base * 2^0 = 1.0, jittered to [0.5, 1.0]
        assert 0.5 <= delays[0] <= 1.0
        # Attempt 1: base * 2^1 = 2.0, jittered to [1.0, 2.0]
        assert 1.0 <= delays[1] <= 2.0

    async def test_exponential_backoff_without_jitter(self) -> None:
        error = LLMProviderError("server error", status_code=500)
        fn = AsyncMock(side_effect=[error, error, "ok"])
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )

        delays: list[float] = []

        async def capture_delay(delay: float) -> None:
            delays.append(delay)

        with patch(SLEEP_PATH, side_effect=capture_delay):
            await retry_with_backoff(fn, policy, classify_error)

        assert delays == [1.0, 2.0]

    async def test_max_delay_caps_backoff(self) -> None:
        error = LLMProviderError("server error", status_code=500)
        fn = AsyncMock(side_effect=[error, error, error, "ok"])
        policy = RetryPolicy(
            max_attempts=4,
            base_delay=10.0,
            max_delay=15.0,
            exponential_base=2.0,
            jitter=False,
        )

        delays: list[float] = []

        async def capture_delay(delay: float) -> None:
            delays.append(delay)

        with patch(SLEEP_PATH, side_effect=capture_delay):
            await retry_with_backoff(fn, policy, classify_error)

        # Attempt 0: min(10*2^0, 15) = 10.0
        # Attempt 1: min(10*2^1, 15) = 15.0 (capped)
        # Attempt 2: min(10*2^2, 15) = 15.0 (capped)
        assert delays == [10.0, 15.0, 15.0]

    async def test_emitter_receives_retry_events(self) -> None:
        error = LLMProviderError("server error", status_code=500)
        fn = AsyncMock(side_effect=[error, "ok"])
        policy = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False)
        emitter = InMemoryEmitter(trace_id="test-trace")

        with patch(SLEEP_PATH, new_callable=AsyncMock):
            await retry_with_backoff(fn, policy, classify_error, emitter=emitter)

        retry_events = [e for e in emitter.events if isinstance(e, ErrorRetryEvent)]
        assert len(retry_events) == 1

        event = retry_events[0]
        assert event.error_type == "LLMProviderError"
        assert event.error_message == "server error"
        assert event.attempt == 1
        assert event.max_attempts == 3
        assert event.category == "retryable"
        assert event.delay_ms == 1000.0
        assert event.trace_id == "test-trace"

    async def test_emitter_receives_multiple_retry_events(
        self,
    ) -> None:
        error = LLMProviderError("server error", status_code=500)
        fn = AsyncMock(side_effect=[error, error, "ok"])
        policy = RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False)
        emitter = InMemoryEmitter(trace_id="test-trace")

        with patch(SLEEP_PATH, new_callable=AsyncMock):
            await retry_with_backoff(fn, policy, classify_error, emitter=emitter)

        retry_events = [e for e in emitter.events if isinstance(e, ErrorRetryEvent)]
        assert len(retry_events) == 2
        assert retry_events[0].attempt == 1
        assert retry_events[1].attempt == 2

    async def test_no_emitter_still_works(self) -> None:
        fn = AsyncMock(
            side_effect=[
                LLMProviderError("fail", status_code=500),
                "ok",
            ]
        )
        policy = RetryPolicy(max_attempts=2, jitter=False)

        with patch(SLEEP_PATH, new_callable=AsyncMock):
            result = await retry_with_backoff(fn, policy, classify_error, emitter=None)

        assert result == "ok"

    async def test_single_attempt_policy_raises_immediately(
        self,
    ) -> None:
        error = LLMProviderError("server error", status_code=500)
        fn = AsyncMock(side_effect=error)
        policy = RetryPolicy(max_attempts=1)

        with pytest.raises(LLMProviderError, match="server error"):
            await retry_with_backoff(fn, policy, classify_error)

        assert fn.call_count == 1
