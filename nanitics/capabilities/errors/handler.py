from __future__ import annotations

from collections.abc import Awaitable, Callable

from nanitics.capabilities.errors.classification import (
    ErrorCategory,
    ErrorClassifier,
    classify_error,
)
from nanitics.capabilities.errors.correction import format_correction_prompt
from nanitics.capabilities.errors.retry import RetryPolicy, retry_with_backoff
from nanitics.infrastructure.observability.emitter import EventEmitter

_DEFAULT_RETRY_POLICY = RetryPolicy()


class ErrorHandler:
    """Default error handling strategy for all agents.

    Ties together classification, retry, correction prompts, and degradation
    into a single strategy. Every agent uses ``ErrorHandler()`` by default,
    providing resilient error handling out of the box: retry with backoff for
    transient failures, self-correction prompts for agent mistakes, graceful
    degradation when correction budgets are exhausted, and propagation for
    fatal errors.

    Use ``ErrorHandler.fail_fast()`` during development or testing when you
    want errors to surface immediately.
    """

    def __init__(
        self,
        *,
        classify: ErrorClassifier = classify_error,
        retry_policy: RetryPolicy = _DEFAULT_RETRY_POLICY,
        max_corrections: int = 3,
        max_total_corrections: int = 5,
    ) -> None:
        """Initialize the error handler.

        Args:
            classify: Error classifier function. Defaults to ``classify_error``.
            retry_policy: Retry configuration for retryable errors.
            max_corrections: Maximum correction attempts per individual tool error.
            max_total_corrections: Maximum total corrections across all tools in a run.
        """
        self._classify = classify
        self._retry_policy = retry_policy
        self._max_corrections = max_corrections
        self._max_total_corrections = max_total_corrections
        self._total_corrections = 0

    @classmethod
    def default(cls) -> ErrorHandler:
        """Create an error handler with standard settings.

        Equivalent to ``ErrorHandler()`` — the same configuration agents use
        by default. Uses default retry policy (3 attempts, exponential backoff),
        3 corrections per tool, and 5 total corrections per run.
        """
        return cls()

    @classmethod
    def fail_fast(cls) -> ErrorHandler:
        """Create an error handler that propagates all errors immediately.

        No retries, no corrections, no degradation. Use this in tests where
        you want errors to surface immediately, or during development when
        debugging specific tool interactions.
        """
        return cls(
            retry_policy=RetryPolicy(max_attempts=1),
            max_corrections=0,
            max_total_corrections=0,
        )

    async def handle_llm_error(
        self,
        error: Exception,
        retry_fn: Callable[[], Awaitable[object]],
        emitter: EventEmitter | None = None,
    ) -> object:
        """Handle an LLM error by retrying if retryable, raising otherwise.

        Args:
            error: The LLM exception to handle.
            retry_fn: Async callable to retry the LLM call.
            emitter: Event emitter for retry event tracking.

        Returns:
            The result of a successful retry.

        Raises:
            Exception: The original error if not retryable or retries exhausted.
        """
        category = self._classify(error)
        if category == ErrorCategory.RETRYABLE:
            if self._retry_policy.max_attempts <= 1:
                raise error
            return await retry_with_backoff(retry_fn, self._retry_policy, self._classify, emitter)
        raise error

    def handle_tool_error(
        self,
        error: Exception,
        attempt: int,
        available_tools: list[str],
    ) -> str | None:
        """Handle a tool error by generating a correction prompt or signaling degradation.

        Args:
            error: The tool exception to handle.
            attempt: Current attempt number for this tool (0-based).
            available_tools: Names of available tools (used in ToolNotFoundError feedback).

        Returns:
            A correction prompt string if the error is correctable and within budget,
            or None if the error is fatal or correction budget is exhausted.
        """
        category = self._classify(error)

        if category == ErrorCategory.FATAL:
            return None

        if category in (
            ErrorCategory.CORRECTABLE,
            ErrorCategory.RETRYABLE,
        ):
            if attempt < self._max_corrections and self._total_corrections < self._max_total_corrections:
                self._total_corrections += 1
                return format_correction_prompt(
                    error,
                    attempt + 1,
                    self._max_corrections,
                    available_tools=available_tools,
                )
            return None

        # Unreachable: FATAL, CORRECTABLE, and RETRYABLE exhaust ErrorCategory.
        return None  # pragma: no cover

    def handle_llm_correction(
        self,
        error: Exception,
        attempt: int,
    ) -> str | None:
        """Handle a correctable LLM error by generating a correction prompt.

        Uses the same shared correction budget as tool corrections.

        Args:
            error: The LLM exception to handle.
            attempt: Current attempt number (0-based).

        Returns:
            A correction prompt string if the error is correctable and within budget,
            or None if the error is fatal or correction budget is exhausted.
        """
        category = self._classify(error)

        if category != ErrorCategory.CORRECTABLE:
            return None

        if attempt < self._max_corrections and self._total_corrections < self._max_total_corrections:
            self._total_corrections += 1
            return format_correction_prompt(
                error,
                attempt + 1,
                self._max_corrections,
            )
        return None

    def should_degrade(self, error: Exception, attempt: int) -> bool:
        """Check whether correction budget is exhausted and degradation is appropriate.

        Only degrades when corrections were actually attempted (max_corrections > 0).
        If max_corrections is 0 (fail-fast), errors propagate instead of degrading.
        """
        if self._max_corrections == 0:
            return False
        return attempt >= self._max_corrections or self._total_corrections >= self._max_total_corrections

    def format_degradation_message(self, error: Exception) -> str:
        """Produce a terminal feedback message for the LLM after exhausting corrections.

        Instructs the LLM to complete the task with available information
        and state what it could not accomplish.

        Args:
            error: The error that triggered degradation.
        """
        tool_name = getattr(error, "tool_name", "unknown")
        return (
            f"Tool '{tool_name}' has failed repeatedly. "
            f"Complete the task using only the information gathered so far, "
            f"and clearly state what you could not accomplish."
        )

    def reset(self) -> None:
        """Reset per-run state. Called at the start of each _execute()."""
        self._total_corrections = 0

    @property
    def max_corrections(self) -> int:
        return self._max_corrections

    @property
    def total_corrections(self) -> int:
        return self._total_corrections

    def restore(self, total_corrections: int) -> None:
        self._total_corrections = total_corrections
