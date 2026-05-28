from enum import Enum
from typing import Protocol

from nanitics.infrastructure.errors import (
    AgentBudgetExceededError,
    AgentEscalationError,
    AgentIterationLimitError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMOverloadedError,
    LLMProviderError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMSchemaViolationError,
    ToolError,
    ToolTimeoutError,
)


class ErrorCategory(Enum):
    """Recovery category for a classified error.

    Determines which recovery strategy the ErrorHandler applies:
    RETRYABLE errors are retried with backoff, CORRECTABLE errors
    trigger self-correction prompts, and FATAL errors propagate or
    trigger degradation.
    """

    RETRYABLE = "retryable"
    CORRECTABLE = "correctable"
    FATAL = "fatal"


class ErrorClassifier(Protocol):
    """Protocol for classifying errors into recovery categories.

    Implement this to override the default classification logic.
    The classifier receives an exception and returns an ErrorCategory
    that determines how the ErrorHandler recovers from the error.
    """

    def __call__(self, error: Exception) -> ErrorCategory: ...


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an error into a recovery category using the default rule set.

    Maps the SDK error hierarchy to ErrorCategory values:
    - RETRYABLE: rate limits, server errors (5xx), tool timeouts,
      provider overload (:class:`LLMOverloadedError`)
    - CORRECTABLE: bad tool parameters, wrong tool name, schema violations,
      and any other ``ToolError`` subclass
    - FATAL: context length exceeded, budget exhausted, client errors (4xx),
      provider auth failures (:class:`LLMAuthenticationError`), and
      provider quota exhaustion (:class:`LLMQuotaExhaustedError`)

    ``ToolError`` itself defaults to CORRECTABLE so app-defined typed
    subclasses route through the correction loop without per-class
    registration. ``ToolTimeoutError`` is the documented exception and
    is classified as RETRYABLE.

    Unknown error types default to FATAL.

    Args:
        error: The exception to classify.

    Returns:
        The ErrorCategory determining recovery strategy.
        :class:`LLMQuotaExhaustedError` is intentionally classified as
        FATAL (not RETRYABLE) — quota exhaustion is a billing-state
        condition that retry cannot resolve within the budget window.
    """
    if isinstance(error, LLMRateLimitError):
        return ErrorCategory.RETRYABLE

    # Typed subclasses of LLMProviderError are checked *before* the parent
    # branch — Python's isinstance matches the first true branch and the
    # parent's status-code fallback would otherwise reclassify them.
    if isinstance(error, LLMAuthenticationError):
        return ErrorCategory.FATAL
    if isinstance(error, LLMQuotaExhaustedError):
        return ErrorCategory.FATAL
    if isinstance(error, LLMOverloadedError):
        return ErrorCategory.RETRYABLE

    if isinstance(error, LLMProviderError):
        if error.status_code is None:
            return ErrorCategory.RETRYABLE
        if error.status_code >= 500:
            return ErrorCategory.RETRYABLE
        return ErrorCategory.FATAL

    if isinstance(error, LLMContextLengthError):
        return ErrorCategory.FATAL

    if isinstance(error, LLMSchemaViolationError):
        return ErrorCategory.CORRECTABLE

    if isinstance(error, EmbeddingRateLimitError):
        return ErrorCategory.RETRYABLE

    if isinstance(error, EmbeddingProviderError):
        if error.status_code is None:
            return ErrorCategory.RETRYABLE
        if error.status_code >= 500:
            return ErrorCategory.RETRYABLE
        return ErrorCategory.FATAL

    if isinstance(error, ToolTimeoutError):
        return ErrorCategory.RETRYABLE

    if isinstance(error, ToolError):
        return ErrorCategory.CORRECTABLE

    if isinstance(error, (AgentIterationLimitError, AgentBudgetExceededError, AgentEscalationError)):
        return ErrorCategory.FATAL

    return ErrorCategory.FATAL
