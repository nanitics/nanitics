from datetime import UTC, datetime
from typing import Any, get_type_hints


class NaniticsError(Exception):
    """Base exception for all SDK errors.

    Carries a human-readable message, a UTC timestamp, and optional
    trace/span IDs for correlation with observability events.
    Supports ``to_dict()`` for serialization of all annotated fields.

    Attributes:
        message: Human-readable error description.
        timestamp: When the error occurred (UTC).
        trace_id: Trace ID for correlation, if available.
        span_id: Span ID for correlation, if available.
    """

    message: str
    timestamp: datetime
    trace_id: str | None
    span_id: str | None

    def __init__(
        self,
        message: str,
        *,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.timestamp = datetime.now(UTC)
        self.trace_id = trace_id
        self.span_id = span_id

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for cls in reversed(type(self).__mro__):
            hints = get_type_hints(cls) if hasattr(cls, "__annotations__") else {}
            for field_name in hints:
                if field_name.startswith("_"):
                    continue
                value = getattr(self, field_name, None)
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[field_name] = value
        return result


# --- LLM Errors ---


class LLMError(NaniticsError):
    """Base exception for LLM-related errors."""


class LLMRateLimitError(LLMError):
    """The LLM provider returned a rate limit (429) response.

    Classified as RETRYABLE. If ``retry_after`` is set, the backoff
    logic respects the provider's requested delay.

    Attributes:
        retry_after: Provider-suggested wait time in seconds, if available.
    """

    retry_after: float | None

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.retry_after = retry_after


class LLMContextLengthError(LLMError):
    """The input exceeds the model's context window.

    Classified as FATAL. Use ContextManager to prevent this.

    Attributes:
        token_count: Number of tokens in the input, if known.
        token_limit: The model's context window limit, if known.
    """

    token_count: int | None
    token_limit: int | None

    def __init__(
        self,
        message: str,
        *,
        token_count: int | None = None,
        token_limit: int | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.token_count = token_count
        self.token_limit = token_limit


class LLMProviderError(LLMError):
    """A provider-specific error from the LLM service.

    Classified as RETRYABLE for 5xx status codes or unknown status,
    FATAL for 4xx client errors.

    Attributes:
        status_code: HTTP status code, if available.
        provider: Provider name (e.g. "anthropic"), if available.
    """

    status_code: int | None
    provider: str | None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.status_code = status_code
        self.provider = provider


class LLMSchemaViolationError(LLMError):
    """The LLM output did not match the expected structured schema.

    Classified as CORRECTABLE. The agent receives a correction prompt
    asking it to match the required format.

    Attributes:
        expected_schema: Description of the expected schema, if available.
        received: The actual output that didn't match, if available.
    """

    expected_schema: str | None
    received: str | None

    def __init__(
        self,
        message: str,
        *,
        expected_schema: str | None = None,
        received: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.expected_schema = expected_schema
        self.received = received


# --- Embedding Errors ---


class EmbeddingError(NaniticsError):
    """Base exception for embedding-related errors."""


class EmbeddingRateLimitError(EmbeddingError):
    """The embedding provider returned a rate limit response.

    Classified as RETRYABLE.

    Attributes:
        retry_after: Provider-suggested wait time in seconds, if available.
    """

    retry_after: float | None

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.retry_after = retry_after


class EmbeddingProviderError(EmbeddingError):
    """A provider-specific error from the embedding service.

    Classified as RETRYABLE for 5xx or unknown status, FATAL for 4xx.

    Attributes:
        status_code: HTTP status code, if available.
        provider: Provider name, if available.
    """

    status_code: int | None
    provider: str | None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.status_code = status_code
        self.provider = provider


# --- Tool Errors ---


class ToolError(NaniticsError):
    """Base exception for tool-related errors."""


class ToolNotFoundError(ToolError):
    """The agent requested a tool that doesn't exist in the registry.

    Classified as CORRECTABLE. The correction prompt lists available tools.

    Attributes:
        tool_name: The name of the tool that was not found.
    """

    tool_name: str

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.tool_name = tool_name


class ToolParameterError(ToolError):
    """The tool received invalid parameters.

    Classified as CORRECTABLE. The correction prompt shows the parameter
    name and reason for rejection.

    Attributes:
        tool_name: The tool that rejected the parameters.
        parameter_name: The specific parameter that failed validation, if known.
        reason: Human-readable explanation of why parameters were invalid.
    """

    tool_name: str
    parameter_name: str | None
    reason: str | None

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        parameter_name: str | None = None,
        reason: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.tool_name = tool_name
        self.parameter_name = parameter_name
        self.reason = reason


class ToolExecutionError(ToolError):
    """The tool raised an exception during execution.

    Classified as CORRECTABLE. Wraps the original exception as ``__cause__``
    and includes it in ``to_dict()`` serialization.

    Attributes:
        tool_name: The tool that failed during execution.
    """

    tool_name: str

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.tool_name = tool_name

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.__cause__ is not None:
            result["original_error_type"] = type(self.__cause__).__name__
            result["original_error_message"] = str(self.__cause__)
        else:
            result["original_error_type"] = None
            result["original_error_message"] = None
        return result


class ToolTimeoutError(ToolError):
    """A tool execution exceeded its timeout.

    Classified as RETRYABLE.

    Attributes:
        tool_name: The tool that timed out.
        timeout_seconds: The timeout that was exceeded.
    """

    tool_name: str
    timeout_seconds: float

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        timeout_seconds: float,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds


# --- Agent Errors ---


class AgentError(NaniticsError):
    """Base exception for agent-level errors."""


class AgentIterationLimitError(AgentError):
    """The agent exceeded its maximum iteration count.

    Classified as FATAL. This is a safety boundary — do not catch
    inside tools.

    Attributes:
        iteration_count: Number of iterations completed.
        iteration_limit: The configured maximum.
    """

    iteration_count: int
    iteration_limit: int

    def __init__(
        self,
        message: str,
        *,
        iteration_count: int,
        iteration_limit: int,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.iteration_count = iteration_count
        self.iteration_limit = iteration_limit


class AgentToolCallLimitError(AgentError):
    """The agent exceeded its maximum tool call count.

    Classified as FATAL. This is a safety boundary — do not catch
    inside tools.

    Attributes:
        tool_call_count: Total tool calls when the limit was exceeded.
        tool_call_limit: The configured maximum.
    """

    tool_call_count: int
    tool_call_limit: int

    def __init__(
        self,
        message: str,
        *,
        tool_call_count: int,
        tool_call_limit: int,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.tool_call_count = tool_call_count
        self.tool_call_limit = tool_call_limit


class AgentBudgetExceededError(AgentError):
    """The agent exceeded its token or cost budget.

    Classified as FATAL. This is a safety boundary — do not catch
    inside tools.

    Attributes:
        budget_type: The type of budget exceeded (e.g. "tokens", "cost").
        budget_limit: The configured budget limit.
        budget_used: The amount consumed when the limit was hit.
    """

    budget_type: str
    budget_limit: float
    budget_used: float

    def __init__(
        self,
        message: str,
        *,
        budget_type: str,
        budget_limit: float,
        budget_used: float,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.budget_type = budget_type
        self.budget_limit = budget_limit
        self.budget_used = budget_used


class AgentEscalationError(AgentError):
    """The agent requested human escalation.

    Raised when an agent determines it cannot complete a task
    and needs human intervention.

    Attributes:
        reason: Why the agent is escalating, if provided.
    """

    reason: str | None

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        super().__init__(message, trace_id=trace_id, span_id=span_id)
        self.reason = reason


# --- Human-Input Provider Errors ---


class HumanInputProviderError(NaniticsError):
    """Base exception for HumanInputProvider failures.

    Raised when a provider cannot obtain a decision for reasons that are
    distinct from the agent's own errors and from a human's explicit
    decision (the approval machinery never ran, or never completed).
    """


class ApprovalUnavailableError(HumanInputProviderError):
    """Raised when the HITL store backend is unreachable or errors.

    The wrapped tool never executed. Callers may retry (transient
    backend failure) or surface the failure as a terminal condition.
    """


class ApprovalTimeoutError(HumanInputProviderError):
    """Raised when a HITL response does not arrive within the window.

    The pending request is removed from the provider before the raise,
    so a subsequent ``resolve`` call for the same ``request_id`` returns
    ``False``. The wrapped tool never executed.
    """
