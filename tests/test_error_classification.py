from nanitics.capabilities.errors.classification import ErrorCategory, classify_error
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
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
    ToolTimeoutError,
)


class TestClassifyError:
    def test_llm_rate_limit_error_is_retryable(self) -> None:
        error = LLMRateLimitError("rate limited", retry_after=5.0)
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_llm_provider_error_5xx_is_retryable(self) -> None:
        error = LLMProviderError("server error", status_code=500)
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_llm_provider_error_503_is_retryable(self) -> None:
        error = LLMProviderError("service unavailable", status_code=503)
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_llm_provider_error_401_is_fatal(self) -> None:
        error = LLMProviderError("unauthorized", status_code=401)
        assert classify_error(error) == ErrorCategory.FATAL

    def test_llm_provider_error_400_is_fatal(self) -> None:
        error = LLMProviderError("bad request", status_code=400)
        assert classify_error(error) == ErrorCategory.FATAL

    def test_llm_provider_error_no_status_code_is_retryable(self) -> None:
        error = LLMProviderError("connection error", status_code=None)
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_llm_context_length_error_is_fatal(self) -> None:
        error = LLMContextLengthError("too long", token_count=200000, token_limit=100000)
        assert classify_error(error) == ErrorCategory.FATAL

    def test_llm_schema_violation_error_is_correctable(self) -> None:
        error = LLMSchemaViolationError("invalid format")
        assert classify_error(error) == ErrorCategory.CORRECTABLE

    def test_tool_parameter_error_is_correctable(self) -> None:
        error = ToolParameterError(
            "bad param",
            tool_name="calc",
            parameter_name="x",
            reason="must be int",
        )
        assert classify_error(error) == ErrorCategory.CORRECTABLE

    def test_tool_execution_error_is_correctable(self) -> None:
        error = ToolExecutionError("failed", tool_name="calc")
        assert classify_error(error) == ErrorCategory.CORRECTABLE

    def test_tool_not_found_error_is_correctable(self) -> None:
        error = ToolNotFoundError("not found", tool_name="nonexistent")
        assert classify_error(error) == ErrorCategory.CORRECTABLE

    def test_tool_timeout_error_is_retryable(self) -> None:
        error = ToolTimeoutError(
            "timed out",
            tool_name="slow_tool",
            timeout_seconds=5.0,
        )
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_app_defined_tool_error_subclass_is_correctable(self) -> None:
        class _AppDefinedToolError(ToolError):
            pass

        error = _AppDefinedToolError("app-specific failure")
        assert classify_error(error) == ErrorCategory.CORRECTABLE

    def test_agent_iteration_limit_error_is_fatal(self) -> None:
        error = AgentIterationLimitError("limit reached", iteration_count=10, iteration_limit=10)
        assert classify_error(error) == ErrorCategory.FATAL

    def test_agent_budget_exceeded_error_is_fatal(self) -> None:
        error = AgentBudgetExceededError(
            "over budget",
            budget_type="tokens",
            budget_limit=1000,
            budget_used=1500,
        )
        assert classify_error(error) == ErrorCategory.FATAL

    def test_agent_escalation_error_is_fatal(self) -> None:
        error = AgentEscalationError("need human help", reason="complex decision")
        assert classify_error(error) == ErrorCategory.FATAL

    def test_unknown_exception_is_fatal(self) -> None:
        error = ValueError("something unexpected")
        assert classify_error(error) == ErrorCategory.FATAL

    def test_unknown_runtime_error_is_fatal(self) -> None:
        error = RuntimeError("unexpected")
        assert classify_error(error) == ErrorCategory.FATAL

    def test_embedding_rate_limit_error_is_retryable(self) -> None:
        error = EmbeddingRateLimitError("rate limited", retry_after=10.0)
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_embedding_provider_error_5xx_is_retryable(self) -> None:
        error = EmbeddingProviderError("server error", status_code=500, provider="voyage")
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_embedding_provider_error_no_status_is_retryable(self) -> None:
        error = EmbeddingProviderError("connection error", status_code=None, provider="voyage")
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_embedding_provider_error_401_is_fatal(self) -> None:
        error = EmbeddingProviderError("unauthorized", status_code=401, provider="voyage")
        assert classify_error(error) == ErrorCategory.FATAL

    def test_llm_authentication_error_is_fatal(self) -> None:
        error = LLMAuthenticationError("Invalid API key", status_code=401, provider="anthropic")
        assert classify_error(error) == ErrorCategory.FATAL

    def test_llm_quota_exhausted_error_is_fatal(self) -> None:
        # Deliberate change from pre-Phase behaviour: an Anthropic
        # 429-with-insufficient_quota now routes to FATAL via the typed
        # subclass instead of RETRYABLE via LLMRateLimitError.
        error = LLMQuotaExhaustedError(
            "Quota exhausted",
            status_code=429,
            provider="anthropic",
            provider_error_type="insufficient_quota",
        )
        assert classify_error(error) == ErrorCategory.FATAL

    def test_llm_overloaded_error_is_retryable(self) -> None:
        error = LLMOverloadedError(
            "Overloaded",
            status_code=529,
            provider="anthropic",
            provider_error_type="overloaded_error",
        )
        assert classify_error(error) == ErrorCategory.RETRYABLE

    def test_raw_llm_provider_error_429_still_falls_through_to_status_branch(self) -> None:
        # Regression guard: the new subclass branches must consume *only*
        # their own subclasses. A raw LLMProviderError(status_code=429),
        # with no subclass identity, must still classify via the existing
        # status-code branch (FATAL for 4xx). This test will fail if a
        # future change accidentally reorders the new branches above the
        # parent check in a way that captures the parent class itself.
        error = LLMProviderError("rate limited", status_code=429, provider="anthropic")
        assert classify_error(error) == ErrorCategory.FATAL

    def test_raw_llm_provider_error_500_still_falls_through_to_status_branch(self) -> None:
        # Regression guard for the 5xx branch on the raw parent class.
        error = LLMProviderError("server error", status_code=500, provider="anthropic")
        assert classify_error(error) == ErrorCategory.RETRYABLE
