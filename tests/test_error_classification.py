from nanitics.capabilities.errors.classification import ErrorCategory, classify_error
from nanitics.infrastructure.errors import (
    AgentBudgetExceededError,
    AgentEscalationError,
    AgentIterationLimitError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaViolationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
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
