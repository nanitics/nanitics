from datetime import datetime

import pytest

from nanitics.infrastructure.errors import (
    AgentBudgetExceededError,
    AgentError,
    AgentEscalationError,
    AgentIterationLimitError,
    EmbeddingProviderError,
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMOverloadedError,
    LLMProviderError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMSchemaViolationError,
    NaniticsError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
)


class TestNaniticsErrorBase:
    def test_message_and_timestamp(self):
        err = NaniticsError("something broke")
        assert err.message == "something broke"
        assert str(err) == "something broke"
        assert isinstance(err.timestamp, datetime)
        assert err.timestamp.tzinfo is not None
        assert err.trace_id is None
        assert err.span_id is None

    def test_trace_context(self):
        err = NaniticsError("msg", trace_id="t1", span_id="s1")
        assert err.trace_id == "t1"
        assert err.span_id == "s1"

    def test_to_dict_base(self):
        err = NaniticsError("msg")
        d = err.to_dict()
        assert d["message"] == "msg"
        assert "timestamp" in d
        assert d["trace_id"] is None
        assert d["span_id"] is None

    def test_is_exception(self):
        with pytest.raises(NaniticsError):
            raise NaniticsError("fail")


class TestLLMErrors:
    def test_rate_limit_error(self):
        err = LLMRateLimitError("Rate limited", retry_after=30.0)
        assert err.retry_after == 30.0
        d = err.to_dict()
        assert d["retry_after"] == 30.0

    def test_rate_limit_error_optional(self):
        err = LLMRateLimitError("Rate limited")
        assert err.retry_after is None
        assert err.to_dict()["retry_after"] is None

    def test_context_length_error(self):
        err = LLMContextLengthError("Too long", token_count=5000, token_limit=4096)
        assert err.token_count == 5000
        assert err.token_limit == 4096
        d = err.to_dict()
        assert d["token_count"] == 5000
        assert d["token_limit"] == 4096

    def test_provider_error(self):
        err = LLMProviderError("Server error", status_code=500, provider="anthropic")
        assert err.status_code == 500
        assert err.provider == "anthropic"

    def test_provider_error_with_provider_error_type(self):
        err = LLMProviderError(
            "Server error",
            status_code=500,
            provider="openai",
            provider_error_type="server_error",
        )
        assert err.provider_error_type == "server_error"
        d = err.to_dict()
        assert d["provider_error_type"] == "server_error"

    def test_provider_error_provider_error_type_defaults_to_none(self):
        err = LLMProviderError("err", status_code=500, provider="openai")
        assert err.provider_error_type is None
        assert err.to_dict()["provider_error_type"] is None

    def test_schema_violation_error(self):
        err = LLMSchemaViolationError("Bad schema", expected_schema='{"type": "object"}', received='{"bad": 1}')
        d = err.to_dict()
        assert d["expected_schema"] == '{"type": "object"}'
        assert d["received"] == '{"bad": 1}'

    def test_category_catch_pattern(self):
        with pytest.raises(LLMError):
            raise LLMRateLimitError("limit")

        with pytest.raises(LLMError):
            raise LLMContextLengthError("ctx")

        with pytest.raises(LLMError):
            raise LLMProviderError("prov")

        with pytest.raises(LLMError):
            raise LLMSchemaViolationError("schema")

    def test_nanitics_error_catch_pattern(self):
        with pytest.raises(NaniticsError):
            raise LLMRateLimitError("limit")


class TestTypedLLMProviderSubclasses:
    """Cover the three typed subclasses of LLMProviderError."""

    def test_authentication_error_construction(self):
        err = LLMAuthenticationError(
            "Invalid API key",
            status_code=401,
            provider="anthropic",
            provider_error_type="authentication_error",
        )
        assert err.message == "Invalid API key"
        assert err.status_code == 401
        assert err.provider == "anthropic"
        assert err.provider_error_type == "authentication_error"

    def test_authentication_error_defaults(self):
        err = LLMAuthenticationError("Invalid API key")
        assert err.status_code is None
        assert err.provider is None
        assert err.provider_error_type is None

    def test_quota_exhausted_error_construction(self):
        err = LLMQuotaExhaustedError(
            "Quota exhausted",
            status_code=429,
            provider="anthropic",
            provider_error_type="insufficient_quota",
        )
        assert err.message == "Quota exhausted"
        assert err.status_code == 429
        assert err.provider == "anthropic"
        assert err.provider_error_type == "insufficient_quota"

    def test_quota_exhausted_error_defaults(self):
        err = LLMQuotaExhaustedError("Quota exhausted")
        assert err.status_code is None
        assert err.provider is None
        assert err.provider_error_type is None

    def test_overloaded_error_construction(self):
        err = LLMOverloadedError(
            "Service overloaded",
            status_code=529,
            provider="anthropic",
            provider_error_type="overloaded_error",
        )
        assert err.message == "Service overloaded"
        assert err.status_code == 529
        assert err.provider == "anthropic"
        assert err.provider_error_type == "overloaded_error"

    def test_overloaded_error_defaults(self):
        err = LLMOverloadedError("Service overloaded")
        assert err.status_code is None
        assert err.provider is None
        assert err.provider_error_type is None

    def test_authentication_error_is_provider_error(self):
        err = LLMAuthenticationError("x", status_code=401, provider="anthropic")
        assert isinstance(err, LLMProviderError)
        assert isinstance(err, LLMError)
        assert isinstance(err, NaniticsError)

    def test_quota_exhausted_error_is_provider_error(self):
        err = LLMQuotaExhaustedError("x", status_code=429, provider="anthropic")
        assert isinstance(err, LLMProviderError)
        assert isinstance(err, LLMError)
        assert isinstance(err, NaniticsError)

    def test_overloaded_error_is_provider_error(self):
        err = LLMOverloadedError("x", status_code=529, provider="anthropic")
        assert isinstance(err, LLMProviderError)
        assert isinstance(err, LLMError)
        assert isinstance(err, NaniticsError)

    def test_authentication_to_dict(self):
        err = LLMAuthenticationError(
            "Invalid",
            status_code=401,
            provider="openai",
            provider_error_type="invalid_api_key",
        )
        d = err.to_dict()
        assert d["message"] == "Invalid"
        assert d["status_code"] == 401
        assert d["provider"] == "openai"
        assert d["provider_error_type"] == "invalid_api_key"

    def test_quota_exhausted_to_dict(self):
        err = LLMQuotaExhaustedError(
            "Quota",
            status_code=429,
            provider="openai",
            provider_error_type="insufficient_quota",
        )
        d = err.to_dict()
        assert d["status_code"] == 429
        assert d["provider"] == "openai"
        assert d["provider_error_type"] == "insufficient_quota"

    def test_overloaded_to_dict(self):
        err = LLMOverloadedError(
            "Overloaded",
            status_code=529,
            provider="anthropic",
            provider_error_type="overloaded_error",
        )
        d = err.to_dict()
        assert d["status_code"] == 529
        assert d["provider"] == "anthropic"
        assert d["provider_error_type"] == "overloaded_error"

    def test_parent_catch_pattern(self):
        # Catching the parent class still catches the new subclasses.
        with pytest.raises(LLMProviderError):
            raise LLMAuthenticationError("x", status_code=401, provider="anthropic")
        with pytest.raises(LLMProviderError):
            raise LLMQuotaExhaustedError("x", status_code=429, provider="anthropic")
        with pytest.raises(LLMProviderError):
            raise LLMOverloadedError("x", status_code=529, provider="anthropic")


class TestEmbeddingProviderErrorField:
    def test_provider_error_type_populated(self):
        err = EmbeddingProviderError(
            "err",
            status_code=500,
            provider="voyage",
            provider_error_type="server_error",
        )
        assert err.provider_error_type == "server_error"
        assert err.to_dict()["provider_error_type"] == "server_error"

    def test_provider_error_type_defaults_to_none(self):
        err = EmbeddingProviderError("err", status_code=500, provider="voyage")
        assert err.provider_error_type is None
        assert err.to_dict()["provider_error_type"] is None


class TestToolErrors:
    def test_not_found_error(self):
        err = ToolNotFoundError("Tool 'search' not found", tool_name="search")
        assert err.tool_name == "search"
        assert err.to_dict()["tool_name"] == "search"

    def test_not_found_requires_tool_name(self):
        with pytest.raises(TypeError):
            ToolNotFoundError("missing arg")  # type: ignore[call-arg]

    def test_parameter_error(self):
        err = ToolParameterError("Bad param", tool_name="calc", parameter_name="x", reason="must be int")
        d = err.to_dict()
        assert d["tool_name"] == "calc"
        assert d["parameter_name"] == "x"
        assert d["reason"] == "must be int"

    def test_parameter_error_optional_fields(self):
        err = ToolParameterError("Bad param", tool_name="calc")
        assert err.parameter_name is None
        assert err.reason is None

    def test_execution_error_with_cause(self):
        original = ValueError("division by zero")
        try:
            raise ToolExecutionError("calc failed", tool_name="calc") from original
        except ToolExecutionError as e:
            d = e.to_dict()
            assert d["original_error_type"] == "ValueError"
            assert d["original_error_message"] == "division by zero"
            assert d["tool_name"] == "calc"

    def test_execution_error_without_cause(self):
        err = ToolExecutionError("calc failed", tool_name="calc")
        d = err.to_dict()
        assert d["original_error_type"] is None
        assert d["original_error_message"] is None

    def test_category_catch_pattern(self):
        with pytest.raises(ToolError):
            raise ToolNotFoundError("x", tool_name="x")

        with pytest.raises(ToolError):
            raise ToolParameterError("x", tool_name="x")

        with pytest.raises(ToolError):
            raise ToolExecutionError("x", tool_name="x")


class TestAgentErrors:
    def test_iteration_limit_error(self):
        err = AgentIterationLimitError("Max iterations", iteration_count=25, iteration_limit=25)
        d = err.to_dict()
        assert d["iteration_count"] == 25
        assert d["iteration_limit"] == 25

    def test_iteration_limit_requires_fields(self):
        with pytest.raises(TypeError):
            AgentIterationLimitError("missing args")  # type: ignore[call-arg]

    def test_budget_exceeded_error(self):
        err = AgentBudgetExceededError("Over budget", budget_type="tokens", budget_limit=1000.0, budget_used=1200.0)
        d = err.to_dict()
        assert d["budget_type"] == "tokens"
        assert d["budget_limit"] == 1000.0
        assert d["budget_used"] == 1200.0

    def test_escalation_error(self):
        err = AgentEscalationError("Need help", reason="ambiguous input")
        assert err.reason == "ambiguous input"
        d = err.to_dict()
        assert d["reason"] == "ambiguous input"

    def test_escalation_error_optional(self):
        err = AgentEscalationError("Need help")
        assert err.reason is None

    def test_category_catch_pattern(self):
        with pytest.raises(AgentError):
            raise AgentIterationLimitError("x", iteration_count=1, iteration_limit=1)

        with pytest.raises(AgentError):
            raise AgentBudgetExceededError("x", budget_type="t", budget_limit=1.0, budget_used=2.0)

        with pytest.raises(AgentError):
            raise AgentEscalationError("x")


class TestToDictNoneHandling:
    def test_all_optional_fields_none(self):
        err = LLMContextLengthError("msg")
        d = err.to_dict()
        assert d["token_count"] is None
        assert d["token_limit"] is None
        assert d["trace_id"] is None
        assert d["span_id"] is None

    def test_timestamp_is_iso_string(self):
        err = NaniticsError("msg")
        d = err.to_dict()
        # Should be parseable as ISO format
        datetime.fromisoformat(d["timestamp"])

    def test_to_dict_skips_private_fields(self):
        class _WithPrivateField(NaniticsError):
            _internal: str

            def __init__(self, message: str) -> None:
                super().__init__(message)
                self._internal = "secret"

        err = _WithPrivateField("msg")
        d = err.to_dict()
        assert "_internal" not in d
