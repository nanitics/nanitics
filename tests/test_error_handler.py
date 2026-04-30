import pytest

from nanitics.capabilities.errors.handler import ErrorHandler
from nanitics.infrastructure.errors import (
    LLMProviderError,
    LLMRateLimitError,
    LLMSchemaViolationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
)


class TestHandleLLMError:
    async def test_retries_retryable_errors(self) -> None:
        handler = ErrorHandler.default()
        call_count = 0

        async def retry_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise LLMRateLimitError("rate limited", retry_after=0.0)
            return "success"

        error = LLMRateLimitError("rate limited", retry_after=0.0)
        result = await handler.handle_llm_error(error, retry_fn)
        assert result == "success"
        assert call_count == 2

    async def test_raises_fatal_errors_immediately(self) -> None:
        handler = ErrorHandler.default()
        call_count = 0

        async def retry_fn():
            nonlocal call_count
            call_count += 1
            return "should not reach"

        error = LLMProviderError("auth failed", status_code=401)
        with pytest.raises(LLMProviderError):
            await handler.handle_llm_error(error, retry_fn)
        assert call_count == 0


class TestHandleToolError:
    def test_returns_correction_for_correctable_errors(self) -> None:
        handler = ErrorHandler.default()
        error = ToolParameterError("bad param", tool_name="search", parameter_name="q", reason="required")
        result = handler.handle_tool_error(error, attempt=0, available_tools=["search"])
        assert result is not None
        assert "search" in result
        assert "Attempt 1/3" in result

    def test_returns_none_when_per_call_budget_exhausted(self) -> None:
        handler = ErrorHandler.default()
        error = ToolParameterError("bad", tool_name="search")
        # Exhaust per-call budget (max_corrections=3)
        for i in range(3):
            handler.handle_tool_error(error, attempt=i, available_tools=["search"])
        result = handler.handle_tool_error(error, attempt=3, available_tools=["search"])
        assert result is None

    def test_returns_none_when_total_budget_exhausted(self) -> None:
        handler = ErrorHandler(max_corrections=10, max_total_corrections=2)
        error = ToolParameterError("bad", tool_name="search")
        # Use up total budget
        handler.handle_tool_error(error, attempt=0, available_tools=[])
        handler.handle_tool_error(error, attempt=1, available_tools=[])
        result = handler.handle_tool_error(error, attempt=2, available_tools=[])
        assert result is None

    def test_returns_none_for_fatal_errors(self) -> None:
        handler = ErrorHandler.default()
        error = LLMProviderError("auth", status_code=401)
        result = handler.handle_tool_error(error, attempt=0, available_tools=[])
        assert result is None

    def test_returns_correction_for_retryable_tool_errors(self) -> None:
        handler = ErrorHandler.default()
        error = ToolExecutionError("failed", tool_name="search")
        result = handler.handle_tool_error(error, attempt=0, available_tools=["search"])
        assert result is not None

    def test_tool_not_found_includes_available_tools(self) -> None:
        handler = ErrorHandler.default()
        error = ToolNotFoundError("not found", tool_name="nonexistent")
        result = handler.handle_tool_error(error, attempt=0, available_tools=["search", "calc"])
        assert result is not None
        assert "search" in result
        assert "calc" in result


class TestReset:
    def test_clears_total_correction_counter(self) -> None:
        handler = ErrorHandler(max_corrections=10, max_total_corrections=2)
        error = ToolParameterError("bad", tool_name="search")
        handler.handle_tool_error(error, attempt=0, available_tools=[])
        handler.handle_tool_error(error, attempt=1, available_tools=[])
        # Budget exhausted
        assert handler.handle_tool_error(error, attempt=2, available_tools=[]) is None
        # Reset
        handler.reset()
        result = handler.handle_tool_error(error, attempt=0, available_tools=[])
        assert result is not None


class TestDegradationMessage:
    def test_includes_tool_name(self) -> None:
        handler = ErrorHandler.default()
        error = ToolExecutionError("failed", tool_name="search")
        msg = handler.format_degradation_message(error)
        assert "search" in msg
        assert "failed repeatedly" in msg

    def test_unknown_tool_name_fallback(self) -> None:
        handler = ErrorHandler.default()
        error = ValueError("something")
        msg = handler.format_degradation_message(error)
        assert "unknown" in msg


class TestFactoryMethods:
    def test_fail_fast_no_retry(self) -> None:
        handler = ErrorHandler.fail_fast()
        assert handler._retry_policy.max_attempts == 1
        assert handler.max_corrections == 0
        assert handler._max_total_corrections == 0

    def test_fail_fast_returns_none_for_tool_errors(self) -> None:
        handler = ErrorHandler.fail_fast()
        error = ToolParameterError("bad", tool_name="search")
        result = handler.handle_tool_error(error, attempt=0, available_tools=["search"])
        assert result is None

    async def test_fail_fast_raises_on_llm_error_without_retrying(self) -> None:
        handler = ErrorHandler.fail_fast()
        call_count = 0

        async def retry_fn():
            nonlocal call_count
            call_count += 1
            raise LLMRateLimitError("rate limited", retry_after=0.0)

        error = LLMRateLimitError("rate limited", retry_after=0.0)
        with pytest.raises(LLMRateLimitError):
            await handler.handle_llm_error(error, retry_fn)
        assert call_count == 0

    def test_default_matches_expected_config(self) -> None:
        handler = ErrorHandler.default()
        assert handler._retry_policy.max_attempts == 5
        assert handler.max_corrections == 3
        assert handler._max_total_corrections == 5


class TestShouldDegrade:
    def test_returns_false_in_fail_fast_mode(self) -> None:
        handler = ErrorHandler.fail_fast()
        error = ToolParameterError("bad", tool_name="search")
        assert handler.should_degrade(error, attempt=5) is False

    def test_returns_true_when_per_call_budget_exhausted(self) -> None:
        handler = ErrorHandler(max_corrections=2)
        error = ToolParameterError("bad", tool_name="search")
        assert handler.should_degrade(error, attempt=2) is True

    def test_returns_true_when_total_budget_exhausted(self) -> None:
        handler = ErrorHandler(max_corrections=10, max_total_corrections=2)
        error = ToolParameterError("bad", tool_name="search")
        # Consume total budget
        handler.handle_tool_error(error, attempt=0, available_tools=[])
        handler.handle_tool_error(error, attempt=0, available_tools=[])
        assert handler.should_degrade(error, attempt=0) is True

    def test_returns_false_when_budget_not_exhausted(self) -> None:
        handler = ErrorHandler.default()
        error = ToolParameterError("bad", tool_name="search")
        assert handler.should_degrade(error, attempt=0) is False


class TestRestore:
    def test_sets_total_corrections(self) -> None:
        handler = ErrorHandler.default()
        handler.restore(3)
        assert handler.total_corrections == 3

    def test_restored_counter_affects_budget(self) -> None:
        handler = ErrorHandler(max_corrections=10, max_total_corrections=5)
        handler.restore(5)
        error = ToolParameterError("bad", tool_name="search")
        result = handler.handle_tool_error(error, attempt=0, available_tools=[])
        assert result is None


class TestHandleLLMCorrection:
    def test_returns_prompt_for_schema_violation(self) -> None:
        handler = ErrorHandler.default()
        error = LLMSchemaViolationError("bad schema", expected_schema="{}", received="{}")
        result = handler.handle_llm_correction(error, attempt=0)
        assert result is not None
        assert "required format" in result
        assert "Attempt 1/3" in result

    def test_returns_none_when_per_error_budget_exhausted(self) -> None:
        handler = ErrorHandler.default()
        error = LLMSchemaViolationError("bad schema")
        for i in range(3):
            handler.handle_llm_correction(error, attempt=i)
        result = handler.handle_llm_correction(error, attempt=3)
        assert result is None


class TestFailFast:
    def test_handle_tool_error_returns_none(self) -> None:
        handler = ErrorHandler.fail_fast()
        result = handler.handle_tool_error(ValueError("err"), attempt=0, available_tools=["tool1"])
        assert result is None

    def test_should_degrade_returns_false(self) -> None:
        handler = ErrorHandler.fail_fast()
        assert handler.should_degrade(ValueError("err"), attempt=5) is False

    def test_format_degradation_message_with_tool_name(self) -> None:
        handler = ErrorHandler.fail_fast()
        error = ToolExecutionError("failed", tool_name="search")
        msg = handler.format_degradation_message(error)
        assert "search" in msg
        assert "failed repeatedly" in msg

    def test_format_degradation_message_without_tool_name(self) -> None:
        handler = ErrorHandler.fail_fast()
        msg = handler.format_degradation_message(ValueError("oops"))
        assert "unknown" in msg

    def test_max_corrections_is_zero(self) -> None:
        handler = ErrorHandler.fail_fast()
        assert handler.max_corrections == 0

    def test_returns_none_when_total_budget_exhausted(self) -> None:
        handler = ErrorHandler(max_corrections=10, max_total_corrections=2)
        error = LLMSchemaViolationError("bad schema")
        handler.handle_llm_correction(error, attempt=0)
        handler.handle_llm_correction(error, attempt=1)
        result = handler.handle_llm_correction(error, attempt=2)
        assert result is None

    def test_shares_budget_with_tool_corrections(self) -> None:
        handler = ErrorHandler(max_corrections=10, max_total_corrections=3)
        tool_error = ToolParameterError("bad", tool_name="search")
        handler.handle_tool_error(tool_error, attempt=0, available_tools=[])
        handler.handle_tool_error(tool_error, attempt=1, available_tools=[])
        # 2 of 3 total budget used by tool corrections
        llm_error = LLMSchemaViolationError("bad schema")
        result = handler.handle_llm_correction(llm_error, attempt=0)
        assert result is not None  # 3rd correction allowed
        result = handler.handle_llm_correction(llm_error, attempt=1)
        assert result is None  # 4th would exceed total budget

    def test_returns_none_for_non_correctable_errors(self) -> None:
        handler = ErrorHandler.default()
        error = LLMProviderError("auth failed", status_code=401)
        result = handler.handle_llm_correction(error, attempt=0)
        assert result is None

    def test_fail_fast_returns_none(self) -> None:
        handler = ErrorHandler.fail_fast()
        error = LLMSchemaViolationError("bad schema")
        result = handler.handle_llm_correction(error, attempt=0)
        assert result is None
