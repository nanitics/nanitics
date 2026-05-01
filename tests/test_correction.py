from nanitics.capabilities.errors.correction import format_correction_prompt
from nanitics.infrastructure.errors import (
    LLMSchemaViolationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
)


class TestFormatCorrectionPrompt:
    def test_tool_parameter_error_includes_tool_name_and_reason(self) -> None:
        error = ToolParameterError(
            "invalid param",
            tool_name="calculator",
            parameter_name="expression",
            reason="must be a string",
        )
        result = format_correction_prompt(error, attempt=1, max_attempts=3)
        assert "calculator" in result
        assert "must be a string" in result
        assert "Review the tool's parameter schema" in result

    def test_tool_parameter_error_uses_message_when_no_reason(self) -> None:
        error = ToolParameterError("bad input", tool_name="calc", parameter_name="x")
        result = format_correction_prompt(error, attempt=1, max_attempts=3)
        assert "bad input" in result

    def test_tool_execution_error_includes_tool_name(self) -> None:
        error = ToolExecutionError("division by zero", tool_name="calculator")
        result = format_correction_prompt(error, attempt=2, max_attempts=3)
        assert "calculator" in result
        assert "division by zero" in result
        assert "different approach" in result

    def test_tool_not_found_error_includes_available_tools(self) -> None:
        error = ToolNotFoundError("not found", tool_name="search")
        result = format_correction_prompt(
            error,
            attempt=1,
            max_attempts=3,
            available_tools=["calculator", "query_dataset"],
        )
        assert "search" in result
        assert "does not exist" in result
        assert "calculator" in result
        assert "query_dataset" in result

    def test_tool_not_found_error_without_available_tools(self) -> None:
        error = ToolNotFoundError("not found", tool_name="search")
        result = format_correction_prompt(error, attempt=1, max_attempts=3)
        assert "none" in result

    def test_llm_schema_violation_error(self) -> None:
        error = LLMSchemaViolationError("missing required field 'name'")
        result = format_correction_prompt(error, attempt=1, max_attempts=2)
        assert "did not match the required format" in result
        assert "missing required field 'name'" in result

    def test_attempt_counter_appears_in_output(self) -> None:
        error = ToolExecutionError("failed", tool_name="tool")
        result = format_correction_prompt(error, attempt=2, max_attempts=3)
        assert "(Attempt 2/3)" in result

    def test_attempt_counter_first_attempt(self) -> None:
        error = ToolExecutionError("failed", tool_name="tool")
        result = format_correction_prompt(error, attempt=1, max_attempts=5)
        assert "(Attempt 1/5)" in result

    def test_fallback_for_unknown_error_type(self) -> None:
        error = ValueError("something went wrong")
        result = format_correction_prompt(error, attempt=1, max_attempts=3)
        assert "ValueError" in result
        assert "something went wrong" in result
        assert "(Attempt 1/3)" in result

    def test_fallback_for_runtime_error(self) -> None:
        error = RuntimeError("unexpected failure")
        result = format_correction_prompt(error, attempt=2, max_attempts=3)
        assert "RuntimeError" in result
        assert "unexpected failure" in result
