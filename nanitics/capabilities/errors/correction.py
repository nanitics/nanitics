from nanitics.infrastructure.errors import (
    LLMSchemaViolationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
)


def format_correction_prompt(
    error: Exception,
    attempt: int,
    max_attempts: int,
    available_tools: list[str] | None = None,
) -> str:
    """Build a structured feedback message for the LLM to self-correct.

    Generates error-specific correction prompts that guide the LLM to
    fix its mistake. The prompt format varies by error type:
    - ToolParameterError: identifies the rejected parameter and reason
    - ToolExecutionError: describes the failure, suggests alternatives
    - ToolNotFoundError: lists available tools to choose from
    - LLMSchemaViolationError: describes the expected format

    Args:
        error: The exception that triggered correction.
        attempt: Current attempt number (1-based).
        max_attempts: Maximum correction attempts allowed.
        available_tools: Tool names the agent can use (shown for ToolNotFoundError).

    Returns:
        A formatted correction prompt string injected as a tool result.
    """
    suffix = f"(Attempt {attempt}/{max_attempts})"

    if isinstance(error, ToolParameterError):
        reason = error.reason or error.message
        return (
            f"Tool '{error.tool_name}' rejected the parameters: {reason}\n"
            f"Review the tool's parameter schema and provide corrected parameters.\n"
            f"{suffix}"
        )

    if isinstance(error, ToolExecutionError):
        return (
            f"Tool '{error.tool_name}' failed during execution: {error.message}\n"
            f"Consider using a different approach or tool.\n{suffix}"
        )

    if isinstance(error, ToolNotFoundError):
        tools_str = ", ".join(available_tools) if available_tools else "none"
        return (
            f"Tool '{error.tool_name}' does not exist. Available tools: {tools_str}\n"
            f"Select a tool from the available list.\n{suffix}"
        )

    if isinstance(error, LLMSchemaViolationError):
        return (
            f"The response did not match the required format: {error.message}\n"
            f"Please provide output matching the specified schema.\n{suffix}"
        )

    # Fallback for unrecognized error types
    error_type = type(error).__name__
    return f"{error_type}: {error}\nAn unexpected error occurred. Try a different approach.\n{suffix}"
