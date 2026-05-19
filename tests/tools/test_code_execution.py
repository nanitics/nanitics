"""Unit tests for :func:`nanitics.tools.code_execution.create_code_execution_tool`.

Covers:

- Successful execution returns content + metadata mirroring
  :class:`~nanitics.safety.sandbox.protocol.ExecutionResult`.
- Failed execution (``success=False`` in ``ExecutionResult``) returns error
  metadata without raising — the LLM reads stderr and tries to correct.
- Sandbox that raises an unexpected exception is wrapped in
  :class:`~nanitics.infrastructure.errors.ToolExecutionError` with the
  original attached as ``__cause__``.
- Parameter validation (empty code, code too long) raises
  :class:`~nanitics.infrastructure.errors.ToolParameterError`.
- The returned object satisfies the structural
  :class:`~nanitics.strategies.tools.protocol.Tool` protocol.

All tests use :class:`~nanitics.safety.sandbox.mock.MockSandbox` — no Docker
and no real subprocess execution.
"""

from __future__ import annotations

from typing import Self

import pytest

from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolParameterError,
)
from nanitics.safety.sandbox.mock import MockSandbox
from nanitics.safety.sandbox.protocol import ExecutionResult
from nanitics.strategies.tools.protocol import Tool, ToolResult
from nanitics.tools.code_execution import create_code_execution_tool

# --- Helpers -----------------------------------------------------------------


def _ok(stdout: str = "hello", stderr: str = "", return_value: str | None = None) -> ExecutionResult:
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        return_value=return_value,
        success=True,
        error=None,
        duration_ms=1.23,
    )


def _fail(error: str = "boom", stderr: str = "traceback...") -> ExecutionResult:
    return ExecutionResult(
        stdout="",
        stderr=stderr,
        return_value=None,
        success=False,
        error=error,
        duration_ms=0.5,
    )


# --- Construction ------------------------------------------------------------


class TestConstruction:
    def test_returns_tool_conforming_object(self) -> None:
        sandbox = MockSandbox([_ok()])
        tool = create_code_execution_tool(sandbox)
        assert isinstance(tool, Tool)

    def test_default_name_and_description(self) -> None:
        sandbox = MockSandbox([_ok()])
        tool = create_code_execution_tool(sandbox)
        assert tool.schema.name == "code_execution"
        assert "sandbox" in tool.schema.description.lower()

    def test_custom_name_and_description(self) -> None:
        sandbox = MockSandbox([_ok()])
        tool = create_code_execution_tool(
            sandbox,
            name="exec",
            description="Custom desc.",
        )
        assert tool.schema.name == "exec"
        assert tool.schema.description == "Custom desc."


# --- Successful execution ----------------------------------------------------


class TestSuccess:
    @pytest.mark.asyncio
    async def test_success_returns_content_and_metadata(self) -> None:
        sandbox = MockSandbox([_ok(stdout="42", stderr="", return_value="42")])
        tool = create_code_execution_tool(sandbox)

        result = await tool.execute(code="print(42)")

        assert isinstance(result, ToolResult)
        assert "stdout:" in result.content
        assert "42" in result.content
        assert "stderr:" in result.content
        assert result.metadata["success"] is True
        assert result.metadata["stdout"] == "42"
        assert result.metadata["return_value"] == "42"
        assert result.metadata["error"] is None

    @pytest.mark.asyncio
    async def test_success_with_stderr_still_does_not_prefix_error(self) -> None:
        # stderr without success=False is just a warning stream.
        sandbox = MockSandbox([_ok(stdout="", stderr="deprecation warning")])
        tool = create_code_execution_tool(sandbox)

        result = await tool.execute(code="import x")

        assert not result.content.startswith("error:")
        assert "deprecation warning" in result.content


# --- Failure (success=False) — no raise --------------------------------------


class TestFailureNotRaised:
    @pytest.mark.asyncio
    async def test_execution_failure_returns_metadata_without_raising(self) -> None:
        sandbox = MockSandbox([_fail(error="NameError: x not defined", stderr="trace")])
        tool = create_code_execution_tool(sandbox)

        result = await tool.execute(code="print(x)")

        # No exception raised.
        assert isinstance(result, ToolResult)
        assert result.metadata["success"] is False
        assert result.metadata["error"] == "NameError: x not defined"
        assert result.metadata["stderr"] == "trace"
        # Content has an error prefix so the LLM sees the failure clearly.
        assert result.content.startswith("error:")
        assert "NameError" in result.content
        assert "stderr:" in result.content


# --- Sandbox raising ---------------------------------------------------------


class _ExplodingSandbox:
    """Sandbox stub that raises on ``execute``."""

    async def start(self) -> None:  # pragma: no cover - not invoked
        pass

    async def execute(self, code: str) -> ExecutionResult:
        raise RuntimeError("sandbox died")

    async def reset(self) -> None:  # pragma: no cover - not invoked
        pass

    async def cleanup(self) -> None:  # pragma: no cover - not invoked
        pass

    async def __aenter__(self) -> Self:  # pragma: no cover - not invoked
        return self

    async def __aexit__(self, *args: object) -> None:  # pragma: no cover - not invoked
        pass


class TestSandboxRaises:
    @pytest.mark.asyncio
    async def test_unexpected_exception_wrapped_in_execution_error(self) -> None:
        sandbox = _ExplodingSandbox()
        tool = create_code_execution_tool(sandbox)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(code="anything")

        assert exc_info.value.tool_name == "code_execution"
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "sandbox died" in str(exc_info.value.__cause__)


# --- Parameter validation ----------------------------------------------------


class TestParameterValidation:
    @pytest.mark.asyncio
    async def test_empty_code_raises_parameter_error(self) -> None:
        sandbox = MockSandbox([_ok()])
        tool = create_code_execution_tool(sandbox)

        with pytest.raises(ToolParameterError):
            await tool.execute(code="")

    @pytest.mark.asyncio
    async def test_code_over_max_length_raises_parameter_error(self) -> None:
        sandbox = MockSandbox([_ok()])
        tool = create_code_execution_tool(sandbox)

        with pytest.raises(ToolParameterError):
            await tool.execute(code="x" * 100_001)

    @pytest.mark.asyncio
    async def test_code_at_max_length_is_accepted(self) -> None:
        sandbox = MockSandbox([_ok(stdout="ok")])
        tool = create_code_execution_tool(sandbox)

        result = await tool.execute(code="x" * 100_000)
        assert result.metadata["success"] is True
