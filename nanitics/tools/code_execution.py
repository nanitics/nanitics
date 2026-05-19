"""``create_code_execution_tool`` — adapter for the ``Sandbox`` protocol.

The factory returns a :class:`~nanitics.strategies.tools.protocol.Tool`-conforming
object that dispatches ``execute(code=...)`` through any object satisfying
the existing :class:`~nanitics.safety.sandbox.protocol.Sandbox` protocol
(e.g. :class:`~nanitics.safety.sandbox.docker.DockerSandbox` or
:class:`~nanitics.safety.sandbox.mock.MockSandbox`).  The tool does NOT
own the sandbox's lifecycle: the caller is expected to enter the sandbox's
async context manager before any agent run and exit it after.

The tool surfaces sandbox-level *execution* failures
(``ExecutionResult.success is False``) through
``ToolResult.metadata.success=False`` and an ``error:`` prefix in
``content`` — the LLM can read stderr and try to fix the code.  Only
unexpected exceptions raised by the sandbox implementation (transport
errors, container crashes, etc.) are wrapped in
:class:`~nanitics.infrastructure.errors.ToolExecutionError`.

This module depends only on the stdlib and the ``Sandbox`` protocol; it
has no optional-dependency guard.  The user's choice of sandbox is what
pulls in ``docker`` (via the existing ``code_execution`` extra when using
:class:`DockerSandbox`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.safety.sandbox.protocol import ExecutionResult, Sandbox
from nanitics.strategies.tools.function_tool import FunctionTool
from nanitics.strategies.tools.protocol import Tool, ToolResult
from nanitics.tools._result_models import CodeExecutionResult

# Guardrail against pathologically long prompts; the sandbox itself enforces
# its own limits on execution time and output size.
_MAX_CODE_CHARS = 100_000


class _CodeExecutionParams(BaseModel):
    """Parameters accepted by the ``code_execution`` tool."""

    code: str = Field(
        min_length=1,
        max_length=_MAX_CODE_CHARS,
        description="Python source code to execute in the sandbox.",
    )


def _render_content(exec_result: ExecutionResult) -> str:
    """Render an :class:`ExecutionResult` as the LLM-visible content string.

    On success, returns ``"stdout:\n{stdout}\n\nstderr:\n{stderr}"``.  On
    failure, prepends an ``"error: {error}"`` line so the LLM immediately
    notices the failure before scanning stderr.
    """
    body = f"stdout:\n{exec_result.stdout}\n\nstderr:\n{exec_result.stderr}"
    if not exec_result.success:
        return f"error: {exec_result.error}\n\n{body}"
    return body


def create_code_execution_tool(
    sandbox: Sandbox,
    *,
    name: str = "code_execution",
    description: str | None = None,
) -> Tool:
    """Create a code-execution tool backed by the given sandbox.

    The returned object satisfies :class:`~nanitics.strategies.tools.protocol.Tool`
    and can be registered in :class:`~nanitics.strategies.ToolRegistry` alongside
    any other tool.  The tool emits
    :class:`~nanitics.events.ToolInvokeEvent` and
    :class:`~nanitics.events.ToolResultEvent` through the registry's
    standard dispatch path.

    Lifecycle contract: the factory holds a reference to *sandbox* but does
    NOT call :meth:`Sandbox.start` or :meth:`Sandbox.cleanup`.  The caller
    must enter the sandbox's async context manager before any agent run
    that uses this tool and exit it after.  Sharing a single sandbox across
    multiple tools or agents is supported.

    Sandbox-level failures (``ExecutionResult.success is False``) do NOT
    raise — they are surfaced through ``ToolResult.metadata.success`` and
    an ``error:`` prefix in ``content`` so the LLM can read the stderr and
    try to correct its code.  Only unexpected exceptions raised by the
    sandbox implementation are wrapped in
    :class:`~nanitics.infrastructure.errors.ToolExecutionError` with the
    original attached as ``__cause__``.

    Args:
        sandbox: Any object satisfying the :class:`Sandbox` protocol.
        name: Tool name exposed to the LLM.  Defaults to ``"code_execution"``.
        description: Optional override of the LLM-facing description.

    Returns:
        A :class:`Tool`-conforming object.
    """
    effective_description = description or (
        "Execute Python code in a sandboxed environment and return stdout, "
        "stderr, and any error. Sandbox state persists across calls until reset."
    )

    async def _execute(code: str) -> ToolResult:
        try:
            exec_result = await sandbox.execute(code)
        except Exception as exc:
            raise ToolExecutionError(
                f"Sandbox execution failed: {exc}",
                tool_name="code_execution",
            ) from exc

        metadata: dict[str, Any] = CodeExecutionResult(
            success=exec_result.success,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            return_value=exec_result.return_value,
            error=exec_result.error,
            duration_ms=exec_result.duration_ms,
        ).model_dump()

        return ToolResult(
            content=_render_content(exec_result),
            metadata=metadata,
        )

    return FunctionTool(
        fn=_execute,
        name=name,
        description=effective_description,
        parameters_model=_CodeExecutionParams,
    )
