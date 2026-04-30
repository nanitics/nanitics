"""``create_file_read_tool`` — a curated file-read reference tool.

The factory returns a :class:`~nanitics.core.tools.protocol.Tool`-conforming
object that reads files under an explicit ``allowed_paths`` allow-list.
Paths are resolved via :meth:`pathlib.Path.resolve` (following symlinks and
collapsing ``..`` segments) and compared against the resolved allow-list
roots.  A request that resolves outside every allowed root raises
:class:`~nanitics.infrastructure.errors.ToolParameterError` so the LLM can
correct by picking a valid path.

UTF-8-decodable files are returned as text; files that fail UTF-8 decoding
are returned as a base64 string in ``content`` with
``metadata.encoding == "base64"`` so application code can detect binary
payloads.  Both paths honour the per-call ``max_bytes`` bound.

This module depends only on the stdlib and has no optional-dependency guard.
"""

from __future__ import annotations

import asyncio
import base64
import pathlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from nanitics.core.tools.function_tool import FunctionTool
from nanitics.core.tools.protocol import Tool, ToolResult
from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolParameterError,
)
from nanitics.tools._result_models import FileReadResult

_DEFAULT_MAX_BYTES = 1_048_576  # 1 MiB
_MAX_BYTES_CEILING = 104_857_600  # 100 MiB


class _FileReadParams(BaseModel):
    """Parameters accepted by the ``file_read`` tool."""

    path: str = Field(
        min_length=1,
        description="Path to the file to read. Must resolve inside one of the configured allowed_paths.",
    )
    max_bytes: int = Field(
        default=_DEFAULT_MAX_BYTES,
        ge=1,
        le=_MAX_BYTES_CEILING,
        description=(
            "Upper bound on the number of bytes returned. "
            "Larger files are truncated; metadata.truncated records the effect."
        ),
    )


def _resolve_allowed_roots(
    entries: list[str | pathlib.Path],
) -> list[pathlib.Path]:
    """Resolve every allow-list entry to an absolute real path once.

    The resolved roots are captured in the factory closure so per-call
    validation is a cheap comparison.
    """
    return [pathlib.Path(entry).resolve() for entry in entries]


def _is_inside_any(path: pathlib.Path, roots: list[pathlib.Path]) -> bool:
    """Return ``True`` if *path* is equal to or a descendant of any root."""
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def create_file_read_tool(
    *,
    allowed_paths: list[str | pathlib.Path],
    name: str = "file_read",
    description: str | None = None,
) -> Tool:
    """Create a file-read tool gated by an explicit path allow-list.

    The returned object satisfies :class:`~nanitics.core.tools.protocol.Tool`
    and can be registered in :class:`~nanitics.core.ToolRegistry` alongside
    any other tool.  The tool emits
    :class:`~nanitics.events.ToolInvokeEvent` and
    :class:`~nanitics.events.ToolResultEvent` through the registry's
    standard dispatch path.

    Security posture: ``allowed_paths`` is required and must be non-empty.
    Every entry is resolved via :meth:`pathlib.Path.resolve` at construction
    time; at call time the requested path is also resolved and compared
    against the resolved roots.  A request
    that resolves outside every root raises
    :class:`~nanitics.infrastructure.errors.ToolParameterError`.

    Args:
        allowed_paths: Non-empty list of allowed roots (string or
            :class:`pathlib.Path`).  Each entry is resolved once at
            construction.
        name: Tool name exposed to the LLM.  Defaults to ``"file_read"``.
        description: Optional override of the LLM-facing description.

    Returns:
        A :class:`Tool`-conforming object.

    Raises:
        ValueError: If ``allowed_paths`` is empty.
    """
    if not allowed_paths:
        raise ValueError("create_file_read_tool requires a non-empty allowed_paths list")

    resolved_roots = _resolve_allowed_roots(list(allowed_paths))
    root_display = ", ".join(str(r) for r in resolved_roots)
    effective_description = description or (
        f"Read the contents of a file under {root_display}. "
        "Returns up to max_bytes of text (UTF-8) or a base64 summary for binary files."
    )

    async def _execute(path: str, max_bytes: int = _DEFAULT_MAX_BYTES) -> ToolResult:
        resolved = pathlib.Path(path).resolve()
        if not _is_inside_any(resolved, resolved_roots):
            raise ToolParameterError(
                f"path '{path}' resolves outside allowed_paths",
                tool_name="file_read",
                parameter_name="path",
                reason="path outside allowed_paths",
            )

        try:
            data: bytes = await asyncio.to_thread(_read_bounded, resolved, max_bytes)
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                f"File not found: {resolved}",
                tool_name="file_read",
            ) from exc
        except IsADirectoryError as exc:
            raise ToolExecutionError(
                f"Path is a directory: {resolved}",
                tool_name="file_read",
            ) from exc
        except PermissionError as exc:
            raise ToolExecutionError(
                f"Permission denied: {resolved}",
                tool_name="file_read",
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                f"OS error reading {resolved}: {exc}",
                tool_name="file_read",
            ) from exc

        size_bytes = resolved.stat().st_size
        truncated = size_bytes > max_bytes
        bytes_read = len(data)

        encoding: Literal["utf-8", "base64"]
        try:
            content = data.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(data).decode("ascii")
            encoding = "base64"

        metadata: dict[str, Any] = FileReadResult(
            path=str(resolved),
            size_bytes=size_bytes,
            bytes_read=bytes_read,
            truncated=truncated,
            encoding=encoding,
        ).model_dump()

        return ToolResult(content=content, metadata=metadata)

    return FunctionTool(
        fn=_execute,
        name=name,
        description=effective_description,
        parameters_model=_FileReadParams,
    )


def _read_bounded(path: pathlib.Path, max_bytes: int) -> bytes:
    """Read up to *max_bytes* bytes from *path*.

    Runs on a worker thread via :func:`asyncio.to_thread`.  The caller
    compares the returned length against ``path.stat().st_size`` to decide
    whether truncation occurred.
    """
    with path.open("rb") as f:
        return f.read(max_bytes)
