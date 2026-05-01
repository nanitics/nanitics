"""Unit tests for :func:`nanitics.tools.file_read.create_file_read_tool`.

Covers:

- UTF-8 text file happy path.
- Non-UTF-8 binary file returns base64-encoded content with
  ``metadata.encoding == "base64"``.
- File whose size exceeds ``max_bytes`` is truncated with
  ``metadata.truncated == True`` and ``bytes_read`` accurate.
- Path outside ``allowed_paths`` raises ``ToolParameterError``.
- Nested allowed directory accepts descendant files.
- Symlink inside an allowed root that resolves inside the allow-list passes.
- Symlink inside an allowed root that points outside the allow-list is
  rejected via ``ToolParameterError`` (because ``.resolve()`` follows it).
- Missing file → ``ToolExecutionError``.
- Directory path → ``ToolExecutionError``.
- Permission denied → ``ToolExecutionError``.
- Empty ``allowed_paths`` → constructor ``ValueError``.
- ``max_bytes`` parameter validation outside bounds → ``ToolParameterError``.

All tests use ``tmp_path`` fixtures; no real network or filesystem access
outside the pytest temporary directory.
"""

from __future__ import annotations

import base64
import os
import stat
import sys
from pathlib import Path

import pytest

from nanitics.core.tools.protocol import Tool, ToolResult
from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolParameterError,
)
from nanitics.tools.file_read import create_file_read_tool

# --- Construction ------------------------------------------------------------


class TestConstruction:
    def test_returns_tool_conforming_object(self, tmp_path: Path) -> None:
        tool = create_file_read_tool(allowed_paths=[tmp_path])
        assert isinstance(tool, Tool)

    def test_default_name_and_description(self, tmp_path: Path) -> None:
        tool = create_file_read_tool(allowed_paths=[tmp_path])
        assert tool.schema.name == "file_read"
        assert "file" in tool.schema.description.lower()

    def test_custom_name_and_description(self, tmp_path: Path) -> None:
        tool = create_file_read_tool(
            allowed_paths=[tmp_path],
            name="my_read",
            description="Custom description.",
        )
        assert tool.schema.name == "my_read"
        assert tool.schema.description == "Custom description."

    def test_empty_allowed_paths_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="allowed_paths"):
            create_file_read_tool(allowed_paths=[])

    def test_accepts_string_and_path_entries(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested"
        nested.mkdir()
        tool = create_file_read_tool(allowed_paths=[str(tmp_path), nested])
        assert isinstance(tool, Tool)


# --- Happy paths -------------------------------------------------------------


class TestReadUtf8:
    @pytest.mark.asyncio
    async def test_reads_utf8_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "hello.txt"
        path.write_text("hello, world!", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path=str(path))

        assert isinstance(result, ToolResult)
        assert result.content == "hello, world!"
        assert result.metadata["encoding"] == "utf-8"
        assert result.metadata["truncated"] is False
        assert result.metadata["size_bytes"] == len(b"hello, world!")
        assert result.metadata["bytes_read"] == len(b"hello, world!")
        # Path in metadata is resolved to absolute form.
        assert result.metadata["path"] == str(path.resolve())

    @pytest.mark.asyncio
    async def test_reads_nested_file_under_allowed_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        path = nested / "x.txt"
        path.write_text("deep", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path=str(path))

        assert result.content == "deep"

    @pytest.mark.asyncio
    async def test_reads_unicode_text(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode.txt"
        path.write_text("héllo • 世界", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path=str(path))

        assert "世界" in result.content
        assert result.metadata["encoding"] == "utf-8"


class TestReadBinary:
    @pytest.mark.asyncio
    async def test_binary_file_returns_base64_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "img.bin"
        # \xff\xfe is not valid UTF-8.
        raw = b"\xff\xfe\x00\x01\x02"
        path.write_bytes(raw)
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path=str(path))

        assert result.metadata["encoding"] == "base64"
        # Content carries the base64 representation so the LLM has *something*
        # legible.
        assert result.content == base64.b64encode(raw).decode("ascii")
        assert result.metadata["size_bytes"] == len(raw)
        assert result.metadata["bytes_read"] == len(raw)
        assert result.metadata["truncated"] is False


# --- Truncation --------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_truncates_when_exceeds_max_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "big.txt"
        # 1000 bytes of ASCII 'a'.
        path.write_bytes(b"a" * 1000)
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path=str(path), max_bytes=100)

        assert result.metadata["truncated"] is True
        assert result.metadata["size_bytes"] == 1000
        assert result.metadata["bytes_read"] == 100
        assert len(result.content) == 100

    @pytest.mark.asyncio
    async def test_no_truncation_when_within_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "small.txt"
        path.write_bytes(b"short")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path=str(path), max_bytes=1024)

        assert result.metadata["truncated"] is False
        assert result.metadata["bytes_read"] == 5


# --- Allow-list enforcement --------------------------------------------------


class TestAllowList:
    @pytest.mark.asyncio
    async def test_path_outside_allowed_paths_raises_parameter_error(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("nope", encoding="utf-8")
        # Allow only a different subdir that does not contain outside.
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        tool = create_file_read_tool(allowed_paths=[allowed])

        with pytest.raises(ToolParameterError) as exc_info:
            await tool.execute(path=str(outside))
        assert "allowed_paths" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_multiple_allowed_roots_any_matches(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        file_b = root_b / "x.txt"
        file_b.write_text("b-content", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[root_a, root_b])

        result = await tool.execute(path=str(file_b))
        assert result.content == "b-content"

    @pytest.mark.asyncio
    async def test_relative_path_resolved_against_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "rel.txt"
        path.write_text("hi", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path="rel.txt")
        assert result.content == "hi"


# --- Symlinks ----------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks flaky on Windows CI")
class TestSymlinks:
    @pytest.mark.asyncio
    async def test_symlink_inside_allowed_root_resolves_correctly(self, tmp_path: Path) -> None:
        target = tmp_path / "target.txt"
        target.write_text("pointed-to", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        result = await tool.execute(path=str(link))
        assert result.content == "pointed-to"

    @pytest.mark.asyncio
    async def test_symlink_pointing_outside_allowed_root_is_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        link = allowed / "escape.txt"
        link.symlink_to(outside)
        tool = create_file_read_tool(allowed_paths=[allowed])

        with pytest.raises(ToolParameterError):
            await tool.execute(path=str(link))


# --- Error mapping -----------------------------------------------------------


class TestErrors:
    @pytest.mark.asyncio
    async def test_missing_file_raises_execution_error(self, tmp_path: Path) -> None:
        tool = create_file_read_tool(allowed_paths=[tmp_path])
        missing = tmp_path / "nonexistent.txt"

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path=str(missing))
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_directory_path_raises_execution_error(self, tmp_path: Path) -> None:
        directory = tmp_path / "dir"
        directory.mkdir()
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path=str(directory))
        assert "director" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0,  # type: ignore[attr-defined]
        reason="chmod-based permission test doesn't apply to root or Windows",
    )
    async def test_permission_denied_raises_execution_error(self, tmp_path: Path) -> None:
        path = tmp_path / "locked.txt"
        path.write_text("hi", encoding="utf-8")
        # Remove read permission.
        path.chmod(0)
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        try:
            with pytest.raises(ToolExecutionError) as exc_info:
                await tool.execute(path=str(path))
            assert "permission" in str(exc_info.value).lower()
        finally:
            # Restore perms so tmp_path cleanup succeeds.
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    @pytest.mark.asyncio
    async def test_other_os_error_raises_execution_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "hi.txt"
        path.write_text("hi", encoding="utf-8")

        from nanitics.tools import file_read as file_read_module

        def bad_read_bounded(p: Path, max_bytes: int) -> bytes:
            raise OSError("disk exploded")

        monkeypatch.setattr(file_read_module, "_read_bounded", bad_read_bounded)
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute(path=str(path))
        assert "os error" in str(exc_info.value).lower()


# --- Parameter validation ----------------------------------------------------


class TestParameterValidation:
    @pytest.mark.asyncio
    async def test_max_bytes_below_lower_bound_raises_parameter_error(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("hi", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        with pytest.raises(ToolParameterError):
            await tool.execute(path=str(path), max_bytes=0)

    @pytest.mark.asyncio
    async def test_max_bytes_above_upper_bound_raises_parameter_error(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("hi", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        with pytest.raises(ToolParameterError):
            await tool.execute(path=str(path), max_bytes=200_000_000)

    @pytest.mark.asyncio
    async def test_default_max_bytes_is_1_mib(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("hi", encoding="utf-8")
        tool = create_file_read_tool(allowed_paths=[tmp_path])

        # Default applies when max_bytes is not provided.
        result = await tool.execute(path=str(path))
        assert result.metadata["truncated"] is False
        # Sanity-check that a file the size of 2 MiB would truncate at default.
        big = tmp_path / "big.bin"
        big.write_bytes(b"\x00" * (1_048_576 + 10))
        result2 = await tool.execute(path=str(big))
        assert result2.metadata["truncated"] is True
        assert result2.metadata["bytes_read"] == 1_048_576
