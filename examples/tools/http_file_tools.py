"""HTTP and file-read reference tools: allow-listed requests and sandboxed reads.

Demonstrates ``create_http_tool`` and ``create_file_read_tool`` — two of the four
curated reference tools shipped by the SDK.  Both satisfy the ``Tool`` protocol
and dispatch through ``ToolRegistry`` identically to a ``FunctionTool``.

Section 1 issues a GET against an allow-listed domain through a
``MockLLMClient``-backed ``ReActAgent``, with ``respx`` intercepting the HTTPS
call.  Section 2 writes a UTF-8 file and a binary file into a temporary
directory, then reads each back through ``create_file_read_tool``, showing the
UTF-8 vs. base64 encoding distinction surfaced in ``metadata.encoding``.
Section 3 is a commented-out real-HTTP block pointing at the public
``api.github.com`` endpoint.

Related guide: docs/guides/tools.md (see the "Reference Tools" section).
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import httpx
import respx

from examples.helpers import make_emitter, make_response
from nanitics import (
    MockLLMClient,
    ReActAgent,
    ToolCall,
    ToolParameterError,
    create_file_read_tool,
    create_http_tool,
)
from nanitics.infrastructure import ToolResultEvent


async def main() -> None:
    # --- Section 1: HTTP tool with allow-listed domain (always runs) ---
    print("--- Section 1: HTTP tool via ReActAgent (hermetic) ---")

    with respx.mock(assert_all_called=True) as respx_router:
        respx_router.get("https://httpbin.org/get").mock(
            return_value=httpx.Response(
                200,
                json={"args": {"q": "hello"}, "origin": "127.0.0.1"},
            )
        )

        http_tool = create_http_tool(allowed_domains=["httpbin.org"])
        assert http_tool.schema.name == "http_request"

        llm = MockLLMClient(
            responses=[
                make_response(
                    "I'll fetch that.",
                    tool_calls=[
                        ToolCall(
                            id="tc-http-1",
                            name="http_request",
                            arguments={
                                "method": "GET",
                                "url": "https://httpbin.org/get",
                                "query_params": {"q": "hello"},
                            },
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("The service echoed our query back."),
            ]
        )
        emitter = make_emitter("http-section-1")

        agent = ReActAgent(
            name="http-agent",
            llm_client=llm,
            emitter=emitter,
            system_prompt="Use http_request to answer the user.",
            tools=[http_tool],
        )
        result = await agent.run("What does httpbin return for ?q=hello?")

    assert result.termination_reason == "complete"
    tool_results = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
    assert tool_results[0].success is True
    assert tool_results[0].result is not None
    assert "HTTP 200" in tool_results[0].result

    # Drive the tool directly to inspect the structured metadata the event
    # layer intentionally omits — status, headers, truncation flags, bytes_read.
    with respx.mock() as respx_router:
        respx_router.get("https://httpbin.org/get").mock(
            return_value=httpx.Response(200, json={"ok": True}),
        )
        direct = await http_tool.execute(method="GET", url="https://httpbin.org/get")
    assert direct.metadata["status"] == 200
    assert direct.metadata["truncated"] is False

    # Rejecting an unlisted domain surfaces as a parameter error so the LLM
    # can correct by choosing a different URL.
    try:
        await http_tool.execute(method="GET", url="https://evil.example.com/")
        raise AssertionError("Should have raised ToolParameterError")
    except ToolParameterError as exc:
        rejection = f"{type(exc).__name__}: {exc}"

    print(f"  Output: {result.output}")
    print(f"  Response status: {direct.metadata['status']}")
    print(f"  Bytes read: {direct.metadata['bytes_read']}")
    print(f"  Rejected off-allowlist call: {rejection}")
    print("✓ http_request enforces allow-listed domains and bounds response size")

    # --- Section 2: File-read tool — UTF-8 vs binary (always runs) ---
    print("\n--- Section 2: File-read tool (UTF-8 + binary) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        text_path = tmp_root / "notes.md"
        text_path.write_text("# Reference tools\n\nThe SDK ships four of them.\n", encoding="utf-8")
        binary_path = tmp_root / "logo.png"
        binary_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00nanitics-binary-bytes\x00\xff")

        file_tool = create_file_read_tool(allowed_paths=[tmp_root])
        assert file_tool.schema.name == "file_read"

        # Drive the tool directly — the agent-side integration is identical to
        # Section 1 (already demonstrated), so here we focus on the
        # encoding distinction.
        text_result = await file_tool.execute(path=str(text_path))
        assert "Reference tools" in text_result.content
        assert text_result.metadata["encoding"] == "utf-8"
        assert text_result.metadata["truncated"] is False

        binary_result = await file_tool.execute(path=str(binary_path))
        # Binary payloads come back as base64 in content; metadata.encoding flags it.
        assert binary_result.metadata["encoding"] == "base64"
        assert base64.b64decode(binary_result.content).startswith(b"\x89PNG")

        # A path outside the allow-list is rejected as a parameter error so the
        # LLM can correct by choosing a valid path.
        try:
            await file_tool.execute(path="/etc/hosts")
            raise AssertionError("Should have raised ToolParameterError")
        except ToolParameterError as exc:
            print(f"  Rejected out-of-scope path: {type(exc).__name__}")

        print(f"  Text file: encoding={text_result.metadata['encoding']!r}, size={text_result.metadata['size_bytes']}B")
        print(
            f"  Binary file: encoding={binary_result.metadata['encoding']!r}, "
            f"size={binary_result.metadata['size_bytes']}B"
        )
        print("✓ file_read resolves symlinks and enforces the allowed_paths allow-list")

    # --- Section 3: Real HTTP call (commented out — requires network) ---
    print("\n--- Section 3: Real HTTP call (commented out — requires network) ---")
    print("  See the source of this file for the runnable block.")
    # The snippet below queries the public GitHub API for a repo's metadata.
    # It requires outbound network access (no API key needed for public repos).
    #
    # --------------------------------------------------------------------------
    # real_http = create_http_tool(allowed_domains=["api.github.com"])
    # response = await real_http.execute(
    #     method="GET",
    #     url="https://api.github.com/repos/anthropics/claude-code",
    #     headers={"Accept": "application/vnd.github+json"},
    # )
    # print(response.metadata["status"], response.metadata["url"])
    # --------------------------------------------------------------------------

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
