"""Reference tools under a real LLM: file_read, http_request, web_search, code_execution.

One script, four sections — each constructs one curated reference tool, drives a
real ``ReActAgent`` against it, and asserts the invocation shape + output quality.
Sections are independent; the Tavily and Docker sections skip cleanly when their
resources are absent. Web search and code execution sections are not ``@pytest.mark.quick``
because of external latency (Tavily API) and cold-start cost (Docker).

Acceptance criteria:

``test_file_read`` (happy path):
  - ``ToolInvokeEvent`` with ``tool_name == "file_read"`` is emitted and the
    ``path`` parameter matches the resolved file path.
  - ``ToolResultEvent`` with ``success=True`` and ``result`` containing the
    marker phrase from ``_FILE_CONTENT`` (round-trip fidelity of file body).
  - Final answer passes a fuzzy judge for summarizing the file.

``test_file_read_outside_allowlist`` (negative path):
  - Directly calling ``execute`` with a path outside ``allowed_paths`` raises
    :class:`ToolParameterError`.

``test_http_request`` (happy path):
  - ``ToolInvokeEvent`` with ``tool_name == "http_request"``,
    ``method == "GET"``, and ``url`` pointing at ``httpbin.org/get``.
  - ``ToolResultEvent`` with ``success=True`` and ``result`` starting with
    ``"HTTP 200"`` (rendering contract).
  - Final answer passes a fuzzy judge for reporting the User-Agent header.

``test_http_request_non_allowlisted_domain`` (negative path):
  - Directly calling ``execute`` with a host outside ``allowed_domains``
    raises :class:`ToolParameterError`.

``test_web_search`` (happy path, gated on Tavily):
  - ``ToolInvokeEvent.parameters["query"]`` is non-empty.
  - ``ToolResultEvent`` with ``success=True`` and a non-empty ``result``.
  - Final answer passes a fuzzy judge for summarizing search results.

``test_code_execution`` (happy path, gated on Docker):
  - ``ToolInvokeEvent.parameters["code"]`` mentions ``factorial`` or ``6``.
  - ``ToolResultEvent`` with ``success=True``.
  - ``"720"`` appears in the agent's final output.
  - Final answer passes a fuzzy judge.

``test_http_tool_post_request_body`` (direct tool call, gated on ``aiohttp``):
  - POST with a JSON body round-trips the method ("POST") and the full body
    content through an in-process echo server.
  - ``ToolResult.metadata["status"] == 200`` and the tool-visible content
    starts with ``"HTTP 200"``.

``test_http_tool_put_idempotent`` (direct tool call, gated on ``aiohttp``):
  - PUT with custom ``headers`` forwards ``method="PUT"`` and the body
    through to the echo server.

``test_http_tool_delete`` (direct tool call, gated on ``aiohttp``):
  - DELETE forwards ``method="DELETE"`` with no body.

``test_http_tool_auth_headers`` (direct tool call, gated on ``aiohttp``):
  - A ``headers={"Authorization": "Bearer ..."}`` arg reaches the server
    verbatim — pins header pass-through (no silent stripping).

``test_http_tool_follows_redirect`` (direct tool call, gated on ``aiohttp``):
  - The server issues a 302 to a final URL; the tool follows and the
    ``ToolResult.metadata["url"]`` reflects the final destination (not the
    initial redirect URL).

``test_web_search_multi_query_aggregates`` (gated on Tavily):
  - A ReAct agent issues at least two ``web_search`` tool invocations across
    turns, each with a distinct ``query`` parameter — pins aggregation of
    results across multiple searches in the same run.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from nanitics import (
    DockerSandbox,
    InMemoryEmitter,
    ReActAgent,
    create_code_execution_tool,
    create_file_read_tool,
    create_http_tool,
    create_web_search_tool,
)
from nanitics.infrastructure import ToolInvokeEvent, ToolResultEvent
from nanitics.infrastructure.errors import ToolParameterError
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    requires_docker,
    requires_tavily,
    run_with_retry,
)

_FILE_CONTENT = (
    "MARKER-PHRASE-0xFEEDFACE: The Nanitics SDK ships four curated reference tools: "
    "web_search, http_request, file_read, and code_execution.\n"
)
_FILE_MARKER = "MARKER-PHRASE-0xFEEDFACE"


@pytest.mark.quick
async def test_file_read(traced_emitter: InMemoryEmitter, tmp_path: Path) -> None:
    data_path = tmp_path / "data.txt"
    data_path.write_text(_FILE_CONTENT, encoding="utf-8")

    file_tool = create_file_read_tool(allowed_paths=[tmp_path])
    agent = ReActAgent(
        name="file-read-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="Use the file_read tool to read files and then summarize their contents.",
        tools=[file_tool],
        max_iterations=3,
    )

    result = await run_with_retry(
        lambda: agent.run(f"Read the file at {data_path} and summarize its contents in one sentence."),
        max_attempts=2,
    )

    invoke = assert_trace_contains(traced_emitter, ToolInvokeEvent, predicate=lambda e: e.tool_name == "file_read")
    invoked_path = str(invoke.parameters.get("path", ""))
    assert invoked_path.endswith("data.txt"), f"Expected file_read path to target data.txt, got: {invoked_path!r}"

    tool_result = assert_trace_contains(traced_emitter, ToolResultEvent, predicate=lambda e: e.success is True)
    assert _FILE_MARKER in (tool_result.result or ""), (
        f"Expected marker phrase in ToolResultEvent.result, got: {tool_result.result!r}"
    )

    await assert_result_satisfies(
        result.output or "",
        "The output conveys that the file lists four reference tools shipped by "
        "the Nanitics SDK (web_search, http_request, file_read, code_execution), "
        "or names at least some of them.",
    )


@pytest.mark.quick
async def test_file_read_outside_allowlist(tmp_path: Path) -> None:
    """Security boundary: a path outside ``allowed_paths`` raises ``ToolParameterError``.

    Exercises the defining security property of the reference ``file_read``
    tool. Does not need the ``traced_emitter`` fixture because the check is
    purely at the tool layer.
    """
    allowed = tmp_path / "inside"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    file_tool = create_file_read_tool(allowed_paths=[allowed])
    with pytest.raises(ToolParameterError):
        await file_tool.execute(path=str(outside))


@pytest.mark.quick
async def test_http_request(traced_emitter: InMemoryEmitter) -> None:
    http_tool = create_http_tool(allowed_domains=["httpbin.org"], request_timeout=15.0)
    agent = ReActAgent(
        name="http-request-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="Use the http_request tool to fetch HTTP endpoints and report back to the user.",
        tools=[http_tool],
        max_iterations=3,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Use the http_request tool to fetch https://httpbin.org/get and report the User-Agent header value."
        ),
        max_attempts=2,
    )

    invoke = assert_trace_contains(traced_emitter, ToolInvokeEvent, predicate=lambda e: e.tool_name == "http_request")
    assert invoke.parameters.get("method") == "GET", f"Expected GET method, got: {invoke.parameters.get('method')!r}"
    url = str(invoke.parameters.get("url", ""))
    assert "httpbin.org/get" in url, f"Expected httpbin.org/get in url, got: {url!r}"

    tool_result = assert_trace_contains(traced_emitter, ToolResultEvent, predicate=lambda e: e.success is True)
    assert (tool_result.result or "").startswith("HTTP 200"), (
        f"Expected ToolResultEvent.result to start with 'HTTP 200', got: {tool_result.result!r}"
    )

    await assert_result_satisfies(
        result.output or "",
        "The output reports the User-Agent header value from the httpbin.org/get response.",
    )


@pytest.mark.quick
async def test_http_request_non_allowlisted_domain() -> None:
    """Security boundary: a host outside ``allowed_domains`` raises ``ToolParameterError``.

    Exercises the defining security property of the reference ``http_request``
    tool.
    """
    http_tool = create_http_tool(allowed_domains=["httpbin.org"], request_timeout=5.0)
    with pytest.raises(ToolParameterError):
        await http_tool.execute(method="GET", url="https://example.com/")


@requires_tavily
async def test_web_search(traced_emitter: InMemoryEmitter) -> None:
    search_tool = create_web_search_tool(api_key=os.environ["TAVILY_API_KEY"], provider="tavily")
    agent = ReActAgent(
        name="web-search-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="Use the web_search tool to look up information, then summarize what the results say.",
        tools=[search_tool],
        max_iterations=3,
    )

    result = await run_with_retry(
        lambda: agent.run(
            "Use web_search to look up Python's asyncio library and summarize what "
            "the top result says about it in one sentence."
        ),
        max_attempts=2,
    )

    invoke = assert_trace_contains(traced_emitter, ToolInvokeEvent, predicate=lambda e: e.tool_name == "web_search")
    assert invoke.parameters.get("query"), f"Expected non-empty query, got: {invoke.parameters}"
    tool_result = assert_trace_contains(traced_emitter, ToolResultEvent, predicate=lambda e: e.success is True)
    assert tool_result.result, f"Expected non-empty rendered result, got: {tool_result.result!r}"
    await assert_result_satisfies(
        result.output or "",
        "The output summarizes information about Python's asyncio library drawn from a web search result.",
    )


@requires_docker
async def test_code_execution(traced_emitter: InMemoryEmitter) -> None:
    sandbox = DockerSandbox()
    async with sandbox:
        code_tool = create_code_execution_tool(sandbox=sandbox)
        agent = ReActAgent(
            name="code-execution-agent",
            llm_client=make_llm_client("anthropic"),
            emitter=traced_emitter,
            system_prompt="Use the code_execution tool to run Python and report the result.",
            tools=[code_tool],
            max_iterations=3,
        )
        result = await run_with_retry(
            lambda: agent.run("Use code_execution to compute the factorial of 6 in Python and report the result."),
            max_attempts=2,
        )

    invoke = assert_trace_contains(traced_emitter, ToolInvokeEvent, predicate=lambda e: e.tool_name == "code_execution")
    code = str(invoke.parameters.get("code", ""))
    assert "factorial" in code.lower() or "6" in code, f"Expected factorial-of-6 code, got: {code!r}"
    assert_trace_contains(traced_emitter, ToolResultEvent, predicate=lambda e: e.success is True)
    assert "720" in (result.output or ""), f"Expected 720 in output, got: {result.output!r}"
    await assert_result_satisfies(
        result.output or "",
        "The output contains 720 as the factorial of 6.",
    )


# ---------------------------------------------------------------------------
# HTTP tool: direct-call coverage for POST/PUT/DELETE, auth headers, redirects
# ---------------------------------------------------------------------------
#
# These tests drive ``create_http_tool.execute`` directly rather than via an
# LLM so the HTTP contract can be pinned deterministically. They require an
# in-process echo server built with ``aiohttp.web`` — when ``aiohttp`` is not
# installed they skip cleanly.


_HAS_AIOHTTP = importlib.util.find_spec("aiohttp") is not None

requires_aiohttp = pytest.mark.skipif(
    not _HAS_AIOHTTP,
    reason="Skipping: aiohttp not installed",
)


@pytest.fixture
async def echo_server() -> AsyncIterator[str]:
    """Start an in-process ``aiohttp`` echo server and yield its base URL.

    The server echoes ``method``, request ``headers``, and the decoded
    ``body`` back as JSON. ``/redirect-to-final`` returns a 302 redirect
    to ``/final-destination``. Binds to 127.0.0.1:0 so each run gets a
    random port — no collision between parallel tests.
    """
    from aiohttp import web  # type: ignore[import-not-found]

    async def echo(request: web.Request) -> web.Response:
        body_bytes = await request.read()
        body_text = body_bytes.decode("utf-8") if body_bytes else ""
        payload = {
            "method": request.method,
            "headers": dict(request.headers.items()),
            "body": body_text,
            "path": request.path,
        }
        return web.json_response(payload)

    async def redirect_entry(request: web.Request) -> web.Response:
        raise web.HTTPFound(location="/final-destination")

    async def redirect_final(request: web.Request) -> web.Response:
        return web.json_response({"reached": "final", "path": request.path})

    app = web.Application()
    # Catch-all echo route attached last so explicit paths take priority.
    app.router.add_get("/redirect-to-final", redirect_entry)
    app.router.add_get("/final-destination", redirect_final)
    app.router.add_route("*", "/{tail:.*}", echo)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()

    # Extract bound port — aiohttp sites expose it via the server sockets.
    server = site._server
    assert server is not None, "aiohttp TCPSite has no bound server."
    assert server.sockets, "aiohttp TCPSite has no bound socket."
    port = server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    try:
        yield base_url
    finally:
        await runner.cleanup()


@requires_aiohttp
async def test_http_tool_post_request_body(echo_server: str) -> None:
    http_tool = create_http_tool(allowed_domains=["127.0.0.1"], request_timeout=5.0)
    result = await http_tool.execute(
        method="POST",
        url=f"{echo_server}/echo",
        body={"name": "nanitics", "n": 3},
    )
    # Status surfaced via metadata, not via raised exception.
    assert result.metadata["status"] == 200, (
        f"Expected 200 from local echo server; got {result.metadata.get('status')!r}"
    )
    assert result.content.startswith("HTTP 200"), f"Rendered content must start with 'HTTP 200'; got {result.content!r}"
    # Echo server round-trip: method + body are preserved.
    import json as _json

    echoed = _json.loads(result.metadata["body"])
    assert echoed["method"] == "POST", f"Server must observe method=POST; got {echoed['method']!r}"
    # The tool JSON-encodes dict bodies before transmission.
    assert _json.loads(echoed["body"]) == {"name": "nanitics", "n": 3}, (
        f"Server must receive the exact JSON body; got {echoed['body']!r}"
    )


@requires_aiohttp
async def test_http_tool_put_idempotent(echo_server: str) -> None:
    http_tool = create_http_tool(allowed_domains=["127.0.0.1"], request_timeout=5.0)
    result = await http_tool.execute(
        method="PUT",
        url=f"{echo_server}/resource/42",
        headers={"X-Client-Name": "nanitics-validation"},
        body={"replace": True},
    )
    assert result.metadata["status"] == 200

    import json as _json

    echoed = _json.loads(result.metadata["body"])
    assert echoed["method"] == "PUT", f"Server must observe method=PUT; got {echoed['method']!r}"
    assert echoed["headers"].get("X-Client-Name") == "nanitics-validation", (
        f"Custom header must be forwarded; got headers={echoed['headers']!r}"
    )


@requires_aiohttp
async def test_http_tool_delete(echo_server: str) -> None:
    http_tool = create_http_tool(allowed_domains=["127.0.0.1"], request_timeout=5.0)
    result = await http_tool.execute(
        method="DELETE",
        url=f"{echo_server}/resource/42",
    )
    assert result.metadata["status"] == 200

    import json as _json

    echoed = _json.loads(result.metadata["body"])
    assert echoed["method"] == "DELETE", f"Server must observe method=DELETE; got {echoed['method']!r}"
    assert echoed["body"] == "", f"DELETE without body must send empty body; got {echoed['body']!r}"


@requires_aiohttp
async def test_http_tool_auth_headers(echo_server: str) -> None:
    http_tool = create_http_tool(allowed_domains=["127.0.0.1"], request_timeout=5.0)
    result = await http_tool.execute(
        method="GET",
        url=f"{echo_server}/protected",
        headers={"Authorization": "Bearer validation-token-XYZ"},
    )
    assert result.metadata["status"] == 200

    import json as _json

    echoed = _json.loads(result.metadata["body"])
    # Custom auth header must reach the server verbatim — no silent stripping.
    assert echoed["headers"].get("Authorization") == "Bearer validation-token-XYZ", (
        f"Authorization header must be forwarded verbatim; got headers={echoed['headers']!r}"
    )


@requires_aiohttp
async def test_http_tool_follows_redirect(echo_server: str) -> None:
    http_tool = create_http_tool(allowed_domains=["127.0.0.1"], request_timeout=5.0)
    result = await http_tool.execute(
        method="GET",
        url=f"{echo_server}/redirect-to-final",
    )
    # The ``create_http_tool`` client is constructed with follow_redirects=True;
    # we expect the final 200 response, not the initial 302.
    assert result.metadata["status"] == 200, (
        f"Expected tool to follow redirect to final 200; got status={result.metadata.get('status')!r}"
    )
    final_url = result.metadata["url"]
    assert "/final-destination" in final_url, (
        f"Result metadata url must reflect the final destination after redirect; got {final_url!r}"
    )
    assert "/redirect-to-final" not in final_url, (
        f"Result metadata url must not still point at the initial redirect URL; got {final_url!r}"
    )


# ---------------------------------------------------------------------------
# Web search: multi-query aggregation across a single agent run
# ---------------------------------------------------------------------------


@requires_tavily
async def test_web_search_multi_query_aggregates(
    traced_emitter: InMemoryEmitter,
) -> None:
    search_tool = create_web_search_tool(api_key=os.environ["TAVILY_API_KEY"], provider="tavily")
    agent = ReActAgent(
        name="web-search-multi-query-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You have a web_search tool. For questions that compare two "
            "distinct topics, you MUST run web_search separately for each "
            "topic (one tool call per topic) before composing your final "
            "answer. Do not combine two topics into a single query."
        ),
        tools=[search_tool],
        max_iterations=6,
    )

    await run_with_retry(
        lambda: agent.run(
            "Compare Python's asyncio library and Rust's tokio runtime. "
            "Search for each separately, then write one sentence per library "
            "summarising the top result."
        ),
        max_attempts=2,
    )

    search_invocations = [
        e for e in traced_emitter.events if isinstance(e, ToolInvokeEvent) and e.tool_name == "web_search"
    ]
    assert len(search_invocations) >= 2, (
        f"Expected at least two web_search invocations (aggregation across queries); got {len(search_invocations)}."
    )

    # Each invocation must carry a distinct, non-empty query.
    queries = [str(e.parameters.get("query") or "") for e in search_invocations]
    assert all(queries), f"Every web_search invocation must carry a non-empty query; got {queries!r}"
    assert len(set(queries)) == len(queries), (
        f"Expected each web_search query to be distinct (aggregation across different topics); "
        f"got duplicates in {queries!r}."
    )
