"""Unit tests for :func:`nanitics.tools.http.create_http_tool`.

Covers:

- GET / POST / PUT / DELETE happy paths.
- Custom headers merge with factory ``default_headers`` (request wins).
- Query params forwarded.
- Disallowed domain raises ``ToolParameterError``.
- ``allow_any_domain=True`` bypasses the host check.
- Constructor without domains and without ``allow_any_domain`` raises
  ``ValueError``.
- 4xx / 5xx statuses are returned via ``metadata.status`` — no exception.
- Timeout maps to ``ToolTimeoutError``.
- Transport failure maps to ``ToolExecutionError``.
- Body greater than ``max_response_bytes`` is truncated; ``truncated=True``
  and ``bytes_read`` are surfaced in metadata.
- Redirects are followed.
- Malformed URL raises ``ToolParameterError`` via Pydantic.
- Default ``max_response_bytes`` is 1 MiB.

All HTTP traffic is intercepted by :mod:`respx`; no real network is touched.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from nanitics.core.tools.protocol import Tool, ToolResult
from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolParameterError,
    ToolTimeoutError,
)
from nanitics.tools.http import create_http_tool

# --- Construction ------------------------------------------------------------


class TestConstruction:
    def test_returns_tool_conforming_object(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        assert isinstance(tool, Tool)

    def test_default_name_and_description_reference_tool(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        assert tool.schema.name == "http_request"
        # Default description should hint at methods and size limit for
        # trace consumers.
        assert "http" in tool.schema.description.lower()

    def test_custom_name_and_description(self) -> None:
        tool = create_http_tool(
            allowed_domains=["example.com"],
            name="my_http",
            description="Custom HTTP description.",
        )
        assert tool.schema.name == "my_http"
        assert tool.schema.description == "Custom HTTP description."

    def test_no_domains_and_no_any_raises(self) -> None:
        with pytest.raises(ValueError, match="allowed_domains"):
            create_http_tool()

    def test_empty_domains_and_no_any_raises(self) -> None:
        with pytest.raises(ValueError, match="allowed_domains"):
            create_http_tool(allowed_domains=[])

    def test_allow_any_domain_skips_domains_check(self) -> None:
        # Construction succeeds with no allowed_domains when
        # allow_any_domain=True.
        tool = create_http_tool(allow_any_domain=True)
        assert isinstance(tool, Tool)

    def test_default_max_response_bytes_is_one_mib(self) -> None:
        # Smoke test: default is 1 MiB as documented.
        tool = create_http_tool(allowed_domains=["example.com"])
        # Introspect via tool description so we don't reach into private
        # state unnecessarily.  The exact default is exercised by the
        # truncation tests below.
        assert tool.schema.name == "http_request"


# --- Parameter validation ----------------------------------------------------


class TestParameterValidation:
    @pytest.mark.asyncio
    async def test_missing_method_raises_parameter_error(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError):
            await tool.execute(url="https://example.com/")

    @pytest.mark.asyncio
    async def test_unknown_method_raises_parameter_error(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError):
            await tool.execute(method="PATCH", url="https://example.com/")

    @pytest.mark.asyncio
    async def test_missing_url_raises_parameter_error(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError):
            await tool.execute(method="GET")

    @pytest.mark.asyncio
    async def test_malformed_url_raises_parameter_error(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError):
            await tool.execute(method="GET", url="not a url")


# --- Domain allow-listing ----------------------------------------------------


class TestDomainAllowList:
    @pytest.mark.asyncio
    @respx.mock
    async def test_host_not_in_allowed_domains_raises_parameter_error(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError) as excinfo:
            await tool.execute(method="GET", url="https://evil.com/")
        assert excinfo.value.tool_name == "http_request"
        assert "allowed_domains" in (excinfo.value.reason or "")
        assert "evil.com" in (excinfo.value.reason or "")

    @pytest.mark.asyncio
    @respx.mock
    async def test_host_match_is_case_insensitive(self) -> None:
        respx.get("https://Example.COM/foo").mock(return_value=httpx.Response(200, text="ok"))
        tool = create_http_tool(allowed_domains=["EXAMPLE.com"])
        result = await tool.execute(method="GET", url="https://Example.COM/foo")
        assert result.metadata["status"] == 200

    @pytest.mark.asyncio
    @respx.mock
    async def test_subdomain_is_not_allowed_by_default(self) -> None:
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError):
            await tool.execute(method="GET", url="https://api.example.com/foo")

    @pytest.mark.asyncio
    @respx.mock
    async def test_allow_any_domain_bypasses_check(self) -> None:
        respx.get("https://anywhere.example.test/path").mock(return_value=httpx.Response(200, text="ok"))
        tool = create_http_tool(allow_any_domain=True)
        result = await tool.execute(method="GET", url="https://anywhere.example.test/path")
        assert result.metadata["status"] == 200


# --- HTTP method happy paths -------------------------------------------------


class TestMethods:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_happy_path(self) -> None:
        respx.get("https://example.com/get").mock(
            return_value=httpx.Response(200, text="hello", headers={"x-test": "1"})
        )
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="GET", url="https://example.com/get")
        assert isinstance(result, ToolResult)
        assert result.metadata["status"] == 200
        assert result.metadata["body"] == "hello"
        assert result.metadata["headers"]["x-test"] == "1"
        assert result.metadata["truncated"] is False
        assert result.metadata["bytes_read"] == len(b"hello")
        assert result.metadata["url"] == "https://example.com/get"
        assert "HTTP 200" in result.content
        assert "hello" in result.content

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_with_json_body(self) -> None:
        route = respx.post("https://example.com/post").mock(return_value=httpx.Response(201, json={"id": 42}))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(
            method="POST",
            url="https://example.com/post",
            body={"name": "alice"},
        )
        assert result.metadata["status"] == 201
        assert route.called
        import json as _json

        sent = _json.loads(route.calls.last.request.content.decode())
        assert sent == {"name": "alice"}
        # JSON bodies carry an application/json content-type header.
        assert route.calls.last.request.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_with_string_body(self) -> None:
        route = respx.post("https://example.com/post").mock(return_value=httpx.Response(200, text="ok"))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(
            method="POST",
            url="https://example.com/post",
            body="raw-string-body",
        )
        assert result.metadata["status"] == 200
        assert route.calls.last.request.content == b"raw-string-body"

    @pytest.mark.asyncio
    @respx.mock
    async def test_put_happy_path(self) -> None:
        respx.put("https://example.com/put").mock(return_value=httpx.Response(204))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="PUT", url="https://example.com/put", body={"x": 1})
        assert result.metadata["status"] == 204

    @pytest.mark.asyncio
    @respx.mock
    async def test_delete_happy_path(self) -> None:
        respx.delete("https://example.com/del").mock(return_value=httpx.Response(200, text="bye"))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="DELETE", url="https://example.com/del")
        assert result.metadata["status"] == 200
        assert result.metadata["body"] == "bye"


# --- Headers and query params ------------------------------------------------


class TestHeadersAndQueryParams:
    @pytest.mark.asyncio
    @respx.mock
    async def test_default_headers_applied(self) -> None:
        route = respx.get("https://example.com/h").mock(return_value=httpx.Response(200))
        tool = create_http_tool(
            allowed_domains=["example.com"],
            default_headers={"Authorization": "Bearer abc", "X-Shared": "yes"},
        )
        await tool.execute(method="GET", url="https://example.com/h")
        req = route.calls.last.request
        assert req.headers["authorization"] == "Bearer abc"
        assert req.headers["x-shared"] == "yes"

    @pytest.mark.asyncio
    @respx.mock
    async def test_request_headers_win_over_default_headers(self) -> None:
        route = respx.get("https://example.com/h").mock(return_value=httpx.Response(200))
        tool = create_http_tool(
            allowed_domains=["example.com"],
            default_headers={"Authorization": "Bearer default"},
        )
        await tool.execute(
            method="GET",
            url="https://example.com/h",
            headers={"Authorization": "Bearer override", "X-Extra": "e"},
        )
        req = route.calls.last.request
        assert req.headers["authorization"] == "Bearer override"
        assert req.headers["x-extra"] == "e"

    @pytest.mark.asyncio
    @respx.mock
    async def test_query_params_forwarded(self) -> None:
        route = respx.get("https://example.com/q").mock(return_value=httpx.Response(200))
        tool = create_http_tool(allowed_domains=["example.com"])
        await tool.execute(
            method="GET",
            url="https://example.com/q",
            query_params={"a": "1", "b": "two"},
        )
        req = route.calls.last.request
        assert req.url.params["a"] == "1"
        assert req.url.params["b"] == "two"


# --- Status-code passthrough -------------------------------------------------


class TestStatusCodePassthrough:
    @pytest.mark.asyncio
    @respx.mock
    async def test_4xx_returned_via_metadata_no_exception(self) -> None:
        respx.get("https://example.com/x").mock(return_value=httpx.Response(404, text="not found"))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="GET", url="https://example.com/x")
        assert result.metadata["status"] == 404
        assert result.metadata["body"] == "not found"
        assert "HTTP 404" in result.content

    @pytest.mark.asyncio
    @respx.mock
    async def test_5xx_returned_via_metadata_no_exception(self) -> None:
        respx.get("https://example.com/x").mock(return_value=httpx.Response(503, text="busy"))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="GET", url="https://example.com/x")
        assert result.metadata["status"] == 503
        assert "HTTP 503" in result.content


# --- Transport errors --------------------------------------------------------


class TestTransportErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_timeout_maps_to_tool_timeout_error(self) -> None:
        respx.get("https://example.com/t").mock(side_effect=httpx.TimeoutException("slow"))
        tool = create_http_tool(allowed_domains=["example.com"], request_timeout=7.5)
        with pytest.raises(ToolTimeoutError) as excinfo:
            await tool.execute(method="GET", url="https://example.com/t")
        assert excinfo.value.tool_name == "http_request"
        assert excinfo.value.timeout_seconds == 7.5

    @pytest.mark.asyncio
    @respx.mock
    async def test_connection_error_maps_to_execution_error(self) -> None:
        respx.get("https://example.com/t").mock(side_effect=httpx.ConnectError("refused"))
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(method="GET", url="https://example.com/t")
        assert excinfo.value.tool_name == "http_request"
        assert "transport" in excinfo.value.message.lower()


# --- Truncation --------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    @respx.mock
    async def test_body_over_max_response_bytes_is_truncated(self) -> None:
        large = "A" * 2048
        respx.get("https://example.com/big").mock(return_value=httpx.Response(200, text=large))
        tool = create_http_tool(allowed_domains=["example.com"], max_response_bytes=1000)
        result = await tool.execute(method="GET", url="https://example.com/big")
        assert result.metadata["truncated"] is True
        assert result.metadata["bytes_read"] == 1000
        assert len(result.metadata["body"]) == 1000
        assert "A" * 1000 in result.content

    @pytest.mark.asyncio
    @respx.mock
    async def test_body_at_exactly_max_response_bytes_is_not_truncated(self) -> None:
        payload = "X" * 512
        respx.get("https://example.com/exact").mock(return_value=httpx.Response(200, text=payload))
        tool = create_http_tool(allowed_domains=["example.com"], max_response_bytes=512)
        result = await tool.execute(method="GET", url="https://example.com/exact")
        assert result.metadata["truncated"] is False
        assert result.metadata["bytes_read"] == 512

    @pytest.mark.asyncio
    @respx.mock
    async def test_default_max_response_bytes_is_one_mib(self) -> None:
        # Body of 1 MiB + 1 triggers truncation at 1 MiB (the default).
        big = "Z" * (1_048_576 + 1)
        respx.get("https://example.com/default").mock(return_value=httpx.Response(200, text=big))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="GET", url="https://example.com/default")
        assert result.metadata["truncated"] is True
        assert result.metadata["bytes_read"] == 1_048_576


# --- Redirects ---------------------------------------------------------------


class TestNonUtf8Body:
    @pytest.mark.asyncio
    @respx.mock
    async def test_non_utf8_body_decoded_with_replacement(self) -> None:
        # 0xff is an invalid UTF-8 start byte — exercise the fallback.
        respx.get("https://example.com/binary").mock(return_value=httpx.Response(200, content=b"\xffhello"))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="GET", url="https://example.com/binary")
        assert result.metadata["status"] == 200
        # Invalid byte is replaced with U+FFFD; the rest survives.
        assert "hello" in result.metadata["body"]
        assert "\ufffd" in result.metadata["body"]


class TestRedirects:
    @pytest.mark.asyncio
    @respx.mock
    async def test_redirects_followed(self) -> None:
        # respx matches absolute URLs; simulate a 302 then 200 follow-up.
        respx.get("https://example.com/start").mock(
            return_value=httpx.Response(302, headers={"Location": "https://example.com/dest"})
        )
        respx.get("https://example.com/dest").mock(return_value=httpx.Response(200, text="arrived"))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="GET", url="https://example.com/start")
        assert result.metadata["status"] == 200
        assert result.metadata["body"] == "arrived"
        assert result.metadata["url"] == "https://example.com/dest"

    @pytest.mark.asyncio
    @respx.mock
    async def test_redirect_to_disallowed_host_raises(self) -> None:
        respx.get("https://example.com/start").mock(
            return_value=httpx.Response(302, headers={"Location": "https://evil.com/dest"})
        )
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError) as excinfo:
            await tool.execute(method="GET", url="https://example.com/start")
        assert "evil.com" in (excinfo.value.reason or "")
        assert "redirect target host" in (excinfo.value.reason or "")

    @pytest.mark.asyncio
    @respx.mock
    async def test_redirect_to_allowed_host_succeeds(self) -> None:
        respx.get("https://api.example.com/start").mock(
            return_value=httpx.Response(302, headers={"Location": "https://data.example.net/final"})
        )
        respx.get("https://data.example.net/final").mock(return_value=httpx.Response(200, text="ok"))
        tool = create_http_tool(allowed_domains=["api.example.com", "data.example.net"])
        result = await tool.execute(method="GET", url="https://api.example.com/start")
        assert result.metadata["status"] == 200
        assert result.metadata["url"] == "https://data.example.net/final"

    @pytest.mark.asyncio
    @respx.mock
    async def test_redirect_limit_exceeded_raises(self) -> None:
        respx.get("https://example.com/a").mock(
            return_value=httpx.Response(302, headers={"Location": "https://example.com/b"})
        )
        respx.get("https://example.com/b").mock(
            return_value=httpx.Response(302, headers={"Location": "https://example.com/c"})
        )
        respx.get("https://example.com/c").mock(return_value=httpx.Response(200, text="done"))
        tool = create_http_tool(allowed_domains=["example.com"], max_redirects=1)
        with pytest.raises(ToolParameterError) as excinfo:
            await tool.execute(method="GET", url="https://example.com/a")
        assert "redirect limit" in (excinfo.value.reason or "")

    @pytest.mark.asyncio
    @respx.mock
    async def test_allow_any_domain_follows_across_hosts_with_hop_cap(self) -> None:
        respx.get("https://h1.example/").mock(
            return_value=httpx.Response(302, headers={"Location": "https://h2.example/"})
        )
        respx.get("https://h2.example/").mock(
            return_value=httpx.Response(302, headers={"Location": "https://h3.example/"})
        )
        respx.get("https://h3.example/").mock(return_value=httpx.Response(200, text="landed"))
        tool = create_http_tool(allow_any_domain=True, max_redirects=5)
        result = await tool.execute(method="GET", url="https://h1.example/")
        assert result.metadata["status"] == 200
        assert result.metadata["url"] == "https://h3.example/"

    @pytest.mark.asyncio
    @respx.mock
    async def test_max_redirects_zero_disables_following(self) -> None:
        respx.get("https://example.com/start").mock(
            return_value=httpx.Response(302, headers={"Location": "https://example.com/dest"})
        )
        tool = create_http_tool(allowed_domains=["example.com"], max_redirects=0)
        result = await tool.execute(method="GET", url="https://example.com/start")
        # A 3xx is surfaced as-is when redirect following is disabled.
        assert result.metadata["status"] == 302
        assert result.metadata["url"] == "https://example.com/start"

    @pytest.mark.asyncio
    @respx.mock
    async def test_invalid_redirect_target_raises(self) -> None:
        # A Location of "http://" resolves to an absolute URL with no host,
        # which is not safe to follow.
        respx.get("https://example.com/start").mock(return_value=httpx.Response(302, headers={"Location": "http://"}))
        tool = create_http_tool(allowed_domains=["example.com"])
        with pytest.raises(ToolParameterError) as excinfo:
            await tool.execute(method="GET", url="https://example.com/start")
        assert "redirect target URL is invalid" in (excinfo.value.reason or "")

    @pytest.mark.asyncio
    @respx.mock
    async def test_negative_max_redirects_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="max_redirects"):
            create_http_tool(allowed_domains=["example.com"], max_redirects=-1)

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_redirect_response_unchanged(self) -> None:
        respx.get("https://example.com/ok").mock(return_value=httpx.Response(200, text="fine"))
        tool = create_http_tool(allowed_domains=["example.com"])
        result = await tool.execute(method="GET", url="https://example.com/ok")
        assert result.metadata["status"] == 200
        assert result.metadata["body"] == "fine"
