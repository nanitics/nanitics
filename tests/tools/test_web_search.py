"""Unit tests for :func:`nanitics.tools.web_search.create_web_search_tool`.

Covers:

- Tavily happy path (200 OK, 3 results).
- Brave happy path (200 OK with differently-shaped JSON).
- Unknown-provider construction raises ``ValueError``.
- Empty-API-key construction raises ``ValueError``.
- Pydantic parameter validation (empty query, ``max_results=0``,
  ``max_results>20``) raises ``ToolParameterError``.
- 4xx and 5xx responses map to ``ToolExecutionError``.
- Timeout maps to ``ToolTimeoutError``.
- Transport (connection) failure maps to ``ToolExecutionError``.
- Malformed JSON response maps to ``ToolExecutionError``.
- Result count respects the factory's ``max_results_default``.
- Per-call ``max_results`` override works.

All HTTP traffic is intercepted by :mod:`respx`; no real network is touched.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from nanitics.infrastructure.errors import (
    ToolExecutionError,
    ToolParameterError,
    ToolTimeoutError,
)
from nanitics.strategies.tools.protocol import Tool, ToolResult
from nanitics.tools.web_search import create_web_search_tool

# --- Fixtures and helpers ---------------------------------------------------

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _tavily_ok(results: list[dict[str, Any]] | None = None) -> httpx.Response:
    """Build a 200 OK Tavily response with three default results."""
    if results is None:
        results = [
            {"title": "Alpha", "url": "https://a.example.com", "content": "Alpha snippet", "score": 0.9},
            {"title": "Beta", "url": "https://b.example.com", "content": "Beta snippet", "score": 0.8},
            {"title": "Gamma", "url": "https://c.example.com", "content": "Gamma snippet", "score": 0.7},
        ]
    return httpx.Response(200, json={"query": "anything", "results": results})


def _brave_ok(results: list[dict[str, Any]] | None = None) -> httpx.Response:
    """Build a 200 OK Brave response with three default results."""
    if results is None:
        results = [
            {"title": "Delta", "url": "https://d.example.com", "description": "Delta desc"},
            {"title": "Epsilon", "url": "https://e.example.com", "description": "Epsilon desc"},
            {"title": "Zeta", "url": "https://f.example.com", "description": "Zeta desc"},
        ]
    return httpx.Response(
        200,
        json={"type": "search", "web": {"results": results}},
    )


# --- Construction ------------------------------------------------------------


class TestConstruction:
    def test_returns_tool_conforming_object(self) -> None:
        tool = create_web_search_tool(api_key="x")
        assert isinstance(tool, Tool)

    def test_default_provider_is_tavily(self) -> None:
        tool = create_web_search_tool(api_key="x")
        # Description should mention the provider for trace-consumer clarity.
        assert "tavily" in tool.schema.description.lower()

    def test_brave_provider_selectable(self) -> None:
        tool = create_web_search_tool(api_key="x", provider="brave")
        assert "brave" in tool.schema.description.lower()

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            create_web_search_tool(api_key="")

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            create_web_search_tool(api_key="x", provider="google")  # type: ignore[arg-type]

    def test_custom_name_and_description(self) -> None:
        tool = create_web_search_tool(
            api_key="x",
            name="my_search",
            description="Custom search description.",
        )
        assert tool.schema.name == "my_search"
        assert tool.schema.description == "Custom search description."


# --- Parameter validation ----------------------------------------------------


class TestParameterValidation:
    @pytest.mark.asyncio
    async def test_empty_query_raises_parameter_error(self) -> None:
        tool = create_web_search_tool(api_key="x")
        with pytest.raises(ToolParameterError):
            await tool.execute(query="")

    @pytest.mark.asyncio
    async def test_max_results_too_small_raises(self) -> None:
        tool = create_web_search_tool(api_key="x")
        with pytest.raises(ToolParameterError):
            await tool.execute(query="hello", max_results=0)

    @pytest.mark.asyncio
    async def test_max_results_too_large_raises(self) -> None:
        tool = create_web_search_tool(api_key="x")
        with pytest.raises(ToolParameterError):
            await tool.execute(query="hello", max_results=21)


# --- Tavily backend ---------------------------------------------------------


class TestTavily:
    @pytest.mark.asyncio
    @respx.mock
    async def test_happy_path_three_results(self) -> None:
        respx.post(TAVILY_URL).mock(return_value=_tavily_ok())
        tool = create_web_search_tool(api_key="secret", provider="tavily")
        result = await tool.execute(query="python sdks")
        assert isinstance(result, ToolResult)
        assert "Alpha" in result.content
        assert "https://a.example.com" in result.content
        assert "Alpha snippet" in result.content
        assert result.metadata["provider"] == "tavily"
        assert result.metadata["query"] == "python sdks"
        assert len(result.metadata["results"]) == 3
        first = result.metadata["results"][0]
        assert first["title"] == "Alpha"
        assert first["url"] == "https://a.example.com"
        assert first["snippet"] == "Alpha snippet"
        assert first["score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    @respx.mock
    async def test_request_body_contains_api_key_and_query(self) -> None:
        route = respx.post(TAVILY_URL).mock(return_value=_tavily_ok())
        tool = create_web_search_tool(api_key="secret", provider="tavily")
        await tool.execute(query="needle")
        assert route.called
        import json

        body = json.loads(route.calls.last.request.content.decode())
        assert body["api_key"] == "secret"
        assert body["query"] == "needle"
        assert body["max_results"] == 5  # default

    @pytest.mark.asyncio
    @respx.mock
    async def test_respects_max_results_default(self) -> None:
        route = respx.post(TAVILY_URL).mock(return_value=_tavily_ok())
        tool = create_web_search_tool(api_key="x", max_results_default=2)
        await tool.execute(query="q")
        import json

        body = json.loads(route.calls.last.request.content.decode())
        assert body["max_results"] == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_per_call_max_results_override(self) -> None:
        route = respx.post(TAVILY_URL).mock(return_value=_tavily_ok())
        tool = create_web_search_tool(api_key="x", max_results_default=2)
        await tool.execute(query="q", max_results=7)
        import json

        body = json.loads(route.calls.last.request.content.decode())
        assert body["max_results"] == 7

    @pytest.mark.asyncio
    @respx.mock
    async def test_4xx_maps_to_execution_error(self) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(401, text="invalid key"))
        tool = create_web_search_tool(api_key="x", provider="tavily")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert excinfo.value.tool_name == "web_search"
        assert "tavily" in excinfo.value.message.lower()
        assert "401" in excinfo.value.message

    @pytest.mark.asyncio
    @respx.mock
    async def test_5xx_maps_to_execution_error(self) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(503, text="unavailable"))
        tool = create_web_search_tool(api_key="x", provider="tavily")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "503" in excinfo.value.message

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_json_maps_to_execution_error(self) -> None:
        respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(
                200,
                content=b"not-a-json-body",
                headers={"content-type": "application/json"},
            )
        )
        tool = create_web_search_tool(api_key="x", provider="tavily")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "unexpected response" in excinfo.value.message.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_results_field_maps_to_execution_error(self) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"query": "q"}))
        tool = create_web_search_tool(api_key="x", provider="tavily")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "unexpected response" in excinfo.value.message.lower()


# --- Brave backend ---------------------------------------------------------


class TestBrave:
    @pytest.mark.asyncio
    @respx.mock
    async def test_happy_path_three_results(self) -> None:
        respx.get(BRAVE_URL).mock(return_value=_brave_ok())
        tool = create_web_search_tool(api_key="secret", provider="brave")
        result = await tool.execute(query="q")
        assert "Delta" in result.content
        assert "https://d.example.com" in result.content
        assert result.metadata["provider"] == "brave"
        assert len(result.metadata["results"]) == 3
        first = result.metadata["results"][0]
        assert first["title"] == "Delta"
        assert first["url"] == "https://d.example.com"
        assert first["snippet"] == "Delta desc"
        # Brave does not supply a per-result score.
        assert first["score"] is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_request_has_subscription_token_header_and_query(self) -> None:
        route = respx.get(BRAVE_URL).mock(return_value=_brave_ok())
        tool = create_web_search_tool(api_key="subkey", provider="brave")
        await tool.execute(query="needle", max_results=4)
        assert route.called
        req = route.calls.last.request
        assert req.headers["X-Subscription-Token"] == "subkey"
        assert req.headers["Accept"] == "application/json"
        assert req.url.params["q"] == "needle"
        assert req.url.params["count"] == "4"

    @pytest.mark.asyncio
    @respx.mock
    async def test_4xx_maps_to_execution_error(self) -> None:
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(403, text="forbidden"))
        tool = create_web_search_tool(api_key="x", provider="brave")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "brave" in excinfo.value.message.lower()
        assert "403" in excinfo.value.message

    @pytest.mark.asyncio
    @respx.mock
    async def test_5xx_maps_to_execution_error(self) -> None:
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(502, text="bad gateway"))
        tool = create_web_search_tool(api_key="x", provider="brave")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "502" in excinfo.value.message

    @pytest.mark.asyncio
    @respx.mock
    async def test_malformed_response_shape(self) -> None:
        # ``web`` key absent — tool cannot extract results.
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(200, json={"type": "search"}))
        tool = create_web_search_tool(api_key="x", provider="brave")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "unexpected response" in excinfo.value.message.lower()


# --- Transport errors -------------------------------------------------------


class TestTransportErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_timeout_maps_to_tool_timeout_error(self) -> None:
        respx.post(TAVILY_URL).mock(side_effect=httpx.TimeoutException("slow"))
        tool = create_web_search_tool(api_key="x", request_timeout=5.0)
        with pytest.raises(ToolTimeoutError) as excinfo:
            await tool.execute(query="q")
        assert excinfo.value.tool_name == "web_search"
        assert excinfo.value.timeout_seconds == 5.0

    @pytest.mark.asyncio
    @respx.mock
    async def test_connection_error_maps_to_execution_error(self) -> None:
        respx.post(TAVILY_URL).mock(side_effect=httpx.ConnectError("refused"))
        tool = create_web_search_tool(api_key="x")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert excinfo.value.tool_name == "web_search"
        assert "transport error" in excinfo.value.message.lower()


# --- Content rendering edge cases -------------------------------------------


class TestContentRendering:
    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_results_renders_no_results(self) -> None:
        respx.post(TAVILY_URL).mock(return_value=_tavily_ok(results=[]))
        tool = create_web_search_tool(api_key="x", provider="tavily")
        result = await tool.execute(query="q")
        assert result.content == "No results."
        assert result.metadata["results"] == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_long_snippet_is_clamped(self) -> None:
        long_snippet = "x" * 1000
        respx.post(TAVILY_URL).mock(
            return_value=_tavily_ok(
                results=[
                    {
                        "title": "Long",
                        "url": "https://long.example.com",
                        "content": long_snippet,
                        "score": 0.5,
                    }
                ]
            )
        )
        tool = create_web_search_tool(api_key="x")
        result = await tool.execute(query="q")
        # ``content`` is clamped but the full snippet remains in metadata.
        assert "x" * 500 in result.content
        assert "x" * 501 not in result.content
        assert result.metadata["results"][0]["snippet"] == long_snippet

    @pytest.mark.asyncio
    @respx.mock
    async def test_long_error_body_is_clamped_in_message(self) -> None:
        # Trigger the error-body clamp path (snippet preview in
        # ``ToolExecutionError.message`` is bounded).
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(500, text="z" * 1000))
        tool = create_web_search_tool(api_key="x", provider="tavily")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        # 200 chars of 'z' are retained, but not the full 1000.
        assert "z" * 200 in excinfo.value.message
        assert "z" * 201 not in excinfo.value.message


# --- Malformed-shape branches ------------------------------------------------


class TestMalformedShapes:
    @pytest.mark.asyncio
    @respx.mock
    async def test_tavily_results_not_a_list(self) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": "not-a-list"}))
        tool = create_web_search_tool(api_key="x", provider="tavily")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "not a list" in excinfo.value.message.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_tavily_result_entry_not_an_object(self) -> None:
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": ["not-an-object"]}))
        tool = create_web_search_tool(api_key="x", provider="tavily")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "not an object" in excinfo.value.message.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_brave_response_not_an_object(self) -> None:
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(200, json=["not", "a", "dict"]))
        tool = create_web_search_tool(api_key="x", provider="brave")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "unexpected response" in excinfo.value.message.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_brave_web_results_not_a_list(self) -> None:
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(200, json={"web": {"results": "not-a-list"}}))
        tool = create_web_search_tool(api_key="x", provider="brave")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "not a list" in excinfo.value.message.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_brave_result_entry_not_an_object(self) -> None:
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(200, json={"web": {"results": ["not-an-object"]}}))
        tool = create_web_search_tool(api_key="x", provider="brave")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "not an object" in excinfo.value.message.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_brave_malformed_json(self) -> None:
        respx.get(BRAVE_URL).mock(
            return_value=httpx.Response(
                200,
                content=b"not-a-json-body",
                headers={"content-type": "application/json"},
            )
        )
        tool = create_web_search_tool(api_key="x", provider="brave")
        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(query="q")
        assert "could not decode json" in excinfo.value.message.lower()
