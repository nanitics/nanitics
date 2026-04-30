"""Web-search reference tool: Tavily and Brave backends consumed by a ReActAgent.

Demonstrates ``create_web_search_tool`` — a curated, batteries-included tool that
satisfies the ordinary ``Tool`` protocol, dispatches through ``ToolRegistry`` the
same way a ``FunctionTool`` does, and emits ``ToolInvokeEvent`` /
``ToolResultEvent`` through the standard path.  No new registry, no new dispatch
path, no new event types.

Section 1 drives the Tavily backend through a ``MockLLMClient``-backed
``ReActAgent`` with ``respx`` intercepting the HTTPS call.  Section 2 does the
same against Brave, showing that the backend is a construction-time choice and
does not change the agent-side integration.  Section 3 is a commented-out
real-API block showing the production shape for readers who have a Tavily key.

Related guide: docs/guides/tools.md (see the "Reference Tools" section).
"""

from __future__ import annotations

import asyncio

import httpx
import respx

from examples.helpers import make_emitter, make_response
from nanitics import (
    MockLLMClient,
    ReActAgent,
    ToolCall,
    create_web_search_tool,
)
from nanitics.infrastructure import ToolInvokeEvent, ToolResultEvent

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def _tavily_payload() -> dict[str, object]:
    return {
        "query": "nanitics sdk",
        "results": [
            {
                "title": "Nanitics SDK",
                "url": "https://example.com/nanitics",
                "content": "A multi-agent SDK focused on curated reference tools.",
                "score": 0.94,
            },
            {
                "title": "Nanitics reference tools",
                "url": "https://example.com/tools",
                "content": "Four curated tools: web_search, http_request, file_read, code_execution.",
                "score": 0.88,
            },
            {
                "title": "Nanitics discussions",
                "url": "https://example.com/discussions",
                "content": "Community Q&A about building agents with the SDK.",
                "score": 0.71,
            },
        ],
    }


def _brave_payload() -> dict[str, object]:
    return {
        "type": "search",
        "web": {
            "results": [
                {
                    "title": "Nanitics on GitHub",
                    "url": "https://example.com/nanitics-github",
                    "description": "Open-source SDK repo.",
                },
                {
                    "title": "Getting started with Nanitics",
                    "url": "https://example.com/getting-started",
                    "description": "Install and run your first agent.",
                },
            ]
        },
    }


async def main() -> None:
    # --- Section 1: Tavily backend via ReActAgent (always runs) ---
    print("--- Section 1: Tavily backend via ReActAgent (hermetic) ---")

    with respx.mock(assert_all_called=True) as respx_router:
        respx_router.post(TAVILY_URL).mock(return_value=httpx.Response(200, json=_tavily_payload()))

        tavily_tool = create_web_search_tool(api_key="fake-tavily-key", provider="tavily")
        assert tavily_tool.schema.name == "web_search"
        assert "tavily" in tavily_tool.schema.description.lower()

        llm = MockLLMClient(
            responses=[
                make_response(
                    "Let me look that up.",
                    tool_calls=[ToolCall(id="tc-1", name="web_search", arguments={"query": "nanitics sdk"})],
                    stop_reason="tool_use",
                ),
                make_response("Nanitics is a multi-agent SDK with curated reference tools."),
            ]
        )
        emitter = make_emitter("web-search-tavily")

        agent = ReActAgent(
            name="search-agent",
            llm_client=llm,
            emitter=emitter,
            system_prompt="Use the web_search tool to answer the user.",
            tools=[tavily_tool],
        )
        result = await agent.run("What is Nanitics?")

    assert result.termination_reason == "complete"
    assert result.output is not None

    invokes = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
    results = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
    assert [e.tool_name for e in invokes] == ["web_search"]
    assert results[0].success is True
    assert results[0].result is not None
    assert "Nanitics SDK" in results[0].result  # rendered bullet list

    print(f"  Output: {result.output}")
    print(f"  Tool invoked: {invokes[0].tool_name}({invokes[0].parameters})")
    print(f"  Events: {len(invokes)} invoke, {len(results)} result")
    print("✓ Tavily-backed web_search dispatches through the standard ToolRegistry path")

    # --- Section 2: Brave backend via ReActAgent (always runs) ---
    print("\n--- Section 2: Brave backend via ReActAgent (hermetic) ---")

    with respx.mock(assert_all_called=True) as respx_router:
        respx_router.get(BRAVE_URL).mock(return_value=httpx.Response(200, json=_brave_payload()))

        brave_tool = create_web_search_tool(api_key="fake-brave-key", provider="brave")
        assert "brave" in brave_tool.schema.description.lower()

        llm = MockLLMClient(
            responses=[
                make_response(
                    "Searching.",
                    tool_calls=[
                        ToolCall(
                            id="tc-2",
                            name="web_search",
                            arguments={"query": "nanitics getting started", "max_results": 2},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                make_response("Two links cover installation and first-run."),
            ]
        )
        emitter = make_emitter("web-search-brave")

        agent = ReActAgent(
            name="search-agent",
            llm_client=llm,
            emitter=emitter,
            system_prompt="Use the web_search tool to answer the user.",
            tools=[brave_tool],
        )
        result = await agent.run("How do I get started with Nanitics?")

    assert result.termination_reason == "complete"
    results = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
    assert results[0].success is True

    # Drive the Brave tool directly to inspect the structured metadata.  The
    # Brave backend normalizes the provider's ``description`` field into the
    # common ``snippet`` shape so application code can consume both backends
    # through one schema.
    with respx.mock() as respx_router:
        respx_router.get(BRAVE_URL).mock(return_value=httpx.Response(200, json=_brave_payload()))
        direct = await brave_tool.execute(query="nanitics getting started")
    assert direct.metadata["provider"] == "brave"
    first = direct.metadata["results"][0]
    assert first["snippet"] == "Open-source SDK repo."
    assert first["score"] is None  # Brave does not supply per-result scores

    print(f"  Output: {result.output}")
    print(f"  First normalized result: title={first['title']!r}, snippet={first['snippet']!r}")
    print("✓ Brave-backed web_search normalizes into the same shape as Tavily")

    # --- Section 3: Real Tavily API (commented out — requires a real key) ---
    print("\n--- Section 3: Real Tavily API (commented out — requires a real key) ---")
    print("  See the source of this file for the runnable block.")
    # Obtain a Tavily API key from https://tavily.com and set TAVILY_API_KEY:
    #
    #     export TAVILY_API_KEY=tvly-xxxxxxxxxxxx
    #
    # --------------------------------------------------------------------------
    # import os
    # from nanitics import AnthropicLLMClient
    #
    # real_llm = AnthropicLLMClient(model="claude-haiku-4-5")  # reads ANTHROPIC_API_KEY
    # real_tool = create_web_search_tool(api_key=os.environ["TAVILY_API_KEY"])
    # agent = ReActAgent(
    #     name="real-search",
    #     llm_client=real_llm,
    #     emitter=make_emitter("real-tavily"),
    #     system_prompt="Use web_search to research, then cite the URLs.",
    #     tools=[real_tool],
    # )
    # result = await agent.run("Summarize today's top AI-safety news in three bullets.")
    # print(result.output)
    # --------------------------------------------------------------------------

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
