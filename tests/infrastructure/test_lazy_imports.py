"""Tests for optional dependency lazy imports in infrastructure __init__ modules."""

import importlib
import sys
from unittest.mock import patch

import pytest


def _reload_without(module_path: str, blocked_modules: list[str]) -> object:
    """Reload a module with certain imports blocked, return the reloaded module."""
    sentinel = object()
    saved = {m: sys.modules.pop(m, sentinel) for m in [module_path, *blocked_modules]}
    try:
        with patch.dict(sys.modules, dict.fromkeys(blocked_modules)):
            return importlib.import_module(module_path)
    finally:
        for m, val in saved.items():
            if val is sentinel:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = val


class TestLLMLazyImports:
    def test_mistral_unavailable_sets_none(self) -> None:
        mod = _reload_without(
            "nanitics.infrastructure.llm",
            ["nanitics.infrastructure.llm.mistral"],
        )
        assert mod.MistralLLMClient is None  # type: ignore[union-attr]

    def test_litellm_unavailable_sets_none(self) -> None:
        mod = _reload_without(
            "nanitics.infrastructure.llm",
            ["nanitics.infrastructure.llm.litellm"],
        )
        assert mod.LiteLLMClient is None  # type: ignore[union-attr]

    def test_optional_providers_unavailable_set_none(self) -> None:
        mod = _reload_without(
            "nanitics.infrastructure.llm",
            [
                "nanitics.infrastructure.llm.litellm",
                "nanitics.infrastructure.llm.mistral",
            ],
        )
        assert mod.LiteLLMClient is None  # type: ignore[union-attr]
        assert mod.MistralLLMClient is None  # type: ignore[union-attr]


class TestEmbeddingsLazyImports:
    def test_voyage_unavailable_sets_none(self) -> None:
        mod = _reload_without(
            "nanitics.infrastructure.embeddings",
            ["nanitics.infrastructure.embeddings.voyage"],
        )
        assert mod.VoyageEmbeddingClient is None  # type: ignore[union-attr]


class TestObservabilityLazyImports:
    def test_postgres_trace_store_unavailable_sets_none(self) -> None:
        mod = _reload_without(
            "nanitics.infrastructure.observability",
            ["nanitics.infrastructure.observability.postgres_store"],
        )
        assert mod.PostgresTraceStore is None  # type: ignore[union-attr]
        assert mod.get_schema_sql is None  # type: ignore[union-attr]


class TestCollaborationLazyImports:
    def test_postgres_hitl_store_unavailable_sets_none(self) -> None:
        mod = _reload_without(
            "nanitics.collaboration",
            ["nanitics.collaboration.postgres_hitl_store"],
        )
        assert mod.PostgresHitlRequestStore is None  # type: ignore[union-attr]
        assert mod.get_hitl_schema_sql is None  # type: ignore[union-attr]


class TestToolsLazyImports:
    def test_http_tools_unavailable_raises_on_call(self) -> None:
        # Block the leaf http-based tool modules so nanitics.tools exercises
        # its try/except fallback: the HTTP-based factory names resolve to
        # stubs that raise ImportError with an install hint on first call.
        # The file_read and code_execution factories stay callable.
        mod = _reload_without(
            "nanitics.tools",
            [
                "nanitics.tools.http",
                "nanitics.tools.web_search",
            ],
        )
        assert callable(mod.create_file_read_tool)  # type: ignore[union-attr]
        assert callable(mod.create_code_execution_tool)  # type: ignore[union-attr]

        with pytest.raises(ImportError, match="http-tools"):
            mod.create_http_tool(allow_any_domain=True)  # type: ignore[union-attr]
        with pytest.raises(ImportError, match="http-tools"):
            mod.create_web_search_tool(api_key="x")  # type: ignore[union-attr]


class TestMCPLazyImports:
    def test_mcp_unavailable_sets_all_none(self) -> None:
        mod = _reload_without(
            "nanitics.infrastructure.mcp",
            [
                "nanitics.infrastructure.mcp.client",
                "nanitics.infrastructure.mcp._tool",
            ],
        )
        assert mod.MCPClient is None  # type: ignore[union-attr]
        assert mod.MCPStdioParameters is None  # type: ignore[union-attr]
        assert mod.MCPTool is None  # type: ignore[union-attr]

    def test_mcp_unavailable_propagates_to_infrastructure(self) -> None:
        # Drop the cached mcp subpackage so reloading nanitics.infrastructure
        # re-imports it; block the leaf modules so the subpackage's __init__
        # exercises its try/except and re-exports the three names as None.
        sentinel = object()
        saved_mcp = sys.modules.pop("nanitics.infrastructure.mcp", sentinel)
        try:
            mod = _reload_without(
                "nanitics.infrastructure",
                [
                    "nanitics.infrastructure.mcp.client",
                    "nanitics.infrastructure.mcp._tool",
                ],
            )
            assert mod.MCPClient is None  # type: ignore[union-attr]
            assert mod.MCPStdioParameters is None  # type: ignore[union-attr]
            assert mod.MCPTool is None  # type: ignore[union-attr]
        finally:
            if saved_mcp is sentinel:
                sys.modules.pop("nanitics.infrastructure.mcp", None)
            else:
                sys.modules["nanitics.infrastructure.mcp"] = saved_mcp  # type: ignore[assignment]
