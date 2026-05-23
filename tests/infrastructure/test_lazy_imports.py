"""Tests for optional dependency lazy imports in infrastructure __init__ modules."""

import importlib
import sys
from unittest.mock import patch

import pytest


def _reload_without(module_path: str, blocked_modules: list[str]) -> object:
    """Reload a module with certain imports blocked, return the reloaded module.

    Snapshots and restores ``sys.modules`` entries AND parent-package
    attributes for the reloaded module and every blocked submodule.  The
    parent-attribute restore matters because ``importlib.import_module``
    has a documented side effect of setting the imported submodule as an
    attribute on its parent package (e.g. importing ``a.b`` sets
    ``a.b = <module>`` on package ``a``).  Without restoration, the fresh
    module created during the reload leaks into the parent's namespace
    even after ``sys.modules`` is reverted, leaving callers walking the
    dotted-attribute chain (``pytest``'s ``monkeypatch.setattr`` resolver,
    most notably) on a stale graph that no longer matches sys.modules.
    """
    sentinel = object()
    targets = [module_path, *blocked_modules]
    saved_modules = {m: sys.modules.pop(m, sentinel) for m in targets}

    # Snapshot the parent-package attribute for each target so we can
    # restore it after the reload (which may overwrite the parent's
    # binding with a fresh module).
    saved_parent_attrs: dict[str, tuple[object, object]] = {}
    for m in targets:
        parent_name, _, attr = m.rpartition(".")
        if not parent_name:
            continue
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        saved_parent_attrs[m] = (parent, getattr(parent, attr, sentinel))

    try:
        with patch.dict(sys.modules, dict.fromkeys(blocked_modules)):
            return importlib.import_module(module_path)
    finally:
        for m, val in saved_modules.items():
            if val is sentinel:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = val
        for m, (parent, original) in saved_parent_attrs.items():
            _, _, attr = m.rpartition(".")
            if original is sentinel:
                # Parent had no such attribute before the reload — remove
                # any binding the reload introduced.
                if hasattr(parent, attr):
                    delattr(parent, attr)
            else:
                setattr(parent, attr, original)


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


class TestReloadWithoutHelperRestoresParentAttributes:
    """Regression guard for ``_reload_without`` parent-attribute restoration.

    ``importlib.import_module`` has a documented side effect: importing
    ``a.b`` binds ``b`` as an attribute on package ``a``.  If the helper
    only restores ``sys.modules`` after a reload (the historical bug),
    the freshly-loaded module leaks into the parent's namespace.  Callers
    that walk dotted-attribute chains afterwards (notably pytest's
    ``monkeypatch.setattr`` resolver) then traverse a graph that no
    longer agrees with ``sys.modules``, producing order-dependent
    ``AttributeError``s.
    """

    def test_parent_attribute_points_to_original_after_reload(self) -> None:
        import nanitics
        import nanitics.infrastructure

        original_infrastructure = nanitics.infrastructure
        original_sys_modules_entry = sys.modules["nanitics.infrastructure"]

        _reload_without(
            "nanitics.infrastructure",
            ["nanitics.infrastructure.mcp.client", "nanitics.infrastructure.mcp._tool"],
        )

        assert nanitics.infrastructure is original_infrastructure, (
            "Parent package's `infrastructure` attribute leaked to the fresh module"
        )
        assert sys.modules["nanitics.infrastructure"] is original_sys_modules_entry, (
            "sys.modules entry diverged from the parent-package attribute"
        )

    def test_parent_attribute_for_submodule_target(self) -> None:
        # When the reloaded module itself is a submodule, the parent's
        # attribute for it must be restored too.
        import nanitics.infrastructure
        import nanitics.infrastructure.llm

        original = nanitics.infrastructure.llm
        _reload_without(
            "nanitics.infrastructure.llm",
            ["nanitics.infrastructure.llm.mistral"],
        )
        assert nanitics.infrastructure.llm is original
