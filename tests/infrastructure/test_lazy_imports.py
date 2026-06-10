"""Tests for optional dependency lazy imports in infrastructure __init__ modules."""

import importlib
import importlib.util
import subprocess
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


class TestCompositionDurabilityLazyImports:
    def test_postgres_checkpoint_store_unavailable_sets_none(self) -> None:
        mod = _reload_without(
            "nanitics.composition.durability",
            ["nanitics.composition.durability.postgres_checkpoint_store"],
        )
        assert mod.PostgresCheckpointStore is None  # type: ignore[union-attr]
        assert mod.get_checkpoint_schema_sql is None  # type: ignore[union-attr]


class TestCompositionThreadsLazyImports:
    def test_postgres_thread_store_unavailable_sets_none(self) -> None:
        mod = _reload_without(
            "nanitics.composition.threads",
            ["nanitics.composition.threads.postgres_thread_store"],
        )
        assert mod.PostgresThreadStore is None  # type: ignore[union-attr]
        assert mod.get_thread_schema_sql is None  # type: ignore[union-attr]


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


class TestMCPExtraDetection:
    """The mcp subpackage maps the *missing-extra* case to ``None`` via an
    explicit ``find_spec`` check, and lets every other import failure (a
    circular import, a genuinely broken module) propagate rather than
    silently nulling out MCP support.
    """

    def test_extra_absent_sets_all_none(self) -> None:
        # Simulate the optional ``mcp`` distribution being absent: find_spec
        # returns None, so the three symbols re-export as None.
        import nanitics.infrastructure.mcp as mcp_pkg

        with patch.object(importlib.util, "find_spec", return_value=None):
            reloaded = importlib.reload(mcp_pkg)
        try:
            assert reloaded.MCPClient is None
            assert reloaded.MCPStdioParameters is None
            assert reloaded.MCPTool is None
        finally:
            importlib.reload(mcp_pkg)  # restore the real symbols

    def test_real_import_failure_propagates(self) -> None:
        # The extra is present (find_spec is truthy) but a real module is
        # broken: the ImportError must surface, not be swallowed into None.
        import nanitics.infrastructure.mcp as mcp_pkg

        try:
            with (
                patch.dict(sys.modules, {"nanitics.infrastructure.mcp._tool": None}),
                pytest.raises(ImportError),
            ):
                importlib.reload(mcp_pkg)
        finally:
            importlib.reload(mcp_pkg)  # restore a clean module


class TestMCPLazyReExportFromInfrastructure:
    """``nanitics.infrastructure`` re-exports the MCP symbols lazily (PEP 562
    ``__getattr__``) to avoid a circular import at load time, while keeping
    ``from nanitics.infrastructure import MCPClient`` working.
    """

    def test_lazy_reexport_matches_subpackage(self) -> None:
        import nanitics.infrastructure as infra
        import nanitics.infrastructure.mcp as mcp_pkg

        assert infra.MCPClient is mcp_pkg.MCPClient
        assert infra.MCPStdioParameters is mcp_pkg.MCPStdioParameters
        assert infra.MCPTool is mcp_pkg.MCPTool

    def test_unknown_attribute_raises(self) -> None:
        import nanitics.infrastructure as infra

        with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
            _ = infra.does_not_exist

    def test_extra_absent_propagates_none_through_lazy_reexport(self) -> None:
        import nanitics.infrastructure as infra
        import nanitics.infrastructure.mcp as mcp_pkg

        try:
            with patch.object(importlib.util, "find_spec", return_value=None):
                importlib.reload(mcp_pkg)
                # Resolved lazily through infrastructure.__getattr__.
                assert infra.MCPClient is None
        finally:
            # Restore the real symbols with find_spec unpatched, so the
            # reload re-imports the genuine classes rather than re-nulling.
            importlib.reload(mcp_pkg)


class TestImportOrderCycle:
    """Regression for the circular import that silently disabled MCP.

    Importing ``nanitics.safety`` before ``nanitics.infrastructure`` once
    tripped a cycle (safety → infrastructure → mcp → strategies → codeact →
    safety) whose ImportError was swallowed, nulling out the MCP symbols.
    Run in a fresh interpreter so the import order is the one under test.
    """

    def test_safety_imported_first_keeps_mcp_available(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import nanitics.safety\n"
                "from nanitics.infrastructure import MCPClient, MCPStdioParameters, MCPTool\n"
                "assert MCPClient is not None, 'MCPClient nulled by import-order cycle'\n"
                "assert MCPStdioParameters is not None\n"
                "assert MCPTool is not None\n"
                "print('ok')\n",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


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
