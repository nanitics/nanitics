"""Per-subpackage `__all__` consistency tests.

Replaces the historical flat-surface invariance test. For every top-level
public subpackage, verifies that:

- every name in `__all__` is actually defined on the module (no drift between
  the import block and `__all__`); and
- the top-level `nanitics.__all__` is the small documented set, not the
  pre-Phase-3 flat 286-name surface.
"""

from __future__ import annotations

import importlib

import pytest

import nanitics

_SUBPACKAGES = [
    "nanitics.collaboration",
    "nanitics.composition",
    "nanitics.context",
    "nanitics.errors",
    "nanitics.evaluation",
    "nanitics.hitl",
    "nanitics.infrastructure",
    "nanitics.memory",
    "nanitics.patterns",
    "nanitics.planning",
    "nanitics.safety",
    "nanitics.specialized",
    "nanitics.strategies",
    "nanitics.tools",
    "nanitics.tracing",
]


@pytest.mark.parametrize("dotted", _SUBPACKAGES)
def test_subpackage_all_matches_module_attributes(dotted: str) -> None:
    """Every name in `<subpackage>.__all__` is actually defined on the module."""
    module = importlib.import_module(dotted)
    all_names = list(getattr(module, "__all__", []))
    missing = [n for n in all_names if not hasattr(module, n)]
    assert missing == [], f"{dotted}.__all__ lists names not defined on the module: {missing}"


def test_top_level_all_is_minimal() -> None:
    """`nanitics.__all__` is the small documented surface (just `__version__`)."""
    assert nanitics.__all__ == ["__version__"]
    assert hasattr(nanitics, "__version__")


def test_top_level_does_not_re_export_flat_symbols() -> None:
    """Symbols that used to be in the flat surface are no longer importable from `nanitics`."""
    for name in ("Agent", "Tool", "AnthropicLLMClient", "NaniticsError", "deprecated"):
        assert not hasattr(nanitics, name), f"`{name}` should no longer live on `nanitics`"
