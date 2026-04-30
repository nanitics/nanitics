"""Tests for the `nanitics.deprecated` re-export (PEP 702)."""

import warnings

import pytest
import typing_extensions

import nanitics


def test_deprecated_is_typing_extensions_re_export() -> None:
    """`nanitics.deprecated` is the same object as `typing_extensions.deprecated`.

    Source is bound to ``typing_extensions`` so that the 3.11 and 3.12
    supported versions receive a PEP 702-exact backport; on 3.13 the
    backport is itself a re-export of ``warnings.deprecated``, so
    behavior is uniform.
    """
    assert nanitics.deprecated is typing_extensions.deprecated


def test_deprecated_listed_in_all() -> None:
    """`deprecated` appears in `nanitics.__all__`. Lint (ruff RUF022) enforces the
    ordering convention at `just check` time; this test enforces membership."""
    assert "deprecated" in nanitics.__all__


def test_decorator_emits_deprecation_warning() -> None:
    """Applying `@deprecated` to a function raises `DeprecationWarning` when warnings are errors."""

    @nanitics.deprecated("use replacement() instead")
    def legacy() -> int:
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="use replacement"):
            legacy()


def test_decorator_sets_deprecated_attribute() -> None:
    """PEP 702 decorator annotates the target with a `__deprecated__` attribute carrying the message."""

    @nanitics.deprecated("reason-text")
    def old() -> None:
        return None

    assert old.__deprecated__ == "reason-text"
