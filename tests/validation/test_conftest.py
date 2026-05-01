"""Unit tests for `validation/conftest.py`.

Invokes a synthetic pytest run via :class:`pytest.Pytester` so we can drive
the conftest with controlled environment and assert the hard-skip gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST_PATH = Path(__file__).resolve().parents[2] / "validation" / "conftest.py"


@pytest.fixture
def conftest_source() -> str:
    return CONFTEST_PATH.read_text()


def test_suite_hard_skips_without_anthropic_key(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, conftest_source: str
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pytester.makeconftest(conftest_source)
    pytester.makepyfile(
        test_example="""
        def test_one() -> None:
            assert True
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=1)
    assert any("ANTHROPIC_API_KEY" in line for line in result.stdout.lines)


def test_suite_runs_with_anthropic_key(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, conftest_source: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    pytester.makeconftest(conftest_source)
    pytester.makepyfile(
        test_example="""
        def test_one() -> None:
            assert True
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
