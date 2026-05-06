"""Unit tests for `validation/conftest.py`.

Invokes a synthetic pytest run via :class:`pytest.Pytester` so we can drive
the conftest with controlled environment and assert the hard-skip gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from validation.helpers import postgres_container

pytest_plugins = ["pytester"]

CONFTEST_PATH = Path(__file__).resolve().parents[2] / "validation" / "conftest.py"


@pytest.fixture
def conftest_source() -> str:
    return CONFTEST_PATH.read_text()


@pytest.fixture
def pgvector_call_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace ``maybe_start_pgvector`` with a counter that returns no container.

    The synthetic suite imports ``validation.helpers.postgres_container``
    from the parent process, so monkeypatching the module attribute here
    is observed by the conftest hook running in the synthetic suite.
    Returning ``None`` means "provisioning failed" — postgres-marked
    tests will be skipped, which is how we assert the hook fires
    without paying a real Docker pull.
    """
    counter = {"calls": 0}

    def fake_start() -> tuple[str, Any] | None:
        counter["calls"] += 1
        return None

    monkeypatch.setattr(postgres_container, "maybe_start_pgvector", fake_start)
    return counter


def test_suite_hard_skips_without_anthropic_key(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    conftest_source: str,
    pgvector_call_counter: dict[str, int],
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
    assert pgvector_call_counter["calls"] == 0, "Should not provision Postgres when the suite is hard-skipped"


def test_suite_runs_with_anthropic_key(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    conftest_source: str,
    pgvector_call_counter: dict[str, int],
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
    assert pgvector_call_counter["calls"] == 0, "Should not provision Postgres when no test needs it"


def test_provisioning_skipped_when_no_postgres_marker(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    conftest_source: str,
    pgvector_call_counter: dict[str, int],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    pytester.makeconftest(conftest_source)
    pytester.makepyfile(
        test_example="""
        def test_no_postgres() -> None:
            assert True
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
    assert pgvector_call_counter["calls"] == 0


def test_provisioning_fires_when_postgres_marker_present(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    conftest_source: str,
    pgvector_call_counter: dict[str, int],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    pytester.makeconftest(conftest_source)
    pytester.makepyfile(
        test_example="""
        import pytest

        @pytest.mark.postgres
        def test_needs_postgres() -> None:
            assert True
        """
    )
    result = pytester.runpytest("-v", "-rs")
    result.assert_outcomes(skipped=1)
    assert pgvector_call_counter["calls"] == 1
    assert any("POSTGRES_URL not set or asyncpg not installed" in line for line in result.stdout.lines)


def test_provisioning_fires_only_once_for_multiple_postgres_items(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    conftest_source: str,
    pgvector_call_counter: dict[str, int],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    pytester.makeconftest(conftest_source)
    pytester.makepyfile(
        test_example="""
        import pytest

        @pytest.mark.postgres
        def test_one() -> None:
            assert True

        @pytest.mark.postgres
        def test_two() -> None:
            assert True
        """
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(skipped=2)
    assert pgvector_call_counter["calls"] == 1, "Should provision once per session, not per item"
