"""Unit tests for `validation.helpers.skips`.

Each decorator must produce a ``pytest.skip`` marker when its precondition
fails and leave the function runnable when it passes.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from validation.helpers import skips


def _has_skipif(fn: Any) -> bool:
    marks = getattr(fn, "pytestmark", [])
    return any(m.name == "skipif" and m.args and m.args[0] for m in marks)


def test_requires_postgres_skips_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_URL", raising=False)

    @skips.requires_postgres
    def fn() -> None: ...

    assert _has_skipif(fn)


def test_requires_postgres_skips_when_driver_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_URL", "postgres://x")
    monkeypatch.setattr(skips, "_has_module", lambda name: name != "asyncpg")

    @skips.requires_postgres
    def fn() -> None: ...

    assert _has_skipif(fn)


def test_requires_postgres_passes_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_URL", "postgres://x")
    monkeypatch.setattr(skips, "_has_module", lambda name: True)

    @skips.requires_postgres
    def fn() -> None: ...

    assert not _has_skipif(fn)


def test_requires_docker_skips_without_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skips, "_has_module", lambda name: False)

    @skips.requires_docker
    def fn() -> None: ...

    assert _has_skipif(fn)


def test_requires_docker_skips_when_daemon_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skips, "_has_module", lambda name: True)

    fake_client = MagicMock()
    fake_client.ping.side_effect = Exception("daemon down")
    fake_docker = MagicMock()
    fake_docker.from_env.return_value = fake_client
    monkeypatch.setitem(sys.modules, "docker", fake_docker)

    @skips.requires_docker
    def fn() -> None: ...

    assert _has_skipif(fn)


def test_requires_docker_passes_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skips, "_has_module", lambda name: True)

    fake_client = MagicMock()
    fake_client.ping.return_value = True
    fake_docker = MagicMock()
    fake_docker.from_env.return_value = fake_client
    monkeypatch.setitem(sys.modules, "docker", fake_docker)

    @skips.requires_docker
    def fn() -> None: ...

    assert not _has_skipif(fn)


def test_requires_voyage_skips_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    @skips.requires_voyage
    def fn() -> None: ...

    assert _has_skipif(fn)


def test_requires_voyage_passes_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "k")

    @skips.requires_voyage
    def fn() -> None: ...

    assert not _has_skipif(fn)


def test_requires_tavily_skips_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    @skips.requires_tavily
    def fn() -> None: ...

    assert _has_skipif(fn)


def test_requires_tavily_does_not_skip_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "k")

    @skips.requires_tavily
    def fn() -> None: ...

    assert not _has_skipif(fn)


def test_has_module_reports_installed_and_missing() -> None:
    # ``os`` is always installed; a deliberately bogus module name is not.
    assert skips._has_module("os") is True
    assert skips._has_module("this_module_definitely_does_not_exist_xyz") is False
