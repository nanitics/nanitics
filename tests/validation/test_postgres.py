"""Unit tests for `validation.helpers.postgres`.

Exercises missing-env and missing-driver error paths, plus the happy-path
pool construction/teardown via a monkeypatched ``asyncpg.create_pool``.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from validation.helpers.postgres import make_postgres_pool


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_URL", raising=False)


async def test_missing_env() -> None:
    with pytest.raises(ValueError, match="POSTGRES_URL") as excinfo:
        async with make_postgres_pool():
            pass
    assert "uv sync --extra postgres" in str(excinfo.value)


async def test_missing_asyncpg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_URL", "postgres://u:p@h/db")
    # Hide asyncpg by injecting an ImportError at import time.
    orig_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "asyncpg":
            raise ImportError("no asyncpg")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ValueError, match="POSTGRES_URL") as excinfo:
        async with make_postgres_pool():
            pass
    assert "uv sync --extra postgres" in str(excinfo.value)


async def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_URL", "postgres://u:p@h/db")

    fake_pool = MagicMock()
    fake_pool.close = AsyncMock()

    async def fake_create_pool(url: str) -> Any:
        assert url == "postgres://u:p@h/db"
        return fake_pool

    fake_asyncpg = MagicMock()
    fake_asyncpg.create_pool = fake_create_pool
    monkeypatch.setitem(sys.modules, "asyncpg", fake_asyncpg)

    async with make_postgres_pool() as pool:
        assert pool is fake_pool

    fake_pool.close.assert_awaited_once()
