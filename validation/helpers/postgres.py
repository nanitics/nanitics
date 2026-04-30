"""Async Postgres pool helper for validation scripts.

Opens an :mod:`asyncpg` pool keyed on the ``POSTGRES_URL`` env var. Fails
loudly when the variable is missing or the driver is not installed.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


@asynccontextmanager
async def make_postgres_pool() -> AsyncIterator[Any]:
    """Open an asyncpg pool from ``POSTGRES_URL``.

    Yields:
        An open :class:`asyncpg.Pool`. The pool is closed on context exit.

    Raises:
        ValueError: If ``POSTGRES_URL`` is unset or ``asyncpg`` is not
            installed. The message names the env var and the install extra.
    """
    url = os.environ.get("POSTGRES_URL")
    if not url:
        raise ValueError("Postgres validation requires POSTGRES_URL. Install with: uv sync --extra postgres")
    try:
        import asyncpg
    except ImportError as exc:
        raise ValueError("Postgres validation requires POSTGRES_URL. Install with: uv sync --extra postgres") from exc

    pool = await asyncpg.create_pool(url)
    try:
        yield pool
    finally:
        await pool.close()
