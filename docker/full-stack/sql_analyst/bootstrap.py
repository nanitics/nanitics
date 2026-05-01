"""Analyst schema + sandbox-role bootstrap.

:func:`ensure_analyst_schema` reads ``schema.sql`` from the package
directory, substitutes the ``{{sandbox_role}}`` / ``{{sandbox_password}}``
placeholders, and applies the whole script in a single transaction under
the app's privileged pool. Idempotent and cheap enough to run on every
startup: schema DDL uses ``CREATE TABLE IF NOT EXISTS``, seed inserts use
``ON CONFLICT DO NOTHING``, role creation is gated on ``pg_roles``, and
``ALTER ROLE`` / ``GRANT`` statements are no-ops when they already hold.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _render_schema_sql(template: str, *, sandbox_role: str, sandbox_password: str) -> str:
    """Substitute the placeholders in ``schema.sql`` with caller values.

    The placeholders are the minimum shape needed to thread the sandbox
    role's credentials through without writing them to the SQL file at
    rest. No other templating is performed — the file is plain SQL.
    """
    return template.replace("{{sandbox_role}}", sandbox_role).replace("{{sandbox_password}}", sandbox_password)


async def ensure_analyst_schema(
    pool: asyncpg.Pool,
    *,
    sandbox_role: str,
    sandbox_password: str,
) -> None:
    """Apply the analyst schema + sandbox role under the app's pool.

    Args:
        pool: The privileged asyncpg pool the shell opened at startup.
        sandbox_role: Login name for the sandbox Postgres role. Created
            if missing; its password is re-pinned on every call so env
            rotation takes effect on the next app start.
        sandbox_password: The sandbox role's password.

    Raises:
        asyncpg.exceptions.PostgresError: When the schema / role / seed
            script fails to apply. The whole script runs in one
            transaction, so any failure rolls back and the app startup
            should fail loud.
    """
    template = _SCHEMA_PATH.read_text(encoding="utf-8")
    sql = _render_schema_sql(
        template,
        sandbox_role=sandbox_role,
        sandbox_password=sandbox_password,
    )
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(sql)
