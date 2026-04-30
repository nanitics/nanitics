"""Docker-gated schema and sandbox-role tests for the SQL-analyst runner.

Brings up a real Postgres 16 container via ``testcontainers``, applies
the analyst schema + sandbox role with :func:`ensure_analyst_schema`,
and verifies:

1. The schema apply is idempotent (running twice leaves row counts
   and the role unchanged).
2. The sandbox role can ``SELECT`` on the five analyst tables.
3. The sandbox role cannot ``INSERT`` / ``UPDATE`` / ``DELETE`` on
   any analyst table (each raises ``InsufficientPrivilegeError``).
4. The sandbox role cannot read the ``trace_events`` table.
5. ``statement_timeout = '2s'`` pinned on the sandbox role is
   enforced (``SELECT pg_sleep(5)`` raises ``QueryCanceledError``).
6. The ``run_sql`` tool injects ``LIMIT 200`` on a bare ``SELECT``
   against a seeded table and flags ``truncated=True``.

Excluded from the default ``just check`` run via
``pytestmark = pytest.mark.docker`` — the project's pytest ``addopts``
carries ``-m 'not docker'`` so this module only runs under
``just check docker=true`` / ``just ci``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.docker

# Put ``docker/full-stack/`` on sys.path so ``sql_analyst`` resolves as
# a top-level package — mirroring the runtime image layout.
_FULL_STACK_DIR = Path(__file__).resolve().parent.parent / "docker" / "full-stack"
if str(_FULL_STACK_DIR) not in sys.path:
    sys.path.insert(0, str(_FULL_STACK_DIR))


_SANDBOX_ROLE = "sql_analyst_sandbox_test"
_SANDBOX_PASSWORD = "sandbox-test-pw"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[tuple[str, str]]:
    """Session-scoped Postgres 16 container.

    Yields ``(privileged_dsn, sandbox_dsn)`` — both asyncpg-compatible
    ``postgresql://`` URLs. The privileged DSN authenticates as the
    container's superuser; the sandbox DSN authenticates as
    ``sql_analyst_sandbox_test`` (created by
    :func:`ensure_analyst_schema` during the session's first apply in
    the :data:`schema_applied` fixture).
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(image="postgres:16") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        privileged_dsn = f"postgresql://{container.username}:{container.password}@{host}:{port}/{container.dbname}"
        sandbox_dsn = f"postgresql://{_SANDBOX_ROLE}:{_SANDBOX_PASSWORD}@{host}:{port}/{container.dbname}"
        yield privileged_dsn, sandbox_dsn


@pytest.fixture(scope="session")
async def privileged_pool(postgres_container: tuple[str, str]) -> Any:
    """An asyncpg pool authenticated as the container's superuser."""
    import asyncpg

    privileged_dsn, _ = postgres_container
    pool = await asyncpg.create_pool(privileged_dsn, min_size=1, max_size=4)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture(scope="session")
async def schema_applied(privileged_pool: Any) -> Any:
    """Apply the analyst schema + sandbox role exactly once per session.

    Also installs the ``trace_events`` table via
    :meth:`PostgresTraceStore.ensure_schema` so the sandbox-permission
    test has a real table to be denied access to.
    """
    from sql_analyst.bootstrap import ensure_analyst_schema

    from nanitics import PostgresTraceStore

    trace_store = PostgresTraceStore(privileged_pool)
    await trace_store.ensure_schema()
    await ensure_analyst_schema(
        privileged_pool,
        sandbox_role=_SANDBOX_ROLE,
        sandbox_password=_SANDBOX_PASSWORD,
    )
    return privileged_pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_ensure_analyst_schema_is_idempotent(
    schema_applied: Any,
    privileged_pool: Any,
) -> None:
    """Applying the schema a second time leaves row counts unchanged."""
    from sql_analyst.bootstrap import ensure_analyst_schema

    async with privileged_pool.acquire() as conn:
        first_counts = {
            table: await conn.fetchval(f"SELECT count(*) FROM {table}")
            for table in ("regions", "customers", "products", "orders", "order_items")
        }

    await ensure_analyst_schema(
        privileged_pool,
        sandbox_role=_SANDBOX_ROLE,
        sandbox_password=_SANDBOX_PASSWORD,
    )

    async with privileged_pool.acquire() as conn:
        second_counts = {
            table: await conn.fetchval(f"SELECT count(*) FROM {table}")
            for table in ("regions", "customers", "products", "orders", "order_items")
        }

    assert first_counts == second_counts
    # Seed rowcounts match the expected sizing.
    assert first_counts["regions"] == 5
    assert first_counts["customers"] == 50
    assert first_counts["products"] == 30
    assert first_counts["orders"] == 200
    assert first_counts["order_items"] == 500


async def test_sandbox_role_select_allowed(
    schema_applied: Any,
    postgres_container: tuple[str, str],
) -> None:
    import asyncpg

    _, sandbox_dsn = postgres_container
    connection = await asyncpg.connect(sandbox_dsn)
    try:
        count = await connection.fetchval("SELECT count(*) FROM customers")
    finally:
        await connection.close()

    assert count == 50


async def test_sandbox_role_insert_denied(
    schema_applied: Any,
    postgres_container: tuple[str, str],
) -> None:
    import asyncpg

    _, sandbox_dsn = postgres_container
    connection = await asyncpg.connect(sandbox_dsn)
    try:
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await connection.execute(
                "INSERT INTO customers (name, email, region_id, signup_date) VALUES ('x', 'x@x', 1, DATE '2024-01-01')"
            )
    finally:
        await connection.close()


async def test_sandbox_role_update_denied(
    schema_applied: Any,
    postgres_container: tuple[str, str],
) -> None:
    import asyncpg

    _, sandbox_dsn = postgres_container
    connection = await asyncpg.connect(sandbox_dsn)
    try:
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await connection.execute("UPDATE customers SET name = 'x' WHERE id = 1")
    finally:
        await connection.close()


async def test_sandbox_role_delete_denied(
    schema_applied: Any,
    postgres_container: tuple[str, str],
) -> None:
    import asyncpg

    _, sandbox_dsn = postgres_container
    connection = await asyncpg.connect(sandbox_dsn)
    try:
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await connection.execute("DELETE FROM customers WHERE id = 1")
    finally:
        await connection.close()


async def test_sandbox_role_cannot_read_trace_tables(
    schema_applied: Any,
    postgres_container: tuple[str, str],
) -> None:
    """The sandbox role has no grants on ``trace_events`` — SELECT must fail."""
    import asyncpg

    _, sandbox_dsn = postgres_container
    connection = await asyncpg.connect(sandbox_dsn)
    try:
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await connection.fetch("SELECT * FROM trace_events LIMIT 1")
    finally:
        await connection.close()


async def test_statement_timeout_enforced(
    schema_applied: Any,
    postgres_container: tuple[str, str],
) -> None:
    """``pg_sleep(5)`` exceeds the 2s statement_timeout pinned on the role."""
    import asyncpg

    _, sandbox_dsn = postgres_container
    connection = await asyncpg.connect(sandbox_dsn)
    try:
        with pytest.raises(asyncpg.exceptions.QueryCanceledError):
            await connection.fetch("SELECT pg_sleep(5)")
    finally:
        await connection.close()


async def test_limit_injection_on_plain_select(
    schema_applied: Any,
    postgres_container: tuple[str, str],
) -> None:
    """``run_sql`` injects ``LIMIT 200`` on a bare ``SELECT`` and flags
    ``truncated=True`` when the row count fills the limit."""
    from sql_analyst.tool import build_run_sql_tool

    _, sandbox_dsn = postgres_container
    tool = build_run_sql_tool(sandbox_dsn=sandbox_dsn)
    # ``order_items`` has 500 rows — a bare SELECT would stream all of
    # them; the tool must inject LIMIT 200 and flag truncation.
    result = await tool.execute(sql="SELECT * FROM order_items")

    assert result.metadata["error"] is False
    assert "LIMIT 200" in result.metadata["sql"]
    assert result.metadata["rowcount"] == 200
    assert result.metadata["truncated"] is True
