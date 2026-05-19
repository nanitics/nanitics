"""``run_sql`` tool factory.

Builds an SDK :class:`~nanitics.Tool` that executes a SQL query against
the sandbox Postgres role with three belt-and-braces safeguards:

1. **Per-call connection** to the sandbox DSN — never the privileged
   app pool. The sandbox role's grants are the hard line; the tool
   opens a fresh asyncpg connection per invocation so test injection
   is trivial and the blast radius stays per-call.
2. **Statement timeout** applied server-side via
   ``server_settings={"statement_timeout": ...}`` so a pathological
   query is cancelled by Postgres itself.
3. **LIMIT injection** for bare SELECTs without an explicit ``LIMIT``
   (skipped for scalar aggregates, ``EXPLAIN``, and queries that
   already carry a top-level ``LIMIT``).

On ``asyncpg`` errors the tool returns a :class:`ToolResult` with
``metadata["error"] is True`` and the exception class name in
``content`` — errors are *not* re-raised so the agent sees the failure
text and can rewrite.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from nanitics import FunctionTool, Tool, ToolResult
from nanitics.strategies.tools.context import ToolContext

# Key the tool writes to ``ToolContext.state`` carrying the latest
# ``ToolResult.metadata`` — the Supervisor's ``PredicateTrigger`` reads
# this to decide whether to retry after a tool-level error or empty
# result. Keeping the key constant-exposed here lets the runner and
# tests import the exact same string (no magic strings).
LAST_TOOL_METADATA_STATE_KEY = "sql_analyst.last_run_sql_metadata"

# Type alias for the async connector. Tests inject a fake connector
# that returns an object with ``fetch`` + ``close`` methods; production
# uses :func:`asyncpg.connect` curried with ``dsn`` + ``server_settings``.
Connector = Callable[[], Awaitable[Any]]

_SQL_PARAMS_MODEL_DOC = "Execute a read-only SELECT against the analyst sandbox."

# Regexes compiled once. The LIMIT-injection rule is deliberately conservative.
_COMMENT_SINGLELINE = re.compile(r"--[^\n]*")
_COMMENT_MULTILINE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_LIMIT = re.compile(r"\bLIMIT\b\s+\d+(?:\s+OFFSET\s+\d+)?\s*;?\s*$", re.IGNORECASE)
_EXPLAIN_PREFIX = re.compile(r"^\s*EXPLAIN\b", re.IGNORECASE)
_SCALAR_AGGREGATE = re.compile(r"^\s*SELECT\s+(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)


def _strip_comments(sql: str) -> str:
    """Remove SQL comments for rule matching. The original text is
    preserved for execution — this is only used by the injection-rule
    inspector."""
    no_multi = _COMMENT_MULTILINE.sub(" ", sql)
    return _COMMENT_SINGLELINE.sub(" ", no_multi)


def _should_inject_limit(sql: str) -> bool:
    """Decide whether to append LIMIT.

    - ``EXPLAIN ...`` → no (user is inspecting a plan, not reading rows).
    - Trailing ``LIMIT N`` at top level → no (already bounded).
    - Scalar aggregate without ``GROUP BY`` → no (single row is returned).
    - Otherwise → yes (``SELECT * FROM t`` would otherwise stream).
    """
    inspected = _strip_comments(sql).strip().rstrip(";").strip()
    if not inspected:
        return False
    if _EXPLAIN_PREFIX.search(inspected):
        return False
    if _TRAILING_LIMIT.search(inspected):
        return False
    return not (_SCALAR_AGGREGATE.search(inspected) and not _GROUP_BY.search(inspected))


def _inject_limit(sql: str, *, row_limit: int) -> str:
    """Append ``LIMIT {row_limit}`` to *sql*.

    Preserves a trailing semicolon on the original statement by
    stripping it before the append and re-appending afterwards.
    """
    stripped = sql.rstrip()
    trailing_semicolon = stripped.endswith(";")
    if trailing_semicolon:
        stripped = stripped[:-1].rstrip()
    return f"{stripped}\nLIMIT {row_limit}" + (";" if trailing_semicolon else "")


def _render_table(columns: list[str], rows: list[list[Any]], *, truncated: bool) -> str:
    """Render a markdown-flavored table of up to ~20 rows.

    Keeps long result sets legible to the agent without dumping
    hundreds of rows into the conversation — the structured data is
    available on ``metadata["rows"]`` for the evaluator and endpoint.
    """
    if not columns:
        return "(no columns)"
    header = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    render_rows = rows[:20]
    body_lines = [" | ".join(str(v) for v in row) for row in render_rows]
    lines = [header, separator, *body_lines]
    if len(rows) > len(render_rows):
        lines.append(f"... and {len(rows) - len(render_rows)} more")
    elif truncated:
        lines.append("(rows truncated by LIMIT)")
    return "\n".join(lines)


async def _execute_query(
    *,
    connector: Connector,
    sql: str,
    row_limit: int,
) -> ToolResult:
    """Execute *sql* via a per-call connection and shape the result."""
    injected_sql = _inject_limit(sql, row_limit=row_limit) if _should_inject_limit(sql) else sql

    try:
        connection = await connector()
    except Exception as exc:  # pragma: no cover — connector errors are caller-side
        return ToolResult(
            content=f"ERROR: {exc.__class__.__name__}: {exc}",
            metadata={
                "error": True,
                "error_type": exc.__class__.__name__,
                "sql": injected_sql,
            },
        )

    try:
        try:
            records = await connection.fetch(injected_sql)
        except Exception as exc:
            return ToolResult(
                content=f"ERROR: {exc.__class__.__name__}: {exc}",
                metadata={
                    "error": True,
                    "error_type": exc.__class__.__name__,
                    "sql": injected_sql,
                },
            )
    finally:
        close = getattr(connection, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    columns: list[str] = list(records[0].keys()) if records else []
    rows: list[list[Any]] = [list(record.values()) for record in records]
    truncated = len(rows) == row_limit

    return ToolResult(
        content=_render_table(columns, rows, truncated=truncated),
        metadata={
            "columns": columns,
            "rows": rows,
            "rowcount": len(rows),
            "sql": injected_sql,
            "truncated": truncated,
            "error": False,
        },
    )


def _default_connector_factory(dsn: str, statement_timeout_ms: int) -> Connector:
    """Build a connector closure around ``asyncpg.connect``."""

    async def connect() -> Any:
        import asyncpg  # local import keeps the hard dependency opt-in

        return await asyncpg.connect(
            dsn,
            server_settings={"statement_timeout": str(statement_timeout_ms)},
        )

    return connect


def build_run_sql_tool(
    *,
    sandbox_dsn: str,
    row_limit: int = 200,
    statement_timeout_ms: int = 2000,
    connector_factory: Callable[[str, int], Connector] = _default_connector_factory,
) -> Tool:
    """Construct the ``run_sql`` tool bound to the sandbox DSN.

    Args:
        sandbox_dsn: DSN for the sandbox Postgres role. Never the app's
            privileged DSN — the sandbox role's grants are the hard
            line between the agent and write access.
        row_limit: Injected into bare SELECTs that do not already carry
            a ``LIMIT``. Default 200.
        statement_timeout_ms: Applied server-side on the per-call
            connection. Default 2000ms.
        connector_factory: Test seam. Production callers should not set
            this; tests inject a fake connector returning an object
            with ``fetch`` and ``close`` methods.

    Returns:
        An SDK :class:`~nanitics.Tool` named ``run_sql`` with a single
        ``sql: str`` parameter.
    """
    connector = connector_factory(sandbox_dsn, statement_timeout_ms)

    async def run_sql(sql: str, ctx: ToolContext | None = None) -> ToolResult:
        result = await _execute_query(
            connector=connector,
            sql=sql,
            row_limit=row_limit,
        )
        # Record the last result's metadata into the per-run tool state
        # so the Supervisor's error-catch PredicateTrigger can inspect it
        # without scraping the tool-result message's content. ``state``
        # is typed ``Mapping`` but the registry hands us a mutable dict —
        # cast through ``Any`` so the runtime write is explicit. On
        # direct-invocation paths (tests) ``ctx`` is ``None`` and this
        # is a no-op.
        if ctx is not None:
            state: Any = ctx.state
            state[LAST_TOOL_METADATA_STATE_KEY] = dict(result.metadata)
        return result

    return FunctionTool(
        fn=run_sql,
        name="run_sql",
        description=(
            "Execute a read-only SELECT against the bundled analyst database. "
            "Returns columns, rows, and rowcount on success. On error, the "
            "result's content starts with 'ERROR:' — read the error message "
            "and rewrite the query."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": _SQL_PARAMS_MODEL_DOC,
                }
            },
            "required": ["sql"],
        },
    )
