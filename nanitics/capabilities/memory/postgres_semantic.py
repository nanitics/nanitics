"""PostgresSemanticStore — persistent semantic storage backed by PostgreSQL + pgvector.

Implements the :class:`SemanticStore` protocol using ``asyncpg``
for persisting semantic memory entries with vector similarity search.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nanitics.capabilities.memory.semantic import SearchResult
from nanitics.infrastructure.embeddings.protocol import EmbeddingClient

if TYPE_CHECKING:
    import asyncpg


def _vector_to_text(vector: list[float]) -> str:
    """Encode a Python float list as pgvector text format."""
    return "[" + ",".join(str(v) for v in vector) + "]"


SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table_name} (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector({dimension}) NOT NULL,
    namespace TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_hnsw
    ON {table_name} USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_{table_name}_namespace
    ON {table_name} (namespace);
"""


def get_semantic_store_schema_sql(*, table_name: str = "semantic_entries", dimension: int = 1024) -> str:
    """Return the CREATE TABLE + INDEX statements for semantic storage.

    Applications can execute this during database migrations to ensure
    the tables exist.  If custom names are used, they must also be
    passed to the :class:`PostgresSemanticStore` constructor.

    .. note::

        For production use, prefer :meth:`PostgresSemanticStore.ensure_schema`
        which handles both initial creation and schema evolution via
        versioned migrations.
    """
    return SCHEMA_SQL.format(table_name=table_name, dimension=dimension)


# ---------------------------------------------------------------------------
# Versioned schema migrations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Migration:
    version: int
    description: str
    up_sql: list[str]


def _build_migrations(table_name: str, dimension: int) -> list[_Migration]:
    """Build the ordered migration list, parameterised by table name and dimension."""
    return [
        _Migration(
            version=1,
            description="Baseline schema — create semantic_entries table with pgvector",
            up_sql=[get_semantic_store_schema_sql(table_name=table_name, dimension=dimension)],
        ),
    ]


LATEST_VERSION = 1


class PostgresSemanticStore:
    """Persistent semantic store backed by PostgreSQL + pgvector via ``asyncpg``.

    Implements the :class:`SemanticStore` protocol for production use.

    Args:
        pool: An initialised ``asyncpg.Pool``.
        embedding_client: Client used to convert text into embedding vectors.
        namespace: Optional namespace for entry isolation.
        table_name: Name of the semantic entries table (default ``"semantic_entries"``).
        dimension: Embedding vector dimension (default ``1024``).
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_client: EmbeddingClient,
        *,
        namespace: str | None = None,
        table_name: str = "semantic_entries",
        dimension: int = 1024,
    ) -> None:
        self._pool = pool
        self._embedding_client = embedding_client
        self._namespace = namespace
        self._table = table_name
        self._dimension = dimension
        self._version_table = f"_{table_name}_schema_version"

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create or migrate semantic tables to the latest schema version.

        Handles three cases:

        - **Fresh database:** Creates all tables from scratch.
        - **Legacy database:** Tables exist from ``CREATE TABLE IF NOT EXISTS``
          but no version tracking — applies corrective migrations.
        - **Current database:** Already at latest version — no-op.

        Uses a PostgreSQL advisory lock to prevent concurrent migration races
        when multiple application instances start simultaneously.
        """
        lock_key = zlib.crc32(self._table.encode()) & 0x7FFFFFFF

        async with self._pool.acquire() as conn:
            await conn.execute("SELECT pg_advisory_lock($1)", lock_key)
            try:
                await self._apply_migrations(conn)
            finally:
                await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)

    async def _apply_migrations(self, conn: asyncpg.Connection) -> None:
        """Detect current state and apply pending migrations."""
        version_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = $1
            )
            """,
            self._version_table,
        )

        if not version_exists:
            table_exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = $1
                )
                """,
                self._table,
            )

            # Create the version tracking table
            await conn.execute(
                f"""
                CREATE TABLE {self._version_table} (
                    version INTEGER NOT NULL
                )
                """
            )

            if table_exists:
                # Legacy database: tables exist but no version tracking.
                await conn.execute(f"INSERT INTO {self._version_table} (version) VALUES (1)")
            else:
                # Fresh database: no tables at all.
                await conn.execute(f"INSERT INTO {self._version_table} (version) VALUES (0)")

        current_version = await conn.fetchval(f"SELECT version FROM {self._version_table}")

        migrations = _build_migrations(self._table, self._dimension)

        for migration in migrations:
            if migration.version <= current_version:
                continue

            async with conn.transaction():
                for sql in migration.up_sql:
                    await conn.execute(sql)
                await conn.execute(
                    f"UPDATE {self._version_table} SET version = $1",
                    migration.version,
                )

    # ------------------------------------------------------------------
    # SemanticStore protocol
    # ------------------------------------------------------------------

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add content to the store."""
        vectors = await self._embedding_client.embed([content])
        entry_id = str(uuid4())
        vector_text = _vector_to_text(vectors[0])

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table}
                    (id, content, metadata, embedding, namespace)
                VALUES ($1, $2, $3, $4::vector, $5)
                """,
                entry_id,
                content,
                json.dumps(metadata) if metadata is not None else None,
                vector_text,
                self._namespace,
            )

        return entry_id

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search for content by semantic similarity."""
        vectors = await self._embedding_client.embed([query])
        vector_text = _vector_to_text(vectors[0])

        if self._namespace is not None:
            sql = f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM {self._table}
                WHERE namespace = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """
            params: list[Any] = [vector_text, self._namespace, limit]
        else:
            sql = f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM {self._table}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            params = [vector_text, limit]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [_row_to_search_result(row) for row in rows]

    async def delete(self, id: str) -> None:
        """Remove an entry by ID."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = $1::uuid",
                id,
            )


def _row_to_search_result(row: asyncpg.Record) -> SearchResult:
    """Convert a database row to a SearchResult."""
    metadata_raw = row["metadata"]
    if isinstance(metadata_raw, str):
        metadata_raw = json.loads(metadata_raw)

    return SearchResult(
        id=str(row["id"]),
        content=row["content"],
        score=float(row["score"]),
        metadata=metadata_raw,
    )
