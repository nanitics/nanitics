"""PostgresThreadStore — persistent thread-prefix storage backed by PostgreSQL.

Implements the :class:`ThreadStore` protocol using ``asyncpg`` for
persisting per-thread message prefixes across process restarts.
"""

from __future__ import annotations

import asyncpg

from nanitics.infrastructure.llm.protocol import Message

THREAD_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS thread_messages (
    thread_key  TEXT NOT NULL,
    seq         BIGSERIAL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data        JSONB NOT NULL,
    PRIMARY KEY (thread_key, seq)
);
"""


def get_thread_schema_sql() -> str:
    """Return the CREATE TABLE statement for the thread_messages table."""
    return THREAD_SCHEMA_SQL


class PostgresThreadStore:
    """Persistent thread-prefix store backed by PostgreSQL via ``asyncpg``.

    Stores one row per :class:`Message`, keyed by ``(thread_key, seq)``.
    ``seq`` is ``BIGSERIAL`` so insertion order is preserved globally and
    therefore also within each ``thread_key``. ``load(thread_key)``
    returns the messages ordered by ``seq ASC`` — the deterministic
    insertion order.

    Cross-process locking is out of scope for this implementation:
    concurrent ``append`` calls against the same ``thread_key`` from
    different processes will interleave their batches at row granularity.
    Consumers running multiple processes against the same logical thread
    must coordinate externally. The advisory-lock story remains a
    follow-up (see :class:`~nanitics.composition.threads.store.ThreadLocks`
    for the in-process equivalent).

    Args:
        pool: An initialised ``asyncpg.Pool``.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def load(self, thread_key: str) -> list[Message]:
        """Return the message prefix for the thread, in insertion order.

        Returns an empty list when the key is unknown. Persistence
        failures propagate.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data FROM thread_messages
                WHERE thread_key = $1
                ORDER BY seq ASC
                """,
                thread_key,
            )
        return [_row_to_message(row["data"]) for row in rows]

    async def append(self, thread_key: str, messages: list[Message]) -> None:
        """Append a batch of messages to the thread.

        Inserts each :class:`Message` as one row with its
        ``model_dump_json()`` payload bound to the ``data`` JSONB
        column. The batch is inserted in a single connection acquisition
        so a single-process append is naturally ordered; cross-process
        ordering across the same key is not guaranteed (see the class
        docstring).
        """
        if not messages:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO thread_messages (thread_key, data)
                VALUES ($1, $2)
                """,
                [(thread_key, message.model_dump_json()) for message in messages],
            )

    async def clear(self, thread_key: str) -> None:
        """Delete all messages for the thread. Silent on unknown keys."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM thread_messages WHERE thread_key = $1",
                thread_key,
            )


def _row_to_message(raw: str | dict[str, object]) -> Message:
    if isinstance(raw, str):
        return Message.model_validate_json(raw)
    return Message.model_validate(raw)
