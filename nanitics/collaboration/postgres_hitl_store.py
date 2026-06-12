"""PostgresHitlRequestStore — persistent HITL storage backed by PostgreSQL.

Implements the :class:`HitlRequestStore` protocol using ``asyncpg``
for persisting human-in-the-loop requests and responses across restarts.
"""

from __future__ import annotations

import json

import asyncpg

from nanitics.collaboration.hitl_store import (
    DuplicateHitlRequestError,
    DuplicateHitlResponseError,
)
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)

HITL_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS hitl_requests (
    request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    request_type TEXT NOT NULL,
    prompt TEXT NOT NULL,
    context TEXT,
    options JSONB,
    metadata JSONB NOT NULL DEFAULT '{}',
    agent_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hitl_responses (
    request_id TEXT PRIMARY KEY REFERENCES hitl_requests(request_id),
    decision TEXT NOT NULL,
    content TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    responded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hitl_requests_run
    ON hitl_requests (run_id);
"""


def get_hitl_schema_sql() -> str:
    """Return the CREATE TABLE + INDEX statements for HITL tables."""
    return HITL_SCHEMA_SQL


class PostgresHitlRequestStore:
    """Persistent HITL request store backed by PostgreSQL via ``asyncpg``.

    Args:
        pool: An initialised ``asyncpg.Pool``.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_request(self, request: HumanInputRequest) -> None:
        """Persist a new HITL request.

        Raises:
            DuplicateHitlRequestError: If ``request.request_id`` already
                exists. The underlying :class:`asyncpg.exceptions.UniqueViolationError`
                is preserved as ``__cause__``.
        """
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO hitl_requests
                        (request_id, run_id, request_type, prompt, context,
                         options, metadata, agent_name)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    request.request_id,
                    request.run_id or "",
                    request.request_type.value,
                    request.prompt,
                    request.context,
                    json.dumps(request.options) if request.options is not None else None,
                    json.dumps(request.metadata),
                    request.agent_name,
                )
            except asyncpg.exceptions.UniqueViolationError as exc:
                raise DuplicateHitlRequestError(request.request_id) from exc

    async def save_response(self, request_id: str, response: HumanInputResponse) -> None:
        """Store a human's response to a request.

        Raises:
            DuplicateHitlResponseError: If a response for ``request_id`` already
                exists. The underlying :class:`asyncpg.exceptions.UniqueViolationError`
                is preserved as ``__cause__``, mirroring :meth:`save_request`.
        """
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO hitl_responses
                        (request_id, decision, content, metadata, responded_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    request_id,
                    response.decision.value,
                    response.content,
                    json.dumps(response.metadata),
                    response.responded_at,
                )
            except asyncpg.exceptions.UniqueViolationError as exc:
                raise DuplicateHitlResponseError(request_id) from exc

    async def get_response(self, request_id: str) -> HumanInputResponse | None:
        """Retrieve a response by request ID, or None if not yet responded."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT request_id, decision, content, metadata, responded_at "
                "FROM hitl_responses WHERE request_id = $1",
                request_id,
            )
        if row is None:
            return None
        return _row_to_response(row)

    async def get_pending_requests(self, run_id: str) -> list[HumanInputRequest]:
        """Return all requests for a run that have no response yet."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT r.request_id, r.run_id, r.request_type, r.prompt,
                       r.context, r.options, r.metadata, r.agent_name
                FROM hitl_requests r
                LEFT JOIN hitl_responses resp ON r.request_id = resp.request_id
                WHERE r.run_id = $1 AND resp.request_id IS NULL
                """,
                run_id,
            )
        return [_row_to_request(row) for row in rows]


def _row_to_request(row: asyncpg.Record) -> HumanInputRequest:
    options_raw = row["options"]
    if isinstance(options_raw, str):
        options_raw = json.loads(options_raw)

    metadata_raw = row["metadata"]
    if isinstance(metadata_raw, str):
        metadata_raw = json.loads(metadata_raw)

    return HumanInputRequest(
        request_id=row["request_id"],
        run_id=row["run_id"] or None,
        request_type=HumanInputType(row["request_type"]),
        prompt=row["prompt"],
        context=row["context"],
        options=options_raw,
        metadata=metadata_raw if metadata_raw else {},
        agent_name=row["agent_name"],
    )


def _row_to_response(row: asyncpg.Record) -> HumanInputResponse:
    metadata_raw = row["metadata"]
    if isinstance(metadata_raw, str):
        metadata_raw = json.loads(metadata_raw)

    return HumanInputResponse(
        request_id=row["request_id"],
        decision=HumanDecision(row["decision"]),
        content=row["content"],
        metadata=metadata_raw if metadata_raw else {},
        responded_at=row["responded_at"],
    )
