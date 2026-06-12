"""Mock-based tests for PostgresHitlRequestStore — no real Postgres needed."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from nanitics.collaboration.hitl_store import (
    DuplicateHitlRequestError,
    DuplicateHitlResponseError,
)
from nanitics.collaboration.postgres_hitl_store import (
    PostgresHitlRequestStore,
    _row_to_request,
    _row_to_response,
    get_hitl_schema_sql,
)
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)


def _make_pool() -> tuple[MagicMock, AsyncMock]:
    pool = MagicMock()
    conn = AsyncMock()
    acquire_cm = AsyncMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_cm
    return pool, conn


class TestGetHitlSchemaSql:
    def test_returns_string_with_create_table(self) -> None:
        sql = get_hitl_schema_sql()
        assert "CREATE TABLE" in sql
        assert "hitl_requests" in sql
        assert "hitl_responses" in sql


class TestPostgresHitlRequestStoreMock:
    async def test_save_request(self) -> None:
        pool, conn = _make_pool()
        store = PostgresHitlRequestStore(pool)
        request = HumanInputRequest(
            request_id="req-1",
            run_id="run-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve this?",
            context="some context",
            options=["yes", "no"],
            metadata={"key": "val"},
            agent_name="agent-1",
        )
        await store.save_request(request)
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "INSERT INTO hitl_requests" in args[0]
        assert args[1] == "req-1"

    async def test_save_request_no_run_id(self) -> None:
        pool, conn = _make_pool()
        store = PostgresHitlRequestStore(pool)
        request = HumanInputRequest(
            request_id="req-2",
            run_id=None,
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        await store.save_request(request)
        args = conn.execute.call_args[0]
        # run_id should be empty string when None
        assert args[2] == ""

    async def test_save_request_duplicate_raises_shared_error(self) -> None:
        pool, conn = _make_pool()
        conn.execute = AsyncMock(
            side_effect=asyncpg.exceptions.UniqueViolationError("duplicate key value violates unique constraint")
        )
        store = PostgresHitlRequestStore(pool)
        request = HumanInputRequest(
            request_id="req-dup",
            run_id="run-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(DuplicateHitlRequestError) as exc_info:
            await store.save_request(request)
        assert exc_info.value.request_id == "req-dup"
        assert isinstance(exc_info.value.__cause__, asyncpg.exceptions.UniqueViolationError)

    async def test_save_request_no_options(self) -> None:
        pool, conn = _make_pool()
        store = PostgresHitlRequestStore(pool)
        request = HumanInputRequest(
            request_id="req-3",
            run_id="run-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
            options=None,
        )
        await store.save_request(request)
        args = conn.execute.call_args[0]
        # options should be None when not provided
        assert args[6] is None

    async def test_save_response(self) -> None:
        pool, conn = _make_pool()
        store = PostgresHitlRequestStore(pool)
        now = datetime.now(tz=UTC)
        response = HumanInputResponse(
            request_id="req-1",
            decision=HumanDecision.APPROVE,
            content="Looks good",
            metadata={"reviewer": "human"},
            responded_at=now,
        )
        await store.save_response("req-1", response)
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "INSERT INTO hitl_responses" in args[0]
        assert args[1] == "req-1"
        assert args[2] == "approve"

    async def test_save_response_duplicate_raises_shared_error(self) -> None:
        pool, conn = _make_pool()
        conn.execute = AsyncMock(
            side_effect=asyncpg.exceptions.UniqueViolationError("duplicate key value violates unique constraint")
        )
        store = PostgresHitlRequestStore(pool)
        response = HumanInputResponse(
            request_id="req-dup",
            decision=HumanDecision.APPROVE,
            content="Looks good",
            responded_at=datetime.now(tz=UTC),
        )
        with pytest.raises(DuplicateHitlResponseError) as exc_info:
            await store.save_response("req-dup", response)
        assert exc_info.value.request_id == "req-dup"
        assert isinstance(exc_info.value.__cause__, asyncpg.exceptions.UniqueViolationError)

    async def test_get_response_found(self) -> None:
        pool, conn = _make_pool()
        now = datetime.now(tz=UTC)
        conn.fetchrow = AsyncMock(
            return_value={
                "request_id": "req-1",
                "decision": "approve",
                "content": "OK",
                "metadata": {"k": "v"},
                "responded_at": now,
            }
        )
        store = PostgresHitlRequestStore(pool)
        result = await store.get_response("req-1")
        assert result is not None
        assert result.decision == HumanDecision.APPROVE
        assert result.content == "OK"

    async def test_get_response_not_found(self) -> None:
        pool, conn = _make_pool()
        conn.fetchrow = AsyncMock(return_value=None)
        store = PostgresHitlRequestStore(pool)
        result = await store.get_response("nonexistent")
        assert result is None

    async def test_get_pending_requests(self) -> None:
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "request_id": "req-1",
                    "run_id": "run-1",
                    "request_type": "approval",
                    "prompt": "Approve?",
                    "context": None,
                    "options": None,
                    "metadata": {},
                    "agent_name": "agent-1",
                },
            ]
        )
        store = PostgresHitlRequestStore(pool)
        results = await store.get_pending_requests("run-1")
        assert len(results) == 1
        assert results[0].request_id == "req-1"

    async def test_get_pending_requests_empty(self) -> None:
        pool, conn = _make_pool()
        conn.fetch = AsyncMock(return_value=[])
        store = PostgresHitlRequestStore(pool)
        results = await store.get_pending_requests("run-1")
        assert results == []


class TestRowToRequest:
    def test_basic_conversion(self) -> None:
        row: dict[str, Any] = {
            "request_id": "req-1",
            "run_id": "run-1",
            "request_type": "approval",
            "prompt": "Approve this?",
            "context": "some context",
            "options": ["yes", "no"],
            "metadata": {"key": "val"},
            "agent_name": "agent-1",
        }
        result = _row_to_request(row)
        assert result.request_id == "req-1"
        assert result.request_type == HumanInputType.APPROVAL
        assert result.options == ["yes", "no"]
        assert result.metadata == {"key": "val"}

    def test_string_options_json_decoded(self) -> None:
        row: dict[str, Any] = {
            "request_id": "req-1",
            "run_id": "run-1",
            "request_type": "approval",
            "prompt": "Approve?",
            "context": None,
            "options": '["a", "b"]',
            "metadata": {"k": "v"},
            "agent_name": None,
        }
        result = _row_to_request(row)
        assert result.options == ["a", "b"]

    def test_string_metadata_json_decoded(self) -> None:
        row: dict[str, Any] = {
            "request_id": "req-1",
            "run_id": "run-1",
            "request_type": "approval",
            "prompt": "Approve?",
            "context": None,
            "options": None,
            "metadata": '{"k": "v"}',
            "agent_name": None,
        }
        result = _row_to_request(row)
        assert result.metadata == {"k": "v"}

    def test_empty_metadata_defaults_to_empty_dict(self) -> None:
        row: dict[str, Any] = {
            "request_id": "req-1",
            "run_id": "",
            "request_type": "approval",
            "prompt": "Approve?",
            "context": None,
            "options": None,
            "metadata": {},
            "agent_name": None,
        }
        result = _row_to_request(row)
        assert result.metadata == {}
        # run_id "" becomes None
        assert result.run_id is None


class TestRowToResponse:
    def test_basic_conversion(self) -> None:
        now = datetime.now(tz=UTC)
        row: dict[str, Any] = {
            "request_id": "req-1",
            "decision": "approve",
            "content": "Looks good",
            "metadata": {"reviewer": "human"},
            "responded_at": now,
        }
        result = _row_to_response(row)
        assert result.request_id == "req-1"
        assert result.decision == HumanDecision.APPROVE
        assert result.content == "Looks good"
        assert result.responded_at == now

    def test_string_metadata_json_decoded(self) -> None:
        now = datetime.now(tz=UTC)
        row: dict[str, Any] = {
            "request_id": "req-1",
            "decision": "reject",
            "content": None,
            "metadata": '{"k": "v"}',
            "responded_at": now,
        }
        result = _row_to_response(row)
        assert result.metadata == {"k": "v"}
        assert result.decision == HumanDecision.REJECT

    def test_empty_metadata_defaults_to_empty_dict(self) -> None:
        now = datetime.now(tz=UTC)
        row: dict[str, Any] = {
            "request_id": "req-1",
            "decision": "approve",
            "content": None,
            "metadata": {},
            "responded_at": now,
        }
        result = _row_to_response(row)
        assert result.metadata == {}
