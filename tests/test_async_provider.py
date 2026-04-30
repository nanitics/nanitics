"""Tests for AsyncHumanInputProvider."""

from __future__ import annotations

import asyncio

import pytest

from nanitics.collaboration.async_provider import AsyncHumanInputProvider
from nanitics.collaboration.hitl_store import InMemoryHitlRequestStore
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)
from nanitics.infrastructure.errors import ApprovalTimeoutError


def _make_request(request_id: str = "req-1", run_id: str = "run-1") -> HumanInputRequest:
    return HumanInputRequest(
        request_id=request_id,
        run_id=run_id,
        request_type=HumanInputType.QUESTION,
        prompt="What do you think?",
    )


def _make_response(request_id: str = "req-1") -> HumanInputResponse:
    return HumanInputResponse(
        request_id=request_id,
        decision=HumanDecision.ANSWER,
        content="Looks good",
    )


class TestRequestAndResolve:
    async def test_request_input_blocks_until_resolved(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()
        response = _make_response()

        async def resolve_later() -> None:
            await asyncio.sleep(0.01)
            await provider.resolve(request.request_id, response)

        _task = asyncio.create_task(resolve_later())  # noqa: RUF006
        result = await provider.request_input(request)
        assert result == response

    async def test_resolve_completes_future_and_returns_true(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()
        response = _make_response()

        task = asyncio.create_task(provider.request_input(request))
        await asyncio.sleep(0)  # let task start

        resolved = await provider.resolve(request.request_id, response)
        assert resolved is True
        assert (await task) == response

    async def test_resolve_returns_false_for_unknown_request(self) -> None:
        provider = AsyncHumanInputProvider()
        response = _make_response(request_id="unknown")
        assert await provider.resolve("unknown", response) is False

    async def test_resolve_returns_false_for_already_resolved(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()
        response = _make_response()

        task = asyncio.create_task(provider.request_input(request))
        await asyncio.sleep(0)

        await provider.resolve(request.request_id, response)
        await task

        # Second resolve should return False (already popped)
        assert await provider.resolve(request.request_id, response) is False


class TestGetPending:
    async def test_returns_unresolved_requests(self) -> None:
        provider = AsyncHumanInputProvider()
        req1 = _make_request(request_id="req-1")
        req2 = _make_request(request_id="req-2")

        asyncio.create_task(provider.request_input(req1))  # noqa: RUF006
        asyncio.create_task(provider.request_input(req2))  # noqa: RUF006
        await asyncio.sleep(0)

        pending = provider.get_pending()
        assert len(pending) == 2
        assert {r.request_id for r in pending} == {"req-1", "req-2"}

    async def test_excludes_resolved_requests(self) -> None:
        provider = AsyncHumanInputProvider()
        req1 = _make_request(request_id="req-1")
        req2 = _make_request(request_id="req-2")

        asyncio.create_task(provider.request_input(req1))  # noqa: RUF006
        asyncio.create_task(provider.request_input(req2))  # noqa: RUF006
        await asyncio.sleep(0)

        await provider.resolve("req-1", _make_response("req-1"))

        # req-1 is popped from _pending on resolve, only req-2 remains
        pending = provider.get_pending()
        assert len(pending) == 1
        assert pending[0].request_id == "req-2"

    async def test_filters_by_run_id(self) -> None:
        provider = AsyncHumanInputProvider()
        req_run1 = _make_request(request_id="req-1", run_id="run-1")
        req_run2 = _make_request(request_id="req-2", run_id="run-2")

        asyncio.create_task(provider.request_input(req_run1))  # noqa: RUF006
        asyncio.create_task(provider.request_input(req_run2))  # noqa: RUF006
        await asyncio.sleep(0)

        pending = provider.get_pending(run_id="run-1")
        assert len(pending) == 1
        assert pending[0].request_id == "req-1"


class TestStoreIntegration:
    async def test_request_input_persists_to_store(self) -> None:
        store = InMemoryHitlRequestStore()
        provider = AsyncHumanInputProvider(store=store)
        request = _make_request()

        task = asyncio.create_task(provider.request_input(request))
        await asyncio.sleep(0)

        assert request.request_id in store._requests

        # Clean up
        await provider.resolve(request.request_id, _make_response())
        await task

    async def test_resolve_persists_response_to_store(self) -> None:
        store = InMemoryHitlRequestStore()
        provider = AsyncHumanInputProvider(store=store)
        request = _make_request()
        response = _make_response()

        task = asyncio.create_task(provider.request_input(request))
        await asyncio.sleep(0)

        await provider.resolve(request.request_id, response)
        await task

        # Response should be persisted (awaited, not fire-and-forget)
        stored = await store.get_response(request.request_id)
        assert stored == response

    async def test_resolve_without_store_still_works(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()
        response = _make_response()

        task = asyncio.create_task(provider.request_input(request))
        await asyncio.sleep(0)

        assert await provider.resolve(request.request_id, response) is True
        assert (await task) == response


class TestDuplicateRequestGuard:
    async def test_duplicate_request_id_raises(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()

        # First request suspends normally
        asyncio.create_task(provider.request_input(request))  # noqa: RUF006
        await asyncio.sleep(0)

        # Second call with same request_id must raise rather than silently orphan the first future
        import pytest

        with pytest.raises(ValueError, match="req-1"):
            await provider.request_input(request)


class TestConcurrentRequests:
    async def test_multiple_requests_resolve_independently(self) -> None:
        provider = AsyncHumanInputProvider()
        req1 = _make_request(request_id="req-1")
        req2 = _make_request(request_id="req-2")
        resp1 = _make_response(request_id="req-1")
        resp2 = HumanInputResponse(
            request_id="req-2",
            decision=HumanDecision.APPROVE,
            content="Approved",
        )

        task1 = asyncio.create_task(provider.request_input(req1))
        task2 = asyncio.create_task(provider.request_input(req2))
        await asyncio.sleep(0)

        # Resolve in reverse order
        await provider.resolve("req-2", resp2)
        result2 = await task2
        assert result2 == resp2
        assert not task1.done()

        await provider.resolve("req-1", resp1)
        result1 = await task1
        assert result1 == resp1

    async def test_two_requests_are_pending_simultaneously(self) -> None:
        """Deterministic proof that two ``ask_human`` suspensions coexist.

        Two ``request_input`` calls run as concurrent tasks; before either is
        resolved, ``get_pending()`` must observe both. Then the second request
        is resolved before the first — ``request_id``-based routing means each
        ``request_input`` returns the response matching its own id, not the
        order in which the futures were created.
        """
        provider = AsyncHumanInputProvider()
        req1 = _make_request(request_id="req-1")
        req2 = _make_request(request_id="req-2")
        resp1 = _make_response(request_id="req-1")
        resp2 = HumanInputResponse(
            request_id="req-2",
            decision=HumanDecision.APPROVE,
            content="Approved",
        )

        task1 = asyncio.create_task(provider.request_input(req1))
        task2 = asyncio.create_task(provider.request_input(req2))
        await asyncio.sleep(0)

        # Both futures are pending simultaneously — the canonical concurrency
        # property that no live-LLM test can assert without racing response
        # timing.
        pending_before = provider.get_pending()
        assert len(pending_before) == 2
        assert {r.request_id for r in pending_before} == {"req-1", "req-2"}

        # Reverse-order resolution: req-2 first. Each request_input returns the
        # response whose request_id matches, regardless of resolution order.
        assert await provider.resolve("req-2", resp2) is True
        result2 = await task2
        assert result2 == resp2
        assert len(provider.get_pending()) == 1
        assert provider.get_pending()[0].request_id == "req-1"

        assert await provider.resolve("req-1", resp1) is True
        result1 = await task1
        assert result1 == resp1
        assert len(provider.get_pending()) == 0


class TestTimeout:
    """Fail-closed timeout semantics — expired requests raise ``ApprovalTimeoutError``."""

    async def test_default_timeout_expires(self) -> None:
        provider = AsyncHumanInputProvider(default_timeout=0.05)
        request = _make_request()
        with pytest.raises(ApprovalTimeoutError):
            await provider.request_input(request)
        # Pending entry must be cleaned up on timeout.
        assert provider.get_pending() == []

    async def test_per_call_timeout_overrides_default(self) -> None:
        provider = AsyncHumanInputProvider(default_timeout=10.0)
        request = _make_request()
        with pytest.raises(ApprovalTimeoutError):
            await provider.request_input(request, timeout=0.05)
        assert provider.get_pending() == []

    async def test_no_timeout_preserves_infinite_wait(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()
        response = _make_response()

        async def resolve_later() -> None:
            await asyncio.sleep(0.01)
            await provider.resolve(request.request_id, response)

        task = asyncio.create_task(resolve_later())
        result = await provider.request_input(request)
        assert result == response
        await task

    async def test_resolve_after_timeout_returns_false(self) -> None:
        provider = AsyncHumanInputProvider(default_timeout=0.05)
        request = _make_request()
        with pytest.raises(ApprovalTimeoutError):
            await provider.request_input(request)
        # The entry was removed; a resolve for the same id returns False.
        delivered = await provider.resolve(request.request_id, _make_response())
        assert delivered is False

    async def test_negative_default_timeout_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="default_timeout"):
            AsyncHumanInputProvider(default_timeout=-1.0)

    async def test_zero_default_timeout_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="default_timeout"):
            AsyncHumanInputProvider(default_timeout=0.0)

    async def test_negative_per_call_timeout_raises_valueerror(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()
        with pytest.raises(ValueError, match="timeout"):
            await provider.request_input(request, timeout=-1.0)

    async def test_zero_per_call_timeout_raises_valueerror(self) -> None:
        provider = AsyncHumanInputProvider()
        request = _make_request()
        with pytest.raises(ValueError, match="timeout"):
            await provider.request_input(request, timeout=0.0)
