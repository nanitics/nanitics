import pytest

from nanitics.collaboration.hitl_store import (
    DuplicateHitlRequestError,
    InMemoryHitlRequestStore,
)
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)


def _make_request(
    request_id: str = "req-1",
    run_id: str = "run-1",
) -> HumanInputRequest:
    return HumanInputRequest(
        request_id=request_id,
        run_id=run_id,
        request_type=HumanInputType.APPROVAL,
        prompt="Approve?",
    )


def _make_response(request_id: str = "req-1") -> HumanInputResponse:
    return HumanInputResponse(
        request_id=request_id,
        decision=HumanDecision.APPROVE,
        content="Approved",
    )


class TestInMemoryHitlRequestStore:
    @pytest.fixture
    def store(self) -> InMemoryHitlRequestStore:
        return InMemoryHitlRequestStore()

    async def test_save_and_get_response(self, store: InMemoryHitlRequestStore) -> None:
        response = _make_response("req-1")
        await store.save_response("req-1", response)
        loaded = await store.get_response("req-1")
        assert loaded is not None
        assert loaded.request_id == "req-1"
        assert loaded.decision == HumanDecision.APPROVE

    async def test_get_response_returns_none_for_missing(self, store: InMemoryHitlRequestStore) -> None:
        result = await store.get_response("nonexistent")
        assert result is None

    async def test_save_request_and_get_pending(self, store: InMemoryHitlRequestStore) -> None:
        req = _make_request("req-1", run_id="run-1")
        await store.save_request(req)
        pending = await store.get_pending_requests("run-1")
        assert len(pending) == 1
        assert pending[0].request_id == "req-1"

    async def test_responded_requests_not_in_pending(self, store: InMemoryHitlRequestStore) -> None:
        req = _make_request("req-1", run_id="run-1")
        await store.save_request(req)
        await store.save_response("req-1", _make_response("req-1"))
        pending = await store.get_pending_requests("run-1")
        assert len(pending) == 0

    async def test_pending_filtered_by_run_id(self, store: InMemoryHitlRequestStore) -> None:
        req1 = _make_request("req-1", run_id="run-1")
        req2 = _make_request("req-2", run_id="run-2")
        await store.save_request(req1)
        await store.save_request(req2)

        pending_run1 = await store.get_pending_requests("run-1")
        assert len(pending_run1) == 1
        assert pending_run1[0].request_id == "req-1"

        pending_run2 = await store.get_pending_requests("run-2")
        assert len(pending_run2) == 1
        assert pending_run2[0].request_id == "req-2"

    async def test_get_pending_returns_empty_for_no_match(self, store: InMemoryHitlRequestStore) -> None:
        pending = await store.get_pending_requests("nonexistent")
        assert pending == []

    async def test_request_without_run_id_not_in_pending(self, store: InMemoryHitlRequestStore) -> None:
        req = HumanInputRequest(
            request_id="req-no-run",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        await store.save_request(req)
        pending = await store.get_pending_requests("run-1")
        assert len(pending) == 0


class TestInMemoryHitlRequestStoreRunIdFallback:
    async def test_store_run_id_applied_to_request_without_run_id(self) -> None:
        store = InMemoryHitlRequestStore(run_id="run-fallback")
        req = HumanInputRequest(
            request_id="req-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        await store.save_request(req)
        pending = await store.get_pending_requests("run-fallback")
        assert len(pending) == 1
        assert pending[0].run_id == "run-fallback"

    async def test_request_run_id_takes_precedence_over_store_run_id(self) -> None:
        store = InMemoryHitlRequestStore(run_id="store-run")
        req = HumanInputRequest(
            request_id="req-1",
            run_id="request-run",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        await store.save_request(req)
        pending = await store.get_pending_requests("request-run")
        assert len(pending) == 1
        assert pending[0].run_id == "request-run"

        pending_store = await store.get_pending_requests("store-run")
        assert len(pending_store) == 0

    async def test_no_store_run_id_preserves_none(self) -> None:
        store = InMemoryHitlRequestStore()
        req = HumanInputRequest(
            request_id="req-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        await store.save_request(req)
        pending = await store.get_pending_requests("any-run")
        assert len(pending) == 0


class TestInMemoryHitlRequestStoreDuplicateSave:
    async def test_duplicate_save_raises_duplicate_hitl_request_error(self) -> None:
        store = InMemoryHitlRequestStore()
        req = _make_request("req-dup", run_id="run-1")
        await store.save_request(req)

        with pytest.raises(DuplicateHitlRequestError) as exc_info:
            await store.save_request(_make_request("req-dup", run_id="run-1"))

        assert exc_info.value.request_id == "req-dup"

    async def test_duplicate_save_does_not_mutate_store(self) -> None:
        store = InMemoryHitlRequestStore()
        original = _make_request("req-dup", run_id="run-1")
        await store.save_request(original)

        replacement = HumanInputRequest(
            request_id="req-dup",
            run_id="run-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Different prompt",
        )
        with pytest.raises(DuplicateHitlRequestError):
            await store.save_request(replacement)

        # The original request instance is still stored, not the replacement.
        assert store._requests["req-dup"] is original
        assert store._requests["req-dup"].prompt == "Approve?"
