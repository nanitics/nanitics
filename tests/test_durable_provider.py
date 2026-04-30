from unittest.mock import AsyncMock

import pytest

from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.hitl_store import InMemoryHitlRequestStore
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.infrastructure.errors import (
    ApprovalTimeoutError,
    ApprovalUnavailableError,
)


class TestDurableHumanInputProvider:
    @pytest.fixture
    def store(self) -> InMemoryHitlRequestStore:
        return InMemoryHitlRequestStore()

    @pytest.fixture
    def provider(self, store: InMemoryHitlRequestStore) -> DurableHumanInputProvider:
        return DurableHumanInputProvider(request_store=store)

    async def test_first_call_persists_and_raises_when_no_stored_response(
        self,
        provider: DurableHumanInputProvider,
        store: InMemoryHitlRequestStore,
    ) -> None:
        request = HumanInputRequest(
            request_id="req-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )

        with pytest.raises(SuspendExecution) as exc_info:
            await provider.request_input(request)

        assert exc_info.value.suspension_info.suspension_id == "req-1"
        # Request was persisted into the store.
        assert store._requests["req-1"] is request

    async def test_returns_stored_response_when_present(
        self,
        provider: DurableHumanInputProvider,
        store: InMemoryHitlRequestStore,
    ) -> None:
        response = HumanInputResponse(
            request_id="req-1",
            decision=HumanDecision.APPROVE,
            content="Approved",
        )
        await store.save_response("req-1", response)

        request = HumanInputRequest(
            request_id="req-1",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        result = await provider.request_input(request)
        assert result == response

    async def test_duplicate_save_on_re_execution_suspends_not_raises(
        self,
        provider: DurableHumanInputProvider,
        store: InMemoryHitlRequestStore,
    ) -> None:
        # Pre-populate the store with a request (no response yet) — simulating
        # a prior run that already persisted the request before suspending.
        first_request = HumanInputRequest(
            request_id="req-dup",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        await store.save_request(first_request)

        # Re-executing with the same request_id (same deterministic id) must
        # swallow the duplicate and still suspend — no response is stored yet.
        replay_request = HumanInputRequest(
            request_id="req-dup",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(SuspendExecution):
            await provider.request_input(replay_request)

    async def test_resume_after_response_saved(
        self,
        store: InMemoryHitlRequestStore,
    ) -> None:
        # A fresh provider (no prior state) resolves the request from the
        # store alone. This pins the statelessness of the provider.
        request = HumanInputRequest(
            request_id="req-resume",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        await store.save_request(request)
        response = HumanInputResponse(
            request_id="req-resume",
            decision=HumanDecision.APPROVE,
        )
        await store.save_response("req-resume", response)

        provider = DurableHumanInputProvider(request_store=store)
        result = await provider.request_input(request)
        assert result == response

    async def test_provider_has_no_state_between_calls(
        self,
        store: InMemoryHitlRequestStore,
    ) -> None:
        # Two distinct provider instances over the same store behave
        # identically — the defining durability property of statelessness.
        provider_a = DurableHumanInputProvider(request_store=store)
        provider_b = DurableHumanInputProvider(request_store=store)

        request = HumanInputRequest(
            request_id="req-shared",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )

        # provider_a suspends and persists.
        with pytest.raises(SuspendExecution):
            await provider_a.request_input(request)

        # A response is recorded on the store.
        response = HumanInputResponse(
            request_id="req-shared",
            decision=HumanDecision.APPROVE,
        )
        await store.save_response("req-shared", response)

        # provider_b — which never saw the first call — resolves via the store.
        result = await provider_b.request_input(request)
        assert result == response


class TestDurableProviderFailClosed:
    """Fail-closed wrap around store backend failures.

    A store exception that is not a known control-flow signal
    (``DuplicateHitlRequestError``, ``SuspendExecution``,
    ``HumanInputProviderError``) is re-raised as
    ``ApprovalUnavailableError`` with ``__cause__`` preserved so the
    caller can distinguish "store unreachable" from "wrong input."
    """

    async def test_get_response_backend_failure_raises_approval_unavailable(
        self,
    ) -> None:
        store = AsyncMock()
        store.get_response.side_effect = RuntimeError("db down")
        provider = DurableHumanInputProvider(request_store=store)
        request = HumanInputRequest(
            request_id="req-backend-get",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(ApprovalUnavailableError) as exc_info:
            await provider.request_input(request)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "req-backend-get" in exc_info.value.message

    async def test_save_request_backend_failure_raises_approval_unavailable(
        self,
    ) -> None:
        store = AsyncMock()
        store.get_response.return_value = None
        store.save_request.side_effect = RuntimeError("disk full")
        provider = DurableHumanInputProvider(request_store=store)
        request = HumanInputRequest(
            request_id="req-backend-save",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(ApprovalUnavailableError) as exc_info:
            await provider.request_input(request)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "req-backend-save" in exc_info.value.message

    async def test_already_typed_error_on_get_response_passes_through(
        self,
    ) -> None:
        store = AsyncMock()
        inner = ApprovalTimeoutError("already typed")
        store.get_response.side_effect = inner
        provider = DurableHumanInputProvider(request_store=store)
        request = HumanInputRequest(
            request_id="req-typed-get",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(ApprovalTimeoutError) as exc_info:
            await provider.request_input(request)
        # Must not be double-wrapped.
        assert exc_info.value is inner

    async def test_already_typed_error_on_save_request_passes_through(
        self,
    ) -> None:
        store = AsyncMock()
        store.get_response.return_value = None
        inner = ApprovalTimeoutError("already typed on save")
        store.save_request.side_effect = inner
        provider = DurableHumanInputProvider(request_store=store)
        request = HumanInputRequest(
            request_id="req-typed-save",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(ApprovalTimeoutError) as exc_info:
            await provider.request_input(request)
        assert exc_info.value is inner

    async def test_suspend_execution_on_get_response_passes_through(
        self,
    ) -> None:
        """``SuspendExecution`` raised by a store is a control-flow signal — do not wrap."""
        store = AsyncMock()
        from nanitics.composition.durability.models import SuspensionInfo

        store.get_response.side_effect = SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id="x",
                request_id="x",
                request_type="APPROVAL",
                prompt="x",
                agent_name=None,
            ),
        )
        provider = DurableHumanInputProvider(request_store=store)
        request = HumanInputRequest(
            request_id="req-suspend",
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(SuspendExecution):
            await provider.request_input(request)
