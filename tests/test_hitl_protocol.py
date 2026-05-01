import asyncio

import pytest
from pydantic import ValidationError

from nanitics.collaboration.protocol import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputProvider,
    HumanInputRequest,
    HumanInputResponse,
    HumanInputType,
)


class TestHumanInputType:
    def test_values(self) -> None:
        assert HumanInputType.APPROVAL.value == "approval"
        assert HumanInputType.QUESTION.value == "question"

    def test_members(self) -> None:
        assert {member.value for member in HumanInputType} == {"approval", "question"}

    def test_is_str(self) -> None:
        assert isinstance(HumanInputType.APPROVAL, str)


class TestHumanDecision:
    def test_values(self) -> None:
        assert HumanDecision.APPROVE.value == "approve"
        assert HumanDecision.REJECT.value == "reject"
        assert HumanDecision.OVERRIDE.value == "override"
        assert HumanDecision.ANSWER.value == "answer"
        assert HumanDecision.ESCALATE.value == "escalate"

    def test_is_str(self) -> None:
        assert isinstance(HumanDecision.APPROVE, str)


class TestHumanInputRequest:
    def test_creates_with_defaults(self) -> None:
        request = HumanInputRequest(
            request_type=HumanInputType.APPROVAL,
            prompt="Approve this action?",
        )
        assert request.request_id  # auto-generated
        assert request.request_type == HumanInputType.APPROVAL
        assert request.prompt == "Approve this action?"
        assert request.context is None
        assert request.options is None
        assert request.metadata == {}
        assert request.agent_name is None

    def test_creates_with_all_fields(self) -> None:
        request = HumanInputRequest(
            request_id="test-id",
            request_type=HumanInputType.QUESTION,
            prompt="What color?",
            context="Choosing a theme",
            options=["red", "blue"],
            metadata={"key": "value"},
            agent_name="researcher",
        )
        assert request.request_id == "test-id"
        assert request.options == ["red", "blue"]
        assert request.agent_name == "researcher"

    def test_frozen(self) -> None:
        request = HumanInputRequest(
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        with pytest.raises(ValidationError):
            request.prompt = "Changed"

    def test_unique_request_ids(self) -> None:
        r1 = HumanInputRequest(request_type=HumanInputType.APPROVAL, prompt="a")
        r2 = HumanInputRequest(request_type=HumanInputType.APPROVAL, prompt="b")
        assert r1.request_id != r2.request_id

    def test_serialization_roundtrip(self) -> None:
        request = HumanInputRequest(
            request_type=HumanInputType.APPROVAL,
            prompt="Review this plan",
            context="Step 1, Step 2",
            options=["approve", "reject"],
            metadata={"steps": 2},
            agent_name="planner",
        )
        data = request.model_dump()
        restored = HumanInputRequest(**data)
        assert restored == request


class TestHumanInputResponse:
    def test_creates_with_defaults(self) -> None:
        response = HumanInputResponse(
            request_id="req-1",
            decision=HumanDecision.APPROVE,
        )
        assert response.request_id == "req-1"
        assert response.decision == HumanDecision.APPROVE
        assert response.content is None
        assert response.metadata == {}
        assert response.responded_at is not None

    def test_creates_with_all_fields(self) -> None:
        response = HumanInputResponse(
            request_id="req-1",
            decision=HumanDecision.OVERRIDE,
            content="Use parameter X instead",
            metadata={"modified_params": {"x": 1}},
        )
        assert response.content == "Use parameter X instead"

    def test_frozen(self) -> None:
        response = HumanInputResponse(
            request_id="req-1",
            decision=HumanDecision.APPROVE,
        )
        with pytest.raises(ValidationError):
            response.decision = HumanDecision.REJECT

    def test_serialization_roundtrip(self) -> None:
        response = HumanInputResponse(
            request_id="req-1",
            decision=HumanDecision.ANSWER,
            content="42",
        )
        data = response.model_dump()
        restored = HumanInputResponse(**data)
        assert restored == response


class TestHumanInputProviderProtocol:
    def test_callback_provider_satisfies_protocol(self) -> None:
        provider = CallbackHumanInputProvider(
            callback=lambda req: HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.APPROVE,
            )
        )
        assert isinstance(provider, HumanInputProvider)


class TestCallbackHumanInputProvider:
    async def test_sync_callback(self) -> None:
        def approve(req: HumanInputRequest) -> HumanInputResponse:
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.APPROVE,
                content="Looks good",
            )

        provider = CallbackHumanInputProvider(callback=approve)
        request = HumanInputRequest(
            request_type=HumanInputType.APPROVAL,
            prompt="Approve?",
        )
        response = await provider.request_input(request)
        assert response.decision == HumanDecision.APPROVE
        assert response.content == "Looks good"
        assert response.request_id == request.request_id

    async def test_async_callback(self) -> None:
        async def answer(req: HumanInputRequest) -> HumanInputResponse:
            await asyncio.sleep(0)  # simulate async work
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.ANSWER,
                content="The answer is 42",
            )

        provider = CallbackHumanInputProvider(callback=answer)
        request = HumanInputRequest(
            request_type=HumanInputType.QUESTION,
            prompt="What is the meaning of life?",
        )
        response = await provider.request_input(request)
        assert response.decision == HumanDecision.ANSWER
        assert response.content == "The answer is 42"

    async def test_callback_receives_full_request(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.APPROVE,
            )

        provider = CallbackHumanInputProvider(callback=capture)
        request = HumanInputRequest(
            request_type=HumanInputType.APPROVAL,
            prompt="Review plan",
            context="Steps: A, B, C",
            agent_name="planner",
            metadata={"step_count": 3},
        )
        await provider.request_input(request)

        assert len(captured) == 1
        assert captured[0].prompt == "Review plan"
        assert captured[0].context == "Steps: A, B, C"
        assert captured[0].agent_name == "planner"
        assert captured[0].metadata == {"step_count": 3}

    async def test_reject_with_reason(self) -> None:
        provider = CallbackHumanInputProvider(
            callback=lambda req: HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.REJECT,
                content="Too risky",
            )
        )
        request = HumanInputRequest(
            request_type=HumanInputType.APPROVAL,
            prompt="Delete all data?",
        )
        response = await provider.request_input(request)
        assert response.decision == HumanDecision.REJECT
        assert response.content == "Too risky"

    async def test_escalate_decision(self) -> None:
        provider = CallbackHumanInputProvider(
            callback=lambda req: HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.ESCALATE,
                content="Need manager approval",
            )
        )
        request = HumanInputRequest(
            request_type=HumanInputType.APPROVAL,
            prompt="Large purchase",
        )
        response = await provider.request_input(request)
        assert response.decision == HumanDecision.ESCALATE
