"""Tests for ApprovalGate: approve, reject, modify, event emission, Sequential integration."""

from typing import Any

import pytest

from nanitics.collaboration.approval_gate import ApprovalGate
from nanitics.collaboration.protocol import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
)
from nanitics.composition.orchestration.adapters import FunctionStep
from nanitics.composition.orchestration.protocol import Step
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.events import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)
from tests.testing_helpers import make_emitter

# ``ApprovalGate`` now derives ``request_id = {run_id}:{name}:{revision_count}``
# and refuses to execute without a ``run_id``. Tests in this module use a
# fixed placeholder; tests that care about identity override it explicitly.
_DEFAULT_RUN_ID = "test-run"


def make_provider(
    decision: HumanDecision = HumanDecision.APPROVE,
    content: str | None = None,
) -> CallbackHumanInputProvider:
    return CallbackHumanInputProvider(
        callback=lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=decision,
            content=content,
        )
    )


def make_gate(**kwargs: Any) -> ApprovalGate:
    """Construct an ``ApprovalGate`` with a default ``run_id`` applied.

    Tests that care about identity override ``run_id`` explicitly; all
    others rely on the shared ``_DEFAULT_RUN_ID`` so the suite does not
    pepper every construction with the same literal.
    """
    kwargs.setdefault("run_id", _DEFAULT_RUN_ID)
    return ApprovalGate(**kwargs)


class TestApprovalGateConstruction:
    def test_satisfies_step_protocol(self) -> None:
        gate = make_gate(provider=make_provider())
        assert isinstance(gate, Step)

    def test_name_property(self) -> None:
        gate = make_gate(provider=make_provider(), name="review_gate")
        assert gate.name == "review_gate"

    def test_default_name(self) -> None:
        gate = make_gate(provider=make_provider())
        assert gate.name == "approval_gate"


class TestApprovalGateExecution:
    async def test_approve_passes_input_through(self) -> None:
        gate = make_gate(provider=make_provider(HumanDecision.APPROVE))
        result = await gate.execute("some data")
        assert result.output == "some data"

    async def test_reject_returns_none_output_with_metadata(self) -> None:
        gate = make_gate(provider=make_provider(HumanDecision.REJECT, content="Not ready"))
        result = await gate.execute("some data")
        assert result.output is None
        assert result.metadata["rejected"] is True
        assert result.metadata["reason"] == "Not ready"

    async def test_override_returns_modified_content(self) -> None:
        gate = make_gate(provider=make_provider(HumanDecision.OVERRIDE, content="modified data"))
        result = await gate.execute("original data")
        assert result.output == "modified data"
        assert result.metadata.get("modified") is True

    async def test_static_prompt(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = make_gate(provider=provider, prompt="Approve this plan?")
        await gate.execute("data")
        assert captured[0].prompt == "Approve this plan?"

    async def test_callable_prompt(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = make_gate(
            provider=provider,
            prompt=lambda inp: f"Agent produced: {inp}\nApprove?",
        )
        await gate.execute("analysis results")
        assert "Agent produced: analysis results" in captured[0].prompt

    async def test_static_context(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = make_gate(
            provider=provider,
            prompt="Approve?",
            context="Here is the data to review.",
        )
        await gate.execute("data")
        assert captured[0].context == "Here is the data to review."

    async def test_callable_context(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = make_gate(
            provider=provider,
            prompt="Approve?",
            context=lambda inp: f"## Analysis\n{inp}",
        )
        await gate.execute("detailed results")
        assert captured[0].context == "## Analysis\ndetailed results"

    async def test_no_context_defaults_to_none(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = make_gate(provider=provider, prompt="Approve?")
        await gate.execute("data")
        assert captured[0].context is None

    async def test_agent_name_from_kwarg_or_none(self) -> None:
        """``agent_name`` on the request reflects the kwarg passed to the gate.

        When the adopter does not pass ``agent_name``, the field is ``None``
        on the request — the gate does not fall back to its own ``name``.
        When passed, the value flows verbatim onto the request.
        """
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate_without = make_gate(provider=provider, name="my_gate")
        await gate_without.execute("data")
        assert captured[0].agent_name is None

        gate_with = make_gate(provider=provider, name="my_gate", agent_name="drafter")
        await gate_with.execute("data")
        assert captured[1].agent_name == "drafter"

    async def test_escalate_treated_as_rejection(self) -> None:
        gate = make_gate(provider=make_provider(HumanDecision.ESCALATE, content="Need manager"))
        result = await gate.execute("data")
        assert result.output is None
        assert result.metadata["rejected"] is True


class TestApprovalGateEvents:
    async def test_emits_request_and_response_events(self) -> None:
        emitter = make_emitter()
        gate = make_gate(
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await gate.execute("data")
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(req_events) == 1
        assert req_events[0].request_type == "approval"
        assert len(resp_events) == 1
        assert resp_events[0].decision == "approve"

    async def test_context_included_in_event(self) -> None:
        emitter = make_emitter()
        gate = make_gate(
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
            prompt="Approve?",
            context=lambda inp: f"Data: {inp}",
        )
        await gate.execute("test-data")
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].context == "Data: test-data"

    async def test_no_context_event_field_is_none(self) -> None:
        emitter = make_emitter()
        gate = make_gate(
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await gate.execute("data")
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert req_events[0].context is None

    async def test_rejection_emits_events(self) -> None:
        emitter = make_emitter()
        gate = make_gate(
            provider=make_provider(HumanDecision.REJECT, content="No"),
            emitter=emitter,
        )
        await gate.execute("data")
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(resp_events) == 1
        assert resp_events[0].decision == "reject"
        assert resp_events[0].has_content is True

    async def test_event_agent_name_reflects_kwarg(self) -> None:
        """``agent_name`` on the emitted event mirrors the gate's kwarg.

        Pins Step 3 contract: the gate threads ``agent_name`` onto
        ``HumanInputRequestEvent`` so adopters can filter HITL events by
        producer agent without re-parsing the request payload.
        """
        emitter = make_emitter()
        gate = make_gate(
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
            agent_name="drafter",
        )
        await gate.execute("data")
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].agent_name == "drafter"

    async def test_event_agent_name_defaults_to_none(self) -> None:
        """Without the kwarg the event's ``agent_name`` is ``None``.

        The gate no longer falls back to ``self._name`` — when the adopter
        does not wire ``agent_name``, the field is honestly ``None`` on the
        event (matches the request-level contract in
        ``test_agent_name_from_kwarg_or_none``).
        """
        emitter = make_emitter()
        gate = make_gate(
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await gate.execute("data")
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].agent_name is None

    async def test_event_metadata_mirrors_request_metadata(self) -> None:
        """The event's ``metadata`` carries the same dict as the request.

        Pins Fork 3: every HITL surface that populates ``HumanInputRequest.metadata``
        also mirrors it onto ``HumanInputRequestEvent.metadata`` so machine
        consumers can read it off the event stream.
        """
        emitter = make_emitter()
        gate = make_gate(
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
            name="my_gate",
            allow_revision=True,
        )
        await gate.execute("data")
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].metadata == {"step_name": "my_gate", "allow_revision": True}


class TestApprovalGateRevision:
    async def test_revise_returns_revision_requested_metadata_with_feedback(self) -> None:
        gate = make_gate(
            provider=make_provider(HumanDecision.REVISE, content="Add more detail"),
            allow_revision=True,
        )
        result = await gate.execute("analysis output")
        assert result.output is None
        assert result.metadata["revision_requested"] is True
        assert result.metadata["feedback"] == "Add more detail"

    async def test_revise_empty_feedback_defaults_to_empty_string(self) -> None:
        gate = make_gate(
            provider=make_provider(HumanDecision.REVISE, content=None),
            allow_revision=True,
        )
        result = await gate.execute("data")
        assert result.metadata["feedback"] == ""

    async def test_allow_revision_true_includes_flag_in_request_metadata(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = make_gate(provider=provider, allow_revision=True)
        await gate.execute("data")
        assert captured[0].metadata["allow_revision"] is True

    async def test_allow_revision_false_does_not_include_flag(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = make_gate(provider=provider, allow_revision=False)
        await gate.execute("data")
        assert "allow_revision" not in captured[0].metadata


class TestApprovalGateDeterministicIdentity:
    async def test_request_id_derived_from_run_id_step_name_and_revision_count(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = ApprovalGate(provider=provider, run_id="r", name="gate-A")
        await gate.execute("data", revision_count=0)
        assert captured[0].request_id == "r:gate-A:0"
        assert captured[0].run_id == "r"

    async def test_request_id_varies_by_revision_count(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = ApprovalGate(provider=provider, run_id="r", name="gate-A")
        await gate.execute("data", revision_count=2)
        assert captured[0].request_id == "r:gate-A:2"

    async def test_request_id_stable_across_re_execution(self) -> None:
        """Deterministic id means the same slot collides on resume re-execute."""
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        gate = ApprovalGate(provider=provider, run_id="r", name="gate-A")
        await gate.execute("data", revision_count=0)
        await gate.execute("data", revision_count=0)
        assert captured[0].request_id == captured[1].request_id == "r:gate-A:0"

    async def test_missing_run_id_raises(self) -> None:
        gate = ApprovalGate(provider=make_provider(HumanDecision.APPROVE))
        with pytest.raises(ValueError, match="run_id"):
            await gate.execute("data")


class TestApprovalGateSequentialIntegration:
    async def test_approve_in_sequential_workflow(self) -> None:
        emitter = make_emitter()

        async def analyze(x):
            return f"analyzed: {x}"

        async def act(x):
            return f"acted on: {x}"

        gate = make_gate(
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
            prompt=lambda inp: f"Review: {inp}",
            name="review",
        )

        seq = Sequential(
            name="workflow",
            steps=[
                FunctionStep(name="analyze", fn=analyze),
                gate,
                FunctionStep(name="act", fn=act),
            ],
            emitter=emitter,
        )
        result = await seq.execute("input data")
        assert result.output == "acted on: analyzed: input data"

    async def test_reject_passes_none_to_subsequent_steps(self) -> None:
        emitter = make_emitter()
        act_called = False

        async def analyze(x):
            return f"analyzed: {x}"

        async def act(x):
            nonlocal act_called
            act_called = True
            return f"acted on: {x}"

        gate = make_gate(
            provider=make_provider(HumanDecision.REJECT, content="Not approved"),
            emitter=emitter,
            name="review",
        )

        seq = Sequential(
            name="workflow",
            steps=[
                FunctionStep(name="analyze", fn=analyze),
                gate,
                FunctionStep(name="act", fn=act),
            ],
            emitter=emitter,
        )
        result = await seq.execute("input data")
        # Sequential passes the gate's None output to the next step — it does not halt
        assert act_called is True
        assert result.output == "acted on: None"
