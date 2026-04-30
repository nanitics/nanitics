from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from nanitics.infrastructure.observability.events import (
    DelegationEvent,
    HandoffEvent,
    SupervisionEvent,
    TraceEvent,
)

TRACE_ID = "trace-001"
SPAN_ID = "span-001"


def _base_fields(**overrides: Any) -> dict[str, Any]:
    defaults = {"trace_id": TRACE_ID, "span_id": SPAN_ID}
    defaults.update(overrides)
    return defaults


class TestDelegationEvent:
    def test_construction(self):
        evt = DelegationEvent(
            **_base_fields(),
            caller_agent="supervisor",
            delegate_agent="researcher",
            task="find papers on multi-agent systems",
            transfer_strategy="RawOutputTransfer",
        )
        assert evt.event_type == "multi_agent.delegation"
        assert evt.caller_agent == "supervisor"
        assert evt.delegate_agent == "researcher"
        assert evt.task == "find papers on multi-agent systems"
        assert evt.transfer_strategy == "RawOutputTransfer"

    def test_serialization_roundtrip(self):
        evt = DelegationEvent(
            **_base_fields(),
            caller_agent="supervisor",
            delegate_agent="researcher",
            task="summarize findings",
            transfer_strategy="SummaryTransfer",
        )
        data = evt.model_dump()
        restored = DelegationEvent.model_validate(data)
        assert restored == evt

    def test_frozen(self):
        evt = DelegationEvent(
            **_base_fields(),
            caller_agent="a",
            delegate_agent="b",
            task="t",
            transfer_strategy="s",
        )
        with pytest.raises(ValidationError):
            evt.caller_agent = "changed"


class TestHandoffEvent:
    def test_construction(self):
        evt = HandoffEvent(
            **_base_fields(),
            from_agent="researcher",
            to_agent="writer",
            payload_fields=["task_state", "findings", "decisions"],
            payload_size=1500,
        )
        assert evt.event_type == "multi_agent.handoff"
        assert evt.from_agent == "researcher"
        assert evt.to_agent == "writer"
        assert evt.payload_fields == ["task_state", "findings", "decisions"]
        assert evt.payload_size == 1500

    def test_serialization_roundtrip(self):
        evt = HandoffEvent(
            **_base_fields(),
            from_agent="researcher",
            to_agent="writer",
            payload_fields=["task_state"],
            payload_size=200,
        )
        data = evt.model_dump()
        restored = HandoffEvent.model_validate(data)
        assert restored == evt

    def test_frozen(self):
        evt = HandoffEvent(
            **_base_fields(),
            from_agent="a",
            to_agent="b",
            payload_fields=[],
            payload_size=0,
        )
        with pytest.raises(ValidationError):
            evt.from_agent = "changed"


class TestSupervisionEvent:
    def test_construction(self):
        evt = SupervisionEvent(
            **_base_fields(),
            supervised_agent="writer",
            action="retry",
            trigger_name="quality",
            feedback="Improve the conclusion",
            attempt=1,
        )
        assert evt.event_type == "multi_agent.supervision"
        assert evt.supervised_agent == "writer"
        assert evt.action == "retry"
        assert evt.trigger_name == "quality"
        assert evt.feedback == "Improve the conclusion"
        assert evt.reassigned_to is None
        assert evt.attempt == 1

    def test_construction_with_reassignment(self):
        evt = SupervisionEvent(
            **_base_fields(),
            supervised_agent="writer",
            action="reassign",
            trigger_name="quality",
            reassigned_to="senior_writer",
            attempt=2,
        )
        assert evt.action == "reassign"
        assert evt.reassigned_to == "senior_writer"

    def test_serialization_roundtrip(self):
        evt = SupervisionEvent(
            **_base_fields(),
            supervised_agent="writer",
            action="escalate",
            trigger_name="budget",
            feedback="Token budget exceeded: 5000/3000",
            attempt=1,
        )
        data = evt.model_dump()
        restored = SupervisionEvent.model_validate(data)
        assert restored == evt

    def test_frozen(self):
        evt = SupervisionEvent(
            **_base_fields(),
            supervised_agent="writer",
            action="retry",
            trigger_name="quality",
            attempt=1,
        )
        with pytest.raises(ValidationError):
            evt.supervised_agent = "changed"


class TestDiscriminatedUnion:
    adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)

    @pytest.mark.parametrize(
        ("event_cls", "extra_fields", "expected_type"),
        [
            (
                DelegationEvent,
                {
                    "caller_agent": "supervisor",
                    "delegate_agent": "worker",
                    "task": "do work",
                    "transfer_strategy": "RawOutputTransfer",
                },
                "multi_agent.delegation",
            ),
            (
                HandoffEvent,
                {
                    "from_agent": "researcher",
                    "to_agent": "writer",
                    "payload_fields": ["task_state"],
                    "payload_size": 100,
                },
                "multi_agent.handoff",
            ),
            (
                SupervisionEvent,
                {
                    "supervised_agent": "writer",
                    "action": "retry",
                    "trigger_name": "quality",
                    "feedback": "Improve the conclusion",
                    "attempt": 1,
                },
                "multi_agent.supervision",
            ),
        ],
    )
    def test_roundtrip(self, event_cls, extra_fields, expected_type):
        evt = event_cls(**_base_fields(), **extra_fields)
        json_data = evt.model_dump_json()
        restored = self.adapter.validate_json(json_data)
        assert type(restored) is event_cls
        assert restored.event_type == expected_type
        assert restored.event_id == evt.event_id
