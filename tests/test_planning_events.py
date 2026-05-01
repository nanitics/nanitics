"""Tests for planning events: construction, event_type literals, TraceEvent deserialization round-trip."""

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from nanitics.infrastructure.observability.events import (
    GoalStatusChangedEvent,
    PlanCreatedEvent,
    PlanRevisedEvent,
    PlanStepUpdatedEvent,
    TraceEvent,
)

TRACE_ID = "trace-001"
SPAN_ID = "span-001"


def _base_fields(**overrides: Any) -> dict[str, Any]:
    defaults = {"trace_id": TRACE_ID, "span_id": SPAN_ID}
    defaults.update(overrides)
    return defaults


# ──────────────────────────────────────────────────────────
# Event Construction
# ──────────────────────────────────────────────────────────


class TestPlanCreatedEvent:
    def test_construction(self) -> None:
        evt = PlanCreatedEvent(
            **_base_fields(),
            plan_id="p1",
            plan_name="My plan",
            step_count=3,
            goal_count=1,
            namespace="ns",
        )
        assert evt.event_type == "planning.plan.created"
        assert evt.plan_id == "p1"
        assert evt.plan_name == "My plan"
        assert evt.step_count == 3
        assert evt.goal_count == 1
        assert evt.namespace == "ns"

    def test_namespace_defaults_to_none(self) -> None:
        evt = PlanCreatedEvent(
            **_base_fields(),
            plan_id="p1",
            plan_name="Plan",
            step_count=0,
            goal_count=0,
        )
        assert evt.namespace is None


class TestPlanStepUpdatedEvent:
    def test_construction(self) -> None:
        evt = PlanStepUpdatedEvent(
            **_base_fields(),
            plan_id="p1",
            step_id="s1",
            step_description="Do something",
            previous_status="not_started",
            new_status="in_progress",
            has_result=False,
        )
        assert evt.event_type == "planning.step.updated"
        assert evt.plan_id == "p1"
        assert evt.step_id == "s1"
        assert evt.has_result is False


class TestPlanRevisedEvent:
    def test_construction(self) -> None:
        evt = PlanRevisedEvent(
            **_base_fields(),
            plan_id="p1",
            steps_before=5,
            steps_after=3,
            steps_preserved=2,
            revision_reason="New information",
        )
        assert evt.event_type == "planning.plan.revised"
        assert evt.steps_before == 5
        assert evt.steps_after == 3
        assert evt.steps_preserved == 2
        assert evt.revision_reason == "New information"


class TestGoalStatusChangedEvent:
    def test_construction(self) -> None:
        evt = GoalStatusChangedEvent(
            **_base_fields(),
            plan_id="p1",
            goal_id="g1",
            goal_description="Achieve X",
            previous_status="active",
            new_status="achieved",
        )
        assert evt.event_type == "planning.goal.status_changed"
        assert evt.goal_id == "g1"
        assert evt.goal_description == "Achieve X"


# ──────────────────────────────────────────────────────────
# TraceEvent Discriminated Union Round-Trip
# ──────────────────────────────────────────────────────────


class TestTraceEventRoundTrip:
    adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)

    @pytest.mark.parametrize(
        ("event_cls", "extra_fields", "expected_type"),
        [
            (
                PlanCreatedEvent,
                {
                    "plan_id": "p1",
                    "plan_name": "Plan",
                    "step_count": 2,
                    "goal_count": 0,
                },
                "planning.plan.created",
            ),
            (
                PlanStepUpdatedEvent,
                {
                    "plan_id": "p1",
                    "step_id": "s1",
                    "step_description": "Step",
                    "previous_status": "not_started",
                    "new_status": "completed",
                    "has_result": True,
                },
                "planning.step.updated",
            ),
            (
                PlanRevisedEvent,
                {
                    "plan_id": "p1",
                    "steps_before": 3,
                    "steps_after": 2,
                    "steps_preserved": 1,
                    "revision_reason": "Changed approach",
                },
                "planning.plan.revised",
            ),
            (
                GoalStatusChangedEvent,
                {
                    "plan_id": "p1",
                    "goal_id": "g1",
                    "goal_description": "Goal",
                    "previous_status": "active",
                    "new_status": "achieved",
                },
                "planning.goal.status_changed",
            ),
        ],
    )
    def test_roundtrip(self, event_cls: type[BaseModel], extra_fields: dict[str, Any], expected_type: str) -> None:
        evt = event_cls(**_base_fields(), **extra_fields)
        json_data = evt.model_dump_json()
        restored = self.adapter.validate_json(json_data)
        assert type(restored) is event_cls
        assert restored.event_type == expected_type
