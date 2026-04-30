"""Trivial-accessor coverage for bound workflow/step wrappers.

Covers ``BoundWorkflow.workflow`` / ``BoundWorkflow.emitter`` and the
``name`` property on ``_BoundAgentStep``, ``_BoundHandoffStep``, and
``_BoundWorkflowStep``. Execution paths are exercised by the heavier
orchestration tests — these cases isolate the pure property getters.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nanitics.composition.orchestration.workflow import (
    BoundWorkflow,
    _BoundAgentStep,
    _BoundHandoffStep,
    _BoundWorkflowStep,
)


class TestBoundWorkflowAccessors:
    def test_workflow_property_returns_injected_workflow(self) -> None:
        workflow = MagicMock()
        emitter = MagicMock()
        bound = BoundWorkflow(workflow, emitter)
        assert bound.workflow is workflow

    def test_emitter_property_returns_injected_emitter(self) -> None:
        workflow = MagicMock()
        emitter = MagicMock()
        bound = BoundWorkflow(workflow, emitter)
        assert bound.emitter is emitter


class TestBoundAgentStepName:
    def test_name_returns_wrapped_step_name(self) -> None:
        step = MagicMock()
        step.name = "wrapped-agent"
        bound_step = _BoundAgentStep(step, MagicMock())
        assert bound_step.name == "wrapped-agent"


class TestBoundHandoffStepName:
    def test_name_returns_wrapped_step_name(self) -> None:
        step = MagicMock()
        step.name = "wrapped-handoff"
        bound_step = _BoundHandoffStep(step, MagicMock(), MagicMock())
        assert bound_step.name == "wrapped-handoff"


class TestBoundWorkflowStepName:
    def test_name_returns_wrapped_step_name(self) -> None:
        step = MagicMock()
        step.name = "wrapped-workflow"
        bound_step = _BoundWorkflowStep(step, MagicMock())
        assert bound_step.name == "wrapped-workflow"
