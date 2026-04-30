"""Tests for planning tools: factory, execution, event emission, validation, auto-complete, namespace isolation."""

import pytest

from nanitics import Tool, ToolCall, ToolRegistry
from nanitics.capabilities.planning.models import (
    Goal,
    GoalStatus,
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from nanitics.capabilities.planning.store import InMemoryPlanStore
from nanitics.capabilities.planning.tools import create_planning_tools
from nanitics.core.tools.function_tool import FunctionTool
from nanitics.infrastructure.errors import ToolParameterError
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    GoalStatusChangedEvent,
    PlanCreatedEvent,
    PlanRevisedEvent,
    PlanStepUpdatedEvent,
)
from tests.testing_helpers import make_emitter


def make_store() -> InMemoryPlanStore:
    return InMemoryPlanStore()


# ──────────────────────────────────────────────────────────
# Tool Factory
# ──────────────────────────────────────────────────────────


class TestCreatePlanningTools:
    def test_returns_six_tools(self) -> None:
        tools = create_planning_tools(make_store())
        assert len(tools) == 6

    def test_tools_are_function_tools(self) -> None:
        tools = create_planning_tools(make_store())
        for t in tools:
            assert isinstance(t, FunctionTool)
            assert isinstance(t, Tool)

    def test_tool_names(self) -> None:
        tools = create_planning_tools(make_store())
        names = {t.schema.name for t in tools}
        assert names == {"create_plan", "get_plan", "update_step", "revise_plan", "update_goal", "create_goal"}

    def test_tool_schemas_have_descriptions(self) -> None:
        tools = create_planning_tools(make_store())
        for t in tools:
            assert t.schema.description


class TestPlanningToolSchemas:
    """Schema-level assertions pinning the Step-4 hardening contract.

    The planning tools expose JSON schema to every LLM provider adapter.
    These tests pin that:

    - ``update_step.status`` / ``update_goal.status`` export an ``enum``
      (via the ``StepStatus`` / ``GoalStatus`` ``StrEnum``s).
    - Every parameter across all six tools carries a non-empty
      ``description`` in the exported schema.
    """

    def _find(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    def test_update_step_status_exports_enum(self) -> None:
        tools = create_planning_tools(make_store())
        schema = self._find(tools, "update_step").schema.parameters
        step_status_def = schema["$defs"]["StepStatus"]
        assert step_status_def["enum"] == [
            "not_started",
            "in_progress",
            "completed",
            "skipped",
            "failed",
        ]
        # status field references the StepStatus $def
        assert schema["properties"]["status"]["$ref"] == "#/$defs/StepStatus"

    def test_update_goal_status_exports_enum(self) -> None:
        tools = create_planning_tools(make_store())
        schema = self._find(tools, "update_goal").schema.parameters
        goal_status_def = schema["$defs"]["GoalStatus"]
        assert goal_status_def["enum"] == ["active", "achieved", "blocked", "abandoned"]
        assert schema["properties"]["status"]["$ref"] == "#/$defs/GoalStatus"

    def test_every_parameter_has_description(self) -> None:
        """Every property on every planning tool carries a non-empty description."""
        tools = create_planning_tools(make_store())
        for t in tools:
            properties = t.schema.parameters.get("properties", {})
            for param_name, param_schema in properties.items():
                assert param_schema.get("description"), (
                    f"Tool '{t.schema.name}' parameter '{param_name}' is missing a description"
                )


# ──────────────────────────────────────────────────────────
# create_plan Tool
# ──────────────────────────────────────────────────────────


class TestCreatePlanTool:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_creates_plan_with_steps(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        create = self._get_tool(tools, "create_plan")
        result = await create.execute(
            name="My Plan",
            steps=["Step 1", "Step 2", "Step 3"],
        )
        plans = await store.list_plans()
        assert len(plans) == 1
        assert plans[0].name == "My Plan"
        assert len(plans[0].steps) == 3
        assert all(s.status == StepStatus.not_started for s in plans[0].steps)

        # Return string must include plan id and each step description + id on its own line,
        # in input order, so the agent can address any step by ID without a get_plan round-trip.
        assert f"(id: {plans[0].id})" in result.content
        for step in plans[0].steps:
            assert f"{step.description} (id: {step.id})" in result.content
        # Steps appear in the same order as the input list.
        step_id_positions = [result.content.index(f"(id: {s.id})") for s in plans[0].steps]
        assert step_id_positions == sorted(step_id_positions)

    async def test_creates_plan_with_description(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        create = self._get_tool(tools, "create_plan")
        await create.execute(
            name="Described Plan",
            steps=["Do something"],
            description="A plan with a purpose",
        )
        plans = await store.list_plans()
        assert plans[0].description == "A plan with a purpose"

    async def test_creates_plan_with_namespace(self) -> None:
        store = make_store()
        tools = create_planning_tools(store, namespace="agent1")
        create = self._get_tool(tools, "create_plan")
        await create.execute(name="NS Plan", steps=["Step 1"])
        plans = await store.list_plans()
        assert plans[0].namespace == "agent1"


# ──────────────────────────────────────────────────────────
# create_plan → update_step without a get_plan round-trip
# ──────────────────────────────────────────────────────────


class TestCreatePlanStepIDVisibility:
    """Pins the ergonomic contract: step IDs are recoverable from create_plan's
    return string, so an agent can call update_step immediately without an
    intervening get_plan."""

    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_update_step_directly_from_create_plan_response(self) -> None:
        import re

        store = make_store()
        tools = create_planning_tools(store)
        create = self._get_tool(tools, "create_plan")
        update = self._get_tool(tools, "update_step")

        create_result = await create.execute(
            name="Round-trip-free",
            steps=["First step", "Second step"],
        )

        # Parse IDs directly out of the create_plan response with the same regex
        # the SDK example uses. The first match is the plan ID (from the header);
        # the remaining matches are step IDs in input order.
        ids = re.findall(r"\(id: ([^)]+)\)", create_result.content)
        assert len(ids) == 3, f"Expected 1 plan id + 2 step ids, got {ids}"
        plan_id, *step_ids = ids
        assert len(step_ids) == 2

        # No intervening get_plan call — update each step directly.
        for step_id in step_ids:
            result = await update.execute(
                plan_id=plan_id,
                step_id=step_id,
                status="completed",
            )
            assert "not found" not in result.content

        plan = await store.load(plan_id)
        assert plan is not None
        assert all(s.status == StepStatus.completed for s in plan.steps)


# ──────────────────────────────────────────────────────────
# get_plan Tool
# ──────────────────────────────────────────────────────────


class TestGetPlanTool:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_get_plan_shows_state(self) -> None:
        store = make_store()
        plan = Plan(
            name="Test Plan",
            steps=[
                PlanStep(description="First step", status=StepStatus.completed, result="Done"),
                PlanStep(description="Second step", status=StepStatus.in_progress),
                PlanStep(description="Third step"),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        get = self._get_tool(tools, "get_plan")
        result = await get.execute(plan_id=plan.id)
        assert "Test Plan" in result.content
        assert "[✓]" in result.content
        assert "[→]" in result.content
        assert "[ ]" in result.content
        assert "1/3 steps completed" in result.content

    async def test_get_plan_not_found(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        get = self._get_tool(tools, "get_plan")
        result = await get.execute(plan_id="nonexistent")
        assert "not found" in result.content

    async def test_get_plan_shows_description(self) -> None:
        store = make_store()
        plan = Plan(
            name="Described Plan",
            description="A useful description",
            steps=[PlanStep(description="Do it")],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        get = self._get_tool(tools, "get_plan")
        result = await get.execute(plan_id=plan.id)
        assert "A useful description" in result.content

    async def test_get_plan_shows_goals(self) -> None:
        store = make_store()
        plan = Plan(
            name="Goal Plan",
            goals=[Goal(description="Main goal", subgoals=[Goal(description="Sub goal")])],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        get = self._get_tool(tools, "get_plan")
        result = await get.execute(plan_id=plan.id)
        assert "Main goal" in result.content
        assert "Sub goal" in result.content


# ──────────────────────────────────────────────────────────
# update_step Tool
# ──────────────────────────────────────────────────────────


class TestUpdateStepTool:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def _create_plan_with_steps(self, store: InMemoryPlanStore) -> Plan:
        plan = Plan(
            name="Test Plan",
            steps=[
                PlanStep(id="s1", description="First"),
                PlanStep(id="s2", description="Second"),
                PlanStep(id="s3", description="Third"),
            ],
        )
        await store.save(plan)
        return plan

    async def test_update_step_status(self) -> None:
        store = make_store()
        plan = await self._create_plan_with_steps(store)
        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        result = await update.execute(plan_id=plan.id, step_id="s1", status="in_progress")
        assert "not_started → in_progress" in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert loaded.steps[0].status == StepStatus.in_progress

    async def test_update_step_with_result(self) -> None:
        store = make_store()
        plan = await self._create_plan_with_steps(store)
        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        await update.execute(plan_id=plan.id, step_id="s1", status="completed", result="All good")
        loaded = await store.load(plan.id)
        assert loaded is not None
        assert loaded.steps[0].result == "All good"

    async def test_update_step_invalid_status_raises_at_model_boundary(self) -> None:
        """An invalid status value is rejected by Pydantic before the function body runs.

        The `status` parameter is typed as `StepStatus` (a `StrEnum`), so the
        `@tool` decorator's generated parameters model validates the enum at
        the boundary and raises `ToolParameterError` — no in-body branch.
        """
        store = make_store()
        plan = await self._create_plan_with_steps(store)
        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        with pytest.raises(ToolParameterError) as exc_info:
            await update.execute(plan_id=plan.id, step_id="s1", status="invalid_status")
        assert exc_info.value.tool_name == "update_step"
        assert "status" in (exc_info.value.reason or "")

    async def test_update_step_not_found(self) -> None:
        store = make_store()
        plan = await self._create_plan_with_steps(store)
        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        result = await update.execute(plan_id=plan.id, step_id="nonexistent", status="completed")
        assert "not found" in result.content

    async def test_update_step_plan_not_found(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        result = await update.execute(plan_id="nonexistent", step_id="s1", status="completed")
        assert "not found" in result.content

    async def test_auto_complete_plan(self) -> None:
        store = make_store()
        plan = Plan(
            name="Short Plan",
            steps=[
                PlanStep(id="s1", description="Only step"),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        result = await update.execute(plan_id=plan.id, step_id="s1", status="completed")
        assert "plan marked as completed" in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert loaded.status == PlanStatus.completed

    async def test_auto_complete_with_skipped_steps(self) -> None:
        store = make_store()
        plan = Plan(
            name="Mixed Plan",
            steps=[
                PlanStep(id="s1", description="Done", status=StepStatus.completed),
                PlanStep(id="s2", description="Skip me"),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        result = await update.execute(plan_id=plan.id, step_id="s2", status="skipped")
        assert "plan marked as completed" in result.content

    async def test_no_auto_complete_with_remaining_steps(self) -> None:
        store = make_store()
        plan = await self._create_plan_with_steps(store)
        tools = create_planning_tools(store)
        update = self._get_tool(tools, "update_step")

        result = await update.execute(plan_id=plan.id, step_id="s1", status="completed")
        assert "plan marked as completed" not in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert loaded.status == PlanStatus.active


# ──────────────────────────────────────────────────────────
# revise_plan Tool
# ──────────────────────────────────────────────────────────


class TestRevisePlanTool:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_revise_replaces_not_started_steps(self) -> None:
        store = make_store()
        plan = Plan(
            name="Original Plan",
            steps=[
                PlanStep(id="s1", description="Done", status=StepStatus.completed),
                PlanStep(id="s2", description="In progress", status=StepStatus.in_progress),
                PlanStep(id="s3", description="Not started"),
                PlanStep(id="s4", description="Also not started"),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        revise = self._get_tool(tools, "revise_plan")

        result = await revise.execute(
            plan_id=plan.id,
            revised_steps=["New step A", "New step B"],
            reason="Changed approach",
        )
        assert "2 steps preserved" in result.content
        assert "2 new steps added" in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert len(loaded.steps) == 4
        assert loaded.steps[0].description == "Done"
        assert loaded.steps[1].description == "In progress"
        assert loaded.steps[2].description == "New step A"
        assert loaded.steps[3].description == "New step B"

    async def test_revise_plan_not_found(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        revise = self._get_tool(tools, "revise_plan")

        result = await revise.execute(
            plan_id="nonexistent",
            revised_steps=["Step"],
            reason="Reason",
        )
        assert "not found" in result.content

    async def test_revise_all_not_started(self) -> None:
        store = make_store()
        plan = Plan(
            name="Fresh Plan",
            steps=[
                PlanStep(description="Old 1"),
                PlanStep(description="Old 2"),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        revise = self._get_tool(tools, "revise_plan")

        result = await revise.execute(
            plan_id=plan.id,
            revised_steps=["New 1", "New 2", "New 3"],
            reason="Complete rethink",
        )
        assert "0 steps preserved" in result.content
        assert "3 new steps added" in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert len(loaded.steps) == 3


# ──────────────────────────────────────────────────────────
# update_goal Tool
# ──────────────────────────────────────────────────────────


class TestUpdateGoalTool:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_update_top_level_goal(self) -> None:
        store = make_store()
        plan = Plan(
            name="Goal Plan",
            goals=[Goal(id="g1", description="Main goal")],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        update_goal = self._get_tool(tools, "update_goal")

        result = await update_goal.execute(plan_id=plan.id, goal_id="g1", status="achieved")
        assert "active → achieved" in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert loaded.goals[0].status == GoalStatus.achieved

    async def test_update_nested_subgoal(self) -> None:
        store = make_store()
        plan = Plan(
            name="Nested Goals",
            goals=[
                Goal(
                    id="g1",
                    description="Parent",
                    subgoals=[Goal(id="g2", description="Child")],
                )
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        update_goal = self._get_tool(tools, "update_goal")

        result = await update_goal.execute(plan_id=plan.id, goal_id="g2", status="achieved")
        assert "active → achieved" in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert loaded.goals[0].subgoals[0].status == GoalStatus.achieved

    async def test_update_goal_invalid_status_raises_at_model_boundary(self) -> None:
        """An invalid goal status is rejected by Pydantic before the function body runs.

        The `status` parameter is typed as `GoalStatus` (a `StrEnum`); the
        `@tool` decorator's generated parameters model validates the enum at
        the boundary and raises `ToolParameterError` — no in-body branch.
        """
        store = make_store()
        plan = Plan(
            name="Goal Plan",
            goals=[Goal(id="g1", description="Goal")],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        update_goal = self._get_tool(tools, "update_goal")

        with pytest.raises(ToolParameterError) as exc_info:
            await update_goal.execute(plan_id=plan.id, goal_id="g1", status="invalid")
        assert exc_info.value.tool_name == "update_goal"
        assert "status" in (exc_info.value.reason or "")

    async def test_update_goal_not_found(self) -> None:
        store = make_store()
        plan = Plan(
            name="Goal Plan",
            goals=[Goal(id="g1", description="Goal")],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        update_goal = self._get_tool(tools, "update_goal")

        result = await update_goal.execute(plan_id=plan.id, goal_id="nonexistent", status="achieved")
        assert "not found" in result.content

    async def test_update_goal_plan_not_found(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        update_goal = self._get_tool(tools, "update_goal")

        result = await update_goal.execute(plan_id="nonexistent", goal_id="g1", status="achieved")
        assert "not found" in result.content

    async def test_update_goal_in_second_sibling_subtree(self) -> None:
        # Covers the else-branch in _find_and_update_goal when sub_found is None
        # (the target is not found in the first sibling's subgoals)
        store = make_store()
        plan = Plan(
            name="Multi-branch Goals",
            goals=[
                Goal(id="g1", description="First", subgoals=[Goal(id="g1a", description="First child")]),
                Goal(id="g2", description="Second", subgoals=[Goal(id="g2a", description="Second child")]),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        update_goal = self._get_tool(tools, "update_goal")

        result = await update_goal.execute(plan_id=plan.id, goal_id="g2a", status="achieved")
        assert "active → achieved" in result.content

        loaded = await store.load(plan.id)
        assert loaded is not None
        assert loaded.goals[1].subgoals[0].status == GoalStatus.achieved


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestPlanningToolEvents:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    def _make_registry(self, tools: list[FunctionTool], emitter: InMemoryEmitter) -> ToolRegistry:
        registry = ToolRegistry(emitter=emitter)
        for t in tools:
            registry.register(t)
        return registry

    async def test_create_plan_emits_event(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_planning_tools(store)
        registry = self._make_registry(tools, emitter)

        await registry.dispatch(
            ToolCall(
                id="1",
                name="create_plan",
                arguments={"name": "Test", "steps": ["A", "B"]},
            )
        )

        events = [e for e in emitter.events if isinstance(e, PlanCreatedEvent)]
        assert len(events) == 1
        assert events[0].plan_name == "Test"
        assert events[0].step_count == 2
        assert events[0].goal_count == 0
        assert events[0].trace_id == "test-trace"

    async def test_create_plan_event_includes_namespace(self) -> None:
        store = make_store()
        emitter = make_emitter()
        tools = create_planning_tools(store, namespace="agent1")
        registry = self._make_registry(tools, emitter)

        await registry.dispatch(
            ToolCall(
                id="1",
                name="create_plan",
                arguments={"name": "NS Plan", "steps": ["Step"]},
            )
        )

        events = [e for e in emitter.events if isinstance(e, PlanCreatedEvent)]
        assert len(events) == 1
        assert events[0].namespace == "agent1"

    async def test_update_step_emits_event(self) -> None:
        store = make_store()
        plan = Plan(
            name="Plan",
            steps=[PlanStep(id="s1", description="Do it")],
        )
        await store.save(plan)

        emitter = make_emitter()
        tools = create_planning_tools(store)
        registry = self._make_registry(tools, emitter)

        await registry.dispatch(
            ToolCall(
                id="1",
                name="update_step",
                arguments={
                    "plan_id": plan.id,
                    "step_id": "s1",
                    "status": "completed",
                    "result": "Done",
                },
            )
        )

        events = [e for e in emitter.events if isinstance(e, PlanStepUpdatedEvent)]
        assert len(events) == 1
        assert events[0].plan_id == plan.id
        assert events[0].step_id == "s1"
        assert events[0].step_description == "Do it"
        assert events[0].previous_status == "not_started"
        assert events[0].new_status == "completed"
        assert events[0].has_result is True

    async def test_revise_plan_emits_event(self) -> None:
        store = make_store()
        plan = Plan(
            name="Plan",
            steps=[
                PlanStep(description="Done", status=StepStatus.completed),
                PlanStep(description="Old"),
            ],
        )
        await store.save(plan)

        emitter = make_emitter()
        tools = create_planning_tools(store)
        registry = self._make_registry(tools, emitter)

        await registry.dispatch(
            ToolCall(
                id="1",
                name="revise_plan",
                arguments={
                    "plan_id": plan.id,
                    "revised_steps": ["New A", "New B"],
                    "reason": "Changed direction",
                },
            )
        )

        events = [e for e in emitter.events if isinstance(e, PlanRevisedEvent)]
        assert len(events) == 1
        assert events[0].plan_id == plan.id
        assert events[0].steps_before == 2
        assert events[0].steps_after == 3
        assert events[0].steps_preserved == 1
        assert events[0].revision_reason == "Changed direction"

    async def test_update_goal_emits_event(self) -> None:
        store = make_store()
        plan = Plan(
            name="Plan",
            goals=[Goal(id="g1", description="Be great")],
        )
        await store.save(plan)

        emitter = make_emitter()
        tools = create_planning_tools(store)
        registry = self._make_registry(tools, emitter)

        await registry.dispatch(
            ToolCall(
                id="1",
                name="update_goal",
                arguments={
                    "plan_id": plan.id,
                    "goal_id": "g1",
                    "status": "achieved",
                },
            )
        )

        events = [e for e in emitter.events if isinstance(e, GoalStatusChangedEvent)]
        assert len(events) == 1
        assert events[0].plan_id == plan.id
        assert events[0].goal_id == "g1"
        assert events[0].goal_description == "Be great"
        assert events[0].previous_status == "active"
        assert events[0].new_status == "achieved"

    async def test_create_goal_emits_event(self) -> None:
        store = make_store()
        plan = Plan(name="Plan")
        await store.save(plan)

        emitter = make_emitter()
        tools = create_planning_tools(store)
        registry = self._make_registry(tools, emitter)

        await registry.dispatch(
            ToolCall(
                id="1",
                name="create_goal",
                arguments={"plan_id": plan.id, "description": "New goal"},
            )
        )

        events = [e for e in emitter.events if isinstance(e, GoalStatusChangedEvent)]
        assert len(events) == 1
        assert events[0].plan_id == plan.id
        assert events[0].goal_description == "New goal"
        assert events[0].new_status == "active"


# ──────────────────────────────────────────────────────────
# create_goal Tool
# ──────────────────────────────────────────────────────────


class TestCreateGoalTool:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_creates_goal_on_plan(self) -> None:
        store = make_store()
        plan = Plan(name="Plan")
        await store.save(plan)

        tools = create_planning_tools(store)
        create_goal = self._get_tool(tools, "create_goal")
        result = await create_goal.execute(
            plan_id=plan.id,
            description="Ship the feature",
        )
        assert "created" in result.content.lower()
        assert "Ship the feature" in result.content

        updated = await store.load(plan.id)
        assert updated is not None
        assert len(updated.goals) == 1
        assert updated.goals[0].description == "Ship the feature"
        assert updated.goals[0].status == GoalStatus.active

    async def test_creates_goal_with_priority_and_criteria(self) -> None:
        store = make_store()
        plan = Plan(name="Plan")
        await store.save(plan)

        tools = create_planning_tools(store)
        create_goal = self._get_tool(tools, "create_goal")
        await create_goal.execute(
            plan_id=plan.id,
            description="High priority goal",
            priority=10,
            success_criteria="All tests pass",
        )

        updated = await store.load(plan.id)
        assert updated is not None
        assert updated.goals[0].priority == 10
        assert updated.goals[0].success_criteria == "All tests pass"

    async def test_creates_subgoal(self) -> None:
        store = make_store()
        parent_goal = Goal(id="g1", description="Parent")
        plan = Plan(name="Plan", goals=[parent_goal])
        await store.save(plan)

        tools = create_planning_tools(store)
        create_goal = self._get_tool(tools, "create_goal")
        result = await create_goal.execute(
            plan_id=plan.id,
            description="Child goal",
            parent_goal_id="g1",
        )
        assert "under parent" in result.content

        updated = await store.load(plan.id)
        assert updated is not None
        assert len(updated.goals) == 1
        assert len(updated.goals[0].subgoals) == 1
        assert updated.goals[0].subgoals[0].description == "Child goal"

    async def test_plan_not_found(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        create_goal = self._get_tool(tools, "create_goal")
        result = await create_goal.execute(
            plan_id="nonexistent",
            description="A goal",
        )
        assert "not found" in result.content

    async def test_parent_goal_not_found(self) -> None:
        store = make_store()
        plan = Plan(name="Plan")
        await store.save(plan)

        tools = create_planning_tools(store)
        create_goal = self._get_tool(tools, "create_goal")
        result = await create_goal.execute(
            plan_id=plan.id,
            description="Orphan",
            parent_goal_id="nonexistent",
        )
        assert "not found" in result.content

    async def test_creates_subgoal_with_flat_sibling(self) -> None:
        # Covers else-branch in _add_subgoal (sibling goal has no subgoals)
        store = make_store()
        plan = Plan(
            name="Plan",
            goals=[
                Goal(id="g1", description="Sibling"),
                Goal(id="g2", description="Parent"),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        create_goal = self._get_tool(tools, "create_goal")
        result = await create_goal.execute(plan_id=plan.id, description="Child goal", parent_goal_id="g2")
        assert "under parent" in result.content

        updated = await store.load(plan.id)
        assert updated is not None
        assert len(updated.goals[1].subgoals) == 1
        assert updated.goals[1].subgoals[0].description == "Child goal"

    async def test_creates_deeply_nested_subgoal_via_sibling_subtree(self) -> None:
        # Covers the elif/else branches in _add_subgoal when a sibling has subgoals
        # but the target parent is nested under a different sibling
        store = make_store()
        plan = Plan(
            name="Plan",
            goals=[
                Goal(id="g1", description="Sibling", subgoals=[Goal(id="g1a", description="Sibling child")]),
                Goal(id="g2", description="Other", subgoals=[Goal(id="g2a", description="Target parent")]),
            ],
        )
        await store.save(plan)

        tools = create_planning_tools(store)
        create_goal = self._get_tool(tools, "create_goal")
        result = await create_goal.execute(plan_id=plan.id, description="Deep child", parent_goal_id="g2a")
        assert "under parent" in result.content

        updated = await store.load(plan.id)
        assert updated is not None
        assert len(updated.goals[1].subgoals[0].subgoals) == 1
        assert updated.goals[1].subgoals[0].subgoals[0].description == "Deep child"


# ──────────────────────────────────────────────────────────
# Namespace Isolation
# ──────────────────────────────────────────────────────────


class TestPlanningToolNamespaceIsolation:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_plans_created_with_namespace(self) -> None:
        store = make_store()
        tools_a = create_planning_tools(store, namespace="agent_a")
        tools_b = create_planning_tools(store, namespace="agent_b")

        create_a = self._get_tool(tools_a, "create_plan")
        create_b = self._get_tool(tools_b, "create_plan")

        await create_a.execute(name="Plan A", steps=["Step A"])
        await create_b.execute(name="Plan B", steps=["Step B"])

        plans_a = await store.list_plans(namespace="agent_a")
        plans_b = await store.list_plans(namespace="agent_b")

        assert len(plans_a) == 1
        assert plans_a[0].name == "Plan A"
        assert len(plans_b) == 1
        assert plans_b[0].name == "Plan B"


# ──────────────────────────────────────────────────────────
# on_plan_created Callback
# ──────────────────────────────────────────────────────────


class TestOnPlanCreatedCallback:
    def _get_tool(self, tools: list[FunctionTool], name: str) -> FunctionTool:
        return next(t for t in tools if t.schema.name == name)

    async def test_callback_receives_plan_id(self) -> None:
        store = make_store()
        received_ids: list[str] = []
        tools = create_planning_tools(store, on_plan_created=lambda pid: received_ids.append(pid))
        create = self._get_tool(tools, "create_plan")

        await create.execute(name="CB Plan", steps=["Step 1"])

        assert len(received_ids) == 1
        plans = await store.list_plans()
        assert received_ids[0] == plans[0].id

    async def test_no_callback_no_error(self) -> None:
        store = make_store()
        tools = create_planning_tools(store)
        create = self._get_tool(tools, "create_plan")

        result = await create.execute(name="No CB", steps=["Step 1"])
        plans = await store.list_plans()
        assert f"(id: {plans[0].id})" in result.content
        assert f"{plans[0].steps[0].description} (id: {plans[0].steps[0].id})" in result.content
