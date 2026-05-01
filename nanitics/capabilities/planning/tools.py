from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated

from pydantic import Field

from nanitics.capabilities.planning.models import (
    Goal,
    GoalStatus,
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from nanitics.capabilities.planning.store import PlanStore
from nanitics.core.tools.context import ToolContext
from nanitics.core.tools.function_tool import FunctionTool, tool
from nanitics.infrastructure.observability.events import (
    GoalStatusChangedEvent,
    PlanCreatedEvent,
    PlanRevisedEvent,
    PlanStepDetail,
    PlanStepUpdatedEvent,
)

_STATUS_INDICATORS = {
    StepStatus.not_started: "[ ]",
    StepStatus.in_progress: "[→]",
    StepStatus.completed: "[✓]",
    StepStatus.skipped: "[~]",
    StepStatus.failed: "[✗]",
}

_TERMINAL_STATUSES = {StepStatus.completed, StepStatus.skipped}


def _format_plan(plan: Plan) -> str:
    lines = [f"Plan: {plan.name} (id: {plan.id})"]
    lines.append(f"Status: {plan.status.value}")
    if plan.description:
        lines.append(f"Description: {plan.description}")

    completed = sum(1 for s in plan.steps if s.status == StepStatus.completed)
    total = len(plan.steps)
    lines.append(f"Progress: {completed}/{total} steps completed")

    if plan.steps:
        lines.append("")
        lines.append("Steps:")
        for step in plan.steps:
            indicator = _STATUS_INDICATORS[step.status]
            line = f"  {indicator} {step.description} (id: {step.id})"
            if step.result:
                line += f" — result: {step.result}"
            lines.append(line)

    if plan.goals:
        lines.append("")
        lines.append("Goals:")
        _format_goals(plan.goals, lines, indent=2)

    return "\n".join(lines)


def _format_goals(goals: list[Goal], lines: list[str], indent: int) -> None:
    for goal in goals:
        prefix = " " * indent
        lines.append(f"{prefix}- [{goal.status.value}] {goal.description} (id: {goal.id})")
        if goal.subgoals:
            _format_goals(goal.subgoals, lines, indent + 2)


def _find_and_update_goal(goals: list[Goal], goal_id: str, new_status: GoalStatus) -> tuple[list[Goal], Goal | None]:
    """Recursively find and update a goal. Returns (updated_goals, found_goal_before_update)."""
    updated = []
    found: Goal | None = None
    for goal in goals:
        if goal.id == goal_id:
            found = goal
            updated.append(goal.model_copy(update={"status": new_status}))
        elif goal.subgoals:
            new_subgoals, sub_found = _find_and_update_goal(goal.subgoals, goal_id, new_status)
            if sub_found is not None:
                found = sub_found
                updated.append(goal.model_copy(update={"subgoals": new_subgoals}))
            else:
                updated.append(goal)
        else:
            updated.append(goal)
    return updated, found


def _add_subgoal(goals: list[Goal], parent_id: str, new_goal: Goal) -> tuple[list[Goal], Goal | None]:
    """Recursively find a parent goal and add a subgoal to it. Returns (updated_goals, found_parent)."""
    updated = []
    found: Goal | None = None
    for goal in goals:
        if goal.id == parent_id:
            found = goal
            updated.append(goal.model_copy(update={"subgoals": [*list(goal.subgoals), new_goal]}))
        elif goal.subgoals:
            new_subgoals, sub_found = _add_subgoal(goal.subgoals, parent_id, new_goal)
            if sub_found is not None:
                found = sub_found
                updated.append(goal.model_copy(update={"subgoals": new_subgoals}))
            else:
                updated.append(goal)
        else:
            updated.append(goal)
    return updated, found


def create_planning_tools(
    store: PlanStore,
    namespace: str | None = None,
    on_plan_created: Callable[[str], None] | None = None,
) -> list[FunctionTool]:
    """Create planning tools for agent use.

    Returns six tools: ``create_plan``, ``get_plan``, ``update_step``,
    ``revise_plan``, ``update_goal``, and ``create_goal``.

    Args:
        store: Plan store for persistence.
        namespace: Optional namespace scoping plans created by these tools.
        on_plan_created: Callback invoked with the plan ID when a plan is
            created. Used by ``PlanningCapability`` for auto-wiring.
    """

    @tool(
        name="create_plan",
        description=(
            "Create a structured plan with ordered steps. "
            "Provide a name and a list of step descriptions. "
            "Returns the plan ID for tracking progress."
        ),
    )
    async def create_plan(
        name: Annotated[str, Field(description="Short human-readable name of the plan, e.g. 'Market Analysis'.")],
        steps: Annotated[list[str], Field(description="Ordered list of step descriptions. Minimum two steps.")],
        context: ToolContext,
        description: Annotated[
            str | None,
            Field(description="Optional summary of what this plan accomplishes overall."),
        ] = None,
    ) -> str:
        plan_steps = [PlanStep(description=desc) for desc in steps]
        plan = Plan(
            name=name,
            description=description,
            steps=plan_steps,
            namespace=namespace,
        )
        await store.save(plan)

        if on_plan_created is not None:
            on_plan_created(plan.id)

        ctx_emitter = context.emitter if context is not None else None
        if ctx_emitter is not None:
            step_details = [
                PlanStepDetail(
                    step_id=ps.id,
                    description=ps.description,
                    metadata={},
                )
                for ps in plan.steps
            ]
            ctx_emitter.emit(
                PlanCreatedEvent(
                    trace_id=ctx_emitter.trace_id,
                    span_id=ctx_emitter.span_id,
                    parent_span_id=ctx_emitter.parent_span_id,
                    plan_id=plan.id,
                    plan_name=plan.name,
                    step_count=len(plan.steps),
                    goal_count=len(plan.goals),
                    namespace=namespace,
                    steps=step_details,
                )
            )

        header = f"Created plan '{plan.name}' (id: {plan.id}) with {len(plan.steps)} steps:"
        step_lines = [f"  {i}. {ps.description} (id: {ps.id})" for i, ps in enumerate(plan.steps, start=1)]
        return "\n".join([header, *step_lines])

    @tool(
        name="get_plan",
        description=("Retrieve the current state of a plan. Shows all steps with their statuses and any results."),
    )
    async def get_plan(
        plan_id: Annotated[str, Field(description="The plan identifier returned by create_plan.")],
    ) -> str:
        plan = await store.load(plan_id)
        if plan is None:
            return f"Plan '{plan_id}' not found."
        return _format_plan(plan)

    @tool(
        name="update_step",
        description=(
            "Update the status of a plan step. "
            "Valid statuses: not_started, in_progress, completed, skipped, failed. "
            "Optionally include a result description. "
            "When all steps are completed or skipped, the plan auto-completes."
        ),
    )
    async def update_step(
        plan_id: Annotated[str, Field(description="The plan containing the step.")],
        step_id: Annotated[str, Field(description="The step identifier shown in create_plan / get_plan output.")],
        status: Annotated[StepStatus, Field(description="New status of the step.")],
        context: ToolContext,
        result: Annotated[
            str | None,
            Field(description="Short result summary to record on the step. Optional."),
        ] = None,
    ) -> str:
        plan = await store.load(plan_id)
        if plan is None:
            return f"Plan '{plan_id}' not found."

        step_index = next((i for i, s in enumerate(plan.steps) if s.id == step_id), None)
        if step_index is None:
            return f"Step '{step_id}' not found in plan '{plan_id}'."

        old_step = plan.steps[step_index]
        previous_status = old_step.status

        update_fields: dict[str, object] = {"status": status}
        if result is not None:
            update_fields["result"] = result
        new_step = old_step.model_copy(update=update_fields)

        new_steps = list(plan.steps)
        new_steps[step_index] = new_step

        plan_update: dict[str, object] = {
            "steps": new_steps,
            "updated_at": datetime.now(UTC),
        }

        # Auto-complete plan when all steps are completed or skipped
        if all(s.status in _TERMINAL_STATUSES for s in new_steps):
            plan_update["status"] = PlanStatus.completed

        updated_plan = plan.model_copy(update=plan_update)
        await store.update(updated_plan)

        ctx_emitter = context.emitter if context is not None else None
        if ctx_emitter is not None:
            ctx_emitter.emit(
                PlanStepUpdatedEvent(
                    trace_id=ctx_emitter.trace_id,
                    span_id=ctx_emitter.span_id,
                    parent_span_id=ctx_emitter.parent_span_id,
                    plan_id=plan_id,
                    step_id=step_id,
                    step_description=old_step.description,
                    previous_status=previous_status.value,
                    new_status=status.value,
                    has_result=result is not None,
                )
            )

        msg = f"Step '{old_step.description}' updated: {previous_status.value} → {status.value}"
        if updated_plan.status == PlanStatus.completed:
            msg += "\nAll steps complete — plan marked as completed."
        return msg

    @tool(
        name="revise_plan",
        description=(
            "Replace remaining (not-started) steps in a plan with new steps. "
            "Completed and in-progress steps are preserved. "
            "Provide the reason for revision."
        ),
    )
    async def revise_plan(
        plan_id: Annotated[str, Field(description="The plan to revise.")],
        revised_steps: Annotated[
            list[str],
            Field(
                description=(
                    "New steps that replace remaining not-started steps. "
                    "Completed and in-progress steps are preserved automatically."
                )
            ),
        ],
        reason: Annotated[
            str,
            Field(description="Why the revision is needed — what changed between the original plan and now."),
        ],
        context: ToolContext,
    ) -> str:
        plan = await store.load(plan_id)
        if plan is None:
            return f"Plan '{plan_id}' not found."

        preserved = [s for s in plan.steps if s.status != StepStatus.not_started]
        new_steps = preserved + [PlanStep(description=desc) for desc in revised_steps]

        updated_plan = plan.model_copy(
            update={
                "steps": new_steps,
                "updated_at": datetime.now(UTC),
            }
        )
        await store.update(updated_plan)

        ctx_emitter = context.emitter if context is not None else None
        if ctx_emitter is not None:
            ctx_emitter.emit(
                PlanRevisedEvent(
                    trace_id=ctx_emitter.trace_id,
                    span_id=ctx_emitter.span_id,
                    parent_span_id=ctx_emitter.parent_span_id,
                    plan_id=plan_id,
                    steps_before=len(plan.steps),
                    steps_after=len(new_steps),
                    steps_preserved=len(preserved),
                    revision_reason=reason,
                )
            )

        return (
            f"Plan revised: {len(preserved)} steps preserved, {len(revised_steps)} new steps added (reason: {reason})."
        )

    @tool(
        name="update_goal",
        description=(
            "Update the status of a goal in a plan's goal hierarchy. "
            "Valid statuses: active, achieved, blocked, abandoned. "
            "Searches recursively through subgoals."
        ),
    )
    async def update_goal(
        plan_id: Annotated[str, Field(description="The plan containing the goal.")],
        goal_id: Annotated[str, Field(description="The goal identifier.")],
        status: Annotated[GoalStatus, Field(description="New status of the goal.")],
        context: ToolContext,
    ) -> str:
        plan = await store.load(plan_id)
        if plan is None:
            return f"Plan '{plan_id}' not found."

        updated_goals, found_goal = _find_and_update_goal(plan.goals, goal_id, status)
        if found_goal is None:
            return f"Goal '{goal_id}' not found in plan '{plan_id}'."

        updated_plan = plan.model_copy(
            update={
                "goals": updated_goals,
                "updated_at": datetime.now(UTC),
            }
        )
        await store.update(updated_plan)

        ctx_emitter = context.emitter if context is not None else None
        if ctx_emitter is not None:
            ctx_emitter.emit(
                GoalStatusChangedEvent(
                    trace_id=ctx_emitter.trace_id,
                    span_id=ctx_emitter.span_id,
                    parent_span_id=ctx_emitter.parent_span_id,
                    plan_id=plan_id,
                    goal_id=goal_id,
                    goal_description=found_goal.description,
                    previous_status=found_goal.status.value,
                    new_status=status.value,
                )
            )

        return f"Goal '{found_goal.description}' updated: {found_goal.status.value} → {status.value}"

    @tool(
        name="create_goal",
        description=(
            "Add a goal to an existing plan. Goals represent desired outcomes, not individual actions. "
            "Optionally set priority (higher = more important), success criteria, "
            "and a parent goal ID to nest this goal as a subgoal."
        ),
    )
    async def create_goal(
        plan_id: Annotated[str, Field(description="The plan to attach the goal to.")],
        description: Annotated[
            str,
            Field(description="What outcome the goal represents — state it as a result, not an action."),
        ],
        context: ToolContext,
        priority: Annotated[int, Field(description="Higher value = higher priority. Defaults to 0.")] = 0,
        success_criteria: Annotated[
            str | None,
            Field(description="Observable criterion that says the goal is achieved. Optional."),
        ] = None,
        parent_goal_id: Annotated[
            str | None,
            Field(description="Parent goal id to nest this goal under. Omit to create a top-level goal."),
        ] = None,
    ) -> str:
        plan = await store.load(plan_id)
        if plan is None:
            return f"Plan '{plan_id}' not found."

        new_goal = Goal(
            description=description,
            priority=priority,
            success_criteria=success_criteria,
        )

        if parent_goal_id is not None:
            updated_goals, parent = _add_subgoal(plan.goals, parent_goal_id, new_goal)
            if parent is None:
                return f"Parent goal '{parent_goal_id}' not found in plan '{plan_id}'."
        else:
            updated_goals = [*list(plan.goals), new_goal]

        updated_plan = plan.model_copy(
            update={
                "goals": updated_goals,
                "updated_at": datetime.now(UTC),
            }
        )
        await store.update(updated_plan)

        ctx_emitter = context.emitter if context is not None else None
        if ctx_emitter is not None:
            ctx_emitter.emit(
                GoalStatusChangedEvent(
                    trace_id=ctx_emitter.trace_id,
                    span_id=ctx_emitter.span_id,
                    parent_span_id=ctx_emitter.parent_span_id,
                    plan_id=plan_id,
                    goal_id=new_goal.id,
                    goal_description=description,
                    previous_status="",
                    new_status="active",
                )
            )

        parent_msg = f" under parent '{parent_goal_id}'" if parent_goal_id else ""
        return f"Goal '{description}' created{parent_msg} (id: {new_goal.id}, priority: {priority})."

    return [create_plan, get_plan, update_step, revise_plan, update_goal, create_goal]
