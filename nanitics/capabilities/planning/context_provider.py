from __future__ import annotations

from typing import Literal

from nanitics.capabilities.memory.context_provider import ContextContent
from nanitics.capabilities.planning.models import (
    Goal,
    Plan,
    StepStatus,
)
from nanitics.capabilities.planning.store import PlanStore
from nanitics.infrastructure.llm.protocol import Message

_STATUS_INDICATORS = {
    StepStatus.not_started: "[ ]",
    StepStatus.in_progress: "[→]",
    StepStatus.completed: "[✓]",
    StepStatus.skipped: "[~]",
    StepStatus.failed: "[✗]",
}


def _format_minimal(plan: Plan) -> str:
    completed = sum(1 for s in plan.steps if s.status == StepStatus.completed)
    total = len(plan.steps)
    return f"[Current Plan: {plan.name}]\nProgress: {completed}/{total} steps completed"


def _format_goals(goals: list[Goal], lines: list[str], indent: int) -> None:
    for goal in goals:
        prefix = " " * indent
        lines.append(f"{prefix}- [{goal.status.value}] {goal.description}")
        if goal.subgoals:
            _format_goals(goal.subgoals, lines, indent + 2)


def _format_normal(plan: Plan) -> str:
    completed = sum(1 for s in plan.steps if s.status == StepStatus.completed)
    total = len(plan.steps)

    lines = [f"[Current Plan: {plan.name}]"]
    lines.append(f"Status: {plan.status.value} ({completed}/{total} steps completed)")

    completed_steps = [s for s in plan.steps if s.status == StepStatus.completed]
    if completed_steps:
        lines.append("")
        lines.append("## Completed")
        for step in completed_steps:
            line = f"- [✓] {step.description}"
            if step.result:
                line += f" (result: {step.result})"
            lines.append(line)

    failed_steps = [s for s in plan.steps if s.status == StepStatus.failed]
    if failed_steps:
        lines.append("")
        lines.append("## Failed")
        for step in failed_steps:
            line = f"- [✗] {step.description}"
            if step.result:
                line += f" (result: {step.result})"
            lines.append(line)

    current_steps = [s for s in plan.steps if s.status == StepStatus.in_progress]
    if current_steps:
        lines.append("")
        lines.append("## Current")
        lines.extend(f"- [→] {step.description} (in_progress)" for step in current_steps)

    remaining_steps = [s for s in plan.steps if s.status == StepStatus.not_started]
    if remaining_steps:
        lines.append("")
        lines.append("## Remaining")
        for step in remaining_steps:
            dep_info = ""
            if step.dependencies:
                dep_info = f" (depends on: {', '.join(step.dependencies)})"
            lines.append(f"- [ ] {step.description}{dep_info}")

    if plan.goals:
        lines.append("")
        lines.append("## Goals")
        _format_goals(plan.goals, lines, indent=0)

    return "\n".join(lines)


def _format_full(plan: Plan) -> str:
    completed = sum(1 for s in plan.steps if s.status == StepStatus.completed)
    total = len(plan.steps)

    lines = [f"[Current Plan: {plan.name}]"]
    lines.append(f"Status: {plan.status.value} ({completed}/{total} steps completed)")
    if plan.description:
        lines.append(f"Description: {plan.description}")

    if plan.steps:
        lines.append("")
        lines.append("## Steps")
        for step in plan.steps:
            indicator = _STATUS_INDICATORS[step.status]
            line = f"- {indicator} {step.description} (id: {step.id})"
            if step.result:
                line += f" — result: {step.result}"
            if step.dependencies:
                line += f" [depends on: {', '.join(step.dependencies)}]"
            lines.append(line)

    if plan.goals:
        lines.append("")
        lines.append("## Goals")
        _format_goals(plan.goals, lines, indent=0)

    return "\n".join(lines)


_FORMATTERS = {
    "minimal": _format_minimal,
    "normal": _format_normal,
    "full": _format_full,
}


class PlanningContextProvider:
    """Injects current plan state into agent context before every LLM call.

    Formats the plan at the configured detail level and returns it as
    a ``ContextContent`` block. Returns ``None`` if the plan is not found.
    """

    def __init__(
        self,
        store: PlanStore,
        plan_id: str,
        *,
        detail: Literal["minimal", "normal", "full"] = "normal",
    ) -> None:
        """Initialize the planning context provider.

        Args:
            store: Plan store to load the plan from.
            plan_id: ID of the plan to inject into context.
            detail: Level of detail in the formatted plan.
                ``minimal`` shows only progress count,
                ``normal`` shows steps grouped by status,
                ``full`` shows all steps with IDs and dependencies.
        """
        self._store = store
        self._plan_id = plan_id
        self._detail = detail

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        plan = await self._store.load(self._plan_id)
        if plan is None:
            return None

        formatter = _FORMATTERS[self._detail]
        content = formatter(plan)
        return ContextContent(content=content, priority=5, protected=False, provider_name="planning")
