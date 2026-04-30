"""Lightweight extraction of plan/goal state from working memory markdown."""

import re

from nanitics.capabilities.planning.models import GoalStatus, StepStatus

_STEP_MARKERS: dict[str, StepStatus] = {
    " ": StepStatus.not_started,
    "→": StepStatus.in_progress,
    "✓": StepStatus.completed,
    "✗": StepStatus.failed,
    "~": StepStatus.skipped,
}

_GOAL_MARKERS: dict[str, GoalStatus] = {
    " ": GoalStatus.active,
    "→": GoalStatus.active,
    "✓": GoalStatus.achieved,
    "✗": GoalStatus.abandoned,
    "~": GoalStatus.blocked,
}

_CHECKBOX_RE = re.compile(r"^[\s]*-\s*\[(.?)\]\s*(.+)$")


def parse_plan_from_working_memory(content: str) -> list[tuple[str, StepStatus]]:
    """Parse markdown plan format from working memory content.

    Recognizes:
        - [ ] description  → NOT_STARTED
        - [→] description  → IN_PROGRESS
        - [✓] description  → COMPLETED
        - [✗] description  → FAILED
        - [~] description  → SKIPPED

    Missing or unrecognized markers default to NOT_STARTED.
    """
    results: list[tuple[str, StepStatus]] = []
    for line in content.splitlines():
        match = _CHECKBOX_RE.match(line)
        if match:
            marker = match.group(1).strip()
            description = match.group(2).strip()
            status = _STEP_MARKERS.get(marker, StepStatus.not_started)
            results.append((description, status))
    return results


def parse_goals_from_working_memory(content: str) -> list[tuple[str, GoalStatus]]:
    """Parse markdown goal format from working memory content.

    Recognizes:
        - [ ] description  → ACTIVE
        - [→] description  → ACTIVE
        - [✓] description  → ACHIEVED
        - [✗] description  → ABANDONED
        - [~] description  → BLOCKED

    Missing or unrecognized markers default to ACTIVE.
    """
    results: list[tuple[str, GoalStatus]] = []
    for line in content.splitlines():
        match = _CHECKBOX_RE.match(line)
        if match:
            marker = match.group(1).strip()
            description = match.group(2).strip()
            status = _GOAL_MARKERS.get(marker, GoalStatus.active)
            results.append((description, status))
    return results
