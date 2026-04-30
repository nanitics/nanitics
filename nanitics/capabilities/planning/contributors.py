"""Strategy-specific system prompt contributors for planning."""

_ADAPTIVE_PLANNING_INSTRUCTIONS = (
    "You have structured planning tools available. Use them to create, track, "
    "and revise plans throughout execution.\n\n"
    "**Workflow:**\n"
    "1. Create a plan at the start with `create_plan` to outline your approach\n"
    "2. Before executing each step, mark it in_progress with `update_step`\n"
    "3. After completing a step, mark it completed with the result\n"
    "4. After each step, evaluate whether the remaining plan still makes sense\n\n"
    "**When to revise:**\n"
    "- A step produces unexpected results that invalidate later steps\n"
    "- New information emerges that changes the approach\n"
    "- A step fails and the fallback requires different subsequent steps\n"
    "- You discover the problem is different from what you initially assumed\n\n"
    "**Revision discipline:**\n"
    "- Do not revise on every minor deviation — small adjustments are normal\n"
    "- When revising, replace the remaining steps entirely with `revise_plan`\n"
    "- Include a clear reason for the revision\n"
    "- Completed and in-progress steps are preserved automatically"
)


class AdaptivePlanningContributor:
    """Instructs the agent to create, track, and revise plans adaptively."""

    def system_prompt_section(self) -> tuple[str, str]:
        return ("adaptive_planning", _ADAPTIVE_PLANNING_INSTRUCTIONS)


_DECOMPOSITION_INSTRUCTIONS = (
    "Break complex tasks into smaller, focused subtasks before acting. "
    "Each subtask should be independently executable and produce a clear result.\n\n"
    "**Decomposition approach:**\n"
    "1. Analyze the overall task and identify its major components\n"
    "2. Create a plan where each step addresses one component\n"
    "3. Identify dependencies — which steps require results from earlier steps\n"
    "4. Execute each subtask with focused attention, recording its result\n"
    "5. Assemble subtask results into the final coherent output\n\n"
    "**When to decompose further:**\n"
    "- A subtask is still too complex to execute in a single action\n"
    "- A subtask has multiple distinct concerns mixed together\n"
    "- You find yourself needing to track intermediate state within a subtask\n\n"
    "**Result assembly:**\n"
    "- After all subtasks complete, synthesize their results\n"
    "- Resolve any conflicts or inconsistencies between subtask outputs\n"
    "- The final result should be coherent, not a concatenation of parts"
)


class DecompositionContributor:
    """Instructs the agent to decompose tasks recursively before execution."""

    def system_prompt_section(self) -> tuple[str, str]:
        return ("decomposition_planning", _DECOMPOSITION_INSTRUCTIONS)


_UPFRONT_PLANNING_INSTRUCTIONS = (
    "Create a complete, detailed plan before taking any action. Once the plan "
    "is created, execute each step mechanically in order.\n\n"
    "**Planning phase:**\n"
    "1. Analyze the full task requirements before planning\n"
    "2. Create a plan with `create_plan` that covers every action needed\n"
    "3. Steps should be specific and self-contained — each step should be "
    "executable without further reasoning\n"
    "4. Order steps to satisfy dependencies naturally\n\n"
    "**Execution phase:**\n"
    "- Execute steps strictly in order, one at a time\n"
    "- Mark each step in_progress before executing, completed after\n"
    "- Minimize reasoning during execution — the plan already has the reasoning\n"
    "- Do NOT revise the plan during execution\n"
    "- If a step fails, record the failure and continue with remaining steps\n\n"
    "**After execution:**\n"
    "- Review all step results together\n"
    "- Synthesize a final answer from the collected results\n"
    "- Note any steps that failed and their impact on the overall result"
)


class UpfrontPlanContributor:
    """Instructs the agent to plan completely before executing (ReWOO-style)."""

    def system_prompt_section(self) -> tuple[str, str]:
        return ("upfront_planning", _UPFRONT_PLANNING_INSTRUCTIONS)


_GOAL_TRACKING_INSTRUCTIONS = (
    "Maintain explicit awareness of your goals throughout execution. Goals "
    "represent desired outcomes, not individual actions.\n\n"
    "**Goal management:**\n"
    "1. Identify all goals at the start — what outcomes are you trying to achieve?\n"
    "2. Track goal status using `update_goal` as you make progress\n"
    "3. After each significant action, evaluate which goals it advanced\n"
    "4. Prioritize actions that advance the highest-priority unmet goal\n\n"
    "**Goal evaluation:**\n"
    "- A goal is achieved when its success criteria are met\n"
    "- A goal is blocked when prerequisites are missing or unavailable\n"
    "- A goal is abandoned when it becomes impossible or irrelevant\n"
    "- Subgoals contribute to their parent — a parent goal may be achieved "
    "when its subgoals are satisfied\n\n"
    "**Conflict resolution:**\n"
    "- When goals conflict, prefer higher-priority goals\n"
    "- Make trade-off decisions explicit — state what you're sacrificing and why\n"
    "- If a lower-priority goal becomes blocked by a higher-priority one, "
    "update its status to reflect the situation"
)


class GoalTrackingContributor:
    """Instructs the agent to track and prioritize goals throughout execution."""

    def system_prompt_section(self) -> tuple[str, str]:
        return ("goal_tracking", _GOAL_TRACKING_INSTRUCTIONS)
