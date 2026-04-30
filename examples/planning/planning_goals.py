"""Planning goals: goal models, goal tools, and the goal satisfaction evaluator.

Demonstrates the goal subsystem within planning: goal models, goal tools
(create_goal, update_goal), goal status events, and the GoalSatisfactionEvaluator.

Goals represent desired outcomes, not individual actions. They can form hierarchies
via subgoals, carry priorities and success criteria, and drive evaluation through
the goal satisfaction evaluator — an alternative to the step-adherence evaluator
shown in examples/planning/planning.py.

Related guide: docs/guides/planning.md
"""

import asyncio
import re

from examples.helpers import make_emitter, make_response
from nanitics import (
    Goal,
    GoalStatus,
    InMemoryPlanStore,
    LLMResponse,
    Message,
    MockLLMClient,
    Plan,
    PlanningCapability,
    PlanStep,
    ReActAgent,
    ToolCall,
)
from nanitics.infrastructure import (
    GoalStatusChangedEvent,
)


async def main() -> None:
    # --- Section 1: Goal Model Basics ---
    print("--- Section 1: Goal Model Basics ---")

    # Goals are part of a Plan — they represent desired outcomes alongside steps.
    plan = Plan(
        name="Product Launch",
        steps=[
            PlanStep(description="Build feature"),
            PlanStep(description="Write docs"),
        ],
        goals=[
            Goal(
                description="Ship MVP to beta users",
                priority=2,
                success_criteria="At least 10 beta users onboarded",
                subgoals=[
                    Goal(description="Core feature complete", priority=2),
                    Goal(description="Documentation ready", priority=1),
                ],
            ),
            Goal(
                description="Gather initial feedback",
                priority=1,
                success_criteria="Feedback from 5+ users collected",
            ),
        ],
    )

    # Goals start as active.
    assert plan.goals[0].status == GoalStatus.active
    assert plan.goals[0].subgoals[0].status == GoalStatus.active

    # GoalStatus has four values.
    statuses = [s.value for s in GoalStatus]
    assert "active" in statuses
    assert "achieved" in statuses
    assert "blocked" in statuses
    assert "abandoned" in statuses

    print(f"  Plan: {plan.name} ({len(plan.goals)} top-level goals)")
    print(f"  Goal 1: {plan.goals[0].description} (priority={plan.goals[0].priority})")
    print(f"    Subgoals: {len(plan.goals[0].subgoals)}")
    print(f"    Success criteria: {plan.goals[0].success_criteria}")
    print(f"  Goal 2: {plan.goals[1].description} (priority={plan.goals[1].priority})")
    print(f"  Statuses: {', '.join(statuses)}")

    # Goals are frozen (immutable) — updates create new instances via model_copy.
    original = plan.goals[0]
    updated = original.model_copy(update={"status": GoalStatus.achieved})
    assert original.status == GoalStatus.active  # original unchanged
    assert updated.status == GoalStatus.achieved

    print(f"  After model_copy: {updated.status.value} (original unchanged: {original.status.value})")
    print("✓ Goals are immutable models with priorities, success criteria, and subgoals")

    # --- Section 2: Goal Tools with Agent ---
    print("\n--- Section 2: Goal Tools with Agent ---")

    # The agent uses create_goal and update_goal tools to manage goals on a plan.
    # We pre-create a plan with steps so we can focus on goal operations.

    store = InMemoryPlanStore()
    plan = Plan(
        name="Research Project",
        steps=[PlanStep(description="Conduct research")],
    )
    await store.save(plan)
    plan_id = plan.id

    # No evaluator — focus on goal tools and events.
    planning = PlanningCapability(store, evaluator=None)
    planning.set_active_plan(plan_id)

    def _extract_goal_id(messages: list[Message]) -> str:
        """Extract goal ID from the most recent create_goal tool result."""
        for msg in reversed(messages):
            if msg.role == "tool_result" and msg.content and "id:" in msg.content:
                match = re.search(r"\(id: ([^,]+)", msg.content)
                if match:
                    return match.group(1)
        return ""

    # Turn 2 (callable): parse the goal_id from create_goal result and mark it achieved.
    def response_achieve_goal(messages: list[Message]) -> LLMResponse:
        goal_id = _extract_goal_id(messages)
        return make_response(
            "Research is complete.",
            tool_calls=[
                ToolCall(
                    id="tc-ug",
                    name="update_goal",
                    arguments={"plan_id": plan_id, "goal_id": goal_id, "status": "achieved"},
                )
            ],
            stop_reason="tool_use",
        )

    # Turn 3 (callable): create a subgoal under the existing goal.
    def response_create_subgoal(messages: list[Message]) -> LLMResponse:
        _extract_goal_id(messages)
        # Find the parent goal ID (from the first create_goal result, not the update).
        parent_id = ""
        for msg in messages:
            if msg.role == "tool_result" and msg.content and "Goal '" in msg.content and "created" in msg.content:
                match = re.search(r"\(id: ([^,]+)", msg.content)
                if match:
                    parent_id = match.group(1)
                    break
        return make_response(
            "Adding a follow-up subgoal.",
            tool_calls=[
                ToolCall(
                    id="tc-cg2",
                    name="create_goal",
                    arguments={
                        "plan_id": plan_id,
                        "description": "Write summary report",
                        "priority": 1,
                        "parent_goal_id": parent_id,
                    },
                )
            ],
            stop_reason="tool_use",
        )

    client = MockLLMClient(
        responses=[
            # Turn 1 (static): create a goal on the plan.
            make_response(
                "Setting a research goal.",
                tool_calls=[
                    ToolCall(
                        id="tc-cg",
                        name="create_goal",
                        arguments={
                            "plan_id": plan_id,
                            "description": "Complete literature review",
                            "priority": 2,
                            "success_criteria": "All key papers identified and summarized",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 2 (callable): mark the goal as achieved.
            response_achieve_goal,
            # Turn 3 (callable): create a subgoal under the achieved goal.
            response_create_subgoal,
            # Turn 4: final answer.
            make_response("Research goals established and literature review completed."),
        ]
    )
    emitter = make_emitter("goals-s2")

    agent = ReActAgent(
        name="researcher",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant.",
        tools=planning.tools,
        context_providers=[planning.context_provider],
    )

    result = await agent.run("Set up and track research goals.")

    # Verify GoalStatusChangedEvent events were emitted.
    goal_events = [e for e in emitter.events if isinstance(e, GoalStatusChangedEvent)]
    assert len(goal_events) == 3  # create (active), update (achieved), create subgoal (active)

    # First event: goal created (previous="" → new="active").
    assert goal_events[0].new_status == "active"
    assert goal_events[0].previous_status == ""

    # Second event: goal achieved.
    assert goal_events[1].new_status == "achieved"
    assert goal_events[1].previous_status == "active"

    # Third event: subgoal created.
    assert goal_events[2].new_status == "active"
    assert "summary" in goal_events[2].goal_description.lower()

    # Verify the plan has the goal hierarchy.
    final_plan = await store.load(plan_id)
    assert final_plan is not None
    assert len(final_plan.goals) == 1
    assert final_plan.goals[0].status == GoalStatus.achieved
    assert len(final_plan.goals[0].subgoals) == 1
    assert final_plan.goals[0].subgoals[0].description == "Write summary report"

    print(f"  Output: {result.output}")
    print(f"  Goal events: {len(goal_events)}")
    print(f"  Event 1: created → {goal_events[0].new_status}")
    print(f"  Event 2: {goal_events[1].previous_status} → {goal_events[1].new_status}")
    print(f"  Event 3: subgoal created → {goal_events[2].new_status}")
    print(f"  Goal hierarchy: {final_plan.goals[0].description} ({final_plan.goals[0].status.value})")
    print(f"    └── {final_plan.goals[0].subgoals[0].description} ({final_plan.goals[0].subgoals[0].status.value})")
    print("✓ create_goal and update_goal tools with GoalStatusChangedEvent tracking")

    # --- Section 3: GoalSatisfactionEvaluator ---
    print("\n--- Section 3: GoalSatisfactionEvaluator ---")

    # GoalSatisfactionEvaluator rejects completion when active goals remain.
    # This is the goal-based alternative to the step-adherence evaluator in 45.

    store = InMemoryPlanStore()
    plan = Plan(
        name="Delivery Plan",
        steps=[PlanStep(description="Execute delivery")],
        goals=[
            Goal(description="Deliver package on time", priority=2),
        ],
    )
    await store.save(plan)
    plan_id = plan.id
    goal_id = plan.goals[0].id

    # evaluator="goal" creates a GoalSatisfactionEvaluator.
    planning = PlanningCapability(store, evaluator="goal")
    planning.set_active_plan(plan_id)

    client = MockLLMClient(
        responses=[
            # Turn 1: agent tries to finish without resolving the goal.
            make_response("Package is ready for delivery."),
            # Turn 2: after REVISE feedback, agent resolves the goal.
            make_response(
                "Resolving the delivery goal.",
                tool_calls=[
                    ToolCall(
                        id="tc-resolve",
                        name="update_goal",
                        arguments={"plan_id": plan_id, "goal_id": goal_id, "status": "achieved"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 3: final answer — evaluator checks all goals resolved → ACCEPT.
            make_response("Package delivered on time. Goal achieved."),
        ]
    )
    emitter = make_emitter("goals-s3")

    agent = ReActAgent(
        name="courier",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a delivery coordinator.",
        tools=planning.tools,
        context_providers=[planning.context_provider],
        output_evaluator=planning.output_evaluator,
    )

    result = await agent.run("Deliver the package.")

    assert result.termination_reason == "complete"
    assert "delivered" in result.output.lower()

    # Verify the goal was resolved.
    final_plan = await store.load(plan_id)
    assert final_plan is not None
    assert final_plan.goals[0].status == GoalStatus.achieved

    # Verify goal status changed event.
    goal_events = [e for e in emitter.events if isinstance(e, GoalStatusChangedEvent)]
    assert len(goal_events) == 1
    assert goal_events[0].previous_status == "active"
    assert goal_events[0].new_status == "achieved"

    print(f"  Output: {result.output}")
    print(f"  Termination: {result.termination_reason} (evaluator accepted after goal resolved)")
    print(f"  Goal status: {final_plan.goals[0].status.value}")
    print(f"  Goal events: {len(goal_events)}")
    print("✓ GoalSatisfactionEvaluator rejects until all goals are resolved")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
