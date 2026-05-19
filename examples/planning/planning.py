"""Planning capability: plan store, agent auto-wiring, adherence evaluation.

Demonstrates the planning capability: plan store, PlanningCapability with agent
auto-wiring, adaptive plan revision, system prompt strategy contributors,
PlanAdherenceEvaluator (direct usage), and create_planning_tools (factory function).

Plans give agents explicit structure for multi-step tasks. The agent creates a plan,
tracks progress through steps, revises when circumstances change, and is evaluated
on whether it completed its plan before finishing.

Related guide: docs/guides/planning.md
"""

import asyncio
import re

from examples.helpers import make_emitter, make_response
from nanitics import (
    AdaptivePlanningContributor,
    EvaluationContext,
    EvaluationVerdict,
    GoalTrackingContributor,
    InMemoryPlanStore,
    LLMResponse,
    Message,
    MockLLMClient,
    Plan,
    PlanAdherenceEvaluator,
    PlanningCapability,
    PlanStatus,
    PlanStep,
    ReActAgent,
    StepStatus,
    SystemPromptBuilder,
    ToolCall,
    UpfrontPlanContributor,
    create_planning_tools,
)
from nanitics.infrastructure import (
    # Events
    PlanCreatedEvent,
    PlanRevisedEvent,
    PlanStepUpdatedEvent,
)
from nanitics.specialized import DecompositionContributor


async def main() -> None:
    # --- Section 1: Plan Store and Data Models ---
    print("--- Section 1: Plan Store and Data Models ---")

    # The plan store is the persistence layer — plans are saved to and loaded from it.
    store = InMemoryPlanStore()

    # Create a plan with three steps. All planning models are frozen (immutable).
    plan = Plan(
        name="Research Project",
        steps=[
            PlanStep(description="Gather sources"),
            PlanStep(description="Analyze data"),
            PlanStep(description="Write report"),
        ],
    )
    await store.save(plan)

    # Load back from the store — returns a new instance with the same data.
    loaded = await store.load(plan.id)
    assert loaded is not None
    assert loaded.name == "Research Project"
    assert len(loaded.steps) == 3
    assert all(s.status == StepStatus.not_started for s in loaded.steps)

    print(f"  Plan: {loaded.name} ({len(loaded.steps)} steps)")
    print(f"  Step statuses: {', '.join(s.status.value for s in loaded.steps)}")

    # Plans are frozen — updates create new instances via model_copy.
    original_step = loaded.steps[0]
    updated_step = original_step.model_copy(update={"status": StepStatus.completed})
    assert original_step.status == StepStatus.not_started  # original unchanged
    assert updated_step.status == StepStatus.completed

    # Save the updated plan with the new step.
    new_steps = [updated_step, *list(loaded.steps[1:])]
    updated_plan = loaded.model_copy(update={"steps": new_steps})
    await store.update(updated_plan)

    reloaded = await store.load(plan.id)
    assert reloaded is not None
    assert reloaded.steps[0].status == StepStatus.completed

    print(f"  After update: {reloaded.steps[0].status.value} (original unchanged: {original_step.status.value})")
    print("✓ Plans and steps are immutable — updates create new instances via model_copy")

    # --- Section 2: PlanningCapability with Agent ---
    print("\n--- Section 2: PlanningCapability with Agent ---")

    # PlanningCapability bundles tools, context provider, and evaluator in one object.
    # When the agent creates a plan via create_plan, auto-wiring links the context
    # provider and evaluator to the new plan automatically — no manual set_active_plan.

    store = InMemoryPlanStore()
    planning = PlanningCapability(store, evaluator="adherence", context_detail="normal")

    # Build system prompt with the adaptive planning contributor.
    adaptive = AdaptivePlanningContributor()
    builder = SystemPromptBuilder()
    builder.add_section("base", "You are a data analyst.")
    section = adaptive.system_prompt_section()
    builder.add_section(section[0], section[1])
    system_prompt = builder.build()

    # Helper: parse every `(id: <uuid>)` occurrence out of the most recent tool_result.
    # create_plan returns the plan ID followed by one line per step with its ID,
    # so one pass over the response recovers everything the agent needs — no get_plan
    # round-trip.
    def _parse_ids_from_latest_tool_result(messages: list[Message]) -> list[str]:
        for msg in reversed(messages):
            if msg.role == "tool_result" and msg.content:
                return re.findall(r"\(id: ([^)]+)\)", msg.content)
        return []

    # Callable response that parses plan_id + step IDs from create_plan's response
    # and completes step 1 directly.
    def response_complete_step1(messages: list[Message]) -> LLMResponse:
        ids = _parse_ids_from_latest_tool_result(messages)
        assert len(ids) >= 3, f"Expected plan id + 2 step ids in create_plan result, got {ids}"
        plan_id, step_ids = ids[0], ids[1:]
        return make_response(
            "Loading the dataset.",
            tool_calls=[
                ToolCall(
                    id="tc-s1",
                    name="update_step",
                    arguments={
                        "plan_id": plan_id,
                        "step_id": step_ids[0],
                        "status": "completed",
                        "result": "Loaded 1000 rows",
                    },
                )
            ],
            stop_reason="tool_use",
        )

    # Callable response that completes step 2 — IDs recovered from the same create_plan
    # response that's still in the conversation.
    def response_complete_step2(messages: list[Message]) -> LLMResponse:
        # Walk back to find the create_plan tool_result (not the update_step result
        # from step 1). It's the one containing multiple `(id: …)` occurrences.
        for msg in reversed(messages):
            if msg.role == "tool_result" and msg.content:
                ids = re.findall(r"\(id: ([^)]+)\)", msg.content)
                if len(ids) >= 3:
                    break
        else:
            raise AssertionError("Could not find create_plan tool_result in messages")
        plan_id, step_ids = ids[0], ids[1:]
        return make_response(
            "Computing statistics.",
            tool_calls=[
                ToolCall(
                    id="tc-s2",
                    name="update_step",
                    arguments={
                        "plan_id": plan_id,
                        "step_id": step_ids[1],
                        "status": "completed",
                        "result": "Mean: 42, Median: 38",
                    },
                )
            ],
            stop_reason="tool_use",
        )

    # Mock LLM flow: create_plan → update steps → final answer.
    # Step IDs come from create_plan's response directly — no get_plan round-trip.
    client = MockLLMClient(
        responses=[
            # Turn 1 (static): agent creates a plan via the tool
            make_response(
                "I'll create a plan for the analysis.",
                tool_calls=[
                    ToolCall(
                        id="tc-cp",
                        name="create_plan",
                        arguments={"name": "Data Analysis", "steps": ["Load dataset", "Compute statistics"]},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 2 (callable): parse plan_id + step IDs from create_plan → complete step 1
            response_complete_step1,
            # Turn 3 (callable): complete step 2 (IDs still visible in history)
            response_complete_step2,
            # Turn 4 (static): final answer — evaluator checks all steps done → ACCEPT
            make_response("Analysis complete. 1000 rows, mean 42, median 38."),
        ]
    )
    emitter = make_emitter("planning-s2")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt=system_prompt,
        tools=planning.tools,
        context_providers=[planning.context_provider],
        output_evaluator=planning.output_evaluator,
    )

    result = await agent.run("Analyze the dataset.")

    # Verify the agent completed and the evaluator accepted.
    assert result.termination_reason == "complete"
    assert "42" in result.output
    assert "38" in result.output

    # Auto-wiring: active_plan_id was set by on_plan_created — no manual set_active_plan.
    assert planning.active_plan_id is not None

    # Verify PlanCreatedEvent was emitted.
    plan_created_events = [e for e in emitter.events if isinstance(e, PlanCreatedEvent)]
    assert len(plan_created_events) == 1
    assert plan_created_events[0].plan_name == "Data Analysis"
    assert plan_created_events[0].step_count == 2

    # Verify plan auto-completed when the last step was marked done.
    final_plan = await store.load(planning.active_plan_id)
    assert final_plan is not None
    assert final_plan.status == PlanStatus.completed
    assert all(s.status == StepStatus.completed for s in final_plan.steps)

    # Verify step update events were emitted.
    step_events = [e for e in emitter.events if isinstance(e, PlanStepUpdatedEvent)]
    assert len(step_events) == 2

    print(f"  Output: {result.output}")
    print(f"  Auto-wired plan ID: {planning.active_plan_id}")
    print(f"  Plan created event: {plan_created_events[0].plan_name} ({plan_created_events[0].step_count} steps)")
    print(f"  Plan status: {final_plan.status.value} (auto-completed)")
    print(f"  Step events: {len(step_events)}")
    print(f"  Termination: {result.termination_reason} (evaluator accepted)")
    print("✓ Agent-driven create_plan with auto-wiring — no manual set_active_plan needed")

    # --- Section 3: Adaptive Plan Revision ---
    print("\n--- Section 3: Adaptive Plan Revision ---")

    # An agent discovers new information and revises its plan.
    # revise_plan preserves completed/in-progress steps, replaces not-started ones.

    store = InMemoryPlanStore()
    plan = Plan(
        name="API Integration",
        steps=[
            PlanStep(description="Read API documentation"),
            PlanStep(description="Build REST client"),
            PlanStep(description="Test REST integration"),
        ],
    )
    await store.save(plan)
    plan_id = plan.id
    step_ids = [s.id for s in plan.steps]

    # No evaluator — focus on revision, not completion enforcement.
    planning = PlanningCapability(store, evaluator=None)
    planning.set_active_plan(plan_id)

    client = MockLLMClient(
        responses=[
            # Turn 1: complete the discovery step — learns API uses GraphQL
            make_response(
                "Reading the API docs...",
                tool_calls=[
                    ToolCall(
                        id="tc-r1",
                        name="update_step",
                        arguments={
                            "plan_id": plan_id,
                            "step_id": step_ids[0],
                            "status": "completed",
                            "result": "API uses GraphQL, not REST",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 2: revise the remaining steps based on discovery
            make_response(
                "The API uses GraphQL. Revising my plan.",
                tool_calls=[
                    ToolCall(
                        id="tc-r2",
                        name="revise_plan",
                        arguments={
                            "plan_id": plan_id,
                            "revised_steps": ["Build GraphQL client", "Test GraphQL queries"],
                            "reason": "API uses GraphQL instead of REST",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            # Turn 3: report the revision
            make_response("Plan revised to use GraphQL based on API documentation."),
        ]
    )
    emitter = make_emitter("planning-s3")

    agent = ReActAgent(
        name="integrator",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are an API integration specialist.",
        tools=planning.tools,
        context_providers=[planning.context_provider],
    )

    result = await agent.run("Integrate with the external API.")

    assert "GraphQL" in result.output

    # The completed step was preserved through the revision.
    revised_plan = await store.load(plan_id)
    assert revised_plan is not None
    assert revised_plan.steps[0].status == StepStatus.completed
    assert revised_plan.steps[0].description == "Read API documentation"

    # New steps replaced the not-started REST steps.
    revised_descriptions = [s.description for s in revised_plan.steps[1:]]
    assert any("GraphQL" in d for d in revised_descriptions)

    # Verify events: 1 step update + 1 revision.
    step_events = [e for e in emitter.events if isinstance(e, PlanStepUpdatedEvent)]
    revision_events = [e for e in emitter.events if isinstance(e, PlanRevisedEvent)]
    assert len(step_events) == 1
    assert len(revision_events) == 1
    assert "GraphQL" in revision_events[0].revision_reason

    print(f"  Output: {result.output}")
    print(f"  Step 1: {revised_plan.steps[0].status.value} (preserved through revision)")
    print(f"  Revised steps: {', '.join(revised_descriptions)}")
    print(f"  Revision events: {len(revision_events)}")
    print("✓ revise_plan preserves completed steps and replaces remaining ones")

    # --- Section 4: Prompt Strategy Contributors ---
    print("\n--- Section 4: Prompt Strategy Contributors ---")

    # Four contributors teach agents different planning strategies.
    # Each returns a (section_name, instructions) tuple for the system prompt.

    adaptive = AdaptivePlanningContributor()
    upfront = UpfrontPlanContributor()
    decomposition = DecompositionContributor()
    goal_tracking = GoalTrackingContributor()

    adaptive_section = adaptive.system_prompt_section()
    upfront_section = upfront.system_prompt_section()
    decomposition_section = decomposition.system_prompt_section()
    goal_section = goal_tracking.system_prompt_section()

    # Each contributor returns a distinct section name.
    section_names = {adaptive_section[0], upfront_section[0], decomposition_section[0], goal_section[0]}
    assert len(section_names) == 4, "Each contributor should have a unique section name"

    # Each contributor's instructions match its planning strategy.
    assert any(word in adaptive_section[1].lower() for word in ("revis", "adapt"))
    assert any(word in upfront_section[1].lower() for word in ("upfront", "before"))
    assert any(word in decomposition_section[1].lower() for word in ("break", "subtask"))
    assert any(word in goal_section[1].lower() for word in ("goal", "priorit"))

    # Build a system prompt using one contributor — the section appears in the output.
    builder = SystemPromptBuilder()
    builder.add_section("base", "You are a research assistant.")
    builder.add_section(adaptive_section[0], adaptive_section[1])
    prompt = builder.build()
    assert adaptive_section[1] in prompt

    print(f"  Adaptive: {adaptive_section[0]} — mentions revision/adaptation")
    print(f"  Upfront: {upfront_section[0]} — mentions complete plan upfront")
    print(f"  Decomposition: {decomposition_section[0]} — mentions subtasks/breakdown")
    print(f"  Goal Tracking: {goal_section[0]} — mentions goals/priorities")
    print("✓ Four contributors teach agents different planning strategies")

    # --- Section 5: PlanAdherenceEvaluator (Direct Usage) ---
    print("\n--- Section 5: PlanAdherenceEvaluator (Direct Usage) ---")

    # PlanAdherenceEvaluator can be used directly — it checks whether all plan
    # steps are completed. Section 2 uses it indirectly via PlanningCapability
    # (evaluator="adherence"). Here we use it standalone.

    store = InMemoryPlanStore()
    plan = Plan(
        name="Deployment Checklist",
        steps=[
            PlanStep(description="Run test suite"),
            PlanStep(description="Build container image"),
            PlanStep(description="Deploy to staging"),
        ],
    )
    await store.save(plan)

    evaluator = PlanAdherenceEvaluator(store, plan.id, max_revisions=2)
    context = EvaluationContext(messages=[], task_input="Deploy the service.")

    # With all steps not_started, the evaluator returns REVISE.
    result_before = await evaluator.evaluate("Deployment complete!", context)
    assert result_before.verdict == EvaluationVerdict.REVISE
    assert result_before.score == 0.0
    assert result_before.evaluator_name == "plan_adherence"
    assert "Run test suite" in (result_before.feedback or "")
    assert evaluator.max_revisions == 2

    print(f"  Before completing steps: {result_before.verdict.value}")
    print(f"  Feedback: {result_before.feedback}")

    # Complete all steps.
    completed_steps = [s.model_copy(update={"status": StepStatus.completed}) for s in plan.steps]
    updated_plan = plan.model_copy(update={"steps": completed_steps})
    await store.update(updated_plan)

    # Now the evaluator returns ACCEPT with score 1.0.
    result_after = await evaluator.evaluate("Deployment complete!", context)
    assert result_after.verdict == EvaluationVerdict.ACCEPT
    assert result_after.score == 1.0

    print(f"  After completing steps: {result_after.verdict.value} (score={result_after.score})")
    print("✓ PlanAdherenceEvaluator rejects until all steps are completed")

    # --- Section 6: create_planning_tools (Factory Function) ---
    print("\n--- Section 6: create_planning_tools (Factory Function) ---")

    # create_planning_tools is the factory function that PlanningCapability uses
    # internally. Use it directly when you want planning tools without the full
    # capability wrapper — e.g., custom auto-wiring or mixing with other tools.

    store = InMemoryPlanStore()

    # Track which plan IDs are created via the on_plan_created callback.
    created_plan_ids: list[str] = []
    tools = create_planning_tools(
        store,
        namespace="my-agent",
        on_plan_created=lambda pid: created_plan_ids.append(pid),
    )

    # Returns six planning tools.
    tool_names = [t.schema.name for t in tools]
    assert "create_plan" in tool_names
    assert "get_plan" in tool_names
    assert "update_step" in tool_names
    assert "revise_plan" in tool_names
    assert "create_goal" in tool_names
    assert "update_goal" in tool_names
    assert len(tools) == 6

    print(f"  Tools: {', '.join(tool_names)}")

    # Use the tools with an agent — same as Section 2, but without PlanningCapability.
    client = MockLLMClient(
        responses=[
            make_response(
                "Creating a plan.",
                tool_calls=[
                    ToolCall(
                        id="tc-cp",
                        name="create_plan",
                        arguments={"name": "Quick Task", "steps": ["Do the thing"]},
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response("Done."),
        ]
    )
    emitter = make_emitter("planning-s6")

    agent = ReActAgent(
        name="worker",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a task worker.",
        tools=tools,
    )

    result = await agent.run("Do the thing.")

    # The on_plan_created callback was invoked.
    assert len(created_plan_ids) == 1

    # The plan was persisted with the namespace.
    saved_plan = await store.load(created_plan_ids[0])
    assert saved_plan is not None
    assert saved_plan.name == "Quick Task"
    assert saved_plan.namespace == "my-agent"

    print(f"  Created plan: {saved_plan.name} (namespace={saved_plan.namespace})")
    print(f"  on_plan_created callback received: {created_plan_ids[0]}")
    print("✓ create_planning_tools provides planning tools without PlanningCapability")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
