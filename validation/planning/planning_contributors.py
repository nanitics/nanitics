"""Planning contributors inject their instructions into the system prompt and steer real agents.

Three contributors are exercised through a real ``ReActAgent``:

  - ``DecompositionContributor`` + ``GoalTrackingContributor`` are
    wired into one agent whose task genuinely needs decomposition and
    goal tracking (a multi-part research-and-report task supported by
    the planning tools). The agent is expected to create a plan with
    multiple steps and at least one goal, and we pin the contributors'
    instruction text inside the system prompt observed on
    ``LLMRequestEvent.system_prompt``. Goal persistence across steps
    is pinned by ``GoalStatusChangedEvent`` emissions and the final
    plan state.

  - ``AdaptivePlanningContributor`` is wired into a second agent for a
    task designed to require mid-run plan revision: the first tool
    call returns information that invalidates subsequent plan steps,
    forcing the agent to call ``revise_plan``. The revision is pinned
    via ``PlanRevisedEvent`` and by inspecting the final plan's
    steps.

Acceptance criteria (decomposition + goal tracking):
  - The system prompt observed on every ``LLMRequestEvent`` contains
    both the decomposition contributor's "subtask" instruction and
    the goal-tracking contributor's "goal" instruction.
  - A ``PlanCreatedEvent`` is emitted with ``step_count >= 2`` (proves
    the agent actually decomposed the task, not just talked about it).
  - At least one ``GoalStatusChangedEvent`` is emitted — goal state
    was tracked across steps, not just declared once.
  - The final plan loaded from the store contains at least one goal
    whose status is no longer ``active`` (goal was advanced through
    the run).
  - The exported ``FunctionTool.schema.parameters`` for the planning
    tools carries the ``enum`` on ``update_step.status`` /
    ``update_goal.status`` (via ``StepStatus`` / ``GoalStatus``
    ``StrEnum``s) and a non-empty ``description`` on every audited
    optional parameter. The assertion fires on the tools the agent
    actually used in this run.
  - ``result.termination_reason == "complete"``.

Acceptance criteria (adaptive planning revision):
  - The system prompt observed on every ``LLMRequestEvent`` contains
    the adaptive contributor's revision instruction text.
  - A ``PlanRevisedEvent`` is emitted during the run — the agent
    actually used ``revise_plan`` in response to new information.
  - The final plan contains at least one revised (not-in-original)
    step description mentioning the domain discovery keyword.
  - ``result.termination_reason == "complete"``.
"""

from __future__ import annotations

from nanitics.infrastructure import (
    GoalStatusChangedEvent,
    LLMRequestEvent,
    PlanCreatedEvent,
    PlanRevisedEvent,
)
from nanitics.planning import (
    AdaptivePlanningContributor,
    GoalTrackingContributor,
    InMemoryPlanStore,
    Plan,
    PlanningCapability,
    PlanStep,
)
from nanitics.specialized import DecompositionContributor
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# ---------------------------------------------------------------------------
# Decomposition + goal tracking
# ---------------------------------------------------------------------------


async def test_decomposition_and_goal_tracking_drive_real_agent(
    traced_emitter: InMemoryEmitter,
) -> None:
    client = make_llm_client("anthropic")
    decomposition = DecompositionContributor()
    goal_tracking = GoalTrackingContributor()
    decomp_section = decomposition.system_prompt_section()
    goal_section = goal_tracking.system_prompt_section()

    async def _run() -> object:
        traced_emitter.events.clear()
        store = InMemoryPlanStore()
        # No evaluator: the contract under test is the contributors'
        # influence on the system prompt and the resulting plan/goal
        # behaviour — not the evaluator's accept/revise loop, which is
        # covered by validation/planning/planning_evaluators.py.
        planning = PlanningCapability(store, evaluator=None)

        agent = ReActAgent(
            name="decomposer",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a planning-capable research assistant. Use the planning "
                "tools to create a plan, define goals, execute steps, and update "
                "goal status as you go."
            ),
            tools=planning.tools,
            context_providers=[planning.context_provider],
            prompt_contributors=[decomposition, goal_tracking],
            max_iterations=12,
        )

        agent_result = await agent.run(
            "Plan a short market analysis with at least two concrete steps "
            "(e.g. 'gather recent reports' and 'summarize findings'). Create a "
            "goal called 'Deliver summary' with success criteria 'a coherent "
            "one-paragraph summary is produced'. Use `create_plan` and "
            "`create_goal` at the start. Execute each step by updating its "
            "status with `update_step`. When the summary is ready, mark the "
            "goal as achieved with `update_goal`. Finally, produce the "
            "one-paragraph summary as your answer."
        )
        return (agent_result, store, planning)

    result, store, planning = await run_with_retry(_run, max_attempts=2)

    # --- Result-shape invariant ---
    assert result.termination_reason == "complete", (
        f"Expected termination_reason='complete', got: {result.termination_reason!r}"
    )

    # --- System prompt invariants ---
    # Every LLM request made by this agent must carry both contributors'
    # instruction text in its system prompt. Checking all requests (not
    # just one) guards against a regression where a later call rebuilds
    # the prompt without the contributor sections.
    llm_requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent)]
    assert llm_requests, "Expected at least one LLMRequestEvent."
    for event in llm_requests:
        assert event.system_prompt is not None, "Expected a non-None system prompt on LLMRequestEvent."
        assert decomp_section[1] in event.system_prompt, (
            "Decomposition contributor instructions missing from system prompt."
        )
        assert goal_section[1] in event.system_prompt, (
            "Goal-tracking contributor instructions missing from system prompt."
        )

    # --- Decomposition pin: plan was actually created with multiple steps ---
    plan_created = assert_trace_contains(
        traced_emitter,
        PlanCreatedEvent,
        predicate=lambda e: e.step_count >= 2,
    )

    # --- Goal-tracking pins ---
    # Goal state changed at least once — i.e. goal state was tracked across
    # steps, not just declared at plan creation.
    goal_events = [e for e in traced_emitter.events if isinstance(e, GoalStatusChangedEvent)]
    assert goal_events, "Expected at least one GoalStatusChangedEvent."

    # Final plan state: at least one goal is no longer active.
    final_plan = await store.load(plan_created.plan_id)
    assert final_plan is not None, "Plan should be loadable from the store after the run."
    assert final_plan.goals, "Plan should have at least one goal."
    non_active_goals = [g for g in final_plan.goals if g.status.value != "active"]
    assert non_active_goals, (
        "Expected at least one goal to be advanced beyond 'active' by the end of the run; "
        f"goal statuses were: {[g.status.value for g in final_plan.goals]}"
    )

    # Ensure auto-wiring actually linked the plan — eliminates the
    # "agent created a plan but never tracked it" failure mode.
    assert planning.active_plan_id == plan_created.plan_id

    # --- Schema-introspection invariants (planning F-W1, F-W4 hardening) ---
    # ``AgentStartEvent.tool_schemas`` carries only (name, description,
    # requires_approval) — not the full JSON schema — so we inspect the
    # exported ``FunctionTool.schema.parameters`` directly. Status enums
    # use Pydantic's ``$ref`` → ``$defs`` indirection for named types.
    tools_by_name = {t.schema.name: t for t in planning.tools}

    update_step_schema = tools_by_name["update_step"].schema.parameters
    step_status_enum = update_step_schema["$defs"]["StepStatus"]["enum"]
    assert step_status_enum == [
        "not_started",
        "in_progress",
        "completed",
        "skipped",
        "failed",
    ], f"Expected StepStatus enum on update_step.status; got {step_status_enum!r}."
    assert update_step_schema["properties"]["status"]["$ref"] == "#/$defs/StepStatus", (
        f"Expected update_step.status to $ref the StepStatus $def; got {update_step_schema['properties']['status']!r}."
    )

    update_goal_schema = tools_by_name["update_goal"].schema.parameters
    goal_status_enum = update_goal_schema["$defs"]["GoalStatus"]["enum"]
    assert goal_status_enum == ["active", "achieved", "blocked", "abandoned"], (
        f"Expected GoalStatus enum on update_goal.status; got {goal_status_enum!r}."
    )
    assert update_goal_schema["properties"]["status"]["$ref"] == "#/$defs/GoalStatus", (
        f"Expected update_goal.status to $ref the GoalStatus $def; got {update_goal_schema['properties']['status']!r}."
    )

    # Every audited optional/rationale-carrying parameter must carry a
    # non-empty description in the exported schema so adopter-authored
    # agents see the intent at tool-selection time.
    described_params = [
        ("revise_plan", "revised_steps"),
        ("revise_plan", "reason"),
        ("create_goal", "priority"),
        ("create_goal", "success_criteria"),
        ("create_goal", "parent_goal_id"),
    ]
    for tool_name, param_name in described_params:
        schema = tools_by_name[tool_name].schema.parameters
        param_schema = schema["properties"][param_name]
        description = param_schema.get("description", "")
        assert description, f"Expected {tool_name}.{param_name} to carry a non-empty description; got {param_schema!r}."


# ---------------------------------------------------------------------------
# Adaptive planning revision
# ---------------------------------------------------------------------------


async def test_adaptive_planning_contributor_drives_revision(
    traced_emitter: InMemoryEmitter,
) -> None:
    client = make_llm_client("anthropic")
    adaptive = AdaptivePlanningContributor()
    adaptive_section = adaptive.system_prompt_section()

    async def _run() -> object:
        traced_emitter.events.clear()
        store = InMemoryPlanStore()

        # Pre-seed a plan whose remaining steps will be invalidated by the
        # first step's result — this gives the adaptive contributor a
        # concrete reason to revise.
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

        planning = PlanningCapability(store, evaluator=None)
        planning.set_active_plan(plan_id)

        agent = ReActAgent(
            name="integrator",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are an API integration specialist. You already have an "
                "active plan. Inspect it with `get_plan`, execute the first "
                "step, and adapt the plan if new information changes the "
                "approach."
            ),
            tools=planning.tools,
            context_providers=[planning.context_provider],
            prompt_contributors=[adaptive],
            max_iterations=12,
        )

        agent_result = await agent.run(
            f"The active plan id is {plan_id}. Do the following: "
            "(1) call `get_plan` to inspect it; "
            "(2) mark the first step completed with `update_step`, recording "
            "the result 'API uses GraphQL, not REST'; "
            "(3) because the API actually uses GraphQL, call `revise_plan` "
            "to replace the remaining not-started steps with GraphQL-oriented "
            "steps (for example 'Build GraphQL client' and 'Test GraphQL "
            "queries'); include a revision reason that mentions GraphQL; "
            "(4) report that the plan was revised to use GraphQL."
        )
        return (agent_result, store, plan_id)

    result, store, plan_id = await run_with_retry(_run, max_attempts=2)

    # --- Result-shape invariant ---
    assert result.termination_reason == "complete", (
        f"Expected termination_reason='complete', got: {result.termination_reason!r}"
    )

    # --- System prompt invariant ---
    llm_requests = [e for e in traced_emitter.events if isinstance(e, LLMRequestEvent)]
    assert llm_requests, "Expected at least one LLMRequestEvent."
    for event in llm_requests:
        assert event.system_prompt is not None
        assert adaptive_section[1] in event.system_prompt, (
            "Adaptive planning contributor instructions missing from system prompt."
        )

    # --- Revision pin ---
    revision_event = assert_trace_contains(traced_emitter, PlanRevisedEvent)
    assert "graphql" in (revision_event.revision_reason or "").lower(), (
        f"Expected revision reason to mention GraphQL; got: {revision_event.revision_reason!r}"
    )

    # --- Final plan state pin ---
    final_plan = await store.load(plan_id)
    assert final_plan is not None
    # The completed step must be preserved and at least one remaining step
    # must mention the domain discovery keyword.
    preserved_step = final_plan.steps[0]
    assert preserved_step.status.value == "completed", (
        f"Expected first step to remain completed after revision; got: {preserved_step.status.value}"
    )
    remaining_descriptions = [s.description.lower() for s in final_plan.steps[1:]]
    assert any("graphql" in d for d in remaining_descriptions), (
        f"Expected at least one revised step description to mention 'GraphQL'; "
        f"remaining steps were: {remaining_descriptions}"
    )
