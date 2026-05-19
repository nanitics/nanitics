"""GoalSatisfactionEvaluator and PlanAdherenceEvaluator accept/revise across plan states.

Both evaluators read plan state from a ``PlanStore`` — they do not call
LLMs themselves. The load-bearing contract is the mapping from plan
state to verdict:

  - ``PlanAdherenceEvaluator`` accepts iff no step remains
    ``not_started`` or ``in_progress`` (or the plan is abandoned /
    missing).
  - ``GoalSatisfactionEvaluator`` accepts iff no active goal remains
    unresolved (or the plan is abandoned / missing).

Deterministic unit-style cases exercise the pass/fail contract in both
directions. Additionally, one real-LLM integration case drives a
``ReActAgent`` through the planning tools to complete a plan and then
runs the evaluator on the resulting store — this pins the "agent
actually satisfied the evaluator" contract rather than just the
store-level mapping.

Acceptance criteria (``PlanAdherenceEvaluator``):
  - All-not-started plan → ``REVISE``, score 0.0, feedback naming at
    least one incomplete step.
  - All-completed plan → ``ACCEPT``, score 1.0.
  - Agent-driven completion (real LLM) → ``ACCEPT``.

Acceptance criteria (``GoalSatisfactionEvaluator``):
  - Plan with an active goal → ``REVISE``, score 0.0, feedback naming
    the active goal.
  - Plan with all goals achieved → ``ACCEPT``, score 1.0.
  - Agent-driven goal resolution (real LLM) → ``ACCEPT``.
"""

from __future__ import annotations

import pytest

from nanitics.evaluation import (
    EvaluationContext,
    EvaluationVerdict,
)
from nanitics.planning import (
    Goal,
    GoalSatisfactionEvaluator,
    GoalStatus,
    InMemoryPlanStore,
    Plan,
    PlanAdherenceEvaluator,
    PlanningCapability,
    PlanStep,
    StepStatus,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from validation.helpers import make_llm_client, run_with_retry


def _context() -> EvaluationContext:
    return EvaluationContext(messages=[], task_input="Complete the plan.")


# ---------------------------------------------------------------------------
# PlanAdherenceEvaluator — parametrized pass/fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("all_completed", "expected_verdict"),
    [
        pytest.param(False, EvaluationVerdict.REVISE, id="unsatisfied_not_started"),
        pytest.param(True, EvaluationVerdict.ACCEPT, id="satisfied_all_completed"),
    ],
)
async def test_plan_adherence_evaluator_direct(
    traced_emitter: InMemoryEmitter,
    all_completed: bool,
    expected_verdict: EvaluationVerdict,
) -> None:
    _ = traced_emitter  # fixture parity — trace save is useful on failure.

    store = InMemoryPlanStore()
    steps = [
        PlanStep(description="Draft outline"),
        PlanStep(description="Write body"),
        PlanStep(description="Review"),
    ]
    if all_completed:
        steps = [s.model_copy(update={"status": StepStatus.completed}) for s in steps]
    plan = Plan(name="Writing Task", steps=steps)
    await store.save(plan)

    evaluator = PlanAdherenceEvaluator(store, plan.id)
    result = await evaluator.evaluate("Output draft", _context())

    assert result.verdict == expected_verdict, (
        f"Expected verdict {expected_verdict}, got {result.verdict} (feedback={result.feedback!r})"
    )
    assert result.evaluator_name == "plan_adherence"
    if expected_verdict == EvaluationVerdict.ACCEPT:
        assert result.score == 1.0
    else:
        assert result.score == 0.0
        # Feedback must reference at least one incomplete step description,
        # pinning the branch that enumerates incomplete steps (not a
        # generic accept-by-fallback).
        assert any(s.description in (result.feedback or "") for s in plan.steps), (
            f"Expected feedback to name at least one incomplete step; got: {result.feedback!r}"
        )


# ---------------------------------------------------------------------------
# GoalSatisfactionEvaluator — parametrized pass/fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("all_resolved", "expected_verdict"),
    [
        pytest.param(False, EvaluationVerdict.REVISE, id="active_goal_remaining"),
        pytest.param(True, EvaluationVerdict.ACCEPT, id="all_goals_resolved"),
    ],
)
async def test_goal_satisfaction_evaluator_direct(
    traced_emitter: InMemoryEmitter,
    all_resolved: bool,
    expected_verdict: EvaluationVerdict,
) -> None:
    _ = traced_emitter

    store = InMemoryPlanStore()
    goals = [
        Goal(description="Deliver a correct summary", priority=2),
        Goal(description="Cite at least one source", priority=1),
    ]
    if all_resolved:
        goals = [g.model_copy(update={"status": GoalStatus.achieved}) for g in goals]
    plan = Plan(name="Summary Task", steps=[PlanStep(description="Write summary")], goals=goals)
    await store.save(plan)

    evaluator = GoalSatisfactionEvaluator(store, plan.id)
    result = await evaluator.evaluate("Summary complete", _context())

    assert result.verdict == expected_verdict, (
        f"Expected verdict {expected_verdict}, got {result.verdict} (feedback={result.feedback!r})"
    )
    assert result.evaluator_name == "goal_satisfaction"
    if expected_verdict == EvaluationVerdict.ACCEPT:
        assert result.score == 1.0
    else:
        assert result.score == 0.0
        assert any(g.description in (result.feedback or "") for g in plan.goals), (
            f"Expected feedback to name at least one active goal; got: {result.feedback!r}"
        )


# ---------------------------------------------------------------------------
# Real agent drives plan to completion and satisfies the adherence evaluator
# ---------------------------------------------------------------------------


async def test_plan_adherence_evaluator_accepts_after_real_agent_completes_plan(
    traced_emitter: InMemoryEmitter,
) -> None:
    client = make_llm_client("anthropic")

    async def _run() -> object:
        traced_emitter.events.clear()
        store = InMemoryPlanStore()
        planning = PlanningCapability(store, evaluator="adherence")

        agent = ReActAgent(
            name="plan-completer",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a plan-completing assistant. Use the planning tools "
                "to create and execute a small plan, marking each step "
                "completed as you finish it."
            ),
            tools=planning.tools,
            context_providers=[planning.context_provider],
            output_evaluator=planning.output_evaluator,
            max_iterations=12,
        )
        agent_result = await agent.run(
            "Create a short plan named 'Daily Standup' with exactly two "
            "steps: 'share status' and 'note blockers'. After creating the "
            "plan, execute both steps by calling `update_step` with status "
            "'completed' and a brief result for each. Then give a short "
            "one-sentence summary."
        )
        return (agent_result, store, planning)

    result, store, planning = await run_with_retry(_run, max_attempts=2)

    assert result.termination_reason == "complete", (
        f"Expected termination_reason='complete', got: {result.termination_reason!r}"
    )
    assert planning.active_plan_id is not None, "Expected planning capability to have auto-wired a plan."

    # Now run the evaluator standalone against the final store state.
    evaluator = PlanAdherenceEvaluator(store, planning.active_plan_id)
    verdict_result = await evaluator.evaluate(result.output or "", _context())

    assert verdict_result.verdict == EvaluationVerdict.ACCEPT, (
        f"Expected adherence evaluator to ACCEPT after plan completion, got "
        f"{verdict_result.verdict} (feedback={verdict_result.feedback!r})"
    )
    assert verdict_result.score == 1.0


# ---------------------------------------------------------------------------
# Real agent drives goals to resolution and satisfies the goal evaluator
# ---------------------------------------------------------------------------


async def test_goal_satisfaction_evaluator_accepts_after_real_agent_resolves_goals(
    traced_emitter: InMemoryEmitter,
) -> None:
    client = make_llm_client("anthropic")

    async def _run() -> object:
        traced_emitter.events.clear()
        store = InMemoryPlanStore()
        planning = PlanningCapability(store, evaluator="goal")

        agent = ReActAgent(
            name="goal-resolver",
            llm_client=client,
            emitter=traced_emitter,
            system_prompt=(
                "You are a goal-tracking assistant. Create a plan, define a "
                "goal, complete the work that resolves the goal, and update "
                "the goal status accordingly."
            ),
            tools=planning.tools,
            context_providers=[planning.context_provider],
            output_evaluator=planning.output_evaluator,
            max_iterations=12,
        )
        agent_result = await agent.run(
            "Create a plan called 'Greeting' with one step 'say hello'. "
            "Then create a goal 'Greet the user'. Execute the step by "
            "calling `update_step` with status 'completed'. Mark the goal "
            "as achieved with `update_goal`. Then respond with a friendly "
            "greeting as your final answer."
        )
        return (agent_result, store, planning)

    result, store, planning = await run_with_retry(_run, max_attempts=2)

    assert result.termination_reason == "complete", (
        f"Expected termination_reason='complete', got: {result.termination_reason!r}"
    )
    assert planning.active_plan_id is not None

    evaluator = GoalSatisfactionEvaluator(store, planning.active_plan_id)
    verdict_result = await evaluator.evaluate(result.output or "", _context())

    assert verdict_result.verdict == EvaluationVerdict.ACCEPT, (
        f"Expected goal evaluator to ACCEPT after goal resolution, got "
        f"{verdict_result.verdict} (feedback={verdict_result.feedback!r})"
    )
    assert verdict_result.score == 1.0
