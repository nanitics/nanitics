"""Supervisor: post-execution monitoring with triggers and corrective actions.

Demonstrates ``Supervisor`` — the post-execution monitor that evaluates agent
results against ordered triggers and takes corrective action. Covers all four
supervision outcomes (accept, retry, reassign, escalate), three built-in trigger
types (``PredicateTrigger``, ``QualityTrigger``, ``BudgetTrigger``), and
multi-trigger composition.

Related guide: docs/guides/multi-agent-coordination.md
"""

import asyncio

from examples.helpers import make_emitter, make_response, make_usage
from nanitics.composition import (
    BudgetTrigger,
    PredicateTrigger,
    QualityTrigger,
    SupervisionAction,
    SupervisionDecision,
    Supervisor,
)
from nanitics.evaluation import (
    EvaluationCheck,
    ProgrammaticEvaluator,
)
from nanitics.infrastructure import (
    MockLLMClient,
    SupervisionEvent,
)
from nanitics.strategies import ReActAgent


async def main() -> None:
    # --- Section 1: Accept — All Triggers Pass ---
    print("--- Section 1: Accept — All Triggers Pass ---")

    # A PredicateTrigger that always passes (returns None = no intervention)
    trigger = PredicateTrigger(
        name="always_pass",
        predicate=lambda result, task: None,
    )

    client = MockLLMClient(
        responses=[
            make_response("The market shows strong growth across all sectors."),
        ]
    )
    emitter = make_emitter("supervisor-s1")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the given topic.",
        tools=[],
    )

    supervisor = Supervisor(
        triggers=[trigger],
        emitter=emitter,
    )

    supervision_result = await supervisor.supervise(agent, "Analyze the market")

    # When no trigger fires, the result is accepted with no interventions
    assert supervision_result.accepted is True
    assert supervision_result.total_attempts == 1
    assert supervision_result.interventions == []
    assert supervision_result.final_agent == "analyst"
    assert supervision_result.result.output == "The market shows strong growth across all sectors."

    # SupervisionEvent emitted with action="accept" and trigger_name="all_passed"
    sup_events = [e for e in emitter.events if isinstance(e, SupervisionEvent)]
    assert len(sup_events) == 1
    assert sup_events[0].action == "accept"
    assert sup_events[0].trigger_name == "all_passed"
    assert sup_events[0].attempt == 1

    print(f"  Result accepted: {supervision_result.accepted}")
    print(f"  Attempts: {supervision_result.total_attempts}")
    print(f"  Event: action={sup_events[0].action}, trigger={sup_events[0].trigger_name}")
    print("✓ No trigger fired — result accepted")

    # --- Section 2: Retry with Feedback ---
    print("\n--- Section 2: Retry with Feedback ---")

    # PredicateTrigger with mutable state: fires RETRY on first call, passes on second
    call_count = 0

    def retry_once(result, task):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return SupervisionDecision(
                action=SupervisionAction.RETRY,
                feedback="Include specific numbers and data points.",
                trigger_name="detail_check",
            )
        return None

    trigger = PredicateTrigger(name="detail_check", predicate=retry_once)

    client = MockLLMClient(
        responses=[
            make_response("The market is growing."),  # First attempt — too vague
            make_response("The market grew 23% YoY with $4.2B in revenue."),  # After feedback
        ]
    )
    emitter = make_emitter("supervisor-s2")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the given topic.",
        tools=[],
    )

    supervisor = Supervisor(
        triggers=[trigger],
        emitter=emitter,
        max_retries=2,
    )

    supervision_result = await supervisor.supervise(agent, "Analyze the market")

    # Second attempt passes after feedback
    assert supervision_result.accepted is True
    assert supervision_result.total_attempts == 2
    assert supervision_result.result.output == "The market grew 23% YoY with $4.2B in revenue."

    # One intervention recorded: the RETRY with feedback
    assert len(supervision_result.interventions) == 1
    assert supervision_result.interventions[0].action == SupervisionAction.RETRY
    assert supervision_result.interventions[0].feedback == "Include specific numbers and data points."
    assert supervision_result.interventions[0].trigger_name == "detail_check"

    # Two SupervisionEvents: retry then accept
    sup_events = [e for e in emitter.events if isinstance(e, SupervisionEvent)]
    assert len(sup_events) == 2
    assert sup_events[0].action == "retry"
    assert sup_events[1].action == "accept"

    print("  First attempt: 'The market is growing.' → retry with feedback")
    print(f"  Second attempt: '{supervision_result.result.output}' → accepted")
    print(f"  Interventions: {len(supervision_result.interventions)}")
    print("✓ Retry appended feedback — second attempt accepted")

    # --- Section 3: Reassign to Different Agent ---
    print("\n--- Section 3: Reassign to Different Agent ---")

    # PredicateTrigger that reassigns to "specialist" on first call
    reassign_count = 0

    def reassign_once(result, task):
        nonlocal reassign_count
        reassign_count += 1
        if reassign_count == 1:
            return SupervisionDecision(
                action=SupervisionAction.REASSIGN,
                feedback="This requires domain expertise.",
                reassign_to="specialist",
                trigger_name="expertise_check",
            )
        return None

    trigger = PredicateTrigger(name="expertise_check", predicate=reassign_once)

    emitter = make_emitter("supervisor-s3")

    agent_a = ReActAgent(
        name="generalist",
        llm_client=MockLLMClient(
            responses=[
                make_response("Generic analysis of the pharmaceutical sector."),
            ]
        ),
        emitter=emitter,
        system_prompt="General-purpose analyst.",
        tools=[],
    )

    agent_b = ReActAgent(
        name="pharma_specialist",
        llm_client=MockLLMClient(
            responses=[
                make_response("FDA approval pipeline shows 3 Phase III candidates with 78% probability."),
            ]
        ),
        emitter=emitter,
        system_prompt="Pharmaceutical industry specialist.",
        tools=[],
    )

    supervisor = Supervisor(
        triggers=[trigger],
        emitter=emitter,
        agents={"specialist": agent_b},
    )

    supervision_result = await supervisor.supervise(agent_a, "Analyze pharma pipeline")

    # Reassignment switched to the specialist agent
    assert supervision_result.accepted is True
    assert supervision_result.total_attempts == 2
    assert supervision_result.final_agent == "pharma_specialist"
    assert "Phase III" in supervision_result.result.output

    # One intervention: REASSIGN to specialist
    assert len(supervision_result.interventions) == 1
    assert supervision_result.interventions[0].action == SupervisionAction.REASSIGN
    assert supervision_result.interventions[0].reassign_to == "specialist"

    print("  Generalist output: 'Generic analysis...' → reassigned to specialist")
    print(f"  Specialist output: '{supervision_result.result.output}'")
    print(f"  Final agent: {supervision_result.final_agent}")
    print("✓ Reassignment switched execution to specialist agent")

    # --- Section 4: Evaluator-Driven Quality Gate ---
    print("\n--- Section 4: Evaluator-Driven Quality Gate ---")

    # QualityTrigger wraps a ProgrammaticEvaluator for supervision-level quality gating.
    # Evaluator REVISE → supervision RETRY. Evaluator REJECT → supervision ESCALATE.
    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="sufficient_detail",
                check=lambda output: len(output) > 30,
                feedback="Output too brief — provide a detailed analysis.",
            ),
        ],
        max_revisions=1,
    )

    quality_trigger = QualityTrigger(evaluator=evaluator)

    client = MockLLMClient(
        responses=[
            make_response("Looks good."),  # Too short — triggers RETRY
            make_response("Comprehensive analysis: revenue up 15%, margins expanding, guidance raised for Q3."),
        ]
    )
    emitter = make_emitter("supervisor-s4")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the given topic.",
        tools=[],
    )

    supervisor = Supervisor(
        triggers=[quality_trigger],
        emitter=emitter,
        max_retries=2,
    )

    supervision_result = await supervisor.supervise(agent, "Analyze quarterly earnings")

    # First attempt failed quality check, second passed
    assert supervision_result.accepted is True
    assert supervision_result.total_attempts == 2

    # Intervention from the quality trigger
    assert len(supervision_result.interventions) == 1
    assert supervision_result.interventions[0].trigger_name == "quality"
    assert supervision_result.interventions[0].action == SupervisionAction.RETRY
    assert "too brief" in supervision_result.interventions[0].feedback

    sup_events = [e for e in emitter.events if isinstance(e, SupervisionEvent)]
    assert sup_events[0].action == "retry"
    assert sup_events[0].trigger_name == "quality"

    print("  First attempt: 'Looks good.' → quality trigger fired (RETRY)")
    print(f"  Feedback: {supervision_result.interventions[0].feedback}")
    print(f"  Second attempt accepted: {supervision_result.result.output[:60]}...")
    print("✓ QualityTrigger drove evaluator-based retry at supervision level")

    # --- Section 5: Multi-Trigger Composition ---
    print("\n--- Section 5: Multi-Trigger Composition ---")

    # Multiple triggers evaluated in order. First trigger that fires wins.
    # BudgetTrigger (cheap check) before QualityTrigger (expensive check).
    budget_trigger = BudgetTrigger(max_tokens=50)

    evaluator = ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="has_data",
                check=lambda output: any(c.isdigit() for c in output),
                feedback="Include numerical data.",
            ),
        ],
        max_revisions=1,
    )
    quality_trigger = QualityTrigger(evaluator=evaluator)

    # Response with high token usage: 40 + 20 = 60 total, exceeds budget of 50
    client = MockLLMClient(
        responses=[
            make_response(
                "Analysis complete with data points.",
                usage=make_usage(input_tokens=40, output_tokens=20),
            ),
        ]
    )
    emitter = make_emitter("supervisor-s5")

    agent = ReActAgent(
        name="analyst",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze the given topic.",
        tools=[],
    )

    # Budget trigger first (cheap), quality trigger second (expensive)
    supervisor = Supervisor(
        triggers=[budget_trigger, quality_trigger],
        emitter=emitter,
    )

    supervision_result = await supervisor.supervise(agent, "Analyze the market")

    # Budget trigger fired first → ESCALATE (non-retryable)
    assert supervision_result.accepted is False
    assert supervision_result.total_attempts == 1

    # Intervention from budget trigger, not quality
    assert len(supervision_result.interventions) == 1
    assert supervision_result.interventions[0].trigger_name == "budget"
    assert supervision_result.interventions[0].action == SupervisionAction.ESCALATE
    assert "60/50" in supervision_result.interventions[0].feedback

    # SupervisionEvent confirms escalation from budget trigger
    sup_events = [e for e in emitter.events if isinstance(e, SupervisionEvent)]
    assert len(sup_events) == 1
    assert sup_events[0].action == "escalate"
    assert sup_events[0].trigger_name == "budget"

    print("  Token usage: 60 (budget: 50) → budget trigger fired first")
    print("  Quality trigger: never evaluated (short-circuit)")
    print(f"  Result accepted: {supervision_result.accepted}")
    print("✓ First-fires-wins: budget trigger short-circuited quality check")


if __name__ == "__main__":
    asyncio.run(main())
