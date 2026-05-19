"""ReWOO agent: plan-first execution with variable substitution, parallel steps, and plan persistence.

Demonstrates the ReWOO (Reasoning Without Observation) three-phase lifecycle: a planner LLM
generates a structured plan, a worker executes tools in dependency order (independent steps
run in parallel), and a solver LLM synthesizes all observations into a final answer. Only
2 LLM calls are made regardless of tool count.

Related guide: docs/guides/agent-types.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    InMemoryPlanStore,
    MockLLMClient,
    tool,
)
from nanitics.infrastructure import (
    PlanCreatedEvent,
    PlanStepUpdatedEvent,
)
from nanitics.specialized import (
    ReWOOAgent,
    ReWOOPlan,
    ReWOOStep,
)

# --- Shared tools ---


@tool("search", "Search for information on a topic")
async def search(query: str) -> str:
    return f"Results for: {query}"


@tool("summarize", "Summarize the given text")
async def summarize(text: str) -> str:
    return f"Summary of: {text}"


@tool("failing_tool", "A tool that always fails")
async def failing_tool(input: str) -> str:
    raise ValueError(f"Tool error: {input}")


async def main() -> None:
    # --- Section 1: Basic Three-Phase Lifecycle ---
    print("--- Section 1: Basic Three-Phase Lifecycle ---")

    plan = ReWOOPlan(
        steps=[
            ReWOOStep(
                step_number=1,
                description="Search for AI agents",
                tool_name="search",
                arguments={"query": "AI agents"},
                depends_on=[],
            ),
            ReWOOStep(
                step_number=2,
                description="Summarize findings",
                tool_name="summarize",
                arguments={"text": "#1"},
                depends_on=[1],
            ),
        ]
    )
    plan_json = plan.model_dump_json()

    client = MockLLMClient(
        responses=[
            make_response(plan_json),  # planner — structured output
            make_response("AI agents are autonomous systems that use tools to accomplish tasks."),  # solver
        ]
    )
    emitter = make_emitter("rewoo-s1")
    plan_store = InMemoryPlanStore()

    agent = ReWOOAgent(
        name="research-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant.",
        tools=[search, summarize],
        plan_store=plan_store,
    )

    result = await agent.run("Research AI agents and summarize findings")

    assert result.output == "AI agents are autonomous systems that use tools to accomplish tasks."
    assert result.total_steps == 4, f"Expected 4 steps (1 planner + 2 worker + 1 solver), got: {result.total_steps}"
    assert result.termination_reason == "complete"
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert len(client.calls) == 2, f"Expected 2 LLM calls, got: {len(client.calls)}"
    assert client.calls[0]["output_schema"] is ReWOOPlan, "Planner should use structured output"
    assert client.calls[1]["output_schema"] is None, "Solver should use free-form text"

    print(f"  Output: {result.output}")
    print(f"  Total steps: {result.total_steps} (1 planner + 2 worker + 1 solver)")
    print(f"  Termination: {result.termination_reason}")
    print(f"  LLM calls: {len(client.calls)} (planner + solver only)")
    print(f"  Planner used structured output: {client.calls[0]['output_schema'] is ReWOOPlan}")
    print("✓ Three-phase lifecycle: plan → execute tools → synthesize answer with only 2 LLM calls")

    # --- Section 2: Variable Substitution and Dependencies ---
    print("\n--- Section 2: Variable Substitution and Dependencies ---")

    plan = ReWOOPlan(
        steps=[
            ReWOOStep(
                step_number=1,
                description="Search for quantum computing",
                tool_name="search",
                arguments={"query": "quantum computing"},
                depends_on=[],
            ),
            ReWOOStep(
                step_number=2,
                description="Summarize search results",
                tool_name="summarize",
                arguments={"text": "#1"},
                depends_on=[1],
            ),
        ]
    )
    plan_json = plan.model_dump_json()

    client = MockLLMClient(
        responses=[
            make_response(plan_json),
            make_response("Quantum computing summary complete."),
        ]
    )
    emitter = make_emitter("rewoo-s2")

    agent = ReWOOAgent(
        name="variable-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant.",
        tools=[search, summarize],
        plan_store=InMemoryPlanStore(),
    )

    result = await agent.run("Research quantum computing")

    # The solver receives observations with resolved variable references
    solver_input = client.calls[1]["messages"][0].content
    assert "Results for: quantum computing" in solver_input, "Solver should see search result"
    assert "Summary of: Results for: quantum computing" in solver_input, (
        "#1 in step 2 should be substituted with step 1's actual result"
    )
    assert "Research quantum computing" in solver_input, "Solver input should contain the original task"

    print("  Solver input (plan + observations):")
    for line in solver_input.split("\n"):
        if line.strip():
            print(f"    {line}")
    print("✓ Variable #1 resolved: summarize received search output, not literal '#1'")

    # --- Section 3: Parallel Execution (Independent Steps) ---
    print("\n--- Section 3: Parallel Execution (Independent Steps) ---")

    plan = ReWOOPlan(
        steps=[
            ReWOOStep(
                step_number=1,
                description="Search topic A",
                tool_name="search",
                arguments={"query": "topic A"},
                depends_on=[],
            ),
            ReWOOStep(
                step_number=2,
                description="Search topic B",
                tool_name="search",
                arguments={"query": "topic B"},
                depends_on=[],
            ),
            ReWOOStep(
                step_number=3,
                description="Summarize both",
                tool_name="summarize",
                arguments={"text": "#1 and #2"},
                depends_on=[1, 2],
            ),
        ]
    )
    plan_json = plan.model_dump_json()

    client = MockLLMClient(
        responses=[
            make_response(plan_json),
            make_response("Combined summary of topics A and B."),
        ]
    )
    emitter = make_emitter("rewoo-s3")

    agent = ReWOOAgent(
        name="parallel-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant.",
        tools=[search, summarize],
        plan_store=InMemoryPlanStore(),
    )

    result = await agent.run("Compare topic A and topic B")

    assert result.total_steps == 5, f"Expected 5 steps (1 planner + 3 worker + 1 solver), got: {result.total_steps}"
    assert len(client.calls) == 2, f"Still only 2 LLM calls, got: {len(client.calls)}"

    # Solver should see all three observations
    solver_input = client.calls[1]["messages"][0].content
    assert "Results for: topic A" in solver_input
    assert "Results for: topic B" in solver_input
    # Step 3 should have both #1 and #2 substituted
    assert "Summary of: Results for: topic A and Results for: topic B" in solver_input

    print(f"  Total steps: {result.total_steps} (1 planner + 3 worker + 1 solver)")
    print(f"  LLM calls: {len(client.calls)} (still only 2)")
    print("  Steps 1 & 2 are independent → same execution level → parallel")
    print("  Step 3 depends on both → next level → runs after both complete")
    print("✓ Independent steps execute in parallel; dependent steps wait")

    # --- Section 4: Plan Persistence ---
    print("\n--- Section 4: Plan Persistence ---")

    plan = ReWOOPlan(
        steps=[
            ReWOOStep(
                step_number=1,
                description="Search for ML",
                tool_name="search",
                arguments={"query": "machine learning"},
                depends_on=[],
            ),
            ReWOOStep(
                step_number=2,
                description="Summarize ML",
                tool_name="summarize",
                arguments={"text": "#1"},
                depends_on=[1],
            ),
        ]
    )
    plan_json = plan.model_dump_json()

    client = MockLLMClient(
        responses=[
            make_response(plan_json),
            make_response("ML summary."),
        ]
    )
    emitter = make_emitter("rewoo-s4")
    plan_store = InMemoryPlanStore()

    agent = ReWOOAgent(
        name="persistence-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant.",
        tools=[search, summarize],
        plan_store=plan_store,
    )

    await agent.run("Research machine learning")

    stored_plans = await plan_store.list_plans()
    assert len(stored_plans) == 1, f"Expected 1 plan, got: {len(stored_plans)}"

    stored_plan = stored_plans[0]
    assert len(stored_plan.steps) == 2, f"Expected 2 steps, got: {len(stored_plan.steps)}"

    for step in stored_plan.steps:
        assert step.status == "completed", f"Expected completed, got: {step.status}"
        assert step.result is not None, f"Step '{step.description}' should have a result"

    print(f"  Plan ID: {stored_plan.id}")
    print(f"  Step count: {len(stored_plan.steps)}")
    for step in stored_plan.steps:
        print(f"    {step.description}: status={step.status}, result={step.result}")
    print("✓ Plan stored and updated — all steps completed with results")

    # --- Section 5: Observability — Plan Events ---
    print("\n--- Section 5: Observability — Plan Events ---")

    plan = ReWOOPlan(
        steps=[
            ReWOOStep(
                step_number=1,
                description="Search for NLP",
                tool_name="search",
                arguments={"query": "NLP"},
                depends_on=[],
            ),
            ReWOOStep(
                step_number=2,
                description="Summarize NLP",
                tool_name="summarize",
                arguments={"text": "#1"},
                depends_on=[1],
            ),
        ]
    )
    plan_json = plan.model_dump_json()

    client = MockLLMClient(
        responses=[
            make_response(plan_json),
            make_response("NLP summary."),
        ]
    )
    emitter = make_emitter("rewoo-s5")

    agent = ReWOOAgent(
        name="events-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant.",
        tools=[search, summarize],
        plan_store=InMemoryPlanStore(),
    )

    await agent.run("Research NLP")

    plan_created_events = [e for e in emitter.events if isinstance(e, PlanCreatedEvent)]
    assert len(plan_created_events) == 1, f"Expected 1 PlanCreatedEvent, got: {len(plan_created_events)}"
    assert plan_created_events[0].step_count == 2

    step_updated_events = [e for e in emitter.events if isinstance(e, PlanStepUpdatedEvent)]
    assert len(step_updated_events) == 2, f"Expected 2 PlanStepUpdatedEvents, got: {len(step_updated_events)}"

    for event in step_updated_events:
        assert event.new_status == "completed"
        assert event.plan_id == plan_created_events[0].plan_id

    print(
        f"  PlanCreatedEvent: plan_id={plan_created_events[0].plan_id}, step_count={plan_created_events[0].step_count}"
    )
    for event in step_updated_events:
        print(f"  PlanStepUpdatedEvent: step={event.step_description}, status={event.new_status}")
    print("✓ Plan lifecycle observable through events")

    # --- Section 6: Error Resilience — Tool Failure ---
    print("\n--- Section 6: Error Resilience — Tool Failure ---")

    plan = ReWOOPlan(
        steps=[
            ReWOOStep(
                step_number=1,
                description="Run failing tool",
                tool_name="failing_tool",
                arguments={"input": "test"},
                depends_on=[],
            ),
            ReWOOStep(
                step_number=2,
                description="Search successfully",
                tool_name="search",
                arguments={"query": "backup"},
                depends_on=[],
            ),
        ]
    )
    plan_json = plan.model_dump_json()

    client = MockLLMClient(
        responses=[
            make_response(plan_json),
            make_response("Partial answer using available results."),
        ]
    )
    emitter = make_emitter("rewoo-s6")
    plan_store = InMemoryPlanStore()

    agent = ReWOOAgent(
        name="error-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant.",
        tools=[search, summarize, failing_tool],
        plan_store=plan_store,
    )

    result = await agent.run("Try failing and backup tools")

    assert result.termination_reason == "complete", "Agent should complete despite tool error"

    # Solver sees [ERROR] for failed step and successful result for step 2
    solver_input = client.calls[1]["messages"][0].content
    assert "[ERROR]" in solver_input, "Failed step should produce [ERROR] observation"
    assert "Results for: backup" in solver_input, "Successful step should have normal observation"

    # Plan store reflects step statuses
    stored_plans = await plan_store.list_plans()
    stored_plan = stored_plans[0]
    step_statuses = {step.description: step.status for step in stored_plan.steps}
    assert step_statuses["#1: Run failing tool"] == "failed"
    assert step_statuses["#2: Search successfully"] == "completed"

    print("  Solver input excerpts:")
    for line in solver_input.split("\n"):
        if "Observation" in line:
            print(f"    {line}")
    print(f"  Plan step statuses: {step_statuses}")
    print("✓ Tool failure captured as [ERROR] observation; agent completes with partial results")


if __name__ == "__main__":
    asyncio.run(main())
