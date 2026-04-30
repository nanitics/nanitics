import asyncio

import pytest
from pydantic import ValidationError

from nanitics import (
    LLMResponse,
    MockLLMClient,
    ReWOOAgent,
    ReWOOPlan,
    ReWOOStep,
    tool,
)
from nanitics.capabilities.planning.store import InMemoryPlanStore
from nanitics.core.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)
from nanitics.infrastructure.observability.events import (
    AgentStartEvent,
    PlanCreatedEvent,
    PlanStepUpdatedEvent,
)
from nanitics.safety.cancellation import CancellationToken
from tests.testing_helpers import make_emitter, make_response


def _plan_json(steps: list[ReWOOStep]) -> str:
    return ReWOOPlan(steps=steps).model_dump_json()


@tool(name="search", description="Search the web")
async def search_tool(query: str) -> str:
    return f"Results for: {query}"


@tool(name="summarize", description="Summarize text")
async def summarize_tool(text: str) -> str:
    return f"Summary of: {text}"


@tool(name="slow", description="A slow tool")
async def slow_tool(input: str) -> str:
    await asyncio.sleep(0.05)
    return f"slow: {input}"


@tool(name="failing", description="Always fails")
async def failing_tool(reason: str) -> str:
    raise ValueError(f"Tool error: {reason}")


class _AcceptEvaluator:
    @property
    def max_revisions(self) -> int:
        return 2

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            evaluator_name="test-evaluator",
        )


class _ReviseOnceEvaluator:
    def __init__(self) -> None:
        self._call_count = 0

    @property
    def max_revisions(self) -> int:
        return 2

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        self._call_count += 1
        if self._call_count == 1:
            return EvaluationResult(
                verdict=EvaluationVerdict.REVISE,
                feedback="Please improve",
                evaluator_name="test-evaluator",
            )
        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            evaluator_name="test-evaluator",
        )


class _RejectEvaluator:
    @property
    def max_revisions(self) -> int:
        return 1

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        return EvaluationResult(
            verdict=EvaluationVerdict.REJECT,
            feedback="Not acceptable",
            evaluator_name="test-evaluator",
        )


# ──────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────


class TestReWOOStep:
    def test_construction(self) -> None:
        step = ReWOOStep(
            step_number=1,
            description="Search for topic",
            tool_name="search",
            arguments={"query": "topic"},
            depends_on=[],
        )
        assert step.step_number == 1
        assert step.tool_name == "search"
        assert step.depends_on == []

    def test_frozen(self) -> None:
        step = ReWOOStep(
            step_number=1,
            description="test",
            tool_name="search",
            arguments={"query": "test"},
            depends_on=[],
        )
        with pytest.raises(ValidationError):
            step.step_number = 2


class TestReWOOPlan:
    def test_construction(self) -> None:
        plan = ReWOOPlan(
            steps=[
                ReWOOStep(
                    step_number=1,
                    description="Search",
                    tool_name="search",
                    arguments={"query": "test"},
                    depends_on=[],
                ),
            ]
        )
        assert len(plan.steps) == 1


# ──────────────────────────────────────────────────────────
# Variable Substitution
# ──────────────────────────────────────────────────────────


class TestVariableSubstitution:
    def test_simple_substitution(self) -> None:
        from nanitics.core.agents.rewoo import _substitute_variables

        result = _substitute_variables({"text": "#1"}, {1: "hello world"})
        assert result["text"] == "hello world"

    def test_embedded_substitution(self) -> None:
        from nanitics.core.agents.rewoo import _substitute_variables

        result = _substitute_variables(
            {"query": "search for #1 and #2"},
            {1: "cats", 2: "dogs"},
        )
        assert result["query"] == "search for cats and dogs"

    def test_missing_reference(self) -> None:
        from nanitics.core.agents.rewoo import _substitute_variables

        result = _substitute_variables({"text": "#99"}, {})
        assert result["text"] == "[Step #99 failed or not available]"

    def test_no_references(self) -> None:
        from nanitics.core.agents.rewoo import _substitute_variables

        result = _substitute_variables({"query": "plain text"}, {1: "unused"})
        assert result["query"] == "plain text"

    def test_multiple_keys(self) -> None:
        from nanitics.core.agents.rewoo import _substitute_variables

        result = _substitute_variables(
            {"a": "#1", "b": "#2"},
            {1: "first", 2: "second"},
        )
        assert result["a"] == "first"
        assert result["b"] == "second"

    def test_non_string_values_pass_through(self) -> None:
        from nanitics.core.agents.rewoo import _substitute_variables

        result = _substitute_variables(
            {"count": 5, "enabled": True, "query": "#1"},
            {1: "hello"},
        )
        assert result["count"] == 5
        assert result["enabled"] is True
        assert result["query"] == "hello"

    def test_mixed_types_with_references(self) -> None:
        from nanitics.core.agents.rewoo import _substitute_variables

        result = _substitute_variables(
            {"text": "summarize #1", "limit": 10, "tags": None},
            {1: "some article"},
        )
        assert result["text"] == "summarize some article"
        assert result["limit"] == 10
        assert result["tags"] is None


# ──────────────────────────────────────────────────────────
# Execution Levels
# ──────────────────────────────────────────────────────────


class TestBuildExecutionLevels:
    def test_independent_steps_same_level(self) -> None:
        from nanitics.core.agents.rewoo import _build_execution_levels

        steps = [
            ReWOOStep(step_number=1, description="a", tool_name="search", arguments={}, depends_on=[]),
            ReWOOStep(step_number=2, description="b", tool_name="search", arguments={}, depends_on=[]),
        ]
        levels = _build_execution_levels(steps)
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_linear_chain(self) -> None:
        from nanitics.core.agents.rewoo import _build_execution_levels

        steps = [
            ReWOOStep(step_number=1, description="a", tool_name="search", arguments={}, depends_on=[]),
            ReWOOStep(step_number=2, description="b", tool_name="summarize", arguments={"text": "#1"}, depends_on=[1]),
            ReWOOStep(step_number=3, description="c", tool_name="summarize", arguments={"text": "#2"}, depends_on=[2]),
        ]
        levels = _build_execution_levels(steps)
        assert len(levels) == 3
        assert [s.step_number for s in levels[0]] == [1]
        assert [s.step_number for s in levels[1]] == [2]
        assert [s.step_number for s in levels[2]] == [3]

    def test_diamond_dependency(self) -> None:
        from nanitics.core.agents.rewoo import _build_execution_levels

        steps = [
            ReWOOStep(step_number=1, description="a", tool_name="search", arguments={}, depends_on=[]),
            ReWOOStep(step_number=2, description="b", tool_name="search", arguments={}, depends_on=[1]),
            ReWOOStep(step_number=3, description="c", tool_name="search", arguments={}, depends_on=[1]),
            ReWOOStep(step_number=4, description="d", tool_name="summarize", arguments={}, depends_on=[2, 3]),
        ]
        levels = _build_execution_levels(steps)
        assert len(levels) == 3
        assert [s.step_number for s in levels[0]] == [1]
        assert sorted(s.step_number for s in levels[1]) == [2, 3]
        assert [s.step_number for s in levels[2]] == [4]

    def test_circular_dependency_raises(self) -> None:
        from nanitics.core.agents.rewoo import _build_execution_levels

        steps = [
            ReWOOStep(step_number=1, description="a", tool_name="search", arguments={}, depends_on=[2]),
            ReWOOStep(step_number=2, description="b", tool_name="search", arguments={}, depends_on=[1]),
        ]
        with pytest.raises(ValueError, match="Circular dependency"):
            _build_execution_levels(steps)


# ──────────────────────────────────────────────────────────
# Plan Generation
# ──────────────────────────────────────────────────────────


class TestPlanGeneration:
    async def test_plan_stored_in_plan_store(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1,
                    description="Search topic",
                    tool_name="search",
                    arguments={"query": "test"},
                    depends_on=[],
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),  # planner
                make_response(content="Final answer"),  # solver
            ]
        )
        emitter = make_emitter()
        plan_store = InMemoryPlanStore()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=plan_store,
        )

        await agent.run("Research topic")

        plans = await plan_store.list_plans()
        assert len(plans) == 1
        assert len(plans[0].steps) == 1
        assert "Search topic" in plans[0].steps[0].description

    async def test_planner_receives_tool_descriptions(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "test"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Final answer"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool, summarize_tool],
            plan_store=InMemoryPlanStore(),
        )

        await agent.run("Test")

        # Planner call should have output_schema=ReWOOPlan
        assert client.calls[0]["output_schema"] is ReWOOPlan
        # System prompt should contain tool descriptions
        planner_system = client.calls[0]["system_prompt"]
        assert "search" in planner_system
        assert "summarize" in planner_system


# ──────────────────────────────────────────────────────────
# Worker Execution
# ──────────────────────────────────────────────────────────


class TestWorkerExecution:
    async def test_sequential_execution(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "topic"}, depends_on=[]
                ),
                ReWOOStep(
                    step_number=2,
                    description="Summarize",
                    tool_name="summarize",
                    arguments={"text": "#1"},
                    depends_on=[1],
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Synthesis"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool, summarize_tool],
            plan_store=InMemoryPlanStore(),
        )

        result = await agent.run("Research topic")

        assert result.output == "Synthesis"
        assert result.termination_reason == "complete"
        # Solver should receive the observations
        solver_input = client.calls[1]["messages"][0].content
        assert "Results for: topic" in solver_input
        assert "Summary of: Results for: topic" in solver_input

    async def test_parallel_execution(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search A", tool_name="slow", arguments={"input": "a"}, depends_on=[]
                ),
                ReWOOStep(
                    step_number=2, description="Search B", tool_name="slow", arguments={"input": "b"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Combined result"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[slow_tool],
            plan_store=InMemoryPlanStore(),
        )

        import time

        start = time.perf_counter()
        result = await agent.run("Do two things")
        elapsed = time.perf_counter() - start

        assert result.output == "Combined result"
        # Both steps take 50ms each. If sequential, would take >=100ms.
        # Parallel should be <100ms (with some margin).
        assert elapsed < 0.15

    async def test_worker_error_handling(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Fail", tool_name="failing", arguments={"reason": "test"}, depends_on=[]
                ),
                ReWOOStep(
                    step_number=2, description="Search", tool_name="search", arguments={"query": "test"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Partial answer"),
            ]
        )
        emitter = make_emitter()
        plan_store = InMemoryPlanStore()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[failing_tool, search_tool],
            plan_store=plan_store,
        )

        result = await agent.run("Try things")

        # Should complete despite tool error
        assert result.output == "Partial answer"
        assert result.termination_reason == "complete"

        # Solver should receive error message
        solver_input = client.calls[1]["messages"][0].content
        assert "Tool error: test" in solver_input

        # Plan store should show failed step
        plans = await plan_store.list_plans()
        step_statuses = {s.description: s.status for s in plans[0].steps}
        assert step_statuses["#1: Fail"] == "failed"
        assert step_statuses["#2: Search"] == "completed"

    async def test_max_observation_length_cap(self) -> None:
        @tool(name="verbose", description="Returns long text")
        async def verbose_tool(x: str) -> str:
            return "A" * 10000

        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Verbose", tool_name="verbose", arguments={"x": "go"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Done"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[verbose_tool],
            plan_store=InMemoryPlanStore(),
            max_observation_length=100,
        )

        await agent.run("Test")
        solver_input = client.calls[1]["messages"][0].content
        # Observation in solver input should be capped
        # The tool returns 10000 A's, capped to 100
        assert "A" * 100 in solver_input
        assert "A" * 101 not in solver_input


# ──────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────


class TestSolver:
    async def test_solver_receives_plan_and_observations(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1,
                    description="Search cats",
                    tool_name="search",
                    arguments={"query": "cats"},
                    depends_on=[],
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Cats are great"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )

        result = await agent.run("Tell me about cats")

        solver_call = client.calls[1]
        solver_input = solver_call["messages"][0].content
        assert "Tell me about cats" in solver_input
        assert "Search cats" in solver_input
        assert "Results for: cats" in solver_input
        assert result.output == "Cats are great"

    async def test_solver_with_evaluator_accept(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "q"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Good answer"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            output_evaluator=_AcceptEvaluator(),
        )

        result = await agent.run("Test")
        assert result.termination_reason == "complete"

    async def test_solver_with_evaluator_revise(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "q"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Bad answer"),
                make_response(content="Better answer"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            output_evaluator=_ReviseOnceEvaluator(),
        )

        result = await agent.run("Test")
        assert result.output == "Better answer"
        assert result.termination_reason == "complete"
        # 3 LLM calls: planner, solver attempt 1, solver attempt 2
        assert len(client.calls) == 3

    async def test_solver_with_evaluator_reject(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "q"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Bad answer"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            output_evaluator=_RejectEvaluator(),
        )

        result = await agent.run("Test")
        assert result.termination_reason == "evaluation_failed"


# ──────────────────────────────────────────────────────────
# Cancellation
# ──────────────────────────────────────────────────────────


class TestCancellation:
    async def test_cancellation_between_levels(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1,
                    description="Search",
                    tool_name="cancel_search",
                    arguments={"query": "a"},
                    depends_on=[],
                ),
                ReWOOStep(
                    step_number=2,
                    description="Summarize",
                    tool_name="summarize",
                    arguments={"text": "#1"},
                    depends_on=[1],
                ),
            ]
        )
        token = CancellationToken()

        @tool(name="cancel_search", description="Search then cancel")
        async def cancel_search_tool(query: str) -> str:
            token.cancel()
            return "found"

        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Partial"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[cancel_search_tool, summarize_tool],
            plan_store=InMemoryPlanStore(),
            cancellation_token=token,
        )

        await agent.run("Do stuff")
        # Step 2 should not execute because cancellation happened after level 0
        solver_input = client.calls[1]["messages"][0].content
        assert "found" in solver_input
        # Step 2 result should be missing (not executed)
        assert "[No result]" in solver_input


# ──────────────────────────────────────────────────────────
# Events
# ──────────────────────────────────────────────────────────


class TestEvents:
    async def test_event_emission(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "test"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Done"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )

        await agent.run("Test")

        event_types = [e.event_type for e in emitter.events]
        assert "agent.start" in event_types
        assert "agent.complete" in event_types
        assert "agent.step" in event_types
        assert "planning.plan.created" in event_types
        assert "planning.step.updated" in event_types
        assert "tool.invoke" in event_types
        assert "tool.result" in event_types

    async def test_plan_created_event_has_step_details(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1,
                    description="Search topic",
                    tool_name="search",
                    arguments={"query": "test"},
                    depends_on=[],
                ),
                ReWOOStep(
                    step_number=2,
                    description="Summarize results",
                    tool_name="summarize",
                    arguments={"text": "#1"},
                    depends_on=[1],
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Done"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool, summarize_tool],
            plan_store=InMemoryPlanStore(),
        )

        await agent.run("Test")

        plan_events = [e for e in emitter.events if isinstance(e, PlanCreatedEvent)]
        assert len(plan_events) == 1
        event = plan_events[0]
        assert len(event.steps) == 2

        step1 = event.steps[0]
        assert step1.description == "Search topic"
        assert step1.metadata["tool"] == "search"
        assert step1.metadata["variable"] == "#1"
        assert step1.metadata["depends_on"] == []
        assert step1.metadata["execution_level"] == 0

        step2 = event.steps[1]
        assert step2.description == "Summarize results"
        assert step2.metadata["tool"] == "summarize"
        assert step2.metadata["variable"] == "#2"
        assert step2.metadata["depends_on"] == [1]
        assert step2.metadata["execution_level"] == 1

    async def test_start_event_has_tools(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "test"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Done"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool, summarize_tool],
            plan_store=InMemoryPlanStore(),
        )

        await agent.run("Test")

        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert sorted(start_events[0].tools_available) == ["search", "summarize"]

    async def test_plan_step_updated_events(self) -> None:
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1, description="Search", tool_name="search", arguments={"query": "a"}, depends_on=[]
                ),
                ReWOOStep(
                    step_number=2, description="Search B", tool_name="search", arguments={"query": "b"}, depends_on=[]
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Done"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )

        await agent.run("Test")

        update_events = [e for e in emitter.events if isinstance(e, PlanStepUpdatedEvent)]
        assert len(update_events) == 2
        assert all(e.new_status == "completed" for e in update_events)

    async def test_planner_emission_carries_artifact_not_formatted_plan(self) -> None:
        """Planner ``agent.step`` has ``thought is None``, ``artifact`` = plan dump.

        The contract drops the hand-formatted ``"#N: desc → tool(args)"``
        string from ``thought`` and carries the full structured plan in
        ``artifact``.
        """
        from nanitics.infrastructure.observability.events import AgentStepEvent

        plan_steps = [
            ReWOOStep(
                step_number=1,
                description="Search",
                tool_name="search",
                arguments={"query": "topic"},
                depends_on=[],
            ),
        ]
        plan_response = _plan_json(plan_steps)
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Done"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )

        await agent.run("Test")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        # Step events: planner (1) + worker steps (1) + solver (1) = 3.
        planner_event = step_events[0]
        assert planner_event.thought is None
        assert planner_event.artifact == ReWOOPlan(steps=plan_steps).model_dump()

    async def test_solver_without_schema_carries_reasoning_text(self) -> None:
        """Solver step without ``output_schema`` emits ``thought = reasoning_text``
        and ``artifact is None``."""
        from nanitics.infrastructure.observability.events import AgentStepEvent

        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1,
                    description="Search",
                    tool_name="search",
                    arguments={"query": "topic"},
                    depends_on=[],
                ),
            ]
        )
        solver_response = LLMResponse(
            content="final synthesis",
            tool_calls=[],
            usage=make_response().usage,
            model="test",
            stop_reason="end_turn",
            reasoning_text="reasoning about the synthesis",
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                solver_response,
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )

        result = await agent.run("Test")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        solver_event = step_events[-1]
        assert solver_event.thought == "reasoning about the synthesis"
        assert solver_event.artifact is None
        assert result.output == "final synthesis"
        # No schema → solver call does not have output_schema.
        assert client.calls[-1]["output_schema"] is None

    async def test_solver_with_schema_emits_artifact(self) -> None:
        """Solver step with ``output_schema`` emits ``artifact = parsed.model_dump()``
        and threads ``output_schema`` to the solver LLM call."""
        from pydantic import BaseModel

        from nanitics.infrastructure.observability.events import AgentStepEvent

        class Final(BaseModel):
            answer: str
            confidence: float

        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1,
                    description="Search",
                    tool_name="search",
                    arguments={"query": "topic"},
                    depends_on=[],
                ),
            ]
        )
        solver_response = LLMResponse(
            content='{"answer": "42", "confidence": 0.9}',
            tool_calls=[],
            usage=make_response().usage,
            model="test",
            stop_reason="end_turn",
            parsed=Final(answer="42", confidence=0.9),
            reasoning_text="reasoning about the final answer",
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                solver_response,
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            output_schema=Final,
        )

        result = await agent.run("Test")

        step_events = [e for e in emitter.events if isinstance(e, AgentStepEvent)]
        solver_event = step_events[-1]
        assert solver_event.thought == "reasoning about the final answer"
        assert solver_event.artifact == {"answer": "42", "confidence": 0.9}
        # output_schema is threaded to the solver LLM call.
        assert client.calls[-1]["output_schema"] is Final
        # AgentResult.parsed flows through.
        assert isinstance(result.parsed, Final)
        assert result.parsed.answer == "42"


# ──────────────────────────────────────────────────────────
# End-to-End
# ──────────────────────────────────────────────────────────


class TestEndToEnd:
    async def test_full_flow(self) -> None:
        """Planner → Worker → Solver with multiple steps and dependencies."""
        plan_response = _plan_json(
            [
                ReWOOStep(
                    step_number=1,
                    description="Search cats",
                    tool_name="search",
                    arguments={"query": "cats"},
                    depends_on=[],
                ),
                ReWOOStep(
                    step_number=2,
                    description="Search dogs",
                    tool_name="search",
                    arguments={"query": "dogs"},
                    depends_on=[],
                ),
                ReWOOStep(
                    step_number=3,
                    description="Compare",
                    tool_name="summarize",
                    arguments={"text": "#1 vs #2"},
                    depends_on=[1, 2],
                ),
            ]
        )
        client = MockLLMClient(
            [
                make_response(content=plan_response),
                make_response(content="Cats and dogs are both great pets"),
            ]
        )
        emitter = make_emitter()
        plan_store = InMemoryPlanStore()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a pet expert.",
            tools=[search_tool, summarize_tool],
            plan_store=plan_store,
        )

        result = await agent.run("Compare cats and dogs")

        assert result.output == "Cats and dogs are both great pets"
        assert result.termination_reason == "complete"
        assert result.total_steps == 5  # 1 planner + 3 worker + 1 solver

        # Verify plan was stored and updated
        plans = await plan_store.list_plans()
        assert len(plans) == 1
        assert all(s.status == "completed" for s in plans[0].steps)

        # Verify solver received observations with substituted variables
        solver_input = client.calls[1]["messages"][0].content
        assert "Results for: cats" in solver_input
        assert "Results for: dogs" in solver_input
        assert "Summary of: Results for: cats vs Results for: dogs" in solver_input

        # Verify usage aggregation (2 LLM calls)
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 10

        # Only 2 LLM calls (planner + solver)
        assert len(client.calls) == 2

    async def test_two_llm_calls_regardless_of_step_count(self) -> None:
        """ReWOO should make exactly 2 LLM calls: planner + solver."""
        many_steps = [
            ReWOOStep(
                step_number=i, description=f"Step {i}", tool_name="search", arguments={"query": f"q{i}"}, depends_on=[]
            )
            for i in range(1, 6)
        ]
        client = MockLLMClient(
            [
                make_response(content=_plan_json(many_steps)),
                make_response(content="Final"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
        )

        await agent.run("Do 5 things")

        # Exactly 2 LLM calls regardless of step count
        assert len(client.calls) == 2


# ──────────────────────────────────────────────────────────
# Truncation + Evaluator Error
# ──────────────────────────────────────────────────────────


class TestReWOOTruncationAndEvaluatorError:
    def _single_step_plan(self) -> list[ReWOOStep]:
        return [
            ReWOOStep(
                step_number=1,
                description="Search",
                tool_name="search",
                arguments={"query": "q"},
                depends_on=[],
            )
        ]

    async def test_truncation_triggers_revision(self) -> None:
        """Truncated solver response with revision budget → retries."""
        plan = self._single_step_plan()
        truncated = LLMResponse(
            content="truncated...",
            tool_calls=[],
            usage=make_response().usage,
            model="test",
            stop_reason="max_tokens",
        )
        client = MockLLMClient(
            [
                make_response(content=_plan_json(plan)),
                truncated,
                make_response(content="complete answer"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            output_evaluator=_AcceptEvaluator(),
        )

        result = await agent.run("Do something")

        assert result.output == "complete answer"
        assert result.termination_reason == "complete"

    async def test_truncation_during_revision_loop(self) -> None:
        """Truncated response inside revision loop → re-emits truncation events."""
        plan = self._single_step_plan()
        truncated = LLMResponse(
            content="truncated...",
            tool_calls=[],
            usage=make_response().usage,
            model="test",
            stop_reason="max_tokens",
        )
        client = MockLLMClient(
            [
                make_response(content=_plan_json(plan)),
                truncated,  # initial truncation
                truncated,  # truncation during revision
                make_response(content="final answer"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            output_evaluator=_AcceptEvaluator(),
        )

        result = await agent.run("Do something")

        assert result.output == "final answer"
        assert result.termination_reason == "complete"

    async def test_evaluator_error_returns_evaluation_skipped(self) -> None:
        """EVALUATOR_ERROR verdict → evaluation_skipped termination."""

        class _EvaluatorErrorEvaluator:
            @property
            def max_revisions(self) -> int:
                return 2

            async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
                return EvaluationResult(
                    verdict=EvaluationVerdict.EVALUATOR_ERROR,
                    feedback="evaluator crashed",
                    evaluator_name="test-evaluator",
                )

        plan = self._single_step_plan()
        client = MockLLMClient(
            [
                make_response(content=_plan_json(plan)),
                make_response(content="solver answer"),
            ]
        )
        emitter = make_emitter()

        agent = ReWOOAgent(
            name="test-rewoo",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[search_tool],
            plan_store=InMemoryPlanStore(),
            output_evaluator=_EvaluatorErrorEvaluator(),
        )

        result = await agent.run("Do something")

        assert result.output == "solver answer"
        assert result.termination_reason == "evaluation_skipped"
