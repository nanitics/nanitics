"""Tests for the resumable agent loop.

Covers SuspendExecution propagation through the agent loop, checkpoint state
building, and agent resume via the :class:`ResumeService` abstraction.

The resume path is driven exclusively through
:class:`nanitics.composition.durability.resume.DurableRun` and
:class:`~nanitics.composition.durability.resume.ResumeService`; tests never
reach into agent-internal resume-wiring — that channel is the orchestrator's
private contract with :class:`~nanitics.core.agents.base.Agent`.
"""

from __future__ import annotations

from typing import Any

import pytest

from nanitics import (
    MockLLMClient,
    ReActAgent,
    ToolCall,
    tool,
)
from nanitics.capabilities.errors.handler import ErrorHandler
from nanitics.capabilities.memory.working_memory import InMemoryWorkingMemory
from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.hitl_store import InMemoryHitlRequestStore
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputResponse,
)
from nanitics.composition.durability.resume import (
    DurableRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
    SuspendedRun,
)
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.adapters import AgentStep
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.events import (
    ExecutionResumedEvent,
    ExecutionSuspendedEvent,
)
from tests.testing_helpers import make_emitter, make_response

# Constant run_id used across the suspend/resume tests in this module. The
# deterministic HITL identity scheme is f"{run_id}:{tool_call_id}" — tests
# route through the agent with this run_id so request ids are predictable.
_TEST_RUN_ID = "test-run"


# ── Helpers ────────────────────────────────────────────────


@tool(name="add", description="Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


@tool(name="greet", description="Greet a person")
async def greet_tool(name: str) -> str:
    return f"Hello, {name}!"


def make_suspending_tool(
    provider: DurableHumanInputProvider,
) -> Any:
    """Create a tool that triggers suspension via DurableHumanInputProvider."""
    return ApprovalWrappedTool(tool=add_tool, provider=provider)


# ── Agent Suspension Tests ─────────────────────────────────


class TestAgentSuspension:
    async def test_suspend_execution_raised_with_checkpoint_data(self) -> None:
        """When a tool triggers suspension, SuspendExecution has checkpoint_data attached."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        emitter = make_emitter()
        # LLM calls the approval-wrapped tool
        client = MockLLMClient(
            [
                make_response(
                    content="I'll add those numbers",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )

        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id=_TEST_RUN_ID,
        )

        with pytest.raises(SuspendExecution) as exc_info:
            await agent.run("add 1 and 2")

        exc = exc_info.value
        assert exc.checkpoint_data is not None
        assert exc.checkpoint_data["agent_type"] == "react"
        assert exc.checkpoint_data["step_number"] == 1
        assert exc.checkpoint_data["suspended_tool_index"] == 0
        assert len(exc.checkpoint_data["messages"]) >= 2  # user + assistant
        assert exc.checkpoint_data["tool_calls"] is not None

    async def test_checkpoint_contains_correct_state(self) -> None:
        """Checkpoint state includes messages, step_number, revision_count, working memory, usages."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        emitter = make_emitter()
        working_memory = InMemoryWorkingMemory()

        # Step 1: LLM calls greet tool
        # Step 2: LLM calls the approval-wrapped tool → suspends
        client = MockLLMClient(
            [
                make_response(
                    content="Let me greet first",
                    tool_calls=[ToolCall(id="tc1", name="greet", arguments={"name": "Alice"})],
                ),
                make_response(
                    content="Now add",
                    tool_calls=[ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )

        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent",
            tools=[greet_tool, wrapped_tool],
            working_memory=working_memory,
            run_id=_TEST_RUN_ID,
        )

        with pytest.raises(SuspendExecution) as exc_info:
            await agent.run("greet Alice then add 1 and 2")

        state = exc_info.value.checkpoint_data
        assert state is not None
        assert state["step_number"] == 2
        assert state["revision_count"] == 0
        assert state["limiter_count"] == 2
        assert len(state["usages"]) == 2
        assert len(state["messages"]) >= 4  # user, asst+tool_call, tool_result, asst+tool_call

    async def test_checkpoint_mid_batch_preserves_completed_results(self) -> None:
        """When tool k suspends in a batch of n tools, tools 0..k-1 results are in checkpoint."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        emitter = make_emitter()
        # LLM calls two tools in parallel: greet succeeds, then add suspends
        client = MockLLMClient(
            [
                make_response(
                    content="I'll do both",
                    tool_calls=[
                        ToolCall(id="tc1", name="greet", arguments={"name": "Bob"}),
                        ToolCall(id="tc2", name="add", arguments={"a": 5, "b": 3}),
                    ],
                ),
            ]
        )

        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent",
            tools=[greet_tool, wrapped_tool],
            run_id=_TEST_RUN_ID,
        )

        with pytest.raises(SuspendExecution) as exc_info:
            await agent.run("do both")

        state = exc_info.value.checkpoint_data
        assert state is not None
        assert state["suspended_tool_index"] == 1
        assert "0" in state["completed_tool_results"]
        assert state["completed_tool_results"]["0"]["content"] == "Hello, Bob!"

    async def test_execution_suspended_event_emitted(self) -> None:
        """Agent.run() emits ExecutionSuspendedEvent when suspension occurs."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        emitter = make_emitter()
        client = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )

        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id=_TEST_RUN_ID,
        )

        with pytest.raises(SuspendExecution):
            await agent.run("add 1 and 2")

        suspended_events = [e for e in emitter.events if isinstance(e, ExecutionSuspendedEvent)]
        assert len(suspended_events) == 1
        assert suspended_events[0].agent_name == "test-agent"
        assert suspended_events[0].suspension_type == "hitl"

    async def test_suspend_execution_is_base_exception(self) -> None:
        """SuspendExecution inherits BaseException and isn't caught by except Exception."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        emitter = make_emitter()
        client = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )

        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id=_TEST_RUN_ID,
        )

        # It should propagate as SuspendExecution, not get caught by error handler
        with pytest.raises(SuspendExecution):
            await agent.run("add 1 and 2")


# ── Agent Resume Tests ─────────────────────────────────────
#
# Each resume-side test follows the canonical shape:
#
#   1. Build the first-run agent, wrap it in ``AgentStep + Sequential``,
#      wrap that in ``DurableRun``, and call ``durable.start(input)``.
#   2. Assert the returned ``SuspendedRun`` and its ``pending_request``.
#   3. Define a factory that reconstructs the same agent + workflow using
#      the second-run LLM responses; capture the resumed agent / handler
#      into a local list when the test needs to observe agent-internal
#      invariants.
#   4. Call ``ResumeService.resume(run_id, response)`` and assert the
#      returned ``ResumeResult`` plus any captured state.
#
# The factory-captured reference is the idiomatic way to observe
# agent-internal state post-resume under the new abstraction.


class TestAgentResume:
    async def test_resume_restores_state_and_continues(self) -> None:
        """Agent resumes from checkpoint, skips completed tools, re-executes suspended tool, continues."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        # --- First run: suspends ---
        client1 = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )

        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("add 1 and 2")
        assert isinstance(suspended, SuspendedRun)

        # --- Resume: fresh agent + emitter + client in the factory ---
        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            # After the resumed tool batch completes, the LLM is called again
            # and returns the final answer.
            client2 = MockLLMClient([make_response(content="The sum is 3")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[wrapped_tool2],
                run_id=ctx.run_id,
            )
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)

        assert isinstance(result, ResumeResult)
        assert result.output == "The sum is 3"

    async def test_resume_emits_execution_resumed_event(self) -> None:
        """ExecutionResumedEvent is emitted when agent resumes from checkpoint."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        client1 = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("add 1 and 2")
        assert isinstance(suspended, SuspendedRun)

        # --- Resume with a shared emitter so we can inspect the events ---
        emitter2 = make_emitter()

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            client2 = MockLLMClient([make_response(content="Done")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=emitter2,
                system_prompt="You are a test agent",
                tools=[wrapped_tool2],
                run_id=ctx.run_id,
            )
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=emitter2,
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        await service.resume(suspended.run_id, response)

        resumed_events = [e for e in emitter2.events if isinstance(e, ExecutionResumedEvent)]
        # The agent emits one ExecutionResumedEvent; the workflow emits
        # another at its resume boundary. Only assert on the agent's.
        agent_resumed = [e for e in resumed_events if e.resumed_from_step == "step-1"]
        assert len(agent_resumed) == 1

    async def test_resume_with_working_memory(self) -> None:
        """Working memory is restored on resume."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        wm = InMemoryWorkingMemory()

        # Step 1: LLM response includes working memory update + greet tool call
        # Step 2: LLM calls add (suspended)
        client1 = MockLLMClient(
            [
                make_response(
                    content="Let me greet first\n<working_memory>\n## Progress\nGreeted Alice\n</working_memory>",
                    tool_calls=[ToolCall(id="tc1", name="greet", arguments={"name": "Alice"})],
                ),
                make_response(
                    content="Now add",
                    tool_calls=[ToolCall(id="tc2", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[greet_tool, wrapped_tool],
            working_memory=wm,
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("greet and add")
        assert isinstance(suspended, SuspendedRun)

        # Working-memory capture via a fresh instance — the resumed agent
        # restores the suspended working memory before its loop runs.
        wm2 = InMemoryWorkingMemory()

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            client2 = MockLLMClient([make_response(content="All done")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[greet_tool, wrapped_tool2],
                working_memory=wm2,
                run_id=ctx.run_id,
            )
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)
        assert isinstance(result, ResumeResult)
        assert result.output == "All done"
        # Working memory should have been restored into the fresh instance.
        assert wm2.read() is not None
        assert "Greeted Alice" in (wm2.read() or "")

    async def test_resume_mid_batch(self) -> None:
        """Resume mid-batch: tools 0..k-1 results injected, tool k re-executes, tools k+1..n run."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        # LLM calls 3 tools: greet succeeds, add suspends, third never runs
        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return f"echo: {text}"

        client1 = MockLLMClient(
            [
                make_response(
                    content="Doing all three",
                    tool_calls=[
                        ToolCall(id="tc1", name="greet", arguments={"name": "Bob"}),
                        ToolCall(id="tc2", name="add", arguments={"a": 2, "b": 3}),
                        ToolCall(id="tc3", name="echo", arguments={"text": "hello"}),
                    ],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[greet_tool, wrapped_tool, echo_tool],
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("do all")
        assert isinstance(suspended, SuspendedRun)

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            client2 = MockLLMClient([make_response(content="All done")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[greet_tool, wrapped_tool2, echo_tool],
                run_id=ctx.run_id,
            )
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)
        assert isinstance(result, ResumeResult)
        assert result.output == "All done"

    async def test_resume_clears_resume_state(self) -> None:
        """After resume, the agent's ``_resume_state`` is cleared."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        client1 = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("add 1 and 2")
        assert isinstance(suspended, SuspendedRun)

        # Capture the reconstructed agent so we can assert on its internal
        # ``_resume_state`` after the resume drives it to completion.
        resumed_agents: list[ReActAgent] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            client2 = MockLLMClient([make_response(content="Done")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[wrapped_tool2],
                run_id=ctx.run_id,
            )
            resumed_agents.append(agent2)
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        await service.resume(suspended.run_id, response)

        assert len(resumed_agents) == 1
        # ``_resume_state`` is consumed on the first agent-loop tick and
        # cleared by ``Agent._run`` — after the full resume it is back
        # to ``None``.
        assert resumed_agents[0]._resume_state is None


# ── Agent + Orchestrator Integration ───────────────────────


class TestAgentOrchestratorIntegration:
    async def test_agent_suspends_in_orchestrated_workflow(self) -> None:
        """Agent suspends → orchestrator checkpoints both orchestration + agent state."""
        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        emitter = make_emitter()
        checkpoint_store = InMemoryCheckpointStore()

        client = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )

        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id="run-1",
        )

        step = AgentStep(agent=agent)

        seq = Sequential(
            name="workflow",
            steps=[step],
            emitter=emitter,
            checkpoint_store=checkpoint_store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("add 1 and 2")

        # Verify checkpoint has agent_checkpoint
        cp = await checkpoint_store.load("run-1")
        assert cp is not None
        assert cp.state["agent_checkpoint"] is not None
        assert cp.state["agent_checkpoint"]["agent_type"] == "react"
        assert cp.state["agent_checkpoint"]["step_number"] == 1

    async def test_orchestrated_agent_resume(self) -> None:
        """Full suspend → resume cycle through ``ResumeService`` with agent checkpoint restoration."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        client1 = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            run_id="run-1",
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id="run-1",
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("add 1 and 2")
        assert isinstance(suspended, SuspendedRun)

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            # On resume: the agent re-executes the suspended tool call
            # (succeeds), then the LLM is called and returns a final answer.
            client2 = MockLLMClient([make_response(content="The result is 3")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[wrapped_tool2],
                run_id=ctx.run_id,
            )
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)

        assert isinstance(result, ResumeResult)
        assert result.output is not None


# ── IterationLimiter Restore ───────────────────────────────


class TestIterationLimiterRestore:
    def test_restore_sets_count(self) -> None:
        from nanitics.safety.iteration_limits import IterationLimiter

        limiter = IterationLimiter(max_iterations=10)
        limiter.restore(5)
        assert limiter.current_iteration == 5
        assert limiter.remaining == 5

    def test_restore_then_step(self) -> None:
        from nanitics.safety.iteration_limits import IterationLimiter

        limiter = IterationLimiter(max_iterations=3)
        limiter.restore(2)
        limiter.step()  # count becomes 3 — at limit
        assert limiter.current_iteration == 3

        from nanitics.infrastructure.errors import AgentIterationLimitError

        with pytest.raises(AgentIterationLimitError):
            limiter.step()


# ── ErrorHandler Restore ───────────────────────────────────


class TestErrorHandlerRestore:
    def test_restore_total_corrections(self) -> None:
        handler = ErrorHandler(max_corrections=3, max_total_corrections=5)
        handler.restore(3)
        assert handler.total_corrections == 3

    def test_expose_total_corrections(self) -> None:
        handler = ErrorHandler()
        assert handler.total_corrections == 0

    async def test_restore_is_scoped_to_resuming_task(self) -> None:
        """``restore(...)`` writes to the resuming task's ContextVar slot
        only — a parallel idle task on the same shared handler must still
        observe its own (zero) count. Guards the per-task isolation that
        ``Agent._emitter_var`` already provides for emitters.
        """
        import asyncio

        handler = ErrorHandler(max_corrections=3, max_total_corrections=5)
        gate_restored = asyncio.Event()
        gate_idle_checked = asyncio.Event()

        async def resuming_task() -> int:
            handler.restore(4)
            gate_restored.set()
            await gate_idle_checked.wait()
            return handler.total_corrections

        async def idle_task() -> int:
            await gate_restored.wait()
            # This task never called ``restore`` itself — its slot must
            # still read the default ``0`` even though the resuming task
            # set ``4`` on the same handler instance.
            observed = handler.total_corrections
            gate_idle_checked.set()
            return observed

        resumed_count, idle_count = await asyncio.gather(resuming_task(), idle_task())
        assert resumed_count == 4
        assert idle_count == 0


# ── Error Handler + Checkpoint Integration ─────────────────


class TestErrorHandlerCheckpointIntegration:
    async def test_correction_budget_preserved_across_suspend_resume(self) -> None:
        """Suspend after a tool correction → resume → correction budget reflects prior corrections."""
        from nanitics.infrastructure.errors import ToolParameterError

        # A tool that fails once with a correctable error, then succeeds
        call_count = 0

        @tool(name="flaky", description="Fails first, then succeeds")
        async def flaky_tool(x: int) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ToolParameterError("bad param", tool_name="flaky", parameter_name="x", reason="must be positive")
            return f"result: {x}"

        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_add = ApprovalWrappedTool(tool=add_tool, provider=provider)

        handler = ErrorHandler(max_corrections=3, max_total_corrections=5)

        # Step 1: LLM calls flaky_tool (fails → correction prompt injected)
        # Step 2: LLM retries flaky_tool (succeeds) + calls wrapped add (suspends)
        client = MockLLMClient(
            [
                make_response(
                    content="Calling flaky",
                    tool_calls=[ToolCall(id="tc1", name="flaky", arguments={"x": -1})],
                ),
                make_response(
                    content="Retrying flaky and adding",
                    tool_calls=[
                        ToolCall(id="tc2", name="flaky", arguments={"x": 5}),
                        ToolCall(id="tc3", name="add", arguments={"a": 1, "b": 2}),
                    ],
                ),
            ]
        )

        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[flaky_tool, wrapped_add],
            error_handler=handler,
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("use flaky then add")
        assert isinstance(suspended, SuspendedRun)

        # Error handler state should record the 1 correction that happened
        cp = await checkpoint_store.load(_TEST_RUN_ID)
        assert cp is not None
        assert cp.state["agent_checkpoint"]["error_handler_state"]["total_corrections"] == 1

        # --- Resume --- capture the fresh handler to assert on its state.
        resumed_handlers: list[ErrorHandler] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_add2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            handler2 = ErrorHandler(max_corrections=3, max_total_corrections=5)
            resumed_handlers.append(handler2)
            client2 = MockLLMClient([make_response(content="Done: 3")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[flaky_tool, wrapped_add2],
                error_handler=handler2,
                run_id=ctx.run_id,
            )
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)
        assert isinstance(result, ResumeResult)
        assert result.output == "Done: 3"
        # Handler should have restored the correction count from checkpoint
        assert len(resumed_handlers) == 1
        assert resumed_handlers[0].total_corrections == 1

    async def test_resume_restores_tool_call_limiter(self) -> None:
        """Tool call limiter count is restored when resuming from checkpoint."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped_tool = make_suspending_tool(provider)

        client1 = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ]
        )
        agent1 = ReActAgent(
            name="test-agent",
            llm_client=client1,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[wrapped_tool],
            max_tool_calls=10,
            run_id=_TEST_RUN_ID,
        )
        workflow1 = Sequential(
            name="workflow",
            steps=[AgentStep(agent1)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_TEST_RUN_ID,
        )
        durable = DurableRun(
            workflow1,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("add 1 and 2")
        assert isinstance(suspended, SuspendedRun)

        # The suspended checkpoint carries the limiter count into the
        # persisted agent_checkpoint. We verify both (a) the checkpoint
        # persists it and (b) the resumed agent's limiter reflects it
        # after the run completes.
        cp = await checkpoint_store.load(_TEST_RUN_ID)
        assert cp is not None
        assert "tool_call_limiter_count" in cp.state["agent_checkpoint"]

        resumed_agents: list[ReActAgent] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            provider2 = DurableHumanInputProvider(request_store=ctx.hitl_store)
            wrapped_tool2 = ApprovalWrappedTool(tool=add_tool, provider=provider2)
            client2 = MockLLMClient([make_response(content="Sum is 3")])
            agent2 = ReActAgent(
                name="test-agent",
                llm_client=client2,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[wrapped_tool2],
                max_tool_calls=10,
                run_id=ctx.run_id,
            )
            resumed_agents.append(agent2)
            workflow2 = Sequential(
                name="workflow",
                steps=[AgentStep(agent2)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)
        assert isinstance(result, ResumeResult)
        assert result.output == "Sum is 3"
        assert len(resumed_agents) == 1
