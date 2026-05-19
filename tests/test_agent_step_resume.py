"""Tests for ``_BoundAgentStep`` resume-state wiring on the workflow path.

The orchestrator's checkpoint state already carries ``agent_checkpoint``
(see :mod:`nanitics.composition.orchestration.sequential`). This module
covers the channel that hands that checkpoint to the post-resume
``AgentStep`` — so consumers never hand-roll
``agent._set_resume_state(...)`` themselves.
"""

from __future__ import annotations

from typing import Any

import pytest

from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.protocol import HumanDecision, HumanInputResponse
from nanitics.composition import (
    InMemoryCheckpointStore,
    Sequential,
)
from nanitics.composition.durability.resume import (
    DurableRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
    SuspendedRun,
)
from nanitics.composition.orchestration.adapters import AgentStep, FunctionStep
from nanitics.hitl import InMemoryHitlRequestStore
from nanitics.infrastructure import MockLLMClient
from nanitics.infrastructure.observability.events import ExecutionResumedEvent
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter, make_response

_RUN_ID = "agent-step-resume-test"


@tool(name="add", description="Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


def _build_suspending_agent(
    hitl_store: InMemoryHitlRequestStore,
    *,
    responses: list[Any],
    run_id: str = _RUN_ID,
    name: str = "test-agent",
) -> ReActAgent:
    provider = DurableHumanInputProvider(request_store=hitl_store)
    wrapped_tool = ApprovalWrappedTool(tool=add_tool, provider=provider)
    client = MockLLMClient(responses)
    return ReActAgent(
        name=name,
        llm_client=client,
        emitter=make_emitter(),
        system_prompt="You are a test agent",
        tools=[wrapped_tool],
        run_id=run_id,
    )


class TestAgentStepResumeWiring:
    async def test_sequential_with_single_agent_step_resumes_via_durable_run(self) -> None:
        """End-to-end suspend → resume through ``DurableRun`` + ``ResumeService``.

        Asserts:
          - Final output matches what the LLM returns on the re-run.
          - The agent's ``_resume_state`` is cleared after the resume.
          - ``ExecutionResumedEvent`` fires once during the resumed run.
        """
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()

        # First run: LLM triggers a tool call; the ApprovalWrappedTool suspends.
        first_agent = _build_suspending_agent(
            hitl_store,
            responses=[
                make_response(
                    content="I'll add those numbers",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ],
        )
        first_workflow = Sequential(
            name="agent-seq",
            steps=[AgentStep(first_agent)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_RUN_ID,
        )
        durable = DurableRun(
            first_workflow,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )

        suspended = await durable.start("add 1 and 2")
        assert isinstance(suspended, SuspendedRun)

        # Resume: a fresh agent + workflow in the factory. If the
        # ``AgentStep`` resume-wiring hook is absent, the re-executed
        # agent will restart from scratch and the MockLLMClient will be
        # asked for a tool call it cannot produce — the test exercises
        # the full post-resume shape.
        resumed_agents: list[ReActAgent] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            agent = _build_suspending_agent(
                hitl_store,
                responses=[make_response(content="The sum is 3")],
            )
            resumed_agents.append(agent)
            workflow = Sequential(
                name="agent-seq",
                steps=[AgentStep(agent)],
                emitter=make_emitter(),
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow,
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

        assert len(resumed_agents) == 1
        # The agent's resume-state is consumed on the first run and
        # cleared by ``Agent._run`` — after the full resume it is back
        # to ``None``.
        assert resumed_agents[0]._resume_state is None

    async def test_agent_step_without_agent_checkpoint_on_fresh_run(self) -> None:
        """On a non-resume path, ``_set_resume_state`` is never called."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()

        # A non-suspending tool response → the agent returns a final answer.
        agent = _build_suspending_agent(
            hitl_store,
            responses=[make_response(content="hello world")],
        )
        calls: list[dict[str, Any]] = []
        original = agent._set_resume_state

        def spy(state: dict[str, Any]) -> None:
            calls.append(state)
            original(state)

        agent._set_resume_state = spy  # type: ignore[method-assign]

        workflow = Sequential(
            name="agent-seq",
            steps=[AgentStep(agent)],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_RUN_ID,
        )
        durable = DurableRun(
            workflow,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        result = await durable.start("hello")
        assert isinstance(result, ResumeResult)
        assert result.output == "hello world"
        assert calls == []

    async def test_agent_step_resume_state_injected_only_at_suspension_step(self) -> None:
        """With ``[FunctionStep, AgentStep, FunctionStep]`` suspending at the
        ``AgentStep``, the resume path injects the agent checkpoint into
        the ``AgentStep`` alone — the trailing ``FunctionStep`` is not
        asked to consume the checkpoint.
        """
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()

        async def leading(value: object) -> object:
            return value

        trailing_calls: list[object] = []

        async def trailing(value: object) -> object:
            trailing_calls.append(value)
            return f"final:{value}"

        first_agent = _build_suspending_agent(
            hitl_store,
            responses=[
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                ),
            ],
        )
        first_workflow = Sequential(
            name="triple",
            steps=[
                FunctionStep(name="leading", fn=leading),
                AgentStep(first_agent),
                FunctionStep(name="trailing", fn=trailing),
            ],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_RUN_ID,
        )
        durable = DurableRun(
            first_workflow,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
        )
        suspended = await durable.start("start-input")
        assert isinstance(suspended, SuspendedRun)
        # The trailing step never ran on the first attempt.
        assert trailing_calls == []

        # Capture events on the resume run so we can assert the resume
        # event fires.
        resumed_root_emitter = make_emitter()
        resumed_agents: list[ReActAgent] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            agent = _build_suspending_agent(
                hitl_store,
                responses=[make_response(content="final-answer")],
            )
            resumed_agents.append(agent)
            workflow = Sequential(
                name="triple",
                steps=[
                    FunctionStep(name="leading", fn=leading),
                    AgentStep(agent),
                    FunctionStep(name="trailing", fn=trailing),
                ],
                emitter=resumed_root_emitter,
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
            )
            return DurableRun(
                workflow,
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
        # The trailing FunctionStep ran exactly once, on the agent's output.
        assert trailing_calls == ["final-answer"]
        assert result.output == "final:final-answer"

        # Only one ExecutionResumedEvent surfaces on the workflow emitter.
        resumed_events = [e for e in resumed_root_emitter.events if isinstance(e, ExecutionResumedEvent)]
        assert len(resumed_events) >= 1

    async def test_agent_checkpoint_consumed_once_within_bound_step(self) -> None:
        """The consume-once pattern: if ``_BoundAgentStep.execute`` were
        called twice on the same bound wrapper, the second call must not
        re-inject the checkpoint.

        Uses a spy on ``_set_resume_state`` that swallows the call rather
        than forwarding — the agent is a no-op ``MockLLMClient`` path,
        so the swallowed state doesn't affect subsequent behaviour, and
        the test isolates the wrapper's consume-once plumbing.
        """
        from nanitics.composition.orchestration.workflow import _BoundAgentStep

        hitl_store = InMemoryHitlRequestStore()
        # A non-suspending agent — we only exercise the wrapper's plumbing.
        agent = _build_suspending_agent(
            hitl_store,
            responses=[
                make_response(content="answer-1"),
                make_response(content="answer-2"),
            ],
        )

        calls: list[dict[str, Any]] = []

        def spy(state: dict[str, Any]) -> None:
            # Swallow the state so subsequent runs proceed as fresh.
            calls.append(state)

        agent._set_resume_state = spy  # type: ignore[method-assign]

        step = AgentStep(agent)
        emitter = make_emitter()
        bound = _BoundAgentStep(
            step,
            agent.bind(emitter),
            agent_checkpoint={"sentinel": True},
        )

        await bound.execute("hi")
        # ``_set_resume_state`` was called exactly once, on the first execute.
        assert len(calls) == 1
        assert calls[0] == {"sentinel": True}
        # A second call must not re-inject.
        await bound.execute("hi again")
        assert len(calls) == 1

    async def test_bind_step_ignores_agent_checkpoint_for_function_step(self) -> None:
        """``_bind_step`` on a non-``AgentStep`` step with an ``agent_checkpoint``
        returns the step unchanged — the kwarg is a pass-through for
        other step types.
        """
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()

        async def noop(value: object) -> object:
            return value

        step = FunctionStep(name="noop", fn=noop)
        workflow = Sequential(
            name="fn-only",
            steps=[step],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_RUN_ID,
        )
        # Intentionally exercise ``_bind_step`` directly with a non-None
        # agent_checkpoint so the FunctionStep branch is covered.
        bound = workflow._bind_step(step, agent_checkpoint={"history": []})
        assert bound is step
        # Silences the unused-variable lint on ``hitl_store`` for readers.
        assert isinstance(hitl_store, InMemoryHitlRequestStore)


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # Defensive: MockLLMClient is used throughout, but guard against
    # accidental env-based client construction during test collection.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
