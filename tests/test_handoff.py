import pytest

from nanitics.composition.multi_agent.context_transfer import RawOutputTransfer
from nanitics.composition.multi_agent.handoff import HandoffStep, create_handoff_chain
from nanitics.composition.multi_agent.handoff_protocol import (
    HandoffPayload,
    HandoffTransfer,
)
from nanitics.composition.orchestration.protocol import Step, StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.observability.events import HandoffEvent
from nanitics.strategies import ReActAgent
from nanitics.strategies.agents.base import Agent
from nanitics.tracing import InMemoryEmitter
from tests.testing_helpers import make_emitter, make_response


def make_agent(name: str, responses: list[LLMResponse], emitter: InMemoryEmitter) -> ReActAgent:
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(responses),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


# ── HandoffStep Tests ──────────────────────────────────────


class TestHandoffStep:
    def test_satisfies_step_protocol(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response()], emitter)
        step = HandoffStep(agent=agent, emitter=emitter)
        assert isinstance(step, Step)

    def test_name_defaults_to_agent_name(self) -> None:
        emitter = make_emitter()
        agent = make_agent("my-agent", [make_response()], emitter)
        step = HandoffStep(agent=agent, emitter=emitter)
        assert step.name == "my-agent"

    def test_custom_name(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response()], emitter)
        step = HandoffStep(agent=agent, emitter=emitter, name="custom-step")
        assert step.name == "custom-step"

    async def test_execute_calls_agent_and_returns_step_result(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("agent output")], emitter)
        step = HandoffStep(agent=agent, emitter=emitter)

        result = await step.execute("do something")

        assert isinstance(result, StepResult)
        assert result.output == "agent output"
        assert result.metadata["agent_name"] == "agent-a"
        assert result.metadata["total_steps"] == 1
        assert result.metadata["termination_reason"] == "complete"

    async def test_execute_applies_transfer_strategy(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("raw output")], emitter)

        def builder(result):
            return HandoffPayload(task_state=result.output or "")

        step = HandoffStep(
            agent=agent,
            emitter=emitter,
            transfer_strategy=HandoffTransfer(builder),
        )
        result = await step.execute("task")

        assert "## Handoff Context" in result.output
        assert "raw output" in result.output

    async def test_execute_emits_handoff_event(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("output")], emitter)
        step = HandoffStep(agent=agent, emitter=emitter)

        await step.execute("task")

        handoff_events = [e for e in emitter.events if isinstance(e, HandoffEvent)]
        assert len(handoff_events) == 1
        evt = handoff_events[0]
        assert evt.from_agent == "agent-a"
        assert evt.to_agent == "unknown"
        assert evt.payload_size == len("output")

    async def test_execute_emits_handoff_event_with_to_agent(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [make_response("output")], emitter)
        step = HandoffStep(agent=agent, emitter=emitter, to_agent="agent-b")

        await step.execute("task")

        handoff_events = [e for e in emitter.events if isinstance(e, HandoffEvent)]
        assert len(handoff_events) == 1
        assert handoff_events[0].to_agent == "agent-b"

    async def test_agent_error_propagates(self) -> None:
        emitter = make_emitter()
        agent = make_agent("agent-a", [], emitter)  # No responses → will raise

        step = HandoffStep(agent=agent, emitter=emitter)

        with pytest.raises(ValueError):
            await step.execute("task")


# ── create_handoff_chain Tests ─────────────────────────────


class TestCreateHandoffChain:
    def test_requires_at_least_two_agents(self) -> None:
        emitter = make_emitter()
        agent = make_agent("solo", [make_response()], emitter)
        with pytest.raises(ValueError, match="at least 2 agents"):
            create_handoff_chain(name="chain", agents=[agent], emitter=emitter)

    def test_returns_sequential(self) -> None:
        emitter = make_emitter()
        a = make_agent("a", [make_response()], emitter)
        b = make_agent("b", [make_response()], emitter)
        chain = create_handoff_chain(name="chain", agents=[a, b], emitter=emitter)
        assert isinstance(chain, Sequential)

    def test_correct_number_of_steps(self) -> None:
        emitter = make_emitter()
        agents: list[Agent] = [make_agent(f"agent-{i}", [make_response()], emitter) for i in range(3)]
        chain = create_handoff_chain(name="chain", agents=agents, emitter=emitter)
        assert chain._step_count() == 3

    def test_last_step_uses_raw_output_transfer(self) -> None:
        emitter = make_emitter()
        a = make_agent("a", [make_response()], emitter)
        b = make_agent("b", [make_response()], emitter)

        def builder(result):
            return HandoffPayload(task_state=result.output or "")

        chain = create_handoff_chain(
            name="chain",
            agents=[a, b],
            emitter=emitter,
            transfer_strategy=HandoffTransfer(builder),
        )
        last_step = chain._steps[-1]
        assert isinstance(last_step, HandoffStep)
        assert isinstance(last_step._transfer_strategy, RawOutputTransfer)

    def test_non_last_steps_use_provided_strategy(self) -> None:
        emitter = make_emitter()
        a = make_agent("a", [make_response()], emitter)
        b = make_agent("b", [make_response()], emitter)
        c = make_agent("c", [make_response()], emitter)

        def builder(result):
            return HandoffPayload(task_state=result.output or "")

        strategy = HandoffTransfer(builder)
        chain = create_handoff_chain(
            name="chain",
            agents=[a, b, c],
            emitter=emitter,
            transfer_strategy=strategy,
        )
        for step in chain._steps[:2]:
            assert isinstance(step, HandoffStep)
            assert step._transfer_strategy is strategy
        last = chain._steps[2]
        assert isinstance(last, HandoffStep)
        assert isinstance(last._transfer_strategy, RawOutputTransfer)

    def test_steps_have_correct_to_agent(self) -> None:
        emitter = make_emitter()
        a = make_agent("researcher", [make_response()], emitter)
        b = make_agent("analyst", [make_response()], emitter)
        c = make_agent("writer", [make_response()], emitter)
        chain = create_handoff_chain(name="chain", agents=[a, b, c], emitter=emitter)
        step0 = chain._steps[0]
        step1 = chain._steps[1]
        step2 = chain._steps[2]
        assert isinstance(step0, HandoffStep)
        assert isinstance(step1, HandoffStep)
        assert isinstance(step2, HandoffStep)
        assert step0._to_agent == "analyst"
        assert step1._to_agent == "writer"
        assert step2._to_agent == "output"


# ── Integration Tests ──────────────────────────────────────


class TestHandoffChainIntegration:
    async def test_two_agent_chain(self) -> None:
        """First agent produces output, second agent receives it as input."""
        emitter = make_emitter()
        agent_a = make_agent("researcher", [make_response("research findings")], emitter)
        agent_b = make_agent("writer", [make_response("final article")], emitter)

        chain = create_handoff_chain(
            name="research-write",
            agents=[agent_a, agent_b],
            emitter=emitter,
        )
        result = await chain.execute("write about AI")

        assert result.output == "final article"

        # Verify the writer received the researcher's output as input
        writer_client: MockLLMClient = agent_b._llm_client  # type: ignore[assignment]
        assert len(writer_client.calls) == 1
        writer_messages = writer_client.calls[0]["messages"]
        assert any("research findings" in str(m.content) for m in writer_messages)

        # Verify handoff events were emitted
        handoff_events = [e for e in emitter.events if isinstance(e, HandoffEvent)]
        assert len(handoff_events) == 2  # One per step

    async def test_three_agent_chain_with_handoff_transfer(self) -> None:
        """Structured handoff payloads flow through a 3-agent chain."""
        emitter = make_emitter()
        agent_a = make_agent("researcher", [make_response("key findings here")], emitter)
        agent_b = make_agent("analyst", [make_response("analysis complete")], emitter)
        agent_c = make_agent("writer", [make_response("final report")], emitter)

        def builder(result):
            return HandoffPayload(
                task_state=result.output or "",
                findings=["finding 1"],
            )

        chain = create_handoff_chain(
            name="research-pipeline",
            agents=[agent_a, agent_b, agent_c],
            emitter=emitter,
            transfer_strategy=HandoffTransfer(builder),
        )
        result = await chain.execute("investigate topic")

        # Final output is raw (last step uses RawOutputTransfer)
        assert result.output == "final report"

        # Analyst should have received structured handoff from researcher
        analyst_client: MockLLMClient = agent_b._llm_client  # type: ignore[assignment]
        analyst_input = str(analyst_client.calls[0]["messages"])
        assert "## Handoff Context" in analyst_input
        assert "key findings here" in analyst_input


# ── thread_key propagation ────────────────────────────────


class TestHandoffStepThreadKey:
    """HandoffStep forwards its ``thread_key`` to the wrapped agent on
    every execution. ``create_handoff_chain``'s ``thread_keys`` list
    distributes per-step keys parallel to ``agents`` and lets the same
    agent appearing twice in the chain accumulate prior turns."""

    def test_thread_key_defaults_to_none(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", [make_response()], emitter)
        step = HandoffStep(agent=agent, emitter=emitter)
        assert step._thread_key is None

    def test_thread_key_stored(self) -> None:
        emitter = make_emitter()
        agent = make_agent("a", [make_response()], emitter)
        step = HandoffStep(agent=agent, emitter=emitter, thread_key="t-1")
        assert step._thread_key == "t-1"

    async def test_thread_key_appends_to_store(self) -> None:
        from nanitics.composition import InMemoryThreadStore

        thread_store = InMemoryThreadStore()
        emitter = make_emitter()
        agent = ReActAgent(
            name="drafter",
            llm_client=MockLLMClient([make_response("draft v1")]),
            emitter=emitter,
            system_prompt="draft.",
            tools=[],
            thread_store=thread_store,
        )
        step = HandoffStep(agent=agent, emitter=emitter, thread_key="t-1")
        await step.execute("write something")

        loaded = await thread_store.load("t-1")
        assert any(m.role == "user" and "write something" in str(m.content) for m in loaded)
        assert any(m.role == "assistant" for m in loaded)

    def test_create_handoff_chain_rejects_mismatched_thread_keys_length(self) -> None:
        emitter = make_emitter()
        a = make_agent("a", [make_response()], emitter)
        b = make_agent("b", [make_response()], emitter)
        with pytest.raises(ValueError, match="thread_keys length"):
            create_handoff_chain(
                name="chain",
                agents=[a, b],
                emitter=emitter,
                thread_keys=["only-one"],
            )

    def test_create_handoff_chain_distributes_thread_keys(self) -> None:
        emitter = make_emitter()
        a = make_agent("a", [make_response()], emitter)
        b = make_agent("b", [make_response()], emitter)
        c = make_agent("c", [make_response()], emitter)
        chain = create_handoff_chain(
            name="chain",
            agents=[a, b, c],
            emitter=emitter,
            thread_keys=["k-a", None, "k-c"],
        )
        steps = list(chain._steps)
        assert isinstance(steps[0], HandoffStep)
        assert isinstance(steps[1], HandoffStep)
        assert isinstance(steps[2], HandoffStep)
        assert steps[0]._thread_key == "k-a"
        assert steps[1]._thread_key is None
        assert steps[2]._thread_key == "k-c"

    def test_create_handoff_chain_no_thread_keys_means_all_none(self) -> None:
        emitter = make_emitter()
        a = make_agent("a", [make_response()], emitter)
        b = make_agent("b", [make_response()], emitter)
        chain = create_handoff_chain(name="chain", agents=[a, b], emitter=emitter)
        for step in chain._steps:
            assert isinstance(step, HandoffStep)
            assert step._thread_key is None

    async def test_same_agent_twice_with_shared_key_accumulates(self) -> None:
        """drafter→critic→drafter: the second drafter step sees the first
        drafter's prior turns when both steps share a thread_key."""
        from nanitics.composition import InMemoryThreadStore

        thread_store = InMemoryThreadStore()
        emitter = make_emitter()

        # One drafter agent appearing twice in the chain, sharing the
        # same thread_key. The critic uses its own thread (or none).
        drafter = ReActAgent(
            name="drafter",
            llm_client=MockLLMClient([make_response("draft v1"), make_response("draft v2")]),
            emitter=emitter,
            system_prompt="draft.",
            tools=[],
            thread_store=thread_store,
        )
        critic = make_agent("critic", [make_response("critique")], emitter)

        chain = create_handoff_chain(
            name="iterate",
            agents=[drafter, critic, drafter],
            emitter=emitter,
            thread_keys=["drafter-thread", None, "drafter-thread"],
        )
        result = await chain.execute("write a poem")
        # The chain runs all three steps. Final step is the second drafter.
        assert result.output == "draft v2"

        # The drafter's thread carries messages from both drafter runs.
        loaded = await thread_store.load("drafter-thread")
        assistant_turns = [m for m in loaded if m.role == "assistant"]
        assert len(assistant_turns) == 2
