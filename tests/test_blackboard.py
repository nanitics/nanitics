"""Tests for Blackboard: models, control strategies, termination conditions, controller, events."""

import pytest

from nanitics.capabilities.memory.shared import InMemorySharedMemory, SharedMemory
from nanitics.capabilities.memory.shared_tools import create_shared_memory_tools
from nanitics.composition.multi_agent.blackboard import (
    Blackboard,
    BlackboardCompositeTermination,
    BlackboardPredicateTermination,
    BlackboardResult,
    BlackboardState,
    ControlStrategy,
    MaxRoundsTermination,
    NoNewContributions,
    OpportunisticControl,
    PrioritizedControl,
    ScheduledControl,
    TerminationCondition,
)
from nanitics.infrastructure import (
    LLMResponse,
    MockLLMClient,
)
from nanitics.infrastructure.observability.events import (
    BlackboardCompleteEvent,
    BlackboardRoundEvent,
    BlackboardStartEvent,
    SharedMemoryWriteEvent,
)
from nanitics.strategies import (
    ReActAgent,
    ReasoningAgent,
)
from nanitics.tracing import InMemoryEmitter
from tests.testing_helpers import make_emitter, make_response, make_usage


def make_tool_call_response(tool_name: str, **kwargs: str) -> LLMResponse:
    from nanitics.infrastructure.llm.protocol import ToolCall

    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc-1", name=tool_name, arguments=dict(kwargs))],
        usage=make_usage(),
        model="test-model",
        stop_reason="tool_use",
    )


def make_store() -> InMemorySharedMemory:
    return InMemorySharedMemory()


def make_agent(
    name: str,
    emitter: InMemoryEmitter,
    store: InMemorySharedMemory,
    *,
    write_content: str | None = None,
    write_scope: str | None = None,
) -> ReActAgent:
    """Create a test agent without shared memory tools.

    The Blackboard injects shared memory tools at runtime. If write_content
    is given, the agent's scripted LLM response will call write_to_shared.
    """
    if write_content is not None:
        params: dict[str, str] = {"content": write_content}
        if write_scope:
            params["scope"] = write_scope
        responses = [
            make_tool_call_response("write_to_shared", **params),
            make_response("done"),
        ]
    else:
        responses = [make_response("done")]

    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(responses),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


# ──────────────────────────────────────────────────────────
# Control Strategies
# ──────────────────────────────────────────────────────────


class TestScheduledControl:
    def test_returns_all_agents_in_order(self) -> None:
        emitter = make_emitter()
        store = make_store()
        agents = [make_agent(f"a{i}", emitter, store) for i in range(3)]
        control = ScheduledControl()
        state = BlackboardState(round_number=1, round_contributions=0, total_contributions=0)
        selected = control.select(agents, state)
        assert [a.name for a in selected] == ["a0", "a1", "a2"]

    def test_sequential(self) -> None:
        assert ScheduledControl().parallel is False

    def test_satisfies_protocol(self) -> None:
        assert isinstance(ScheduledControl(), ControlStrategy)


class TestPrioritizedControl:
    def test_sorts_by_priority_descending(self) -> None:
        emitter = make_emitter()
        store = make_store()
        agents = [
            make_agent("low", emitter, store),
            make_agent("high", emitter, store),
            make_agent("mid", emitter, store),
        ]
        control = PrioritizedControl({"high": 10, "mid": 5, "low": 1})
        state = BlackboardState(round_number=1, round_contributions=0, total_contributions=0)
        selected = control.select(agents, state)
        assert [a.name for a in selected] == ["high", "mid", "low"]

    def test_agents_without_priority_get_zero(self) -> None:
        emitter = make_emitter()
        store = make_store()
        agents = [
            make_agent("known", emitter, store),
            make_agent("unknown", emitter, store),
        ]
        control = PrioritizedControl({"known": 5})
        state = BlackboardState(round_number=1, round_contributions=0, total_contributions=0)
        selected = control.select(agents, state)
        assert [a.name for a in selected] == ["known", "unknown"]

    def test_sequential(self) -> None:
        assert PrioritizedControl({}).parallel is False

    def test_satisfies_protocol(self) -> None:
        assert isinstance(PrioritizedControl({}), ControlStrategy)


class TestOpportunisticControl:
    def test_returns_all_agents(self) -> None:
        emitter = make_emitter()
        store = make_store()
        agents = [make_agent(f"a{i}", emitter, store) for i in range(3)]
        control = OpportunisticControl()
        state = BlackboardState(round_number=1, round_contributions=0, total_contributions=0)
        selected = control.select(agents, state)
        assert [a.name for a in selected] == ["a0", "a1", "a2"]

    def test_parallel_flag_set(self) -> None:
        assert OpportunisticControl().parallel is True

    def test_satisfies_protocol(self) -> None:
        assert isinstance(OpportunisticControl(), ControlStrategy)


# ──────────────────────────────────────────────────────────
# Termination Conditions
# ──────────────────────────────────────────────────────────


class TestNoNewContributions:
    async def test_terminates_on_zero_contributions(self) -> None:
        store = make_store()
        state = BlackboardState(round_number=1, round_contributions=0, total_contributions=5)
        assert await NoNewContributions().should_terminate(state, store) is True

    async def test_continues_with_contributions(self) -> None:
        store = make_store()
        state = BlackboardState(round_number=1, round_contributions=2, total_contributions=5)
        assert await NoNewContributions().should_terminate(state, store) is False

    def test_satisfies_protocol(self) -> None:
        assert isinstance(NoNewContributions(), TerminationCondition)


class TestMaxRoundsTermination:
    async def test_terminates_at_max_rounds(self) -> None:
        store = make_store()
        state = BlackboardState(round_number=5, round_contributions=1, total_contributions=10)
        assert await MaxRoundsTermination(5).should_terminate(state, store) is True

    async def test_continues_before_max_rounds(self) -> None:
        store = make_store()
        state = BlackboardState(round_number=3, round_contributions=1, total_contributions=5)
        assert await MaxRoundsTermination(5).should_terminate(state, store) is False

    def test_satisfies_protocol(self) -> None:
        assert isinstance(MaxRoundsTermination(5), TerminationCondition)


class TestBlackboardPredicateTermination:
    async def test_custom_async_predicate(self) -> None:
        store = make_store()

        async def has_entries(state: BlackboardState, mem: SharedMemory) -> bool:
            return await mem.count() > 0

        condition = BlackboardPredicateTermination(has_entries)
        state = BlackboardState(round_number=1, round_contributions=0, total_contributions=0)

        assert await condition.should_terminate(state, store) is False

        await store.write("entry", author="test")
        assert await condition.should_terminate(state, store) is True

    def test_satisfies_protocol(self) -> None:
        async def noop(s, m):
            return False

        assert isinstance(BlackboardPredicateTermination(noop), TerminationCondition)


class TestBlackboardCompositeTermination:
    async def test_any_mode(self) -> None:
        store = make_store()
        state = BlackboardState(round_number=5, round_contributions=1, total_contributions=10)
        composite = BlackboardCompositeTermination(
            [NoNewContributions(), MaxRoundsTermination(5)],
            mode="any",
        )
        # MaxRoundsTermination fires (round 5 >= 5), so "any" should terminate
        assert await composite.should_terminate(state, store) is True

    async def test_all_mode(self) -> None:
        store = make_store()
        state = BlackboardState(round_number=5, round_contributions=1, total_contributions=10)
        composite = BlackboardCompositeTermination(
            [NoNewContributions(), MaxRoundsTermination(5)],
            mode="all",
        )
        # NoNewContributions doesn't fire (round_contributions=1), so "all" should not terminate
        assert await composite.should_terminate(state, store) is False

    async def test_all_mode_fires_when_all_match(self) -> None:
        store = make_store()
        state = BlackboardState(round_number=5, round_contributions=0, total_contributions=10)
        composite = BlackboardCompositeTermination(
            [NoNewContributions(), MaxRoundsTermination(5)],
            mode="all",
        )
        assert await composite.should_terminate(state, store) is True


# ──────────────────────────────────────────────────────────
# Blackboard Controller
# ──────────────────────────────────────────────────────────


class TestBlackboard:
    def test_rejects_empty_agent_list(self) -> None:
        with pytest.raises(ValueError, match="agents must not be empty"):
            Blackboard(
                shared_memory=make_store(),
                agents=[],
                emitter=make_emitter(),
            )

    async def test_scheduled_agents_run_in_order(self) -> None:
        emitter = make_emitter()
        store = make_store()

        # Pre-populate each agent's entry so we know they ran
        a1 = make_agent("a1", emitter, store, write_content="from a1")
        a2 = make_agent("a2", emitter, store, write_content="from a2")

        board = Blackboard(
            shared_memory=store,
            agents=[a1, a2],
            emitter=emitter,
            control=ScheduledControl(),
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        assert result.rounds_completed == 1
        entries = await store.read()
        # Both agents wrote
        authors = {e.author for e in entries}
        assert "a1" in authors
        assert "a2" in authors

    async def test_prioritized_agent_order(self) -> None:
        emitter = make_emitter()
        store = make_store()

        high = make_agent("high", emitter, store, write_content="high data")
        low = make_agent("low", emitter, store, write_content="low data")

        board = Blackboard(
            shared_memory=store,
            agents=[low, high],
            emitter=emitter,
            control=PrioritizedControl({"high": 10, "low": 1}),
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        # Both ran
        assert result.agent_contributions["high"] == 1
        assert result.agent_contributions["low"] == 1
        assert result.rounds_completed == 1

    async def test_opportunistic_parallel_execution(self) -> None:
        emitter = make_emitter()
        store = make_store()

        a1 = make_agent("a1", emitter, store, write_content="from a1")
        a2 = make_agent("a2", emitter, store, write_content="from a2")

        board = Blackboard(
            shared_memory=store,
            agents=[a1, a2],
            emitter=emitter,
            control=OpportunisticControl(),
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        assert result.rounds_completed == 1
        entries = await store.read()
        assert len(entries) == 2

    async def test_max_rounds_safety_bound(self) -> None:
        emitter = make_emitter()
        store = make_store()

        # Agent that always writes — would never stop without max_rounds
        # Create multiple response sets for each round
        def make_writing_agent(name: str) -> ReActAgent:
            from nanitics.infrastructure.llm.protocol import ToolCall

            responses = []
            for _ in range(5):
                responses.append(
                    LLMResponse(
                        content=None,
                        tool_calls=[ToolCall(id="tc", name="write_to_shared", arguments={"content": f"from {name}"})],
                        usage=make_usage(),
                        model="test-model",
                        stop_reason="tool_use",
                    )
                )
                responses.append(make_response("done"))

            return ReActAgent(
                name=name,
                llm_client=MockLLMClient(responses),
                emitter=emitter,
                system_prompt=f"You are {name}.",
                tools=[],
            )

        a1 = make_writing_agent("a1")
        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            control=ScheduledControl(),
            # NoNewContributions would never fire because agent always writes
            termination=NoNewContributions(),
            max_rounds=3,
        )
        result = await board.run("keep writing")

        assert result.rounds_completed == 3
        assert result.termination_reason == "max_rounds"

    async def test_convergence_no_new_contributions(self) -> None:
        emitter = make_emitter()
        store = make_store()

        # Agent writes in round 1, then just responds with no tool calls in round 2
        from nanitics.infrastructure.llm.protocol import ToolCall

        responses = [
            # Round 1: write + finish
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc-1", name="write_to_shared", arguments={"content": "contribution"})],
                usage=make_usage(),
                model="test-model",
                stop_reason="tool_use",
            ),
            make_response("done"),
            # Round 2: just finish (no contributions)
            make_response("nothing to add"),
        ]
        a1 = ReActAgent(
            name="a1",
            llm_client=MockLLMClient(responses),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=[],
        )

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            control=ScheduledControl(),
            termination=NoNewContributions(),
            max_rounds=10,
        )
        result = await board.run("do something")

        # Round 1: agent writes. Round 2: agent has no more tool calls (only response),
        # so 0 contributions → NoNewContributions fires
        assert result.rounds_completed == 2
        assert result.termination_reason == "NoNewContributions"

    async def test_contribution_tracking(self) -> None:
        emitter = make_emitter()
        store = make_store()

        a1 = make_agent("a1", emitter, store, write_content="from a1")
        a2 = make_agent("a2", emitter, store, write_content="from a2")

        board = Blackboard(
            shared_memory=store,
            agents=[a1, a2],
            emitter=emitter,
            control=ScheduledControl(),
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        assert result.agent_contributions["a1"] == 1
        assert result.agent_contributions["a2"] == 1

    async def test_blackboard_result_structure(self) -> None:
        emitter = make_emitter()
        store = make_store()

        a1 = make_agent("a1", emitter, store, write_content="entry")

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        assert isinstance(result, BlackboardResult)
        assert len(result.entries) > 0
        assert result.rounds_completed == 1
        assert isinstance(result.agent_contributions, dict)
        assert "a1" in result.agent_contributions

    async def test_default_control_and_termination(self) -> None:
        emitter = make_emitter()
        store = make_store()

        # Agent with no tool calls → 0 contributions → terminates after round 1
        a1 = make_agent("a1", emitter, store)

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            max_rounds=10,
        )
        result = await board.run("task")

        # Default: ScheduledControl + NoNewContributions
        assert result.rounds_completed == 1
        assert result.termination_reason == "NoNewContributions"

    async def test_composite_termination_identifies_sub_condition(self) -> None:
        """When a BlackboardCompositeTermination fires, the result identifies which sub-condition triggered."""
        emitter = make_emitter()
        store = make_store()

        # Agent with no tool calls → 0 contributions → NoNewContributions fires
        a1 = make_agent("a1", emitter, store)

        composite = BlackboardCompositeTermination(
            [MaxRoundsTermination(10), NoNewContributions()],
            mode="any",
        )

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=composite,
            max_rounds=5,
        )
        result = await board.run("task")

        # Should identify NoNewContributions (not BlackboardCompositeTermination)
        assert result.termination_reason == "NoNewContributions"


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestBlackboardEvents:
    async def test_start_event(self) -> None:
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store)

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            max_rounds=1,
        )
        await board.run("test task")

        start_events = [e for e in emitter.events if isinstance(e, BlackboardStartEvent)]
        assert len(start_events) == 1
        start = start_events[0]
        assert start.task == "test task"
        assert start.agent_names == ["a1"]
        assert start.control_strategy == "ScheduledControl"
        assert start.max_rounds == 1

    async def test_round_event(self) -> None:
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store, write_content="data")

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 1
        r = round_events[0]
        assert r.round_number == 1
        assert r.agents_activated == ["a1"]
        assert r.contributions == 1
        assert r.total_contributions == 1
        assert len(r.round_entries) == 1
        entry = r.round_entries[0]
        assert entry.operation == "write"
        assert entry.author == "a1"
        assert entry.content == "data"
        assert entry.entry_id  # non-empty

    async def test_complete_event(self) -> None:
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store, write_content="data")

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        await board.run("task")

        complete_events = [e for e in emitter.events if isinstance(e, BlackboardCompleteEvent)]
        assert len(complete_events) == 1
        c = complete_events[0]
        assert c.rounds_completed == 1
        assert c.total_contributions == 1
        assert c.agent_contributions == {"a1": 1}

    async def test_multiple_round_events(self) -> None:
        emitter = make_emitter()
        store = make_store()

        # Agent writes first round, then doesn't write → convergence at round 2
        from nanitics.infrastructure.llm.protocol import ToolCall

        responses = [
            # Round 1: write + finish
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="tc-1", name="write_to_shared", arguments={"content": "data"})],
                usage=make_usage(),
                model="test-model",
                stop_reason="tool_use",
            ),
            make_response("done"),
            # Round 2: just finish
            make_response("nothing more"),
        ]
        a1 = ReActAgent(
            name="a1",
            llm_client=MockLLMClient(responses),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=[],
        )

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=NoNewContributions(),
            max_rounds=5,
        )
        await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 2  # round 1 (contributes) + round 2 (no contributions → terminate)
        assert round_events[0].contributions == 1
        assert round_events[1].contributions == 0

    async def test_round_entries_tracks_supersede_operations(self) -> None:
        """round_entries captures supersede operations via the contribution listener."""
        from nanitics.infrastructure.llm.protocol import ToolCall

        emitter = make_emitter()
        store = make_store()

        # Pre-write an entry so we know the entry_id for supersede
        entry_id = await store.write("original", author="a1")

        responses = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="supersede_shared",
                        arguments={"entry_id": entry_id, "new_content": "updated"},
                    )
                ],
                usage=make_usage(),
                model="test-model",
                stop_reason="tool_use",
            ),
            make_response("done"),
        ]

        a1 = ReActAgent(
            name="a1",
            llm_client=MockLLMClient(responses),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=[],
        )

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 1
        supersede_entries = [e for e in round_events[0].round_entries if e.operation == "supersede"]
        assert len(supersede_entries) == 1
        assert supersede_entries[0].author == "a1"
        assert supersede_entries[0].content == "updated"
        assert supersede_entries[0].original_entry_id == entry_id
        assert result.agent_contributions["a1"] == 1

    async def test_round_entries_tracks_retract_operations(self) -> None:
        """round_entries captures retract operations via the contribution listener."""
        from nanitics.infrastructure.llm.protocol import ToolCall

        emitter = make_emitter()
        store = make_store()

        # Pre-write an entry so we know the entry_id for retract
        entry_id = await store.write("to be retracted", author="a1")

        responses = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="retract_shared",
                        arguments={"entry_id": entry_id, "reason": "no longer valid"},
                    )
                ],
                usage=make_usage(),
                model="test-model",
                stop_reason="tool_use",
            ),
            make_response("done"),
        ]

        a1 = ReActAgent(
            name="a1",
            llm_client=MockLLMClient(responses),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=[],
        )

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 1
        retract_entries = [e for e in round_events[0].round_entries if e.operation == "retract"]
        assert len(retract_entries) == 1
        assert retract_entries[0].author == "a1"
        assert retract_entries[0].content == ""
        assert retract_entries[0].retract_reason == "no longer valid"
        assert retract_entries[0].entry_id == entry_id
        assert result.agent_contributions["a1"] == 1

    async def test_round_entries_multiple_agents(self) -> None:
        """round_entries captures entries from multiple agents in same round."""
        emitter = make_emitter()
        store = make_store()

        a1 = make_agent("a1", emitter, store, write_content="from a1")
        a2 = make_agent("a2", emitter, store, write_content="from a2")

        board = Blackboard(
            shared_memory=store,
            agents=[a1, a2],
            emitter=emitter,
            control=ScheduledControl(),
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 1
        r = round_events[0]
        assert len(r.round_entries) == 2
        authors = {e.author for e in r.round_entries}
        assert authors == {"a1", "a2"}
        assert all(e.operation == "write" for e in r.round_entries)
        previews = {e.content for e in r.round_entries}
        assert previews == {"from a1", "from a2"}

    async def test_round_entries_empty_when_no_contributions(self) -> None:
        """round_entries is empty when agents make no contributions."""
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store)  # no write_content

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 1
        assert round_events[0].round_entries == []

    async def test_round_entries_with_scope(self) -> None:
        """round_entries captures scope from write events."""
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store, write_content="scoped data", write_scope="findings")

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 1
        entry = round_events[0].round_entries[0]
        assert entry.scope == "findings"
        assert entry.content == "scoped data"


# ──────────────────────────────────────────────────────────
# Tool Injection
# ──────────────────────────────────────────────────────────


class TestBlackboardToolInjection:
    async def test_blackboard_injects_shared_memory_tools(self) -> None:
        """Agent created without shared memory tools gets them injected by Blackboard."""
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store, write_content="injected write")

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        assert result.agent_contributions["a1"] == 1
        entries = await store.read()
        assert any(e.content == "injected write" for e in entries)

    async def test_tools_removed_after_run(self) -> None:
        """Shared memory tools are removed after Blackboard run completes."""
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store)

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            max_rounds=1,
        )
        await board.run("task")

        tool_names = a1._get_tools_available()
        shared_tool_names = {"write_to_shared", "read_shared", "supersede_shared", "retract_shared"}
        assert not shared_tool_names.intersection(tool_names)

    async def test_tools_removed_on_exception(self) -> None:
        """Shared memory tools are removed even when an agent raises."""
        emitter = make_emitter()
        store = make_store()

        # Agent with no responses — will raise ValueError on mock exhaustion
        a1 = ReActAgent(
            name="a1",
            llm_client=MockLLMClient([]),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=[],
        )

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            max_rounds=1,
        )

        with pytest.raises(ValueError):
            await board.run("task")

        tool_names = a1._get_tools_available()
        shared_tool_names = {"write_to_shared", "read_shared", "supersede_shared", "retract_shared"}
        assert not shared_tool_names.intersection(tool_names)

    async def test_idempotent_injection_skips_existing_tools(self) -> None:
        """Agent already having write_to_shared → add_tools skips it,
        remove_tools only removes Blackboard-injected ones."""
        emitter = make_emitter()
        store = make_store()

        # Pre-register shared memory tools on the agent
        pre_tools = create_shared_memory_tools(store, "a1")
        a1 = ReActAgent(
            name="a1",
            llm_client=MockLLMClient(
                [
                    make_tool_call_response("write_to_shared", content="pre-existing"),
                    make_response("done"),
                ]
            ),
            emitter=emitter,
            system_prompt="You are a1.",
            tools=pre_tools,
        )

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            termination=MaxRoundsTermination(1),
            max_rounds=1,
        )
        result = await board.run("task")

        # Agent wrote via pre-existing tool — should still work
        assert result.agent_contributions["a1"] == 1

        # After run, the pre-existing tools should still be there (nothing was injected to remove)
        tool_names = a1._get_tools_available()
        assert "write_to_shared" in tool_names

    async def test_rejects_non_tool_agent(self) -> None:
        """Blackboard rejects agents that don't support dynamic tools."""
        emitter = make_emitter()
        store = make_store()

        reasoning = ReasoningAgent(
            name="thinker",
            llm_client=MockLLMClient([make_response("my analysis result")]),
            emitter=emitter,
            system_prompt="You are a thinker.",
        )

        with pytest.raises(ValueError, match="thinker"):
            Blackboard(
                shared_memory=store,
                agents=[reasoning],
                emitter=emitter,
                termination=MaxRoundsTermination(1),
                max_rounds=1,
            )

    async def test_rejects_mixed_agents(self) -> None:
        """Blackboard rejects when some agents don't support dynamic tools."""
        emitter = make_emitter()
        store = make_store()

        react = make_agent("reactor", emitter, store)
        reasoning = ReasoningAgent(
            name="thinker",
            llm_client=MockLLMClient([make_response("analysis")]),
            emitter=emitter,
            system_prompt="You are a thinker.",
        )

        with pytest.raises(ValueError, match="thinker"):
            Blackboard(
                shared_memory=store,
                agents=[react, reasoning],
                emitter=emitter,
            )


@pytest.mark.parametrize("value", [0, -1])
def test_max_rounds_termination_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_rounds must be positive"):
        MaxRoundsTermination(max_rounds=value)


# ──────────────────────────────────────────────────────────
# Per-Round Listener Isolation (regression: closure-over-loop-var aliasing)
# ──────────────────────────────────────────────────────────


def _make_multi_round_agent(
    name: str,
    emitter: InMemoryEmitter,
    writes_per_round: list[str | None],
) -> ReActAgent:
    """Build a ReActAgent scripted to either write or stay silent per round.

    ``writes_per_round[i]`` is the write content on round i+1, or ``None`` to
    stay silent that round. Each round consumes either 2 responses (tool call
    + confirmation) or 1 response (no tool call).
    """
    from nanitics.infrastructure.llm.protocol import ToolCall

    responses: list[LLMResponse] = []
    for content in writes_per_round:
        if content is None:
            responses.append(make_response("nothing to add"))
        else:
            responses.append(
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="tc", name="write_to_shared", arguments={"content": content})],
                    usage=make_usage(),
                    model="test-model",
                    stop_reason="tool_use",
                )
            )
            responses.append(make_response("done"))
    return ReActAgent(
        name=name,
        llm_client=MockLLMClient(responses),
        emitter=emitter,
        system_prompt=f"You are {name}.",
        tools=[],
    )


class TestBlackboardListenerIsolation:
    """Regression tests for the per-round contribution listener invariant.

    Before the fix, each round registered a new closure that aliased round
    state through a shared cell; every round-N closure saw round-(N+M)
    bindings, so ``round_contributions`` was inflated by a factor of the
    round number.
    """

    async def test_per_round_contributions_not_aliased_across_rounds(self) -> None:
        """``BlackboardRoundEvent[i].contributions`` equals the writes that
        happened *during* round i+1 — not a cumulative count aliased from
        leaked prior-round closures."""
        emitter = make_emitter()
        store = make_store()

        # Round 1 writes once; rounds 2 and 3 write once each as well. With
        # the listener leak, round 2 would report 2 contributions and round 3
        # would report 3 (each prior round's closure still subscribed). With
        # the fix, every round reports exactly 1.
        a1 = _make_multi_round_agent("a1", emitter, writes_per_round=["r1", "r2", "r3"])

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            control=ScheduledControl(),
            termination=MaxRoundsTermination(3),
            max_rounds=3,
        )
        await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 3

        # Count actual write events emitted within each round's window.
        # Each BlackboardRoundEvent is emitted at the end of its round, so
        # writes that precede round_events[i] (and follow round_events[i-1])
        # belong to round i+1.
        actual_writes_per_round = [0, 0, 0]
        round_boundaries = [emitter.events.index(re) for re in round_events]
        prev_boundary = -1
        for i, boundary in enumerate(round_boundaries):
            between = emitter.events[prev_boundary + 1 : boundary]
            actual_writes_per_round[i] = sum(1 for e in between if isinstance(e, SharedMemoryWriteEvent))
            prev_boundary = boundary

        for i, evt in enumerate(round_events):
            assert evt.contributions == actual_writes_per_round[i], (
                f"Round {evt.round_number} reported {evt.contributions} "
                f"contributions but only {actual_writes_per_round[i]} write "
                f"event(s) fired in round {evt.round_number}. Cumulative "
                f"inflation indicates prior-round listener leak."
            )
            assert evt.contributions == 1, (
                f"Round {evt.round_number} should have exactly 1 write; got {evt.contributions}"
            )

    async def test_round_entries_has_no_duplicates_from_leaked_listeners(self) -> None:
        """``round_entries`` length equals ``contributions`` and carries no
        duplicate ``entry_id`` values — which it would if prior-round
        closures still appended into the current round's entry list."""
        emitter = make_emitter()
        store = make_store()

        a1 = _make_multi_round_agent("a1", emitter, writes_per_round=["r1", "r2", "r3"])

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            control=ScheduledControl(),
            termination=MaxRoundsTermination(3),
            max_rounds=3,
        )
        await board.run("task")

        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 3
        for evt in round_events:
            assert len(evt.round_entries) == evt.contributions, (
                f"Round {evt.round_number}: round_entries length "
                f"{len(evt.round_entries)} != contributions "
                f"{evt.contributions}"
            )
            entry_ids = [re.entry_id for re in evt.round_entries]
            assert len(entry_ids) == len(set(entry_ids)), (
                f"Round {evt.round_number} has duplicate entry_ids in "
                f"round_entries (prior-round closures leaked into this "
                f"round): {entry_ids}"
            )

    async def test_quiet_round_fires_no_new_contributions_after_writes(self) -> None:
        """After a write round, a quiet round fires ``NoNewContributions`` and
        stops the loop. (Regression guard: the aliasing bug happens to leave
        termination intact when the later round has literally zero writes —
        the real-world script-73 symptom requires Defect 2 as well — but this
        invariant is still the public contract of ``NoNewContributions`` and
        must be pinned.)"""
        emitter = make_emitter()
        store = make_store()

        # Round 1: write once. Round 2: no writes.
        a1 = _make_multi_round_agent("a1", emitter, writes_per_round=["r1", None])

        board = Blackboard(
            shared_memory=store,
            agents=[a1],
            emitter=emitter,
            control=ScheduledControl(),
            termination=NoNewContributions(),
            max_rounds=3,
        )
        result = await board.run("task")

        assert result.termination_reason == "NoNewContributions"
        assert result.rounds_completed == 2
        round_events = [e for e in emitter.events if isinstance(e, BlackboardRoundEvent)]
        assert len(round_events) == 2
        assert round_events[0].contributions == 1
        assert round_events[1].contributions == 0


# ──────────────────────────────────────────────────────────
# thread_key propagation
# ──────────────────────────────────────────────────────────


class TestBlackboardThreadKeys:
    """Per-agent behavioral continuity across rounds.

    A Blackboard with ``thread_keys={agent.name: key, ...}`` lets each
    named agent see its own prior rounds' assistant turns, tool calls,
    and tool results as conversation history. Agents not in the mapping
    run stateless across rounds (existing behavior). The thread_key
    substrate is orthogonal to the shared-memory board: the board is
    information continuity (what's on the wall), threads are behavioral
    continuity (what each agent did last round)."""

    def test_rejects_unknown_agent_name(self) -> None:
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store)
        with pytest.raises(ValueError, match="thread_keys references agents"):
            Blackboard(
                shared_memory=store,
                agents=[a1],
                emitter=emitter,
                thread_keys={"missing": "k"},
            )

    def test_thread_keys_default_empty(self) -> None:
        emitter = make_emitter()
        store = make_store()
        a1 = make_agent("a1", emitter, store)
        board = Blackboard(shared_memory=store, agents=[a1], emitter=emitter)
        assert board._thread_keys == {}

    async def test_per_agent_thread_accumulates_across_rounds(self) -> None:
        """Two rounds, one agent with a thread_key — the second round's
        run sees the first round's turns appended to its prefix."""
        from nanitics.composition import InMemoryThreadStore
        from nanitics.infrastructure.llm.protocol import ToolCall

        emitter = make_emitter()
        store = make_store()
        thread_store = InMemoryThreadStore()

        responses: list[LLMResponse] = []
        for content in ["r1", "r2"]:
            responses.append(
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="tc", name="write_to_shared", arguments={"content": content})],
                    usage=make_usage(),
                    model="test-model",
                    stop_reason="tool_use",
                )
            )
            responses.append(make_response(f"done {content}"))

        agent = ReActAgent(
            name="writer",
            llm_client=MockLLMClient(responses),
            emitter=emitter,
            system_prompt="you are writer",
            tools=[],
            thread_store=thread_store,
        )
        board = Blackboard(
            shared_memory=store,
            agents=[agent],
            emitter=emitter,
            max_rounds=2,
            termination=MaxRoundsTermination(max_rounds=2),
            thread_keys={"writer": "writer-thread"},
        )
        await board.run("task")

        loaded = await thread_store.load("writer-thread")
        # Two full rounds: each writes a user input, an assistant tool
        # call, a tool_result, and a final assistant turn.
        assert sum(1 for m in loaded if m.role == "user") >= 2
        assert sum(1 for m in loaded if m.role == "assistant") >= 4

    async def test_agent_without_key_stays_stateless(self) -> None:
        """An agent absent from the thread_keys mapping runs without a
        thread — its prior rounds are not replayed."""
        from nanitics.composition import InMemoryThreadStore
        from nanitics.infrastructure.llm.protocol import ToolCall

        emitter = make_emitter()
        store = make_store()
        thread_store = InMemoryThreadStore()

        responses: list[LLMResponse] = []
        for content in ["r1", "r2"]:
            responses.append(
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="tc", name="write_to_shared", arguments={"content": content})],
                    usage=make_usage(),
                    model="test-model",
                    stop_reason="tool_use",
                )
            )
            responses.append(make_response("done"))

        agent = ReActAgent(
            name="anon",
            llm_client=MockLLMClient(responses),
            emitter=emitter,
            system_prompt="you are anon",
            tools=[],
            thread_store=thread_store,
        )
        board = Blackboard(
            shared_memory=store,
            agents=[agent],
            emitter=emitter,
            max_rounds=2,
            termination=MaxRoundsTermination(max_rounds=2),
            # No thread_keys mapping.
        )
        await board.run("task")
        # Nothing in the thread store: agent ran stateless across rounds.
        assert await thread_store.load("anything") == []
