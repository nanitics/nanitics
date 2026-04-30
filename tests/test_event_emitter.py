import asyncio

from nanitics import (
    EventEmitter,
    InMemoryEmitter,
    LLMResponse,
    MockLLMClient,
    ReasoningAgent,
    Usage,
)
from nanitics.infrastructure import (
    AgentStartEvent,
    BaseEvent,
    SpanEndEvent,
    SpanStartEvent,
)


class TestInMemoryEmitterConstruction:
    def test_trace_id_set(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        assert emitter.trace_id == "trace-1"

    def test_root_span_exists(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        assert emitter.span_id is not None
        assert emitter.parent_span_id is None

    def test_events_starts_empty(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        assert emitter.events == []


class TestEmit:
    def test_appends_event(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        event = AgentStartEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            agent_name="test",
            task_input="hello",
            tools_available=[],
        )
        emitter.emit(event)
        assert len(emitter.events) == 1
        assert emitter.events[0] is event

    def test_preserves_order(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        for i in range(3):
            emitter.emit(
                AgentStartEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    agent_name=f"agent-{i}",
                    task_input="hello",
                    tools_available=[],
                )
            )
        names = [e.agent_name for e in emitter.events]  # type: ignore[union-attr]
        assert names == ["agent-0", "agent-1", "agent-2"]


class TestSpan:
    def test_span_changes_ids(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        root_span = emitter.span_id

        with emitter.span("child"):
            child_span = emitter.span_id
            assert child_span != root_span
            assert emitter.parent_span_id == root_span

        assert emitter.span_id == root_span
        assert emitter.parent_span_id is None

    def test_span_emits_start_and_end_events(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")

        with emitter.span("my_span"):
            pass

        assert len(emitter.events) == 2
        start = emitter.events[0]
        end = emitter.events[1]

        assert isinstance(start, SpanStartEvent)
        assert start.name == "my_span"
        assert isinstance(end, SpanEndEvent)
        assert end.name == "my_span"
        assert end.duration_ms >= 0

    def test_span_events_carry_correct_ids(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        root_span = emitter.span_id

        with emitter.span("child"):
            child_span = emitter.span_id

        start = emitter.events[0]
        end = emitter.events[1]

        assert start.trace_id == "trace-1"
        assert start.span_id == child_span
        assert start.parent_span_id == root_span
        assert end.span_id == child_span
        assert end.parent_span_id == root_span

    def test_nested_spans(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        root_span = emitter.span_id

        with emitter.span("outer"):
            outer_span = emitter.span_id
            assert emitter.parent_span_id == root_span

            with emitter.span("inner"):
                inner_span = emitter.span_id
                assert emitter.parent_span_id == outer_span

            assert emitter.span_id == outer_span

        assert emitter.span_id == root_span

        assert len(emitter.events) == 4
        assert isinstance(emitter.events[0], SpanStartEvent)
        assert emitter.events[0].name == "outer"
        assert isinstance(emitter.events[1], SpanStartEvent)
        assert emitter.events[1].name == "inner"
        assert emitter.events[1].span_id == inner_span
        assert isinstance(emitter.events[2], SpanEndEvent)
        assert emitter.events[2].name == "inner"
        assert isinstance(emitter.events[3], SpanEndEvent)
        assert emitter.events[3].name == "outer"

    def test_span_restores_on_exception(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        root_span = emitter.span_id

        try:
            with emitter.span("failing"):
                raise ValueError("boom")
        except ValueError:
            pass

        assert emitter.span_id == root_span
        assert len(emitter.events) == 2
        assert isinstance(emitter.events[1], SpanEndEvent)

    def test_events_within_span_use_span_ids(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")

        with emitter.span("work"):
            event = AgentStartEvent(
                trace_id=emitter.trace_id,
                span_id=emitter.span_id,
                parent_span_id=emitter.parent_span_id,
                agent_name="test",
                task_input="hello",
                tools_available=[],
            )
            emitter.emit(event)

        # SpanStart, AgentStart, SpanEnd
        assert len(emitter.events) == 3
        agent_event = emitter.events[1]
        span_start = emitter.events[0]
        assert agent_event.span_id == span_start.span_id


class TestAsyncSafety:
    def test_concurrent_spans_isolated(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")

        async def task_a() -> str:
            with emitter.span("task_a"):
                await asyncio.sleep(0.01)
                return emitter.span_id

        async def task_b() -> str:
            with emitter.span("task_b"):
                await asyncio.sleep(0.01)
                return emitter.span_id

        async def run() -> tuple[str, str]:
            a, b = await asyncio.gather(task_a(), task_b())
            return a, b

        span_a, span_b = asyncio.run(run())
        assert span_a != span_b


class TestListeners:
    def test_single_listener_receives_event(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        received: list[object] = []
        emitter.add_listener(lambda e: received.append(e))

        event = AgentStartEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            agent_name="test",
            task_input="hello",
            tools_available=[],
        )
        emitter.emit(event)

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_listeners_called_in_order(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        order: list[str] = []
        emitter.add_listener(lambda _: order.append("first"))
        emitter.add_listener(lambda _: order.append("second"))

        emitter.emit(
            AgentStartEvent(
                trace_id=emitter.trace_id,
                span_id=emitter.span_id,
                agent_name="test",
                task_input="hello",
                tools_available=[],
            )
        )

        assert order == ["first", "second"]

    def test_listener_exception_does_not_remove_listener(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        calls: list[str] = []

        def bad_listener(e: object) -> None:
            raise RuntimeError("boom")

        def good_listener(e: object) -> None:
            calls.append("good")

        emitter.add_listener(bad_listener)
        emitter.add_listener(good_listener)

        event = AgentStartEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            agent_name="test",
            task_input="hello",
            tools_available=[],
        )
        # First emit: bad_listener raises but stays, good_listener runs
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emitter.emit(event)
        assert calls == ["good"]
        # Event was still appended
        assert len(emitter.events) == 1
        assert len(w) == 1
        assert "Event listener failed: boom" in str(w[0].message)

        # Second emit: both listeners still present, bad_listener raises again
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emitter.emit(event)
        assert calls == ["good", "good"]

    def test_listener_called_after_event_appended(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        events_at_callback: list[int] = []
        emitter.add_listener(lambda _: events_at_callback.append(len(emitter.events)))

        emitter.emit(
            AgentStartEvent(
                trace_id=emitter.trace_id,
                span_id=emitter.span_id,
                agent_name="test",
                task_input="hello",
                tools_available=[],
            )
        )

        # Listener should see the event already in the list
        assert events_at_callback == [1]


class TestMaxEvents:
    def _emit_n(self, emitter: InMemoryEmitter, n: int) -> None:
        for i in range(n):
            emitter.emit(
                AgentStartEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    agent_name=f"agent-{i}",
                    task_input="hello",
                    tools_available=[],
                )
            )

    def test_default_unbounded(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        self._emit_n(emitter, 100)
        assert len(emitter.events) == 100

    def test_max_events_drops_oldest(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1", max_events=3)
        self._emit_n(emitter, 5)
        assert len(emitter.events) == 3
        names = [e.agent_name for e in emitter.events]  # type: ignore[union-attr]
        assert names == ["agent-2", "agent-3", "agent-4"]

    def test_max_events_not_exceeded(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1", max_events=10)
        self._emit_n(emitter, 3)
        assert len(emitter.events) == 3

    def test_listeners_still_called_when_capped(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1", max_events=2)
        received: list[str] = []
        emitter.add_listener(lambda e: received.append(e.agent_name))  # type: ignore[union-attr, arg-type]
        self._emit_n(emitter, 4)
        assert len(received) == 4
        assert len(emitter.events) == 2


class TestProtocolConformance:
    def test_isinstance_check(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace-1")
        assert isinstance(emitter, EventEmitter)


class TestCreateChild:
    def test_child_shares_trace_id(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1")
        child = parent.create_child()
        assert child.trace_id == parent.trace_id

    def test_child_root_span_links_to_parent_span(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1")
        parent_span = parent.span_id
        child = parent.create_child()
        assert child.parent_span_id == parent_span

    def test_child_inside_parent_span_links_to_that_span(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1")
        with parent.span("outer"):
            outer_span = parent.span_id
            child = parent.create_child()
        assert child.parent_span_id == outer_span

    def test_child_has_independent_span_stack(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1")
        child = parent.create_child()

        parent_root = parent.span_id
        child_root = child.span_id
        assert parent_root != child_root

        with parent.span("p"):
            assert parent.span_id != parent_root
            # Child should be unaffected
            assert child.span_id == child_root

    def test_child_copies_listeners(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1")
        received: list[str] = []
        parent.add_listener(lambda e: received.append("heard"))
        child = parent.create_child()

        child.emit(
            AgentStartEvent(
                trace_id=child.trace_id,
                span_id=child.span_id,
                agent_name="test",
                task_input="hello",
                tools_available=[],
            )
        )
        assert received == ["heard"]

    def test_child_events_forward_to_parent(self) -> None:
        """Child-emitter events are forwarded into the parent's ``events`` list
        so composite-agent inner events surface in the outer emitter's trace.

        The child retains its own copy for backward-compatible in-child
        inspection, but the parent's ``events`` list is authoritative for
        trace consumers (e.g., ``save_trace``).
        """
        parent = InMemoryEmitter(trace_id="trace-1")
        child = parent.create_child()
        child.emit(
            AgentStartEvent(
                trace_id=child.trace_id,
                span_id=child.span_id,
                agent_name="test",
                task_input="hello",
                tools_available=[],
            )
        )
        assert len(child.events) == 1
        assert len(parent.events) == 1
        assert parent.events[0] is child.events[0]

    def test_forwarded_events_do_not_re_run_listeners(self) -> None:
        """Events forwarded from child to parent must not re-invoke the
        parent's listeners — listener copies on the child already fire once,
        and re-running them on the parent would duplicate callbacks and,
        via forwarding, cause infinite recursion through grandchild chains.
        """
        parent = InMemoryEmitter(trace_id="trace-1")
        received: list[str] = []
        parent.add_listener(lambda e: received.append("heard"))
        child = parent.create_child()

        child.emit(
            AgentStartEvent(
                trace_id=child.trace_id,
                span_id=child.span_id,
                agent_name="test",
                task_input="hello",
                tools_available=[],
            )
        )
        # Exactly one listener invocation, from the copied listener on the
        # child — not duplicated by the forwarding sink on the parent.
        assert received == ["heard"]

    def test_concurrent_children_have_isolated_spans(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1")
        child_a = parent.create_child()
        child_b = parent.create_child()

        async def task_a() -> str:
            with child_a.span("a_work"):
                await asyncio.sleep(0.01)
                return child_a.span_id

        async def task_b() -> str:
            with child_b.span("b_work"):
                await asyncio.sleep(0.01)
                return child_b.span_id

        async def run() -> tuple[str, str]:
            a, b = await asyncio.gather(task_a(), task_b())
            return a, b

        span_a, span_b = asyncio.run(run())
        assert span_a != span_b

    def test_child_inherits_max_events(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1", max_events=3)
        child = parent.create_child()
        for i in range(5):
            child.emit(
                AgentStartEvent(
                    trace_id=child.trace_id,
                    span_id=child.span_id,
                    agent_name=f"agent-{i}",
                    task_input="hello",
                    tools_available=[],
                )
            )
        assert len(child.events) == 3


class TestAgentBind:
    """Verify the non-mutating ``Agent.bind`` contract.

    ``bind(parent)`` returns a ``BoundAgent`` handle carrying a fresh
    child emitter. It does not mutate the agent — the agent's default
    emitter is unchanged — so concurrent tasks can each obtain their
    own handle on a shared agent.
    """

    async def test_bind_returns_handle_without_mutating_agent(self) -> None:
        parent = InMemoryEmitter(trace_id="trace-1")
        client = MockLLMClient(
            [
                LLMResponse(
                    content="done",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=1, output_tokens=1),
                    model="test",
                )
            ]
        )
        agent = ReasoningAgent(name="test-agent", llm_client=client, emitter=parent, system_prompt="test")

        original_default = agent._default_emitter
        assert original_default is parent

        handle = agent.bind(parent)

        # Agent's default emitter is unchanged.
        assert agent._default_emitter is parent
        # Handle holds a child emitter linked to parent.
        assert handle.emitter.trace_id == parent.trace_id
        assert handle.emitter.parent_span_id == parent.span_id

    async def test_bound_run_routes_tool_events_to_child_emitter(self) -> None:
        """Tool dispatch inside a bound run emits to the bound child emitter."""
        from nanitics import ReActAgent, ToolCall, tool
        from nanitics.infrastructure.observability.events import ToolInvokeEvent

        @tool(name="stub", description="Stub tool for testing bind()")
        async def stub_tool(query: str) -> str:
            return f"ok:{query}"

        original_emitter = InMemoryEmitter(trace_id="original-trace")
        tool_use_response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc1", name="stub", arguments={"query": "hi"})],
            usage=Usage(input_tokens=1, output_tokens=1),
            model="test",
            stop_reason="tool_use",
        )
        final_response = LLMResponse(
            content="done",
            tool_calls=[],
            usage=Usage(input_tokens=1, output_tokens=1),
            model="test",
            stop_reason="end_turn",
        )
        agent = ReActAgent(
            name="bindable",
            llm_client=MockLLMClient([tool_use_response, final_response]),
            emitter=original_emitter,
            system_prompt="p",
            tools=[stub_tool],
        )

        parent = InMemoryEmitter(trace_id="outer-trace")
        await agent.bind(parent).run("hi")

        invoke_events = [e for e in parent.events if isinstance(e, ToolInvokeEvent)]
        assert len(invoke_events) == 1
        # Tool event was forwarded from the child emitter into the parent.
        assert invoke_events[0].trace_id == parent.trace_id


class TestEmitterPropagationIntegration:
    """Integration test: Sequential workflow with AgentSteps shares one trace_id."""

    async def test_sequential_workflow_single_trace(self) -> None:
        from nanitics.composition.orchestration.adapters import AgentStep
        from nanitics.composition.orchestration.sequential import Sequential

        emitter = InMemoryEmitter(trace_id="integration-trace")
        all_events: list[BaseEvent] = []
        emitter.add_listener(lambda e: all_events.append(e))

        client_a = MockLLMClient(
            [
                LLMResponse(
                    content="output-a",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=1, output_tokens=1),
                    model="test",
                )
            ]
        )
        client_b = MockLLMClient(
            [
                LLMResponse(
                    content="output-b",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=1, output_tokens=1),
                    model="test",
                )
            ]
        )

        agent_a = ReasoningAgent(name="agent-a", llm_client=client_a, emitter=emitter, system_prompt="A")
        agent_b = ReasoningAgent(name="agent-b", llm_client=client_b, emitter=emitter, system_prompt="B")

        workflow = Sequential(
            name="test-seq",
            steps=[AgentStep(agent_a), AgentStep(agent_b)],
            emitter=emitter,
        )

        result = await workflow.execute("start")
        assert result.output == "output-b"

        # All events from the listener share one trace_id
        for event in all_events:
            assert event.trace_id == "integration-trace"

        # Find AgentStartEvents — should have both agents
        agent_starts = [e for e in all_events if isinstance(e, AgentStartEvent)]
        assert len(agent_starts) == 2
        assert {e.agent_name for e in agent_starts} == {"agent-a", "agent-b"}

        # Verify span tree is connected: agent spans have parent_span_ids
        # that trace back to the workflow span
        for start in agent_starts:
            assert start.parent_span_id is not None
