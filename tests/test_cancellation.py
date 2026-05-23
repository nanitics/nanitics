import asyncio
import threading
from unittest.mock import AsyncMock, Mock

from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.composition.multi_agent.orchestrator import create_orchestrator
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, ToolCall
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import Usage
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import AgentResult
from nanitics.strategies.agents.reasoning import ReasoningAgent


def _make_usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5)


def _make_emitter() -> InMemoryEmitter:
    return InMemoryEmitter(trace_id="test-trace")


def _make_result(output: str = "done", termination_reason: str = "complete") -> AgentResult:
    return AgentResult(
        output=output,
        total_steps=1,
        termination_reason=termination_reason,
        messages=[],
        usage=_make_usage(),
    )


def _make_response(
    content: str | None = "response",
    tool_calls: list[ToolCall] | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=_make_usage(),
        model="test-model",
        stop_reason="end_turn" if not tool_calls else "tool_use",
    )


class TestCancellationTokenInitial:
    def test_starts_uncancelled(self):
        token = CancellationToken()
        assert token.is_cancelled is False


class TestCancellationTokenCancel:
    def test_cancel_sets_cancelled(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_cancel_is_idempotent(self):
        token = CancellationToken()
        token.cancel()
        token.cancel()
        token.cancel()
        assert token.is_cancelled is True


class TestCancellationTokenThreadSafety:
    def test_cancel_from_another_thread(self):
        token = CancellationToken()
        assert token.is_cancelled is False

        thread = threading.Thread(target=token.cancel)
        thread.start()
        thread.join()

        assert token.is_cancelled is True


class TestCancellationTokenWaitAsync:
    async def test_already_cancelled_resolves_immediately(self):
        token = CancellationToken()
        token.cancel()
        # No timeout needed — if this hangs, the test framework's own
        # timeout will surface the bug. We assert it returns without
        # awaiting any external signal.
        await asyncio.wait_for(token.wait_async(), timeout=1.0)

    async def test_same_loop_cancel_wakes_pending_waiter(self):
        token = CancellationToken()

        async def _cancel_soon() -> None:
            # Yield once so the waiter is parked on the asyncio.Event first.
            await asyncio.sleep(0)
            token.cancel()

        await asyncio.gather(
            asyncio.wait_for(token.wait_async(), timeout=1.0),
            _cancel_soon(),
        )
        assert token.is_cancelled

    async def test_cross_thread_cancel_wakes_pending_waiter(self):
        token = CancellationToken()

        async def _trigger() -> None:
            # ``asyncio.to_thread`` schedules ``token.cancel()`` on a worker
            # thread, exercising the ``call_soon_threadsafe`` branch.
            await asyncio.to_thread(token.cancel)

        # Park the waiter, then fire ``cancel()`` from another thread.
        waiter = asyncio.create_task(token.wait_async())
        await asyncio.sleep(0)  # let the waiter bind the loop
        trigger = asyncio.create_task(_trigger())
        await asyncio.wait_for(waiter, timeout=1.0)
        await trigger
        assert token.is_cancelled

    async def test_wait_async_from_different_loop_raises(self):
        token = CancellationToken()
        # First-loop bind happens here.
        bind_loop = asyncio.get_running_loop()
        await asyncio.sleep(0)  # establish loop running

        # Force a bind by calling wait_async briefly (already-cancelled fast
        # path still binds the loop in the current implementation).
        async def _first_wait() -> None:
            # Schedule a cancel so the wait resolves quickly.
            asyncio.get_running_loop().call_soon(token.cancel)
            await token.wait_async()

        await _first_wait()
        assert token._bound_loop is bind_loop

        # Now try to use it from a different loop.
        def _other_loop_attempt() -> Exception | None:
            other = asyncio.new_event_loop()
            try:

                async def _wait() -> None:
                    await token.wait_async()

                try:
                    other.run_until_complete(_wait())
                    return None
                except Exception as exc:
                    return exc
            finally:
                other.close()

        err = await asyncio.to_thread(_other_loop_attempt)
        assert isinstance(err, RuntimeError)
        assert "different event loop" in str(err)

    async def test_cancel_before_wait_async_binds_fast_path(self):
        token = CancellationToken()
        token.cancel()
        # The first call after cancel should bind, see the already-cancelled
        # state, and return immediately.
        await asyncio.wait_for(token.wait_async(), timeout=1.0)
        # Second call still works.
        await asyncio.wait_for(token.wait_async(), timeout=1.0)

    async def test_cancel_from_loop_thread_is_direct_set(self):
        # Same-loop ``cancel()`` after binding goes through ``event.set()``
        # directly, not ``call_soon_threadsafe``. Cover that branch.
        token = CancellationToken()
        waiter = asyncio.create_task(token.wait_async())
        await asyncio.sleep(0)  # bind
        token.cancel()
        await asyncio.wait_for(waiter, timeout=1.0)

    def test_cancel_before_any_async_use_is_noop_for_asyncio_event(self):
        # Without a bound loop, ``cancel()`` must still set the thread event
        # and not crash. This is the "no loop yet" branch in ``cancel``.
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True
        assert token._bound_loop is None
        assert token._asyncio_event is None


class TestAgentCancellationTokenAccess:
    def test_cancellation_token_property_returns_token(self):
        token = CancellationToken()
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            system_prompt="You are a test agent.",
            cancellation_token=token,
        )
        assert agent.cancellation_token is token

    def test_cancellation_token_property_returns_none_by_default(self):
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            system_prompt="You are a test agent.",
        )
        assert agent.cancellation_token is None

    def test_set_cancellation_token(self):
        emitter = _make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            system_prompt="You are a test agent.",
        )
        token = CancellationToken()
        agent.set_cancellation_token(token)
        assert agent.cancellation_token is token


class TestAgentToolCancellationPropagation:
    async def test_agent_tool_propagates_token_to_delegate(self):
        token = CancellationToken()
        emitter = _make_emitter()
        agent = AsyncMock()
        agent.name = "delegate"
        agent.run = AsyncMock(return_value=_make_result())
        handle = Mock()
        handle.run = agent.run
        agent.bind = Mock(return_value=handle)
        agent.set_cancellation_token = Mock()

        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="A delegate agent",
            cancellation_token=token,
        )
        await tool.execute(task="do something")
        agent.set_cancellation_token.assert_called_once_with(token)

    async def test_agent_tool_without_token_does_not_call_set(self):
        emitter = _make_emitter()
        agent = AsyncMock()
        agent.name = "delegate"
        agent.run = AsyncMock(return_value=_make_result())
        handle = Mock()
        handle.run = agent.run
        agent.bind = Mock(return_value=handle)
        agent.set_cancellation_token = Mock()

        tool = AgentTool(
            agent=agent,
            emitter=emitter,
            description="A delegate agent",
        )
        await tool.execute(task="do something")
        agent.set_cancellation_token.assert_not_called()


class TestOrchestratorCancellationPropagation:
    def test_create_orchestrator_distributes_token_to_specialists(self):
        token = CancellationToken()
        emitter = _make_emitter()

        s1 = AgentTool(
            agent=ReasoningAgent(
                name="s1",
                llm_client=MockLLMClient([_make_response()]),
                emitter=emitter,
                system_prompt="You are s1.",
            ),
            emitter=emitter,
            description="Specialist 1",
        )
        s2 = AgentTool(
            agent=ReasoningAgent(
                name="s2",
                llm_client=MockLLMClient([_make_response()]),
                emitter=emitter,
                system_prompt="You are s2.",
            ),
            emitter=emitter,
            description="Specialist 2",
        )

        create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            specialists=[s1, s2],
            cancellation_token=token,
        )

        assert s1.cancellation_token is token
        assert s2.cancellation_token is token

    def test_create_orchestrator_without_token_leaves_specialists_unset(self):
        emitter = _make_emitter()

        s1 = AgentTool(
            agent=ReasoningAgent(
                name="s1",
                llm_client=MockLLMClient([_make_response()]),
                emitter=emitter,
                system_prompt="You are s1.",
            ),
            emitter=emitter,
            description="Specialist 1",
        )

        create_orchestrator(
            name="orch",
            llm_client=MockLLMClient([_make_response()]),
            emitter=emitter,
            specialists=[s1],
        )

        assert s1.cancellation_token is None

    async def test_cancelled_token_stops_delegate_agent(self):
        token = CancellationToken()
        emitter = _make_emitter()
        from nanitics.strategies.agents.react import ReActAgent

        # ReActAgent checks _is_cancelled at the start of each loop iteration
        delegate = ReActAgent(
            name="delegate",
            llm_client=MockLLMClient([_make_response("I should not finish")]),
            emitter=emitter,
            system_prompt="You are a delegate.",
            tools=[],
        )

        token.cancel()
        delegate.set_cancellation_token(token)

        result = await delegate.run("do work")
        assert result.termination_reason == "cancelled"

    async def test_three_level_hierarchy_cancellation_cascade(self):
        """3-level hierarchy shares one token; cancellation cascades to all agents."""
        token = CancellationToken()
        emitter = _make_emitter()

        # Leaf agents
        leaf1 = ReasoningAgent(
            name="leaf1",
            llm_client=MockLLMClient([_make_response("leaf1 output")]),
            emitter=emitter,
            system_prompt="You are leaf1.",
        )
        leaf2 = ReasoningAgent(
            name="leaf2",
            llm_client=MockLLMClient([_make_response("leaf2 output")]),
            emitter=emitter,
            system_prompt="You are leaf2.",
        )

        # Mid-level orchestrator specialists
        mid_s1 = AgentTool(agent=leaf1, emitter=emitter, description="Leaf 1")
        mid_s2 = AgentTool(agent=leaf2, emitter=emitter, description="Leaf 2")

        mid_orchestrator = create_orchestrator(
            name="mid-orch",
            llm_client=MockLLMClient([_make_response("mid result")]),
            emitter=emitter,
            specialists=[mid_s1, mid_s2],
            cancellation_token=token,
        )

        # Top-level orchestrator
        top_specialist = AgentTool(agent=mid_orchestrator, emitter=emitter, description="Mid orchestrator")

        create_orchestrator(
            name="top-orch",
            llm_client=MockLLMClient([_make_response("top result")]),
            emitter=emitter,
            specialists=[top_specialist],
            cancellation_token=token,
        )

        # Token distributed to mid-level specialists
        assert mid_s1.cancellation_token is token
        assert mid_s2.cancellation_token is token
        # Token distributed to top-level specialist
        assert top_specialist.cancellation_token is token

        # Cancel and verify all agents stop
        token.cancel()

        result = await mid_orchestrator.run("do work")
        assert result.termination_reason == "cancelled"
