"""Unit tests for the thread-identity primitive.

Covers the ``ThreadStore`` protocol surface, the ``InMemoryThreadStore``
reference implementation, the ``ThreadLocks`` active-key serialization,
the ``ThreadInUseError`` shape, the ``Agent.run`` plumbing
(lock-acquire / prefix-load / append-on-success / no-append-on-failure),
the message-ordering rule (``initial_messages`` → prefix → new input),
the unwrapped-replay semantics (replayed messages bypass the
``<nanitics:context>`` envelope), the ``AgentResult.thread_key`` and
``AgentStartEvent.thread_key`` / ``replayed_message_count`` payload, and
the checkpoint-frozen-view-on-resume contract.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nanitics.capabilities.errors.handler import ErrorHandler
from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.hitl_store import InMemoryHitlRequestStore
from nanitics.collaboration.protocol import HumanDecision, HumanInputResponse
from nanitics.composition import InMemoryThreadStore, ThreadLocks, ThreadStore
from nanitics.errors import ThreadInUseError
from nanitics.infrastructure import LLMRequestEvent, MockLLMClient
from nanitics.infrastructure.observability.events import AgentStartEvent
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import Message, ToolCall
from tests.testing_helpers import make_emitter, make_response

# ── ThreadInUseError ────────────────────────────────────────


class TestThreadInUseError:
    def test_carries_thread_key(self) -> None:
        err = ThreadInUseError(thread_key="t1")
        assert err.thread_key == "t1"
        assert "t1" in str(err)

    def test_is_nanitics_error(self) -> None:
        from nanitics.infrastructure.errors import NaniticsError

        err = ThreadInUseError(thread_key="t1")
        assert isinstance(err, NaniticsError)

    def test_to_dict_includes_thread_key(self) -> None:
        err = ThreadInUseError(thread_key="t1", trace_id="trace-1", span_id="span-1")
        d = err.to_dict()
        assert d["thread_key"] == "t1"
        assert d["trace_id"] == "trace-1"


# ── InMemoryThreadStore ─────────────────────────────────────


class TestInMemoryThreadStore:
    async def test_load_unknown_key_returns_empty(self) -> None:
        store = InMemoryThreadStore()
        assert await store.load("missing") == []

    async def test_append_then_load(self) -> None:
        store = InMemoryThreadStore()
        messages = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
        await store.append("t1", messages)
        loaded = await store.load("t1")
        assert loaded == messages

    async def test_append_extends(self) -> None:
        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="a")])
        await store.append("t1", [Message(role="assistant", content="b")])
        loaded = await store.load("t1")
        assert [m.content for m in loaded] == ["a", "b"]

    async def test_load_returns_copy(self) -> None:
        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="x")])
        loaded = await store.load("t1")
        loaded.append(Message(role="user", content="mutation"))
        again = await store.load("t1")
        assert len(again) == 1

    async def test_clear(self) -> None:
        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="x")])
        await store.clear("t1")
        assert await store.load("t1") == []

    async def test_clear_unknown_key_is_noop(self) -> None:
        store = InMemoryThreadStore()
        await store.clear("never-existed")

    def test_satisfies_protocol(self) -> None:
        store = InMemoryThreadStore()
        assert isinstance(store, ThreadStore)


# ── ThreadLocks ─────────────────────────────────────────────


class TestThreadLocks:
    async def test_hold_releases_on_exit(self) -> None:
        locks = ThreadLocks()
        async with locks.hold("t1"):
            pass
        # Acquiring again succeeds — the previous hold was released.
        async with locks.hold("t1"):
            pass

    async def test_hold_releases_on_exception(self) -> None:
        locks = ThreadLocks()
        with pytest.raises(RuntimeError):
            async with locks.hold("t1"):
                raise RuntimeError("boom")
        # Subsequent acquire succeeds.
        async with locks.hold("t1"):
            pass

    async def test_concurrent_same_key_raises(self) -> None:
        locks = ThreadLocks()
        gate = asyncio.Event()

        async def first() -> None:
            async with locks.hold("t1"):
                gate.set()
                await asyncio.sleep(0.05)

        async def second() -> str:
            await gate.wait()
            async with locks.hold("t1"):
                return "ok"

        results = await asyncio.gather(first(), second(), return_exceptions=True)
        assert results[0] is None
        assert isinstance(results[1], ThreadInUseError)
        assert results[1].thread_key == "t1"

    async def test_concurrent_different_keys_coexist(self) -> None:
        locks = ThreadLocks()

        async def hold(key: str) -> str:
            async with locks.hold(key):
                await asyncio.sleep(0.01)
                return key

        results = await asyncio.gather(hold("a"), hold("b"))
        assert set(results) == {"a", "b"}


# ── Agent.run with thread_key ──────────────────────────────


@tool(name="echo", description="Echo input")
async def echo_tool(text: str) -> str:
    return text


class TestAgentRunThreadKey:
    async def test_run_without_thread_key_unchanged(self) -> None:
        emitter = make_emitter()
        client = MockLLMClient([make_response(content="ok")])
        agent = ReActAgent(name="a", llm_client=client, emitter=emitter, system_prompt="sys", tools=[])
        result = await agent.run("hello")
        assert result.thread_key is None
        assert result.output == "ok"

    async def test_run_with_thread_key_no_store_passes_through(self) -> None:
        emitter = make_emitter()
        client = MockLLMClient([make_response(content="ok")])
        agent = ReActAgent(name="a", llm_client=client, emitter=emitter, system_prompt="sys", tools=[])
        result = await agent.run("hello", thread_key="t1")
        assert result.thread_key == "t1"

    async def test_first_call_loads_empty_appends_new(self) -> None:
        store = InMemoryThreadStore()
        client = MockLLMClient([make_response(content="hi back")])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="sys",
            tools=[],
            thread_store=store,
        )
        await agent.run("hello", thread_key="t1")
        loaded = await store.load("t1")
        # New messages: the user input and the assistant final answer.
        assert [m.role for m in loaded] == ["user", "assistant"]
        assert loaded[0].content == "hello"
        assert loaded[1].content == "hi back"

    async def test_second_call_sees_first_call_history(self) -> None:
        store = InMemoryThreadStore()
        emitter = make_emitter()
        client = MockLLMClient(
            [
                make_response(content="response-1"),
                make_response(content="response-2"),
            ]
        )
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=emitter,
            system_prompt="sys",
            tools=[],
            thread_store=store,
        )
        await agent.run("first", thread_key="t1")
        await agent.run("second", thread_key="t1")

        # The second call's LLM input should include the first call's
        # user/assistant pair plus the new user input. Read from
        # ``LLMRequestEvent`` (which serializes messages at emission
        # time) rather than ``MockLLMClient.calls`` (which stores the
        # list by reference and so reflects post-call mutations).
        req_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        second_call_messages = req_events[1].messages
        assert [m["role"] for m in second_call_messages] == ["user", "assistant", "user"]
        assert second_call_messages[0]["content"] == "first"
        assert second_call_messages[1]["content"] == "response-1"
        assert second_call_messages[2]["content"] == "second"

    async def test_ordering_initial_messages_then_prefix_then_input(self) -> None:
        store = InMemoryThreadStore()
        await store.append(
            "t1",
            [Message(role="user", content="prefix-u"), Message(role="assistant", content="prefix-a")],
        )
        emitter = make_emitter()
        client = MockLLMClient([make_response(content="ok")])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=emitter,
            system_prompt="sys",
            tools=[],
            thread_store=store,
            initial_messages=[Message(role="user", content="seed")],
        )
        await agent.run("now", thread_key="t1")
        req_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        messages = req_events[0].messages
        assert [m["content"] for m in messages] == ["seed", "prefix-u", "prefix-a", "now"]

    async def test_replayed_messages_are_unwrapped(self) -> None:
        """Replayed prefix messages must NOT be wrapped in <nanitics:context>."""
        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="assistant", content="prior answer")])
        emitter = make_emitter()
        client = MockLLMClient([make_response(content="next")])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=emitter,
            system_prompt="sys",
            tools=[],
            thread_store=store,
        )
        await agent.run("follow", thread_key="t1")
        req_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        assert req_events
        # The prior-assistant message appears verbatim (no <nanitics:context>
        # wrapper around it).
        msgs = req_events[0].messages
        contents = [m["content"] for m in msgs]
        assert "prior answer" in contents
        for c in contents:
            if c == "prior answer":
                assert "<nanitics:context" not in c

    async def test_concurrent_same_key_raises(self) -> None:
        """Concurrent same-key runs serialize via ``ThreadLocks.hold`` and raise on the second."""
        store = InMemoryThreadStore()
        shared_locks = ThreadLocks()

        # First task: hold the lock via a manual ``hold`` context so we
        # have a deterministic "in flight" window without needing an LLM
        # call to yield. Then assert the second concurrent acquire fails
        # by routing it through ``Agent.run``.
        client = MockLLMClient([make_response(content="ok")])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="sys",
            tools=[],
            thread_store=store,
            thread_locks=shared_locks,
        )

        gate = asyncio.Event()
        runner_done = asyncio.Event()

        async def holder() -> None:
            async with shared_locks.hold("t1"):
                gate.set()
                await runner_done.wait()

        async def runner() -> Any:
            await gate.wait()
            try:
                return await agent.run("hello", thread_key="t1")
            finally:
                runner_done.set()

        results = await asyncio.gather(holder(), runner(), return_exceptions=True)
        assert results[0] is None
        assert isinstance(results[1], ThreadInUseError)
        assert results[1].thread_key == "t1"

    async def test_concurrent_different_keys_succeed(self) -> None:
        store = InMemoryThreadStore()
        client = MockLLMClient([make_response(content="a"), make_response(content="b")])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="sys",
            tools=[],
            thread_store=store,
        )

        async def call(key: str) -> Any:
            return await agent.run(key, thread_key=key)

        results = await asyncio.gather(call("ta"), call("tb"))
        assert all(r.thread_key in {"ta", "tb"} for r in results)

    async def test_failed_run_does_not_advance_thread(self) -> None:
        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="pre")])
        client = MockLLMClient([])  # Will raise on first call.
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="sys",
            tools=[],
            thread_store=store,
            error_handler=ErrorHandler(max_corrections=0),
        )
        with pytest.raises(ValueError):
            await agent.run("hello", thread_key="t1")
        # Store unchanged.
        loaded = await store.load("t1")
        assert len(loaded) == 1
        assert loaded[0].content == "pre"

    async def test_failed_run_releases_lock(self) -> None:
        store = InMemoryThreadStore()
        client = MockLLMClient([])  # Will raise.
        client_2 = MockLLMClient([make_response(content="ok")])
        locks = ThreadLocks()
        agent_a = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="sys",
            tools=[],
            thread_store=store,
            thread_locks=locks,
            error_handler=ErrorHandler(max_corrections=0),
        )
        agent_b = ReActAgent(
            name="b",
            llm_client=client_2,
            emitter=make_emitter(),
            system_prompt="sys",
            tools=[],
            thread_store=store,
            thread_locks=locks,
        )
        with pytest.raises(ValueError):
            await agent_a.run("first", thread_key="t1")
        # Lock released — second run on same key succeeds.
        result = await agent_b.run("second", thread_key="t1")
        assert result.thread_key == "t1"


# ── AgentStartEvent and AgentResult fields ────────────────


class TestAgentStartPayload:
    async def test_no_thread_key_payload_defaults(self) -> None:
        emitter = make_emitter()
        client = MockLLMClient([make_response(content="x")])
        agent = ReActAgent(name="a", llm_client=client, emitter=emitter, system_prompt="s", tools=[])
        await agent.run("hi")
        starts = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(starts) == 1
        assert starts[0].thread_key is None
        assert starts[0].replayed_message_count == 0

    async def test_thread_key_payload_populated(self) -> None:
        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="a"), Message(role="assistant", content="b")])
        emitter = make_emitter()
        client = MockLLMClient([make_response(content="x")])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=emitter,
            system_prompt="s",
            tools=[],
            thread_store=store,
        )
        await agent.run("hi", thread_key="t1")
        starts = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert starts[0].thread_key == "t1"
        assert starts[0].replayed_message_count == 2


# ── Agent base helpers (direct unit tests for coverage) ───


class TestLoadThreadPrefix:
    async def test_returns_empty_when_no_thread_key(self) -> None:
        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="x")])
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
            thread_store=store,
        )
        assert await agent._load_thread_prefix(None) == []

    async def test_returns_empty_when_no_store(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
        )
        assert await agent._load_thread_prefix("t1") == []

    async def test_returns_prefix(self) -> None:
        store = InMemoryThreadStore()
        msg = Message(role="user", content="hi")
        await store.append("t1", [msg])
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
            thread_store=store,
        )
        prefix = await agent._load_thread_prefix("t1")
        assert prefix == [msg]


class TestNewMessagesAfterPrefix:
    def test_slices_past_initial_and_prefix(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
            initial_messages=[Message(role="user", content="seed")],
        )
        run_messages = [
            Message(role="user", content="seed"),
            Message(role="user", content="prefix-u"),
            Message(role="assistant", content="prefix-a"),
            Message(role="user", content="now"),
            Message(role="assistant", content="answer"),
        ]
        prefix = [Message(role="user", content="prefix-u"), Message(role="assistant", content="prefix-a")]
        new = agent._new_messages_after_prefix(run_messages, prefix)
        assert [m.content for m in new] == ["now", "answer"]

    def test_no_initial_no_prefix_returns_all(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
        )
        run_messages = [Message(role="user", content="x"), Message(role="assistant", content="y")]
        new = agent._new_messages_after_prefix(run_messages, [])
        assert new == run_messages


class TestPeekResumeThreadContext:
    def test_no_resume_state(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
        )
        assert agent._peek_resume_thread_context() == (None, None)

    def test_resume_state_without_thread(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
        )
        agent._set_resume_state({"messages": []})
        key, prefix = agent._peek_resume_thread_context()
        assert key is None
        assert prefix is None

    def test_resume_state_with_thread_and_prefix(self) -> None:
        agent = ReActAgent(
            name="a",
            llm_client=MockLLMClient([]),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
        )
        agent._set_resume_state(
            {
                "thread_key": "t1",
                "thread_prefix": [Message(role="user", content="frozen").model_dump()],
            }
        )
        key, prefix = agent._peek_resume_thread_context()
        assert key == "t1"
        assert prefix is not None
        assert prefix[0].content == "frozen"


# ── Checkpoint × thread (frozen-view-on-resume) ──────────


_TEST_RUN_ID = "thread-resume-run"


@tool(name="add", description="Add two numbers")
async def thread_add_tool(a: int, b: int) -> str:
    return str(a + b)


class TestThreadFrozenPrefixOnDirectResume:
    """Direct-suspend test using bare ``Agent.run`` + ``_set_resume_state``.

    Bypasses ``DurableRun`` (whose ``start`` does not yet forward
    ``thread_key``) to exercise the agent-level contract directly:
    suspend writes ``thread_key`` + ``thread_prefix`` into
    ``SuspendExecution.checkpoint_data``; on resume, the agent uses the
    frozen prefix and ignores live store mutations between
    suspend and resume.
    """

    async def test_suspend_writes_thread_state_into_checkpoint_data(self) -> None:
        from nanitics.composition.durability.suspension import SuspendExecution

        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="snapshot")])

        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped = ApprovalWrappedTool(tool=thread_add_tool, provider=provider)

        responses = [
            make_response(
                content="calling",
                tool_calls=[ToolCall(id="tc-1", name="add", arguments={"a": 1, "b": 2})],
            ),
        ]
        client = MockLLMClient(responses)
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="s",
            tools=[wrapped],
            thread_store=store,
            run_id=_TEST_RUN_ID,
        )

        with pytest.raises(SuspendExecution) as exc_info:
            await agent.run("hello", thread_key="t1")

        data = exc_info.value.checkpoint_data
        assert data is not None
        assert data["thread_key"] == "t1"
        assert data["thread_prefix"]
        assert data["thread_prefix"][0]["content"] == "snapshot"
        # The thread was NOT advanced on suspend.
        assert (await store.load("t1"))[0].content == "snapshot"
        assert len(await store.load("t1")) == 1

    async def test_resume_uses_frozen_prefix_and_ignores_live_store(self) -> None:
        from nanitics.composition.durability.suspension import SuspendExecution

        store = InMemoryThreadStore()
        await store.append("t1", [Message(role="user", content="orig")])

        hitl_store = InMemoryHitlRequestStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped = ApprovalWrappedTool(tool=thread_add_tool, provider=provider)

        responses = [
            make_response(
                content="calling",
                tool_calls=[ToolCall(id="tc-2", name="add", arguments={"a": 3, "b": 4})],
            ),
            make_response(content="final"),
        ]
        client = MockLLMClient(responses)
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="s",
            tools=[wrapped],
            thread_store=store,
            run_id=_TEST_RUN_ID,
        )

        # Drive to suspend.
        try:
            await agent.run("hello", thread_key="t1")
        except SuspendExecution as exc:
            checkpoint_data = exc.checkpoint_data
            assert checkpoint_data is not None
            pending = await hitl_store.get_pending_requests(_TEST_RUN_ID)
            request_id = pending[0].request_id
        else:
            pytest.fail("expected SuspendExecution")

        # Live store mutates between suspend and resume — should be ignored.
        await store.append("t1", [Message(role="user", content="external-mutation")])

        # Preload the human response and resume.
        await hitl_store.save_response(
            request_id,
            HumanInputResponse(
                request_id=request_id,
                decision=HumanDecision.APPROVE,
                response_value=None,
            ),
        )

        # Re-inject checkpoint state into the agent.
        agent._set_resume_state(checkpoint_data)
        result = await agent.run("hello")  # No thread_key — picked up from resume state.

        assert result.thread_key == "t1"
        # The frozen prefix carried through: result.messages starts with the
        # original "orig" message (not the externally-appended one).
        prefix_contents = [m.content for m in result.messages[:1]]
        assert prefix_contents == ["orig"]

        # The live store now has: orig, external-mutation, AND the resumed
        # run's new messages.
        loaded = await store.load("t1")
        assert loaded[0].content == "orig"
        assert loaded[1].content == "external-mutation"
        # The resumed run's new messages append after the external mutation.
        assert any(m.content == "final" for m in loaded[2:])
