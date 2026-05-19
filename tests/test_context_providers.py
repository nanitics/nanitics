from typing import Literal

import pytest
from pydantic import ValidationError

from nanitics.capabilities.context.token_counter import EstimateTokenCounter
from nanitics.capabilities.context.truncation import TruncationPolicy
from nanitics.capabilities.memory.context_provider import (
    ContextContent,
    ContextProvider,
)
from nanitics.infrastructure import MockLLMClient
from nanitics.strategies import (
    ReActAgent,
    ReasoningAgent,
    tool,
)
from nanitics.strategies.agents.base import _render_context_wrapper
from nanitics.tracing import (
    Message,
    ToolCall,
)
from tests.testing_helpers import make_emitter, make_response


def _wrap(
    content: str,
    *,
    provider_name: str = "",
    priority: int = 0,
    protected: bool = False,
) -> str:
    """Shorthand for the canonical wrapper string."""
    return _render_context_wrapper(
        ContextContent(
            content=content,
            priority=priority,
            protected=protected,
            provider_name=provider_name,
        )
    )


def _msg(content: str, role: Literal["user", "assistant"] = "user") -> Message:
    return Message(role=role, content=content)


def _make_groups(count: int, char_length: int = 40) -> list[list[Message]]:
    roles: list[Literal["user", "assistant"]] = ["user", "assistant"]
    return [[_msg("x" * char_length, role=roles[i % 2])] for i in range(count)]


# ──────────────────────────────────────────────────────────
# Mock Context Providers
# ──────────────────────────────────────────────────────────


class StaticProvider:
    def __init__(
        self,
        content: str,
        priority: int = 0,
        protected: bool = False,
    ) -> None:
        self._content = content
        self._priority = priority
        self._protected = protected

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        return ContextContent(
            content=self._content,
            priority=self._priority,
            protected=self._protected,
        )


class NoneProvider:
    async def provide(self, messages: list[Message]) -> ContextContent | None:
        return None


class TrackingProvider:
    """Provider that records the messages it receives."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[list[Message]] = []

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        self.calls.append(list(messages))
        return ContextContent(content=self._content, priority=0, protected=False)


# ──────────────────────────────────────────────────────────
# ContextProvider Protocol Tests
# ──────────────────────────────────────────────────────────


class TestContextProviderProtocol:
    def test_static_provider_is_context_provider(self) -> None:
        provider = StaticProvider("test")
        assert isinstance(provider, ContextProvider)

    def test_none_provider_is_context_provider(self) -> None:
        provider = NoneProvider()
        assert isinstance(provider, ContextProvider)


# ──────────────────────────────────────────────────────────
# ContextContent Model Tests
# ──────────────────────────────────────────────────────────


class TestContextContent:
    def test_defaults(self) -> None:
        cc = ContextContent(content="test")
        assert cc.content == "test"
        assert cc.priority == 0
        assert cc.protected is False

    def test_custom_values(self) -> None:
        cc = ContextContent(content="data", priority=5, protected=True)
        assert cc.priority == 5
        assert cc.protected is True

    def test_frozen(self) -> None:
        cc = ContextContent(content="test")
        with pytest.raises(ValidationError):
            cc.content = "changed"


# ──────────────────────────────────────────────────────────
# Context Injection Unit Tests
# ──────────────────────────────────────────────────────────


class TestContextInjection:
    async def test_none_results_skipped(self) -> None:
        """Providers returning None don't inject any messages."""
        tracking_none = NoneProvider()

        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[tracking_none],
        )

        result = await agent.run("hi")
        assert result.output == "answer"
        # Result messages: user + assistant (no injected)
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    async def test_single_provider_injects_message(self) -> None:
        """A single provider injects its wrapped content before the user message."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[StaticProvider("injected context")],
        )

        await agent.run("question")

        sent_messages = client.calls[0]["messages"]
        assert len(sent_messages) == 2
        # Pinned against the exact canonical wrapper string.
        assert sent_messages[0].content == (
            '<nanitics:context provider="" priority="0" protected="false">\ninjected context\n</nanitics:context>'
        )
        assert sent_messages[0].role == "user"
        assert sent_messages[1].content == "question"
        assert sent_messages[1].role == "user"

    async def test_ordering_by_priority(self) -> None:
        """Providers are sorted by priority — lowest first, highest closest to recent messages."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[
                StaticProvider("high", priority=10),
                StaticProvider("low", priority=1),
                StaticProvider("mid", priority=5),
            ],
        )

        await agent.run("question")

        sent_messages = client.calls[0]["messages"]
        assert len(sent_messages) == 4
        assert sent_messages[0].content == _wrap("low", priority=1)
        assert sent_messages[1].content == _wrap("mid", priority=5)
        assert sent_messages[2].content == _wrap("high", priority=10)
        assert sent_messages[3].content == "question"

    async def test_protected_metadata_set(self) -> None:
        """Protected providers produce messages with metadata.protected=True."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[
                StaticProvider("protected", protected=True),
                StaticProvider("unprotected", protected=False),
            ],
        )

        await agent.run("question")

        sent_messages = client.calls[0]["messages"]
        protected_msgs = [m for m in sent_messages if m.metadata is not None and m.metadata.get("protected")]
        unprotected_msgs = [m for m in sent_messages if m.metadata is not None and not m.metadata.get("protected")]
        assert len(protected_msgs) == 1
        assert protected_msgs[0].content == _wrap("protected", protected=True)
        # The protected=true attribute also lives inside the wrapper string.
        assert 'protected="true"' in protected_msgs[0].content
        assert len(unprotected_msgs) == 1
        assert unprotected_msgs[0].content == _wrap("unprotected", protected=False)
        assert 'protected="false"' in unprotected_msgs[0].content

    async def test_injection_position_with_tool_results(self) -> None:
        """Injected messages appear before the most recent turn's messages (tool results)."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hello"})
        responses = [
            make_response(content="calling", tool_calls=[tool_call]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        tracking = TrackingProvider("context")
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[tracking],
        )

        await agent.run("test")

        # Second call: messages should be [user, context, assistant+tc, tool_result]
        second_call_messages = client.calls[1]["messages"]
        roles = [m.role for m in second_call_messages]
        # Context injected before tool_result (the most recent turn's messages)
        assert "user" in roles
        wrapped_context = _wrap("context")
        context_idx = next(i for i, m in enumerate(second_call_messages) if m.content == wrapped_context)
        tool_result_idx = next(i for i, m in enumerate(second_call_messages) if m.role == "tool_result")
        assert context_idx < tool_result_idx

    async def test_injection_does_not_split_tool_use_and_tool_result(self) -> None:
        """Injected context must not be placed between assistant(tool_use) and its tool_results."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hello"})
        responses = [
            make_response(content="calling", tool_calls=[tool_call]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[StaticProvider("injected")],
        )

        await agent.run("test")

        # In the second LLM call, the assistant(tool_use) must be immediately
        # followed by tool_result — no injected message in between.
        second_call_messages = client.calls[1]["messages"]
        for i, msg in enumerate(second_call_messages):
            if msg.role == "assistant" and msg.tool_calls:
                next_msg = second_call_messages[i + 1]
                assert next_msg.role == "tool_result", (
                    f"Expected tool_result after assistant(tool_use) at index {i}, "
                    f"got {next_msg.role} with content={next_msg.content!r}"
                )

    async def test_no_providers_does_not_modify_messages(self) -> None:
        """When context_providers is None, messages pass through unchanged."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
        )

        result = await agent.run("question")

        assert len(result.messages) == 2
        assert result.messages[0].content == "question"
        assert result.messages[0].role == "user"

    async def test_mixed_none_and_content_providers(self) -> None:
        """Mix of providers returning None and content."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[
                NoneProvider(),
                StaticProvider("visible"),
                NoneProvider(),
            ],
        )

        await agent.run("question")

        sent_messages = client.calls[0]["messages"]
        assert len(sent_messages) == 2
        assert sent_messages[0].content == _wrap("visible")
        assert sent_messages[1].content == "question"


# ──────────────────────────────────────────────────────────
# Wrapper Format Tests — pin the canonical wrapper character-for-character
# ──────────────────────────────────────────────────────────


class TestWrapperFormat:
    async def test_wrapper_is_provider_agnostic(self) -> None:
        """Shape observed on the mock LLM matches the spec literal — regardless of caller."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()

        class NamedProvider:
            async def provide(self, messages: list[Message]) -> ContextContent | None:
                return ContextContent(
                    content="[Working Memory]\n- item",
                    priority=0,
                    protected=True,
                    provider_name="working_memory",
                )

        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[],
            context_providers=[NamedProvider()],
        )
        await agent.run("question")

        sent = client.calls[0]["messages"]
        injected = next(m for m in sent if "<nanitics:context" in (m.content or ""))
        # Literal-string assertion — the wire shape is pinned.
        assert injected.content == (
            '<nanitics:context provider="working_memory" priority="0" protected="true">\n'
            "[Working Memory]\n"
            "- item\n"
            "</nanitics:context>"
        )

    async def test_empty_provider_name_renders_empty_attribute(self) -> None:
        """``provider_name=""`` (default) renders as ``provider=""`` verbatim."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[StaticProvider("body")],
        )
        await agent.run("question")

        sent = client.calls[0]["messages"]
        assert sent[0].content == (
            '<nanitics:context provider="" priority="0" protected="false">\nbody\n</nanitics:context>'
        )

    async def test_negative_priority_rendered_verbatim(self) -> None:
        """``priority=-5`` renders as ``priority="-5"``."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()

        class NegativePriorityProvider:
            async def provide(self, messages: list[Message]) -> ContextContent | None:
                return ContextContent(content="body", priority=-5, provider_name="p")

        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[NegativePriorityProvider()],
        )
        await agent.run("question")

        sent = client.calls[0]["messages"]
        assert 'priority="-5"' in (sent[0].content or "")

    async def test_multi_provider_each_gets_own_wrapper(self) -> None:
        """Two providers → two distinct wrapped messages, not merged."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()

        class FirstProvider:
            async def provide(self, messages: list[Message]) -> ContextContent | None:
                return ContextContent(content="[Working Memory]\n- a", priority=0, provider_name="wm")

        class SecondProvider:
            async def provide(self, messages: list[Message]) -> ContextContent | None:
                return ContextContent(content="[Past Experiences]\n- b", priority=10, provider_name="em")

        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[FirstProvider(), SecondProvider()],
        )
        await agent.run("question")

        sent = client.calls[0]["messages"]
        assert len(sent) == 3
        assert sent[0].content == (
            '<nanitics:context provider="wm" priority="0" protected="false">\n'
            "[Working Memory]\n- a\n"
            "</nanitics:context>"
        )
        assert sent[1].content == (
            '<nanitics:context provider="em" priority="10" protected="false">\n'
            "[Past Experiences]\n- b\n"
            "</nanitics:context>"
        )
        assert sent[2].content == "question"


# ──────────────────────────────────────────────────────────
# ContextAssemblyEvent Tests
# ──────────────────────────────────────────────────────────


class TestContextAssemblyEvent:
    async def test_emits_assembly_event(self) -> None:
        """When providers return content, a ContextAssemblyEvent is emitted.

        ``ContextContribution.content`` and ``content_length`` carry the
        **rendered wrapped string** — the same bytes the LLM sees — so
        trace renderings stay faithful to the wire shape.
        """
        provider = StaticProvider("injected context", priority=3, protected=True)

        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent.",
            context_providers=[provider],
        )
        with emitter.span("test"):
            await agent.run("question")

        from nanitics.infrastructure.observability.events import ContextAssemblyEvent

        assembly_events = [e for e in emitter.events if isinstance(e, ContextAssemblyEvent)]
        assert len(assembly_events) == 1
        evt = assembly_events[0]
        assert evt.total_injected == 1
        assert len(evt.contributions) == 1
        wrapped = _wrap("injected context", priority=3, protected=True)
        assert evt.contributions[0].content == wrapped
        assert evt.contributions[0].content_length == len(wrapped)
        assert evt.contributions[0].priority == 3
        assert evt.contributions[0].protected is True

    async def test_no_assembly_event_when_no_providers(self) -> None:
        """No ContextAssemblyEvent when no providers are configured."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent.",
        )
        with emitter.span("test"):
            await agent.run("question")

        from nanitics.infrastructure.observability.events import ContextAssemblyEvent

        assembly_events = [e for e in emitter.events if isinstance(e, ContextAssemblyEvent)]
        assert len(assembly_events) == 0

    async def test_no_assembly_event_when_all_providers_return_none(self) -> None:
        """No ContextAssemblyEvent when all providers return None."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent.",
            context_providers=[NoneProvider()],
        )
        with emitter.span("test"):
            await agent.run("question")

        from nanitics.infrastructure.observability.events import ContextAssemblyEvent

        assembly_events = [e for e in emitter.events if isinstance(e, ContextAssemblyEvent)]
        assert len(assembly_events) == 0

    async def test_assembly_event_includes_provider_name(self) -> None:
        """Provider name flows through to the assembly event."""

        class NamedProvider:
            async def provide(self, messages: list[Message]) -> ContextContent | None:
                return ContextContent(content="data", provider_name="my_provider")

        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="You are a test agent.",
            context_providers=[NamedProvider()],
        )
        with emitter.span("test"):
            await agent.run("question")

        from nanitics.infrastructure.observability.events import ContextAssemblyEvent

        assembly_events = [e for e in emitter.events if isinstance(e, ContextAssemblyEvent)]
        assert len(assembly_events) == 1
        assert assembly_events[0].contributions[0].provider_name == "my_provider"


# ──────────────────────────────────────────────────────────
# Integration: Agent with Context Provider
# ──────────────────────────────────────────────────────────


class TestContextProviderIntegration:
    async def test_react_agent_with_provider(self) -> None:
        """Full ReActAgent run with a context provider injecting content."""

        @tool(name="greet", description="Greet someone")
        async def greet_tool(name: str) -> str:
            return f"Hello, {name}!"

        tool_call = ToolCall(id="tc1", name="greet", arguments={"name": "World"})
        responses = [
            make_response(content="greeting", tool_calls=[tool_call]),
            make_response(content="Done greeting"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="Be helpful.",
            tools=[greet_tool],
            context_providers=[StaticProvider("Remember: be polite", priority=0, protected=True)],
        )

        result = await agent.run("Greet the world")

        assert result.output == "Done greeting"
        assert result.termination_reason == "complete"

        # Verify context was injected in both LLM calls
        wrapped = _wrap("Remember: be polite", priority=0, protected=True)
        for call in client.calls:
            sent = call["messages"]
            context_msgs = [m for m in sent if m.content == wrapped]
            assert len(context_msgs) == 1

    async def test_provider_receives_current_messages(self) -> None:
        """Provider's provide() receives the current message list."""
        client = MockLLMClient([make_response(content="answer")])
        emitter = make_emitter()
        tracking = TrackingProvider("ctx")
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            context_providers=[tracking],
        )

        await agent.run("hello")

        assert len(tracking.calls) == 1
        assert len(tracking.calls[0]) == 1
        assert tracking.calls[0][0].content == "hello"


# ──────────────────────────────────────────────────────────
# Truncation with Protected Messages
# ──────────────────────────────────────────────────────────


class TestTruncationProtectedMessages:
    def test_protected_message_survives_truncation(self) -> None:
        """A group containing a protected message is never truncated."""
        policy = TruncationPolicy(preserve_first=True, preserve_recent=1)
        counter = EstimateTokenCounter()

        # Use 40-char content so each message = 4 + 10 = 14 tokens
        groups: list[list[Message]] = [
            [_msg("a" * 40)],  # group 0: preserved (first)
            [_msg("b" * 40)],  # group 1: expendable
            [
                Message(
                    role="user",
                    content="c" * 40,  # group 2: protected via metadata
                    metadata={"protected": True},
                )
            ],
            [_msg("d" * 40)],  # group 3: expendable
            [_msg("e" * 40, role="assistant")],  # group 4: preserved (recent)
        ]

        # 5 groups * 14 = 70 total. Budget for 3 groups = 42
        # Protected: 0 (first), 2 (metadata), 4 (recent) = 42 tokens exactly
        # Expendable: 1, 3 — no room
        result = policy.truncate(groups, token_budget=42, counter=counter)

        assert len(result) == 3
        assert result[0] is groups[0]
        assert result[1] is groups[2]
        assert result[2] is groups[4]

    def test_unprotected_metadata_does_not_protect(self) -> None:
        """Messages with metadata.protected=False are still expendable."""
        policy = TruncationPolicy(preserve_first=False, preserve_recent=1)
        counter = EstimateTokenCounter()

        # 40-char content: each = 14 tokens
        groups: list[list[Message]] = [
            [Message(role="user", content="a" * 40, metadata={"protected": False})],
            [_msg("b" * 40)],
            [_msg("c" * 40, role="assistant")],
        ]

        # Budget for only 1 group (14 tokens)
        result = policy.truncate(groups, token_budget=14, counter=counter)
        assert len(result) == 1
        assert result[0] is groups[2]

    def test_no_metadata_is_expendable(self) -> None:
        """Messages without metadata are treated as expendable (backward compat)."""
        policy = TruncationPolicy(preserve_first=False, preserve_recent=1)
        counter = EstimateTokenCounter()

        # 40-char content: each = 14 tokens
        groups: list[list[Message]] = [
            [_msg("a" * 40)],
            [_msg("b" * 40, role="assistant")],
        ]

        # Budget for only 1 group (14 tokens)
        result = policy.truncate(groups, token_budget=14, counter=counter)
        assert len(result) == 1
        assert result[0] is groups[1]

    def test_multiple_protected_groups_all_survive(self) -> None:
        """Multiple groups with protected messages all survive."""
        policy = TruncationPolicy(preserve_first=False, preserve_recent=1)
        counter = EstimateTokenCounter()

        # 40-char content: each = 14 tokens
        groups: list[list[Message]] = [
            [Message(role="user", content="a" * 40, metadata={"protected": True})],
            [_msg("b" * 40)],
            [Message(role="user", content="c" * 40, metadata={"protected": True})],
            [_msg("d" * 40, role="assistant")],
        ]

        # Budget for 3 groups (42 tokens). Protected: groups 0, 2, 3.
        result = policy.truncate(groups, token_budget=42, counter=counter)

        assert len(result) == 3
        assert result[0] is groups[0]
        assert result[1] is groups[2]
        assert result[2] is groups[3]

    def test_protected_over_budget_still_kept(self) -> None:
        """When protected groups alone exceed budget, they're all still returned."""
        policy = TruncationPolicy(preserve_first=True, preserve_recent=1)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("a" * 40)],  # protected (first), 14 tokens
            [Message(role="user", content="b" * 200, metadata={"protected": True})],  # protected (metadata), 54 tokens
            [_msg("c" * 40, role="assistant")],  # protected (recent), 14 tokens
        ]

        # Very small budget — protected groups (82 tokens) exceed it
        result = policy.truncate(groups, token_budget=10, counter=counter)

        # All groups are protected, so all are returned despite exceeding budget
        assert len(result) == 3
