from typing import Literal

import pytest

from nanitics.capabilities.context.manager import ContextManager, ContextUsage
from nanitics.capabilities.context.summarization import SummarizationPolicy
from nanitics.capabilities.context.token_counter import EstimateTokenCounter
from nanitics.capabilities.context.truncation import TruncationPolicy
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, Message, ToolCall, ToolSchema
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    ContextSummarizationEvent,
    ContextTruncationEvent,
    RemovedMessageInfo,
    Usage,
)


def _msg(content: str, role: Literal["user", "assistant"] = "user") -> Message:
    return Message(role=role, content=content)


def _make_messages(count: int, char_length: int = 40) -> list[Message]:
    roles: list[Literal["user", "assistant"]] = ["user", "assistant"]
    return [_msg("x" * char_length, role=roles[i % 2]) for i in range(count)]


class TestContextManagerInit:
    def test_requires_truncation_or_summarization(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ContextManager(context_limit=10000)

    def test_rejects_non_positive_context_limit(self) -> None:
        with pytest.raises(ValueError, match=r"context_limit.*must be positive"):
            ContextManager(context_limit=0, truncation=TruncationPolicy())
        with pytest.raises(ValueError, match=r"context_limit.*must be positive"):
            ContextManager(context_limit=-1, truncation=TruncationPolicy())

    def test_rejects_invalid_reserve_tokens(self) -> None:
        with pytest.raises(ValueError, match="reserve_tokens"):
            ContextManager(context_limit=100, reserve_tokens=-1, truncation=TruncationPolicy())
        with pytest.raises(ValueError, match="reserve_tokens"):
            ContextManager(context_limit=100, reserve_tokens=100, truncation=TruncationPolicy())
        with pytest.raises(ValueError, match="reserve_tokens"):
            ContextManager(context_limit=100, reserve_tokens=200, truncation=TruncationPolicy())

    def test_rejects_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ContextManager(context_limit=10000, threshold=0.0, truncation=TruncationPolicy())
        with pytest.raises(ValueError, match="threshold"):
            ContextManager(context_limit=10000, threshold=-0.1, truncation=TruncationPolicy())
        with pytest.raises(ValueError, match="threshold"):
            ContextManager(context_limit=10000, threshold=1.1, truncation=TruncationPolicy())


class TestContextUsage:
    def test_current_usage_returns_correct_breakdown(self) -> None:
        manager = ContextManager(
            context_limit=10000,
            reserve_tokens=1000,
            truncation=TruncationPolicy(),
        )
        messages = _make_messages(3, char_length=40)
        usage = manager.current_usage(
            system_prompt="System prompt here",
            messages=messages,
        )
        assert isinstance(usage, ContextUsage)
        assert usage.context_limit == 10000
        assert usage.system_tokens > 0
        assert usage.message_tokens > 0
        assert usage.tools_tokens == 0
        assert usage.total_tokens == usage.system_tokens + usage.message_tokens
        assert usage.available_tokens == 9000 - usage.total_tokens
        assert usage.utilization == pytest.approx(usage.total_tokens / 9000, rel=1e-3)

    def test_current_usage_with_tools(self) -> None:
        manager = ContextManager(
            context_limit=10000,
            reserve_tokens=1000,
            truncation=TruncationPolicy(),
        )
        tools = [
            ToolSchema(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {}},
            )
        ]
        usage = manager.current_usage(
            system_prompt="System",
            messages=[_msg("hello")],
            tools=tools,
        )
        assert usage.tools_tokens > 0


class TestContextManagerPrepare:
    async def test_returns_messages_unchanged_when_under_threshold(self) -> None:
        manager = ContextManager(
            context_limit=100000,
            reserve_tokens=1000,
            truncation=TruncationPolicy(),
        )
        messages = _make_messages(3)
        result = await manager.prepare("System", messages, None, None)
        assert result == messages

    async def test_applies_truncation_when_over_threshold(self) -> None:
        counter = EstimateTokenCounter()
        # Each message ~14 tokens (4 + 40/4). 10 messages = ~140 tokens.
        # context_limit=100, reserve=10 → available=90, threshold triggers at 81.
        # message_budget = 90 - system(1) = 89. But threshold exceeded at 141 > 81.
        # With message_budget=60, truncation must drop messages.
        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        messages = _make_messages(10)
        result = await manager.prepare("Sys", messages, None, None)
        assert len(result) < len(messages)
        # First message preserved
        assert result[0] is messages[0]
        # Last 2 preserved
        assert result[-1] is messages[-1]
        assert result[-2] is messages[-2]

    async def test_emits_truncation_event(self) -> None:
        counter = EstimateTokenCounter()
        emitter = InMemoryEmitter(trace_id="test-trace")
        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        messages = _make_messages(10)
        with emitter.span("test"):
            await manager.prepare("Sys", messages, None, emitter)

        truncation_events = [e for e in emitter.events if isinstance(e, ContextTruncationEvent)]
        assert len(truncation_events) == 1
        evt = truncation_events[0]
        assert evt.messages_before == 10
        assert evt.messages_after < 10
        assert evt.tokens_before > evt.tokens_after

    async def test_truncation_event_includes_removed_messages(self) -> None:
        counter = EstimateTokenCounter()
        emitter = InMemoryEmitter(trace_id="test-trace")
        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        messages = _make_messages(10)
        with emitter.span("test"):
            await manager.prepare("Sys", messages, None, emitter)

        truncation_events = [e for e in emitter.events if isinstance(e, ContextTruncationEvent)]
        assert len(truncation_events) == 1
        evt = truncation_events[0]
        removed_count = evt.messages_before - evt.messages_after
        assert len(evt.removed_messages) == removed_count
        assert all(isinstance(r, RemovedMessageInfo) for r in evt.removed_messages)
        for r in evt.removed_messages:
            assert r.role in ("user", "assistant")
            assert 0 <= r.original_index < 10

    async def test_no_event_when_truncation_removes_nothing(self) -> None:
        """Reproducer for the no-op emission defect.

        With ``context_limit=2000``, ``reserve_tokens=200``, ``threshold=0.8``:
        ``available=1800``, threshold fires at ``1440``. 15 messages of 400
        chars each produce ~1560 message tokens — over the threshold but
        under the 1799 message budget (system prompt ~1 token). Truncation
        runs but removes nothing; no ``ContextTruncationEvent`` should fire.
        """
        counter = EstimateTokenCounter()
        emitter = InMemoryEmitter(trace_id="test-trace")
        manager = ContextManager(
            context_limit=2000,
            reserve_tokens=200,
            threshold=0.8,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        messages = _make_messages(15, char_length=400)

        # Confirm the reproducer window: above threshold, below message budget.
        system_tokens = counter.count_text("Sys")
        message_tokens = sum(4 + counter.count_text(m.content) for m in messages if isinstance(m.content, str))
        available = 2000 - 200
        assert system_tokens + message_tokens > available * 0.8
        assert system_tokens + message_tokens <= available

        with emitter.span("test"):
            result = await manager.prepare("Sys", messages, None, emitter)

        truncation_events = [e for e in emitter.events if isinstance(e, ContextTruncationEvent)]
        assert truncation_events == []
        # Returned messages are unchanged: same length, same objects, same order.
        assert len(result) == len(messages)
        for original, returned in zip(messages, result, strict=True):
            assert returned is original

    async def test_original_messages_not_modified(self) -> None:
        counter = EstimateTokenCounter()
        manager = ContextManager(
            context_limit=200,
            reserve_tokens=20,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        messages = _make_messages(10)
        original_count = len(messages)
        await manager.prepare("Sys", messages, None, None)
        assert len(messages) == original_count

    async def test_fallthrough_when_truncation_does_not_resolve(self) -> None:
        """When truncation can't bring total under budget and no summarization, return grouped messages."""
        counter = EstimateTokenCounter()
        # 3 long messages (~50 tokens each = ~150 total), budget 49.
        # Truncation keeps first + last 1 (~100 tokens), still over budget.
        # No summarization → falls through to return flatten_groups(groups).
        manager = ContextManager(
            context_limit=50,
            reserve_tokens=0,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=1),
        )
        messages = _make_messages(3, char_length=200)
        result = await manager.prepare("S", messages, None, None)
        # Truncation removed middle message but still over budget; no summarization
        assert len(result) < len(messages)

    def test_reset_does_not_raise(self) -> None:
        manager = ContextManager(
            context_limit=10000,
            truncation=TruncationPolicy(),
        )
        manager.reset()  # Should not raise

    def test_reset_clears_summarization_state(self) -> None:
        client = MockLLMClient(responses=[])
        manager = ContextManager(
            context_limit=10000,
            summarization=SummarizationPolicy(llm_client=client),
        )
        manager.reset()  # exercises summarization.reset() path

    async def test_tools_reduce_message_budget(self) -> None:
        """Tool token counts reduce the message budget, causing more truncation."""
        counter = EstimateTokenCounter()
        # 6 messages of 40 chars each → ~14 tokens each → ~84 message tokens
        # System "Sys" ≈ 1 token
        messages = _make_messages(6)

        # Without tools: context_limit=120, reserve=10 → available=110, threshold=0.5.
        # Total ≈ 85 > 55 threshold → triggers. message_budget = 110 - 1 = 109.
        # All 6 groups fit in 109.
        manager_no_tools = ContextManager(
            context_limit=120,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=1),
        )
        result_no_tools = await manager_no_tools.prepare("Sys", messages, None, None)

        # With tools: same setup but tools consume ~30 tokens.
        # message_budget = 110 - 1 - 30 = 79. Still fits all 6 (~84 > 79 is borderline).
        # Use tighter limit: context_limit=100, reserve=10 → available=90.
        # message_budget = 90 - 1 - 30 = 59. Only 4 groups fit in 59.
        tools = [
            ToolSchema(
                name="search_web",
                description="Search the web for information",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ]
        manager_with_tools = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=1),
        )
        result_with_tools = await manager_with_tools.prepare("Sys", messages, tools, None)

        # Tools should cause more messages to be dropped
        assert len(result_with_tools) < len(result_no_tools)


def _make_summary_response(summary: str) -> LLMResponse:
    return LLMResponse(
        content=summary,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
        model="test-model",
        stop_reason="end_turn",
    )


class TestContextManagerSummarization:
    async def test_prepare_with_summarization_only(self) -> None:
        counter = EstimateTokenCounter()
        client = MockLLMClient(responses=[_make_summary_response("Summarized conversation.")])
        summarization = SummarizationPolicy(llm_client=client)
        # 10 messages of 40 chars each → ~14 tokens each → ~140 message tokens
        # context_limit=100, reserve=10 → available=90, threshold=0.5 → triggers at 45
        # System "Sys" ≈ 1 token. Total ≈ 141 > 45, so management triggers.
        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            summarization=summarization,
        )
        messages = _make_messages(10)
        result = await manager.prepare("Sys", messages, None, None)

        # Should have first message + summary + 2 preserved recent groups
        assert len(result) < len(messages)
        # First message preserved (preserve_first=True default)
        assert result[0] is messages[0]
        # Summary is the second message
        assert result[1].content is not None
        assert "[Summary of prior conversation]" in result[1].content

    async def test_combined_truncation_then_summarization(self) -> None:
        """When truncation alone can't fit under budget, summarization kicks in
        using the original (pre-truncation) groups."""
        counter = EstimateTokenCounter()
        client = MockLLMClient(responses=[_make_summary_response("Combined summary.")])
        summarization = SummarizationPolicy(llm_client=client)
        # Each message ~14 tokens (4 + 40/4). 10 messages = ~140 tokens.
        # context_limit=20, reserve=2 → available=18, threshold=0.1 → triggers at 1.8.
        # System "Sys" ~1 token. message_budget = 18 - 1 = 17.
        # Truncation preserves first + recent 1 = ~28 tokens > 17, still over budget.
        # Summarization receives original 10 groups, compresses middle ones.
        manager = ContextManager(
            context_limit=20,
            reserve_tokens=2,
            threshold=0.1,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=1),
            summarization=summarization,
        )
        messages = _make_messages(10)
        emitter = InMemoryEmitter(trace_id="test-combined")
        with emitter.span("test"):
            result = await manager.prepare("Sys", messages, None, emitter)

        # Should have applied summarization
        assert len(result) < len(messages)
        # First message preserved, summary follows
        assert result[0] is messages[0]
        assert result[1].content is not None
        assert "[Summary of prior conversation]" in result[1].content

        # Both events should have been emitted
        truncation_events = [e for e in emitter.events if isinstance(e, ContextTruncationEvent)]
        summarization_events = [e for e in emitter.events if isinstance(e, ContextSummarizationEvent)]
        assert len(truncation_events) == 1
        assert len(summarization_events) == 1
        assert summarization_events[0].summary_text == "Combined summary."

    async def test_emits_summarization_event(self) -> None:
        counter = EstimateTokenCounter()
        client = MockLLMClient(responses=[_make_summary_response("Summary.")])
        summarization = SummarizationPolicy(llm_client=client)
        emitter = InMemoryEmitter(trace_id="test-trace")
        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            summarization=summarization,
        )
        messages = _make_messages(10)
        with emitter.span("test"):
            await manager.prepare("Sys", messages, None, emitter)

        summarization_events = [e for e in emitter.events if isinstance(e, ContextSummarizationEvent)]
        assert len(summarization_events) == 1
        evt = summarization_events[0]
        assert evt.messages_summarized > 0
        assert evt.summary_tokens > 0
        assert evt.original_tokens > 0
        assert evt.summary_text == "Summary."

    async def test_reset_clears_summarization_state(self) -> None:
        counter = EstimateTokenCounter()
        client = MockLLMClient(
            responses=[
                _make_summary_response("First summary."),
                _make_summary_response("Second summary."),
            ]
        )
        summarization = SummarizationPolicy(llm_client=client)
        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            summarization=summarization,
        )
        messages = _make_messages(10)
        await manager.prepare("Sys", messages, None, None)

        manager.reset()

        # After reset, next call should do full summarization (not delta)
        await manager.prepare("Sys", messages, None, None)
        assert len(client.calls) == 2
        # Second call should not contain "Previous summary:"
        second_call_content = client.calls[1]["messages"][0].content
        assert "Previous summary:" not in second_call_content


def _make_tool_exchange_messages() -> list[Message]:
    """Create a realistic ReAct conversation with tool exchanges."""
    tc1 = ToolCall(id="tc1", name="search", arguments={"q": "topic1"})
    tc2 = ToolCall(id="tc2", name="search", arguments={"q": "topic2"})
    tc3 = ToolCall(id="tc3", name="get_article", arguments={"id": "a1"})
    return [
        Message(role="user", content="Research topic1 and topic2"),
        Message(role="assistant", content="Searching for topic1", tool_calls=[tc1]),
        Message(role="tool_result", content="Found: topic1 data " * 20, tool_call_id="tc1"),
        Message(role="assistant", content="Searching for topic2", tool_calls=[tc2]),
        Message(role="tool_result", content="Found: topic2 data " * 20, tool_call_id="tc2"),
        Message(role="assistant", content="Getting article", tool_calls=[tc3]),
        Message(role="tool_result", content="Article content " * 30, tool_call_id="tc3"),
        Message(role="assistant", content="Here is my analysis."),
    ]


class TestContextManagerGroupIntegration:
    async def test_truncation_preserves_tool_exchange_integrity(self) -> None:
        """After truncation, no tool_result message appears without its assistant+tool_call."""
        counter = EstimateTokenCounter()
        manager = ContextManager(
            context_limit=300,
            reserve_tokens=20,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        messages = _make_tool_exchange_messages()
        result = await manager.prepare("System prompt", messages, None, None)

        # Verify structural integrity: every tool_result must be preceded by
        # an assistant message with tool_calls
        for i, msg in enumerate(result):
            if msg.role == "tool_result":
                assert i > 0, "tool_result at index 0 without preceding assistant"
                preceding = result[i - 1]
                assert (preceding.role == "assistant" and preceding.tool_calls) or preceding.role == "tool_result", (
                    f"tool_result at index {i} preceded by {preceding.role} without tool_calls"
                )

    async def test_summarization_preserves_tool_exchange_integrity(self) -> None:
        """After summarization, no tool_result appears without its assistant+tool_call."""
        counter = EstimateTokenCounter()
        client = MockLLMClient(
            responses=[
                LLMResponse(
                    content="Summary of research.",
                    tool_calls=[],
                    usage=Usage(input_tokens=10, output_tokens=5),
                    model="test",
                    stop_reason="end_turn",
                ),
            ]
        )
        manager = ContextManager(
            context_limit=300,
            reserve_tokens=20,
            threshold=0.5,
            token_counter=counter,
            summarization=SummarizationPolicy(llm_client=client),
        )
        messages = _make_tool_exchange_messages()
        result = await manager.prepare("System prompt", messages, None, None)

        for i, msg in enumerate(result):
            if msg.role == "tool_result":
                assert i > 0, "tool_result at index 0 without preceding assistant"
                preceding = result[i - 1]
                assert (preceding.role == "assistant" and preceding.tool_calls) or preceding.role == "tool_result"

    async def test_custom_grouper(self) -> None:
        """Custom grouper can be provided to ContextManager."""
        counter = EstimateTokenCounter()

        # Custom grouper that puts every message in its own group
        def single_message_grouper(messages: list[Message]) -> list[list[Message]]:
            return [[m] for m in messages]

        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=1),
            grouper=single_message_grouper,
        )
        messages = _make_messages(10)
        result = await manager.prepare("Sys", messages, None, None)
        assert len(result) < len(messages)
        assert result[0] is messages[0]
        assert result[-1] is messages[-1]


class TestDeltaSummarization:
    async def test_second_prepare_uses_delta_summarization(self) -> None:
        """Second prepare() call should send 'Previous summary:' + only new messages."""
        counter = EstimateTokenCounter()
        client = MockLLMClient(
            responses=[
                _make_summary_response("First summary."),
                _make_summary_response("Updated summary."),
            ]
        )
        summarization = SummarizationPolicy(llm_client=client)
        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            summarization=summarization,
        )
        messages = _make_messages(10)
        await manager.prepare("Sys", messages, None, None)

        # Add more messages and call again
        extended = [*messages, _msg("new user message"), _msg("new response", role="assistant")]
        result = await manager.prepare("Sys", extended, None, None)

        assert len(client.calls) == 2
        # Second call should contain delta context
        second_input = client.calls[1]["messages"][0].content
        assert "Previous summary:" in second_input
        assert "First summary." in second_input
        # Result should contain the updated summary
        summary_msgs = [m for m in result if m.content and "[Summary of prior conversation]" in m.content]
        assert len(summary_msgs) == 1
        assert isinstance(summary_msgs[0].content, str)
        assert "Updated summary." in summary_msgs[0].content


class TestProtectedMessages:
    async def test_protected_message_survives_truncation(self) -> None:
        """Messages with metadata={'protected': True} survive truncation even in the expendable region."""
        counter = EstimateTokenCounter()
        # 10 messages of 40 chars each. Mark message 4 (middle, expendable) as protected.
        messages = _make_messages(10)
        protected_msg = Message(
            role=messages[4].role,
            content=messages[4].content,
            metadata={"protected": True},
        )
        messages[4] = protected_msg

        manager = ContextManager(
            context_limit=100,
            reserve_tokens=10,
            threshold=0.5,
            token_counter=counter,
            truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        )
        result = await manager.prepare("Sys", messages, None, None)

        # Some messages should have been dropped
        assert len(result) < len(messages)
        # But the protected message must survive
        assert protected_msg in result
