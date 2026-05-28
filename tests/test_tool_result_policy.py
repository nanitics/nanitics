"""Tests for ToolResultPolicy and its three default impls."""

from typing import Any

import pytest

from nanitics.capabilities.context.token_counter import EstimateTokenCounter
from nanitics.capabilities.context.tool_result import (
    DEFAULT_TOOL_SUMMARY_PROMPT,
    ErrorOnLargeToolResult,
    SummarizeToolResult,
    ToolResultContext,
    ToolResultPolicy,
    TruncateToolResult,
)
from nanitics.infrastructure.errors import ToolResultTooLargeError
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, ToolCall
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    ToolResultPolicyAppliedEvent,
    Usage,
)
from nanitics.strategies.tools.protocol import ToolResult


def _ctx(emitter: InMemoryEmitter | None = None, *, name: str = "search") -> ToolResultContext:
    return ToolResultContext(
        tool_call=ToolCall(id="tc-1", name=name, arguments={"q": "test"}),
        token_counter=EstimateTokenCounter(chars_per_token=4.0),
        emitter=emitter,
    )


def _events_of_type(emitter: InMemoryEmitter, event_type: type) -> list[Any]:
    return [e for e in emitter.events if isinstance(e, event_type)]


# --- Protocol surface ---


class TestProtocol:
    def test_runtime_checkable(self) -> None:
        assert isinstance(ErrorOnLargeToolResult(max_tokens=10), ToolResultPolicy)
        assert isinstance(TruncateToolResult(max_tokens=10), ToolResultPolicy)
        assert isinstance(
            SummarizeToolResult(max_tokens=10, llm_client=MockLLMClient(responses=[])),
            ToolResultPolicy,
        )

    def test_context_is_frozen(self) -> None:
        from pydantic import ValidationError

        ctx = _ctx()
        with pytest.raises(ValidationError):
            ctx.emitter = InMemoryEmitter(trace_id="x")  # type: ignore[misc]

    def test_default_prompt_constant_is_a_nonempty_str(self) -> None:
        assert isinstance(DEFAULT_TOOL_SUMMARY_PROMPT, str)
        assert "Summarize" in DEFAULT_TOOL_SUMMARY_PROMPT


# --- ErrorOnLargeToolResult ---


class TestErrorOnLargeToolResult:
    async def test_under_budget_passes_through(self) -> None:
        policy = ErrorOnLargeToolResult(max_tokens=100)
        result = ToolResult(content="hello")
        out = await policy.apply(result, _ctx())
        assert out is result

    async def test_over_budget_raises(self) -> None:
        policy = ErrorOnLargeToolResult(max_tokens=2)
        result = ToolResult(content="x" * 100)  # ~25 tokens
        with pytest.raises(ToolResultTooLargeError) as exc:
            await policy.apply(result, _ctx())
        assert exc.value.tool_name == "search"
        assert exc.value.max_tokens == 2
        assert exc.value.result_tokens >= 2

    async def test_emits_event_before_raising(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        policy = ErrorOnLargeToolResult(max_tokens=2)
        result = ToolResult(content="x" * 100)
        with pytest.raises(ToolResultTooLargeError):
            await policy.apply(result, _ctx(emitter))
        events = _events_of_type(emitter, ToolResultPolicyAppliedEvent)
        assert len(events) == 1
        assert events[0].action == "errored"
        assert events[0].final_tokens == 0
        assert events[0].error is not None

    async def test_no_event_when_no_emitter(self) -> None:
        policy = ErrorOnLargeToolResult(max_tokens=2)
        with pytest.raises(ToolResultTooLargeError):
            await policy.apply(ToolResult(content="x" * 100), _ctx())

    async def test_no_event_on_pass_through(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        policy = ErrorOnLargeToolResult(max_tokens=100)
        await policy.apply(ToolResult(content="ok"), _ctx(emitter))
        assert _events_of_type(emitter, ToolResultPolicyAppliedEvent) == []

    def test_reset_is_a_noop(self) -> None:
        policy = ErrorOnLargeToolResult(max_tokens=10)
        policy.reset()
        policy.reset()


# --- TruncateToolResult ---


class TestTruncateToolResult:
    async def test_under_budget_passes_through(self) -> None:
        policy = TruncateToolResult(max_tokens=100)
        result = ToolResult(content="hello")
        out = await policy.apply(result, _ctx())
        assert out is result

    async def test_tail_only_when_no_head_tokens(self) -> None:
        policy = TruncateToolResult(max_tokens=10)
        content = "HEAD" + ("body " * 200) + "TAIL"
        result = ToolResult(content=content, metadata={"keep": "yes"})
        out = await policy.apply(result, _ctx())
        assert out.content.startswith("[…truncated…]")
        assert out.content.endswith("TAIL")
        assert out.metadata["truncated"] is True
        assert out.metadata["original_tokens"] > 10
        assert out.metadata["keep"] == "yes"  # preserves original metadata

    async def test_head_and_tail_when_head_tokens_set(self) -> None:
        policy = TruncateToolResult(max_tokens=20, head_tokens=2)
        content = "HEAD" + ("middle " * 200) + "TAIL"
        result = ToolResult(content=content)
        out = await policy.apply(result, _ctx())
        assert out.content.startswith("HEAD")
        assert out.content.endswith("TAIL")
        assert "[…truncated…]" in out.content

    async def test_marker_too_large_falls_back_to_flat_tail(self) -> None:
        # marker_tokens >= max_tokens triggers the no-marker fallback path
        policy = TruncateToolResult(max_tokens=1, marker="this marker is huge")
        content = "x" * 1000
        out = await policy.apply(ToolResult(content=content), _ctx())
        assert "this marker is huge" not in out.content
        assert out.content.endswith("x")
        assert out.metadata["truncated"] is True

    async def test_head_only_when_tail_budget_zero(self) -> None:
        # head_tokens fills the entire budget — tail slice is empty
        policy = TruncateToolResult(max_tokens=2, head_tokens=2, marker="|M|")
        content = "ABCD" + ("z" * 200)
        out = await policy.apply(ToolResult(content=content), _ctx())
        assert out.content.startswith("AB")
        assert out.content.endswith("|M|")

    async def test_empty_content_pass_through(self) -> None:
        # Even at budget 0, an under-budget content passes; the count_text
        # floor is 1 so a single char is always counted >= 1
        policy = TruncateToolResult(max_tokens=10)
        out = await policy.apply(ToolResult(content=""), _ctx())
        assert out.content == ""

    async def test_emits_event_when_truncating(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        policy = TruncateToolResult(max_tokens=5)
        await policy.apply(ToolResult(content="x" * 1000), _ctx(emitter))
        events = _events_of_type(emitter, ToolResultPolicyAppliedEvent)
        assert len(events) == 1
        assert events[0].action == "truncated"
        assert events[0].original_tokens > 5

    async def test_no_event_on_pass_through(self) -> None:
        emitter = InMemoryEmitter(trace_id="t")
        policy = TruncateToolResult(max_tokens=100)
        await policy.apply(ToolResult(content="hi"), _ctx(emitter))
        assert _events_of_type(emitter, ToolResultPolicyAppliedEvent) == []

    def test_reset_is_a_noop(self) -> None:
        TruncateToolResult(max_tokens=10).reset()


# --- SummarizeToolResult ---


def _summary_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
        model="m",
        stop_reason="end_turn",
    )


class TestSummarizeToolResult:
    async def test_under_budget_passes_through(self) -> None:
        client = MockLLMClient(responses=[])
        policy = SummarizeToolResult(max_tokens=100, llm_client=client)
        result = ToolResult(content="hello")
        out = await policy.apply(result, _ctx())
        assert out is result
        assert client.calls == []

    async def test_success_path_returns_summary(self) -> None:
        client = MockLLMClient(responses=[_summary_response("short summary")])
        policy = SummarizeToolResult(max_tokens=100, llm_client=client)
        emitter = InMemoryEmitter(trace_id="t")
        out = await policy.apply(
            ToolResult(content="x" * 1000, metadata={"keep": "yes"}),
            _ctx(emitter),
        )
        assert out.content == "short summary"
        assert out.metadata["summarized"] is True
        assert out.metadata["original_tokens"] > 100
        assert out.metadata["keep"] == "yes"
        events = _events_of_type(emitter, ToolResultPolicyAppliedEvent)
        assert len(events) == 1 and events[0].action == "summarized"
        # LLM was called with the documented shape
        assert len(client.calls) == 1
        assert client.calls[0]["system_prompt"] == DEFAULT_TOOL_SUMMARY_PROMPT

    async def test_custom_summary_prompt(self) -> None:
        client = MockLLMClient(responses=[_summary_response("S")])
        policy = SummarizeToolResult(max_tokens=100, llm_client=client, summary_prompt="CUSTOM")
        await policy.apply(ToolResult(content="x" * 1000), _ctx())
        assert client.calls[0]["system_prompt"] == "CUSTOM"

    async def test_fallback_when_llm_raises(self) -> None:
        # MockLLMClient raises ValueError when responses are exhausted; we
        # configure it with no responses so the first call raises.
        client = MockLLMClient(responses=[])
        policy = SummarizeToolResult(max_tokens=5, llm_client=client)
        emitter = InMemoryEmitter(trace_id="t")
        out = await policy.apply(ToolResult(content="x" * 1000), _ctx(emitter))
        assert out.metadata["summarized"] is False
        assert out.metadata["truncated"] is True
        assert out.metadata["fell_back"] is True
        events = _events_of_type(emitter, ToolResultPolicyAppliedEvent)
        assert len(events) == 1
        assert events[0].action == "truncated"
        assert events[0].fell_back is True
        assert events[0].error is not None

    async def test_fallback_when_summary_still_over_budget(self) -> None:
        # Summary is huge — still exceeds the budget. Falls back.
        client = MockLLMClient(responses=[_summary_response("y" * 1000)])
        policy = SummarizeToolResult(max_tokens=5, llm_client=client)
        emitter = InMemoryEmitter(trace_id="t")
        out = await policy.apply(ToolResult(content="x" * 1000), _ctx(emitter))
        assert out.metadata["fell_back"] is True
        assert out.metadata["truncated"] is True
        events = _events_of_type(emitter, ToolResultPolicyAppliedEvent)
        assert len(events) == 1
        assert events[0].fell_back is True
        assert events[0].error is not None and "still exceeds budget" in events[0].error

    async def test_fallback_when_llm_raises_with_empty_message(self) -> None:
        # Exercises the `str(exc) or type(exc).__name__` branch.
        class BlankError(Exception):
            pass

        class _RaisingClient:
            model = None

            async def generate(self, **kwargs: Any) -> LLMResponse:
                raise BlankError()

        policy = SummarizeToolResult(max_tokens=5, llm_client=_RaisingClient())  # type: ignore[arg-type]
        emitter = InMemoryEmitter(trace_id="t")
        out = await policy.apply(ToolResult(content="x" * 1000), _ctx(emitter))
        assert out.metadata["fell_back"] is True
        events = _events_of_type(emitter, ToolResultPolicyAppliedEvent)
        assert events[0].error == "BlankError"

    def test_reset_is_a_noop(self) -> None:
        SummarizeToolResult(max_tokens=10, llm_client=MockLLMClient(responses=[])).reset()
