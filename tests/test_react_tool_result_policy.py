"""ReAct integration tests for ToolResultPolicy + error-handler composition."""

from nanitics.context import (
    ErrorOnLargeToolResult,
    SummarizeToolResult,
    TruncateToolResult,
)
from nanitics.infrastructure import LLMResponse, MockLLMClient
from nanitics.infrastructure.observability.events import (
    ToolResultPolicyAppliedEvent,
)
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter, make_response, make_usage


@tool(name="dump", description="Returns a large dump")
async def dump_tool() -> str:
    return "BIG" + ("payload " * 1000)


def _summary_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=make_usage(),
        model="m",
        stop_reason="end_turn",
    )


class TestReActWithToolResultPolicy:
    async def test_error_policy_surfaces_to_llm_via_error_handler(self) -> None:
        """ErrorOnLargeToolResult routes through ErrorHandler as a correction prompt."""
        tc = ToolCall(id="tc1", name="dump", arguments={})
        # 1st call: ask to dump (over budget → ToolResultTooLargeError).
        # The ErrorHandler catches the ToolError, emits a correction
        # prompt, and the LLM is given another turn — answer "ok".
        responses = [
            make_response(content="dumping", tool_calls=[tc]),
            make_response(content="ok"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react",
            llm_client=client,
            emitter=emitter,
            system_prompt="be terse",
            tools=[dump_tool],
            tool_result_policy=ErrorOnLargeToolResult(max_tokens=20),
        )
        result = await agent.run("dump it")
        assert result.output == "ok"
        # The tool_result message reflects the error correction surface
        tool_results = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_results) == 1
        # The policy emitted an "errored" event
        events = [e for e in emitter.events if isinstance(e, ToolResultPolicyAppliedEvent)]
        assert len(events) == 1
        assert events[0].action == "errored"

    async def test_truncate_policy_content_appears_in_messages(self) -> None:
        tc = ToolCall(id="tc1", name="dump", arguments={})
        responses = [
            make_response(content="dumping", tool_calls=[tc]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react",
            llm_client=client,
            emitter=emitter,
            system_prompt="be terse",
            tools=[dump_tool],
            tool_result_policy=TruncateToolResult(max_tokens=10),
        )
        result = await agent.run("dump it")
        tool_results = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_results) == 1
        content = tool_results[0].content
        assert isinstance(content, str)
        assert "[…truncated…]" in content
        assert tool_results[0].metadata.get("truncated") is True
        events = [e for e in emitter.events if isinstance(e, ToolResultPolicyAppliedEvent)]
        assert len(events) == 1
        assert events[0].action == "truncated"

    async def test_summarize_policy_summary_appears_in_messages(self) -> None:
        tc = ToolCall(id="tc1", name="dump", arguments={})
        # The summary LLM client is separate from the agent's LLM client.
        summary_llm = MockLLMClient([_summary_response("3 lines, ok")])
        responses = [
            make_response(content="dumping", tool_calls=[tc]),
            make_response(content="done"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="react",
            llm_client=client,
            emitter=emitter,
            system_prompt="be terse",
            tools=[dump_tool],
            tool_result_policy=SummarizeToolResult(max_tokens=50, llm_client=summary_llm),
        )
        result = await agent.run("dump it")
        tool_results = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0].content == "3 lines, ok"
        assert tool_results[0].metadata.get("summarized") is True
        events = [e for e in emitter.events if isinstance(e, ToolResultPolicyAppliedEvent)]
        assert len(events) == 1
        assert events[0].action == "summarized"


class TestPolicyResetPerAgent:
    """Confirms policy.reset() is called by each agent type on each run."""

    async def test_react_resets_policy(self) -> None:
        calls: list[str] = []

        class _Spy:
            async def apply(self, result, context):  # type: ignore[no-untyped-def]
                return result

            def reset(self) -> None:
                calls.append("reset")

        responses = [make_response(content="hi")]
        agent = ReActAgent(
            name="r",
            llm_client=MockLLMClient(responses),
            emitter=make_emitter(),
            system_prompt="s",
            tools=[],
            tool_result_policy=_Spy(),
        )
        await agent.run("go")
        assert calls == ["reset"]
