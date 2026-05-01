from nanitics.composition.multi_agent.context_transfer import (
    ContextTransferStrategy,
    CustomTransfer,
    RawOutputTransfer,
    SummaryTransfer,
    TrajectoryTransfer,
)
from nanitics.core.agents.base import AgentResult
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, Message, ToolCall
from nanitics.infrastructure.observability.events import Usage


def _make_result(
    output: str | None = "final answer",
    messages: list[Message] | None = None,
) -> AgentResult:
    return AgentResult(
        output=output,
        total_steps=1,
        termination_reason="completed",
        messages=messages or [],
        usage=Usage(input_tokens=10, output_tokens=5),
    )


class TestRawOutputTransfer:
    async def test_returns_output(self):
        result = _make_result(output="hello world")
        assert await RawOutputTransfer().extract(result) == "hello world"

    async def test_returns_empty_when_none(self):
        result = _make_result(output=None)
        assert await RawOutputTransfer().extract(result) == ""

    def test_satisfies_protocol(self):
        assert isinstance(RawOutputTransfer(), ContextTransferStrategy)


class TestTrajectoryTransfer:
    async def test_formats_messages(self):
        messages = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="The answer is 4."),
        ]
        result = _make_result(messages=messages)
        text = await TrajectoryTransfer().extract(result)
        assert "USER: What is 2+2?" in text
        assert "ASSISTANT: The answer is 4." in text

    async def test_formats_tool_calls(self):
        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="tc1", name="calc", arguments={"expr": "2+2"})],
            ),
            Message(role="tool_result", content="4"),
        ]
        result = _make_result(messages=messages)
        text = await TrajectoryTransfer().extract(result)
        assert "ASSISTANT [tool_call]: calc(" in text
        assert "TOOL_RESULT: 4" in text

    async def test_empty_messages(self):
        result = _make_result(messages=[])
        assert await TrajectoryTransfer().extract(result) == ""

    def test_satisfies_protocol(self):
        assert isinstance(TrajectoryTransfer(), ContextTransferStrategy)


class TestSummaryTransfer:
    async def test_summarizes_trajectory(self):
        mock = MockLLMClient(
            responses=[
                LLMResponse(
                    content="Summary: user asked 2+2, answer is 4.",
                    usage=Usage(input_tokens=50, output_tokens=20),
                    model="mock",
                    stop_reason="end_turn",
                ),
            ]
        )
        messages = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="The answer is 4."),
        ]
        result = _make_result(messages=messages)
        transfer = SummaryTransfer(llm_client=mock)
        text = await transfer.extract(result)
        assert text == "Summary: user asked 2+2, answer is 4."
        assert len(mock.calls) == 1
        assert "USER: What is 2+2?" in mock.calls[0]["messages"][0].content

    async def test_returns_empty_on_none_content(self):
        mock = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    usage=Usage(input_tokens=10, output_tokens=0),
                    model="mock",
                    stop_reason="end_turn",
                ),
            ]
        )
        result = _make_result(messages=[])
        transfer = SummaryTransfer(llm_client=mock)
        text = await transfer.extract(result)
        assert text == ""


class TestCustomTransfer:
    async def test_custom_function(self):
        result = _make_result(output="raw output")
        transfer = CustomTransfer(fn=lambda r: f"Custom: {r.output}")
        assert await transfer.extract(result) == "Custom: raw output"

    def test_satisfies_protocol(self):
        assert isinstance(CustomTransfer(fn=lambda r: ""), ContextTransferStrategy)
