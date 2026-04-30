from typing import Literal

from nanitics.capabilities.context.grouping import default_message_grouper
from nanitics.infrastructure.llm.protocol import Message, ToolCall


def _msg(
    role: Literal["user", "assistant", "tool_result"],
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> Message:
    return Message(role=role, content=content, tool_calls=tool_calls)


def _tc(name: str = "tool", tc_id: str = "tc1") -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments={})


class TestDefaultMessageGrouper:
    def test_empty_list(self) -> None:
        assert default_message_grouper([]) == []

    def test_messages_without_tool_calls(self) -> None:
        msgs = [
            _msg("user", "hello"),
            _msg("assistant", "hi"),
            _msg("user", "bye"),
        ]
        groups = default_message_grouper(msgs)
        assert len(groups) == 3
        assert groups[0] == [msgs[0]]
        assert groups[1] == [msgs[1]]
        assert groups[2] == [msgs[2]]

    def test_react_sequence(self) -> None:
        """assistant+tool_call followed by tool_result forms one group."""
        msgs = [
            _msg("user", "task"),
            _msg("assistant", "thinking", tool_calls=[_tc("search", "tc1")]),
            _msg("tool_result", "result1", tool_calls=None),
            _msg("assistant", "done"),
        ]
        groups = default_message_grouper(msgs)
        assert len(groups) == 3
        assert groups[0] == [msgs[0]]
        assert groups[1] == [msgs[1], msgs[2]]
        assert groups[2] == [msgs[3]]

    def test_multiple_tool_results_per_assistant(self) -> None:
        """Multiple tool_results attach to the same preceding group."""
        msgs = [
            _msg("user", "task"),
            _msg("assistant", "calling tools", tool_calls=[_tc("a", "tc1"), _tc("b", "tc2")]),
            _msg("tool_result", "result_a"),
            _msg("tool_result", "result_b"),
            _msg("assistant", "final"),
        ]
        groups = default_message_grouper(msgs)
        assert len(groups) == 3
        assert groups[0] == [msgs[0]]
        assert groups[1] == [msgs[1], msgs[2], msgs[3]]
        assert groups[2] == [msgs[4]]

    def test_tool_result_at_start(self) -> None:
        """A tool_result at the start (no preceding group) becomes its own group."""
        msgs = [
            _msg("tool_result", "orphan"),
            _msg("user", "hello"),
        ]
        groups = default_message_grouper(msgs)
        assert len(groups) == 2
        assert groups[0] == [msgs[0]]
        assert groups[1] == [msgs[1]]

    def test_multiple_react_iterations(self) -> None:
        """Multiple tool-use iterations each form their own group."""
        msgs = [
            _msg("user", "task"),
            _msg("assistant", "step1", tool_calls=[_tc("search", "tc1")]),
            _msg("tool_result", "r1"),
            _msg("assistant", "step2", tool_calls=[_tc("read", "tc2")]),
            _msg("tool_result", "r2"),
            _msg("assistant", "done"),
        ]
        groups = default_message_grouper(msgs)
        assert len(groups) == 4
        assert groups[0] == [msgs[0]]
        assert groups[1] == [msgs[1], msgs[2]]
        assert groups[2] == [msgs[3], msgs[4]]
        assert groups[3] == [msgs[5]]

    def test_single_message(self) -> None:
        msgs = [_msg("user", "hi")]
        groups = default_message_grouper(msgs)
        assert len(groups) == 1
        assert groups[0] == [msgs[0]]

    def test_reasoning_only_sequence(self) -> None:
        """User/assistant alternation without tools — each is its own group."""
        msgs = [
            _msg("user", "question"),
            _msg("assistant", "answer"),
            _msg("user", "followup"),
            _msg("assistant", "response"),
        ]
        groups = default_message_grouper(msgs)
        assert len(groups) == 4
        for i, group in enumerate(groups):
            assert group == [msgs[i]]
