from typing import Literal

from nanitics.capabilities.context.summarization import SummarizationPolicy
from nanitics.capabilities.context.token_counter import EstimateTokenCounter
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import LLMResponse, Message, ToolCall
from nanitics.infrastructure.observability.events import Usage


def _msg(content: str, role: Literal["user", "assistant"] = "user") -> Message:
    return Message(role=role, content=content)


def _make_summary_response(summary: str) -> LLMResponse:
    return LLMResponse(
        content=summary,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
        model="test-model",
        stop_reason="end_turn",
    )


class TestSummarizationPolicy:
    async def test_initial_summarization_produces_summary_and_recent(self) -> None:
        client = MockLLMClient(responses=[_make_summary_response("The user asked about math.")])
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("What is 2+2?")],
            [_msg("2+2 equals 4.", role="assistant")],
            [_msg("What is 3+3?")],
            [_msg("3+3 equals 6.", role="assistant")],
        ]

        result = await policy.summarize(groups, preserve_recent=2, counter=counter)

        # first_group + summary_group + 2 recent groups = 4 groups
        assert len(result.groups) == 4
        # First group preserved
        assert result.groups[0] == groups[0]
        # Summary group
        assert result.groups[1][0].role == "user"
        assert isinstance(result.groups[1][0].content, str)
        assert result.groups[1][0].content.startswith("[Summary of prior conversation]")
        assert "The user asked about math." in result.groups[1][0].content
        # Last 2 groups preserved
        assert result.groups[2] is groups[2]
        assert result.groups[3] is groups[3]
        # summary_text populated
        assert result.summary_text == "The user asked about math."

    async def test_delta_summarization_only_new_groups(self) -> None:
        client = MockLLMClient(
            responses=[
                _make_summary_response("Summary of first exchange."),
                _make_summary_response("Updated summary with second exchange."),
            ]
        )
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        # First call: 4 groups (first preserved), preserve_recent=2
        # Middle groups to summarize: groups[1] only (group 0 is first, groups 2-3 are recent)
        groups_first: list[list[Message]] = [
            [_msg("msg1")],
            [_msg("msg2", role="assistant")],
            [_msg("msg3")],
            [_msg("msg4", role="assistant")],
        ]
        await policy.summarize(groups_first, preserve_recent=2, counter=counter)

        # Second call: 6 groups total, preserve_recent=2
        # Middle groups: groups[1:4], policy already summarized 1 group, delta = groups[2:4]
        groups_second: list[list[Message]] = [
            [_msg("msg1")],
            [_msg("msg2", role="assistant")],
            [_msg("msg3")],
            [_msg("msg4", role="assistant")],
            [_msg("msg5")],
            [_msg("msg6", role="assistant")],
        ]
        result = await policy.summarize(groups_second, preserve_recent=2, counter=counter)

        # first_group + summary + 2 recent = 4 groups
        assert len(result.groups) == 4
        assert "Updated summary" in (result.groups[1][0].content or "")
        # The second LLM call should include the previous summary
        second_call = client.calls[1]
        call_content = second_call["messages"][0].content
        assert "Previous summary:" in call_content
        assert "Summary of first exchange." in call_content

    async def test_delta_no_new_groups_returns_existing_summary(self) -> None:
        client = MockLLMClient(responses=[_make_summary_response("Existing summary.")])
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("msg1")],
            [_msg("msg2", role="assistant")],
            [_msg("msg3")],
            [_msg("msg4", role="assistant")],
        ]
        # First call summarizes middle groups
        await policy.summarize(groups, preserve_recent=2, counter=counter)

        # Second call with same groups — no delta
        result = await policy.summarize(groups, preserve_recent=2, counter=counter)

        # Should not make another LLM call
        assert len(client.calls) == 1
        assert result.groups[1][0].content is not None
        assert "Existing summary." in result.groups[1][0].content

    async def test_reset_clears_state(self) -> None:
        client = MockLLMClient(
            responses=[
                _make_summary_response("First summary."),
                _make_summary_response("Fresh summary after reset."),
            ]
        )
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("msg1")],
            [_msg("msg2", role="assistant")],
            [_msg("msg3")],
            [_msg("msg4", role="assistant")],
        ]
        await policy.summarize(groups, preserve_recent=2, counter=counter)
        policy.reset()

        # After reset, should do full summarization again (not delta)
        result = await policy.summarize(groups, preserve_recent=2, counter=counter)

        assert len(client.calls) == 2
        # Second call should NOT contain "Previous summary:" since state was reset
        second_call_content = client.calls[1]["messages"][0].content
        assert "Previous summary:" not in second_call_content
        assert "Fresh summary after reset." in (result.groups[1][0].content or "")

    async def test_all_groups_preserved_when_count_lte_recent(self) -> None:
        client = MockLLMClient(responses=[])
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("msg1")],
            [_msg("msg2", role="assistant")],
        ]
        result = await policy.summarize(groups, preserve_recent=3, counter=counter)

        # Nothing to summarize — all groups fit in preserve_recent + first
        assert result.groups == groups
        assert result.summary_text is None
        assert len(client.calls) == 0

    async def test_empty_groups(self) -> None:
        client = MockLLMClient(responses=[])
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        result = await policy.summarize([], preserve_recent=2, counter=counter)
        assert result.groups == []
        assert result.summary_text is None
        assert len(client.calls) == 0

    async def test_llm_returns_empty_content(self) -> None:
        client = MockLLMClient(responses=[_make_summary_response("")])
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("msg1")],
            [_msg("msg2", role="assistant")],
            [_msg("msg3")],
            [_msg("msg4", role="assistant")],
        ]
        result = await policy.summarize(groups, preserve_recent=2, counter=counter)

        # first_group + summary + 2 recent = 4
        assert len(result.groups) == 4
        assert result.groups[1][0].content == "[Summary of prior conversation]\n"

    async def test_preserve_first_false(self) -> None:
        """When preserve_first=False, the first group is not preserved separately."""
        client = MockLLMClient(responses=[_make_summary_response("Full summary.")])
        policy = SummarizationPolicy(llm_client=client, preserve_first=False)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("msg1")],
            [_msg("msg2", role="assistant")],
            [_msg("msg3")],
            [_msg("msg4", role="assistant")],
        ]
        result = await policy.summarize(groups, preserve_recent=2, counter=counter)

        # summary + 2 recent = 3 groups (no first group preserved)
        assert len(result.groups) == 3
        assert result.groups[0][0].content is not None
        assert "[Summary of prior conversation]" in result.groups[0][0].content
        assert result.groups[1] is groups[2]
        assert result.groups[2] is groups[3]

    async def test_tool_exchange_groups_summarized(self) -> None:
        """Groups with tool exchanges are flattened properly for summarization."""
        client = MockLLMClient(responses=[_make_summary_response("User searched for data.")])
        policy = SummarizationPolicy(llm_client=client)
        counter = EstimateTokenCounter()

        tc = ToolCall(id="tc1", name="search", arguments={"q": "test"})
        groups: list[list[Message]] = [
            [_msg("find data")],
            [
                Message(role="assistant", content="searching", tool_calls=[tc]),
                Message(role="tool_result", content="found: test data"),
            ],
            [
                Message(role="assistant", content="more searching", tool_calls=[tc]),
                Message(role="tool_result", content="found: more data"),
            ],
            [_msg("done", role="assistant")],
        ]

        result = await policy.summarize(groups, preserve_recent=1, counter=counter)

        # first_group + summary + 1 recent = 3 groups
        assert len(result.groups) == 3
        assert result.groups[0] is groups[0]
        assert "[Summary of prior conversation]" in (result.groups[1][0].content or "")
        assert result.groups[2] is groups[3]

        # Verify the LLM received flattened messages including tool content
        call_content = client.calls[0]["messages"][0].content
        assert "searching" in call_content
        assert "search" in call_content
