import pytest

from nanitics.capabilities.context.token_counter import (
    EstimateTokenCounter,
    TokenCounter,
    count_message_tokens,
)
from nanitics.infrastructure.llm.protocol import ImageContentBlock, Message, TextContentBlock, ToolCall


class TestEstimateTokenCounter:
    def test_empty_string_returns_one(self) -> None:
        counter = EstimateTokenCounter()
        assert counter.count_text("") == 1

    def test_short_text(self) -> None:
        counter = EstimateTokenCounter()
        # "hello" = 5 chars / 4.0 = 1.25 -> int(1.25) = 1
        assert counter.count_text("hello") == 1

    def test_long_text(self) -> None:
        counter = EstimateTokenCounter()
        text = "a" * 400  # 400 chars / 4.0 = 100 tokens
        assert counter.count_text(text) == 100

    def test_configurable_chars_per_token(self) -> None:
        counter = EstimateTokenCounter(chars_per_token=2.0)
        text = "a" * 100  # 100 chars / 2.0 = 50 tokens
        assert counter.count_text(text) == 50

    def test_satisfies_token_counter_protocol(self) -> None:
        counter = EstimateTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_custom_object_satisfies_protocol(self) -> None:
        class CustomCounter:
            def count_text(self, text: str) -> int:
                return len(text)

        counter = CustomCounter()
        assert isinstance(counter, TokenCounter)


class TestCountMessageTokens:
    def test_counts_tool_calls(self) -> None:
        counter = EstimateTokenCounter()
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msg = Message(role="assistant", tool_calls=[tc])
        tokens = count_message_tokens(msg, counter)
        # 4 (overhead) + count("search") + count(json.dumps({"q": "test"}))
        assert tokens > 4

    def test_counts_content_blocks(self) -> None:
        counter = EstimateTokenCounter()
        msg = Message(
            role="user",
            content=[
                TextContentBlock(text="hello world"),
                ImageContentBlock(media_type="image/png", data="abc"),
            ],
        )
        tokens = count_message_tokens(msg, counter)
        # 4 (overhead) + text tokens + 85 (image estimate)
        assert tokens >= 4 + 1 + 85


@pytest.mark.parametrize("value", [0, -1, -0.5])
def test_estimate_token_counter_rejects_non_positive_chars_per_token(value: float) -> None:
    with pytest.raises(ValueError, match="chars_per_token must be positive"):
        EstimateTokenCounter(chars_per_token=value)
