from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nanitics.infrastructure.llm.protocol import TextContentBlock

if TYPE_CHECKING:
    from nanitics.infrastructure.llm.protocol import Message


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens in a text string.

    Implement this to provide accurate token counting for a specific
    model's tokenizer. The default EstimateTokenCounter uses a
    character-based approximation.
    """

    def count_text(self, text: str) -> int: ...


class EstimateTokenCounter:
    """Token counter that estimates tokens from character count.

    Uses a configurable characters-per-token ratio (default: 4.0).
    Fast and sufficient for budget management. For exact counts,
    implement ``TokenCounter`` with a model-specific tokenizer.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    def count_text(self, text: str) -> int:
        return max(1, int(len(text) / self._chars_per_token))


def count_message_tokens(message: Message, counter: TokenCounter) -> int:
    """Count tokens for a single message including content and tool calls.

    Adds 4 tokens per-message overhead for framing.
    """
    tokens = 4  # per-message overhead
    if message.content:
        if isinstance(message.content, str):
            tokens += counter.count_text(message.content)
        else:
            for block in message.content:
                if isinstance(block, TextContentBlock):
                    tokens += counter.count_text(block.text)
                else:
                    tokens += 85  # estimate for image blocks
    if message.tool_calls:
        for tc in message.tool_calls:
            tokens += counter.count_text(tc.name)
            tokens += counter.count_text(json.dumps(tc.arguments))
    return tokens


def count_group_tokens(group: list[Message], counter: TokenCounter) -> int:
    """Count total tokens for a group of messages."""
    return sum(count_message_tokens(m, counter) for m in group)
