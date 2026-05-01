from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from nanitics.infrastructure.llm.protocol import Message

MessageGrouper: TypeAlias = Callable[[list[Message]], list[list[Message]]]
"""Callable that partitions a message list into atomic groups that must not be split during truncation."""


def flatten_groups(groups: list[list[Message]]) -> list[Message]:
    """Flatten grouped messages back into a flat list."""
    return [msg for group in groups for msg in group]


def default_message_grouper(messages: list[Message]) -> list[list[Message]]:
    """Group messages into atomic units that should not be split.

    A tool_result attaches to the preceding group. All other roles start a
    new group. This keeps assistant+tool_call messages together with their
    corresponding tool_result responses.
    """
    if not messages:
        return []

    groups: list[list[Message]] = []
    for msg in messages:
        if msg.role == "tool_result" and groups:
            groups[-1].append(msg)
        else:
            groups.append([msg])
    return groups
