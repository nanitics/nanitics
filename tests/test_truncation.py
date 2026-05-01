from typing import Literal

from nanitics.capabilities.context.token_counter import EstimateTokenCounter
from nanitics.capabilities.context.truncation import TruncationPolicy
from nanitics.infrastructure.llm.protocol import Message, ToolCall


def _msg(content: str, role: Literal["user", "assistant"] = "user") -> Message:
    return Message(role=role, content=content)


def _make_groups(count: int, char_length: int = 40) -> list[list[Message]]:
    """Create single-message groups alternating user/assistant roles."""
    roles: list[Literal["user", "assistant"]] = ["user", "assistant"]
    return [[_msg("x" * char_length, role=roles[i % 2])] for i in range(count)]


class TestTruncationPolicy:
    def test_no_truncation_when_within_budget(self) -> None:
        policy = TruncationPolicy()
        counter = EstimateTokenCounter()
        groups = _make_groups(3, char_length=20)
        # Each message: 4 overhead + 20/4=5 = 9 tokens, 3 groups = 27
        result = policy.truncate(groups, token_budget=100, counter=counter)
        assert result == groups

    def test_oldest_expendable_groups_dropped(self) -> None:
        policy = TruncationPolicy(preserve_first=True, preserve_recent=2)
        counter = EstimateTokenCounter()
        # 6 groups, each ~14 tokens (4 overhead + 40/4=10)
        groups = _make_groups(6)
        # Budget for 4 groups: 56 tokens
        result = policy.truncate(groups, token_budget=56, counter=counter)
        # Should keep: first (0), and recent 2 (4, 5)
        # Plus most recent expendable that fits: index 3
        assert len(result) == 4
        assert result[0] is groups[0]
        assert result[1] is groups[3]
        assert result[2] is groups[4]
        assert result[3] is groups[5]

    def test_first_group_preserved(self) -> None:
        policy = TruncationPolicy(preserve_first=True, preserve_recent=1)
        counter = EstimateTokenCounter()
        groups = _make_groups(5)
        # Very tight budget: only room for 2 groups
        result = policy.truncate(groups, token_budget=28, counter=counter)
        assert result[0] is groups[0]
        assert result[-1] is groups[-1]

    def test_recent_groups_always_preserved(self) -> None:
        policy = TruncationPolicy(preserve_first=False, preserve_recent=3)
        counter = EstimateTokenCounter()
        groups = _make_groups(6)
        # Budget for 3 groups: 42
        result = policy.truncate(groups, token_budget=42, counter=counter)
        assert len(result) == 3
        assert result == groups[3:]

    def test_only_protected_groups_fit(self) -> None:
        policy = TruncationPolicy(preserve_first=True, preserve_recent=2)
        counter = EstimateTokenCounter()
        groups = _make_groups(6)
        # Budget for only 3 groups (protected set): 42 tokens
        result = policy.truncate(groups, token_budget=42, counter=counter)
        assert len(result) == 3
        assert result[0] is groups[0]
        assert result[1] is groups[4]
        assert result[2] is groups[5]

    def test_empty_group_list(self) -> None:
        policy = TruncationPolicy()
        counter = EstimateTokenCounter()
        result = policy.truncate([], token_budget=100, counter=counter)
        assert result == []

    def test_preserve_recent_larger_than_count(self) -> None:
        policy = TruncationPolicy(preserve_first=True, preserve_recent=10)
        counter = EstimateTokenCounter()
        groups = _make_groups(3)
        result = policy.truncate(groups, token_budget=100, counter=counter)
        assert result == groups

    def test_preserves_original_order(self) -> None:
        policy = TruncationPolicy(preserve_first=True, preserve_recent=1)
        counter = EstimateTokenCounter()
        groups = _make_groups(5)
        result = policy.truncate(groups, token_budget=100, counter=counter)
        # All should fit, order preserved
        assert result == groups

    def test_tool_exchange_groups_never_split(self) -> None:
        """Groups with multiple messages (assistant+tool_result) are kept or dropped atomically."""
        policy = TruncationPolicy(preserve_first=True, preserve_recent=1)
        counter = EstimateTokenCounter()

        tc = ToolCall(id="tc1", name="search", arguments={"q": "test"})
        groups: list[list[Message]] = [
            [_msg("task")],  # group 0: user
            [  # group 1: assistant + tool_result
                Message(role="assistant", content="calling", tool_calls=[tc]),
                Message(role="tool_result", content="result data " * 10),
            ],
            [  # group 2: assistant + tool_result
                Message(role="assistant", content="calling again", tool_calls=[tc]),
                Message(role="tool_result", content="more data " * 10),
            ],
            [_msg("final answer", role="assistant")],  # group 3
        ]

        # Tight budget: only protected groups fit (group 0 + group 3)
        result = policy.truncate(groups, token_budget=50, counter=counter)
        assert len(result) == 2
        assert result[0] is groups[0]
        assert result[1] is groups[3]
        # Verify groups are intact — no partial splits
        for group in result:
            assert isinstance(group, list)
            for msg in group:
                assert isinstance(msg, Message)

    def test_preserve_recent_counts_groups_not_messages(self) -> None:
        """preserve_recent=2 keeps 2 groups, even if they contain multiple messages."""
        policy = TruncationPolicy(preserve_first=False, preserve_recent=2)
        counter = EstimateTokenCounter()

        tc = ToolCall(id="tc1", name="tool", arguments={})
        groups: list[list[Message]] = [
            [_msg("task")],
            [
                Message(role="assistant", content="step1", tool_calls=[tc]),
                Message(role="tool_result", content="r1"),
            ],
            [
                Message(role="assistant", content="step2", tool_calls=[tc]),
                Message(role="tool_result", content="r2"),
            ],
            [_msg("done", role="assistant")],
        ]

        # Budget tight enough that only 2 groups (the protected recent) fit.
        # Protected (groups 2, 3) ≈ 17 tokens. Budget=18 leaves 1 for expendable.
        # Neither expendable group fits.
        result = policy.truncate(groups, token_budget=18, counter=counter)
        assert len(result) == 2
        assert result[0] is groups[2]
        assert result[1] is groups[3]

    def test_metadata_protected_groups_kept(self) -> None:
        """Groups with metadata.protected=True are never dropped."""
        policy = TruncationPolicy(preserve_first=False, preserve_recent=1)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("expendable1")],
            [Message(role="user", content="important", metadata={"protected": True})],
            [_msg("expendable2")],
            [_msg("recent", role="assistant")],
        ]
        # Protected (group 1 + group 3) = ~11 tokens. Budget=12 leaves 1 for expendable.
        # Neither expendable group (6 tokens each) fits.
        result = policy.truncate(groups, token_budget=12, counter=counter)
        assert len(result) == 2
        assert result[0] is groups[1]  # protected kept
        assert result[1] is groups[3]  # recent kept

    def test_metadata_protected_false_is_expendable(self) -> None:
        """metadata.protected=False does not protect a group."""
        policy = TruncationPolicy(preserve_first=False, preserve_recent=1)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [Message(role="user", content="x" * 40, metadata={"protected": False})],
            [_msg("x" * 40, role="assistant")],
            [_msg("recent")],
        ]
        # Budget for 2 groups: 28 tokens. Only recent fits as protected.
        result = policy.truncate(groups, token_budget=28, counter=counter)
        assert len(result) == 2
        # Most recent expendable kept (group 1), plus recent (group 2)
        assert result[0] is groups[1]
        assert result[1] is groups[2]

    def test_metadata_protected_kept_even_when_over_budget(self) -> None:
        """Protected groups are kept even when they alone exceed the budget."""
        policy = TruncationPolicy(preserve_first=False, preserve_recent=1)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [Message(role="user", content="x" * 200, metadata={"protected": True})],  # ~54 tokens
            [_msg("expendable")],
            [_msg("recent")],
        ]
        # Budget of 20 tokens — protected group alone exceeds it
        result = policy.truncate(groups, token_budget=20, counter=counter)
        assert len(result) == 2
        assert result[0] is groups[0]  # protected kept despite exceeding budget
        assert result[1] is groups[2]  # recent kept

    def test_skips_large_expendable_keeps_smaller_older(self) -> None:
        """A large expendable group that doesn't fit is skipped; smaller older groups are still kept."""
        policy = TruncationPolicy(preserve_first=True, preserve_recent=1)
        counter = EstimateTokenCounter()

        groups: list[list[Message]] = [
            [_msg("x" * 20)],  # group 0: first, ~9 tokens
            [_msg("x" * 20, role="assistant")],  # group 1: expendable, ~9 tokens
            [_msg("x" * 400, role="user")],  # group 2: expendable, ~104 tokens (large)
            [_msg("x" * 20, role="assistant")],  # group 3: expendable, ~9 tokens
            [_msg("x" * 20)],  # group 4: recent, ~9 tokens
        ]
        # Budget: 40 tokens. Protected (0, 4) = 18 tokens. Remaining = 22.
        # Walking recent-to-old: group 3 (9) fits, group 2 (104) doesn't fit (skipped),
        # group 1 (9) fits.
        result = policy.truncate(groups, token_budget=40, counter=counter)
        assert len(result) == 4
        assert result[0] is groups[0]
        assert result[1] is groups[1]  # smaller older group kept
        assert result[2] is groups[3]  # recent expendable kept
        assert result[3] is groups[4]
