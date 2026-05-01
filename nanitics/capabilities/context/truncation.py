from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from nanitics.capabilities.context.token_counter import TokenCounter, count_group_tokens
from nanitics.infrastructure.llm.protocol import Message


class TruncationPolicy(BaseModel):
    """Configuration for dropping old message groups to fit a token budget.

    Preserves the first group (initial context) and the most recent groups
    (current conversation state), dropping expendable groups from the middle.
    Groups containing messages with ``metadata={"protected": True}`` are
    never dropped.

    Expendable groups are kept from most recent to oldest until the budget
    is exhausted, so the oldest messages are dropped first.

    Attributes:
        preserve_first: Whether to always keep the first message group.
        preserve_recent: Number of most-recent groups to always keep.
    """

    model_config = ConfigDict(frozen=True)

    preserve_first: bool = True
    preserve_recent: int = 2

    def truncate(
        self,
        groups: list[list[Message]],
        token_budget: int,
        counter: TokenCounter,
    ) -> list[list[Message]]:
        """Truncate message groups to fit within the token budget.

        Args:
            groups: Message groups from the grouper.
            token_budget: Maximum tokens allowed for messages.
            counter: Token counter for measuring group sizes.

        Returns:
            Filtered groups that fit within the budget.
        """
        if not groups:
            return []

        total = len(groups)

        # Identify protected group indices
        protected: set[int] = set()
        if self.preserve_first and total > 0:
            protected.add(0)

        recent_start = max(0, total - self.preserve_recent)
        for i in range(recent_start, total):
            protected.add(i)

        # Also protect groups that contain messages with metadata.protected
        for i, group in enumerate(groups):
            if any(m.metadata is not None and m.metadata.get("protected") for m in group):
                protected.add(i)

        # Calculate tokens for protected groups
        protected_tokens = sum(count_group_tokens(groups[i], counter) for i in sorted(protected))

        if protected_tokens >= token_budget:
            return [groups[i] for i in sorted(protected)]

        remaining_budget = token_budget - protected_tokens

        # Walk expendable groups from most recent to oldest
        expendable_indices = [i for i in range(total) if i not in protected]
        expendable_indices.reverse()  # most recent first

        kept_expendable: set[int] = set()
        for i in expendable_indices:
            group_tokens = count_group_tokens(groups[i], counter)
            if group_tokens <= remaining_budget:
                kept_expendable.add(i)
                remaining_budget -= group_tokens

        # Merge preserving original order
        kept_indices = sorted(protected | kept_expendable)
        return [groups[i] for i in kept_indices]
