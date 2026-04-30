from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from nanitics.capabilities.context.grouping import (
    MessageGrouper,
    default_message_grouper,
    flatten_groups,
)
from nanitics.capabilities.context.summarization import SummarizationPolicy
from nanitics.capabilities.context.token_counter import (
    EstimateTokenCounter,
    TokenCounter,
    count_message_tokens,
)
from nanitics.capabilities.context.truncation import TruncationPolicy
from nanitics.infrastructure.llm.protocol import Message, ToolSchema
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    ContextSummarizationEvent,
    ContextTruncationEvent,
    RemovedMessageInfo,
)


class ContextUsage(BaseModel):
    """Token usage snapshot for the current conversation state.

    Provides a breakdown of token consumption across system prompt,
    tool schemas, and messages, plus utilization relative to the
    configured context limit.

    Attributes:
        total_tokens: Combined tokens for system prompt, tools, and messages.
        message_tokens: Tokens used by conversation messages only.
        system_tokens: Tokens used by the system prompt only.
        tools_tokens: Tokens used by tool schemas only.
        context_limit: The configured context window limit.
        available_tokens: Remaining tokens (limit - reserve - total).
        utilization: Ratio of total tokens to available budget (0.0–1.0+).
    """

    model_config = ConfigDict(frozen=True)

    total_tokens: int
    message_tokens: int
    system_tokens: int
    tools_tokens: int
    context_limit: int
    available_tokens: int
    utilization: float


class ContextManager:
    """Coordinates token tracking, truncation, and summarization.

    Runs automatically before each LLM call via ``prepare()``. Checks
    whether the conversation exceeds the configured threshold, then
    applies truncation and/or summarization to fit within the token budget.

    At least one of ``truncation`` or ``summarization`` must be provided.
    """

    def __init__(
        self,
        *,
        context_limit: int,
        reserve_tokens: int = 4096,
        threshold: float = 0.9,
        token_counter: TokenCounter | None = None,
        truncation: TruncationPolicy | None = None,
        summarization: SummarizationPolicy | None = None,
        grouper: MessageGrouper | None = None,
    ) -> None:
        """Initialize the context manager.

        Args:
            context_limit: The model's total context window in tokens.
            reserve_tokens: Tokens reserved for the LLM's response output.
            threshold: Utilization ratio (0.0–1.0) that triggers context management.
            token_counter: Token counting implementation. Defaults to EstimateTokenCounter.
            truncation: Truncation strategy for dropping old messages.
            summarization: Summarization strategy for compressing history.
            grouper: Message grouping function. Defaults to default_message_grouper.

        Raises:
            ValueError: If neither truncation nor summarization is provided.
            ValueError: If context_limit, reserve_tokens, or threshold are out of range.
        """
        if context_limit <= 0:
            raise ValueError(f"'context_limit' must be positive, got {context_limit}")
        if reserve_tokens < 0 or reserve_tokens >= context_limit:
            raise ValueError(
                f"'reserve_tokens' must be >= 0 and < context_limit ({context_limit}), got {reserve_tokens}"
            )
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(f"'threshold' must be in (0.0, 1.0], got {threshold}")
        if truncation is None and summarization is None:
            raise ValueError("At least one of 'truncation' or 'summarization' must be provided")
        self._context_limit = context_limit
        self._reserve_tokens = reserve_tokens
        self._threshold = threshold
        self._token_counter: TokenCounter = token_counter if token_counter is not None else EstimateTokenCounter()
        self._truncation = truncation
        self._summarization = summarization
        self._grouper = grouper if grouper is not None else default_message_grouper

    def _count_messages_tokens(self, messages: list[Message]) -> int:
        return sum(count_message_tokens(m, self._token_counter) for m in messages)

    def _count_tools_tokens(self, tools: list[ToolSchema] | None) -> int:
        if not tools:
            return 0
        tokens = 0
        for tool in tools:
            tokens += self._token_counter.count_text(tool.name)
            tokens += self._token_counter.count_text(tool.description)
            tokens += self._token_counter.count_text(json.dumps(tool.parameters))
        return tokens

    def current_usage(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ContextUsage:
        """Calculate current token usage without triggering management.

        Args:
            system_prompt: The system prompt text.
            messages: Current conversation messages.
            tools: Tool schemas, if any.

        Returns:
            A ContextUsage snapshot with token breakdown and utilization.
        """
        system_tokens = self._token_counter.count_text(system_prompt)
        tools_tokens = self._count_tools_tokens(tools)
        message_tokens = self._count_messages_tokens(messages)
        total_tokens = system_tokens + tools_tokens + message_tokens
        available = self._context_limit - self._reserve_tokens
        return ContextUsage(
            total_tokens=total_tokens,
            message_tokens=message_tokens,
            system_tokens=system_tokens,
            tools_tokens=tools_tokens,
            context_limit=self._context_limit,
            available_tokens=available - total_tokens,
            utilization=total_tokens / available if available > 0 else 1.0,
        )

    async def prepare(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        emitter: EventEmitter | None,
    ) -> list[Message]:
        """Prepare messages for an LLM call, applying truncation and/or summarization if needed.

        Returns messages unchanged if token usage is below the threshold.
        Otherwise, groups messages into atomic units, applies truncation first
        (if configured), then summarization (if still over budget).

        Emits ``ContextTruncationEvent`` only when at least one message is
        removed; a no-op pass through truncation produces no event, mirroring
        the summarization branch. ``ContextSummarizationEvent`` is emitted
        only when a summary is actually produced.

        Args:
            system_prompt: The system prompt text.
            messages: Current conversation messages.
            tools: Tool schemas, if any.
            emitter: Optional event emitter for observability.

        Returns:
            The (potentially reduced) message list ready for the LLM.
        """
        available = self._context_limit - self._reserve_tokens
        system_tokens = self._token_counter.count_text(system_prompt)
        tools_tokens = self._count_tools_tokens(tools)
        message_tokens = self._count_messages_tokens(messages)
        total = system_tokens + tools_tokens + message_tokens

        if total <= available * self._threshold:
            return messages

        message_budget = available - system_tokens - tools_tokens

        # Group messages into atomic units
        groups = self._grouper(messages)
        original_groups = groups

        # Apply truncation if configured
        if self._truncation is not None:
            tokens_before = message_tokens
            groups = self._truncation.truncate(groups, message_budget, self._token_counter)
            managed = flatten_groups(groups)
            tokens_after = self._count_messages_tokens(managed)

            if emitter is not None and len(managed) != len(messages):
                managed_set = {id(m) for m in managed}
                removed = [
                    RemovedMessageInfo(
                        role=m.role,
                        original_index=i,
                        content=m.content or "",
                    )
                    for i, m in enumerate(messages)
                    if id(m) not in managed_set
                ]
                emitter.emit(
                    ContextTruncationEvent(
                        trace_id=emitter.trace_id,
                        span_id=emitter.span_id,
                        parent_span_id=emitter.parent_span_id,
                        messages_before=len(messages),
                        messages_after=len(managed),
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                        removed_messages=removed,
                    )
                )

            # Check if we're now under budget
            new_total = system_tokens + tools_tokens + tokens_after
            if new_total <= available:
                return managed

        # Apply summarization if configured and still over budget.
        # Use original_groups so summarization has access to the full
        # middle content that truncation may have discarded.
        if self._summarization is not None:
            preserve_recent = self._truncation.preserve_recent if self._truncation else 2
            original_tokens = self._count_messages_tokens(flatten_groups(original_groups))
            summarization_result = await self._summarization.summarize(
                original_groups, preserve_recent, self._token_counter
            )
            managed = flatten_groups(summarization_result.groups)
            summary_tokens = self._count_messages_tokens(managed)

            if emitter is not None and summarization_result.summary_text is not None:
                emitter.emit(
                    ContextSummarizationEvent(
                        trace_id=emitter.trace_id,
                        span_id=emitter.span_id,
                        parent_span_id=emitter.parent_span_id,
                        messages_summarized=len(messages) - preserve_recent,
                        summary_tokens=summary_tokens,
                        original_tokens=original_tokens,
                        summary_text=summarization_result.summary_text,
                        summarization_input=summarization_result.summarization_input or "",
                    )
                )

            return managed

        return flatten_groups(groups)

    def reset(self) -> None:
        """Reset summarization state. Call between agent runs."""
        if self._summarization is not None:
            self._summarization.reset()
