from __future__ import annotations

from dataclasses import dataclass

from nanitics.capabilities.context.grouping import flatten_groups
from nanitics.capabilities.context.token_counter import TokenCounter
from nanitics.infrastructure.llm.protocol import LLMClient, Message

DEFAULT_SUMMARY_PROMPT = """Summarize the following conversation concisely. Preserve:
- Key decisions and their rationale
- Important findings and facts
- Task progress (what's done, what remains)
- Errors encountered and how they were resolved

Omit verbose tool outputs, redundant reasoning, and failed approaches
whose lessons are already captured. Be concise but complete."""

_SUMMARY_PREFIX = "[Summary of prior conversation]\n"


@dataclass(frozen=True)
class SummarizationResult:
    """Result of a summarization operation.

    Attributes:
        groups: The message groups after summarization (may include a summary message).
        summary_text: The generated summary text, or None if no summarization occurred.
        summarization_input: The text sent to the LLM for summarization, or None.
    """

    groups: list[list[Message]]
    summary_text: str | None
    summarization_input: str | None


class SummarizationPolicy:
    """Compresses conversation history via LLM summarization.

    Splits messages into first (preserved), middle (summarized), and recent
    (preserved) groups. The middle groups are formatted as text and sent to
    an LLM to produce a concise summary. On subsequent calls, uses delta
    summarization — only new messages since the last summary are processed.

    The summary replaces the middle messages with a single
    ``[Summary of prior conversation]`` message.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        preserve_first: bool = True,
    ) -> None:
        """Initialize the summarization policy.

        Args:
            llm_client: LLM client used to generate summaries.
            summary_prompt: System prompt for the summarization LLM call.
            preserve_first: Whether to keep the first message group intact.
        """
        self._llm_client = llm_client
        self._summary_prompt = summary_prompt
        self._preserve_first = preserve_first
        self._current_summary: str | None = None
        self._summarized_up_to: int = 0

    async def summarize(
        self,
        groups: list[list[Message]],
        preserve_recent: int,
        counter: TokenCounter,
    ) -> SummarizationResult:
        """Summarize message groups, preserving first and recent groups.

        On the first call, summarizes all middle groups. On subsequent calls,
        performs delta summarization — incorporates only new groups since
        the last summary.

        Args:
            groups: Message groups from the grouper.
            preserve_recent: Number of most-recent groups to keep intact.
            counter: Token counter (unused currently, reserved for budget-aware summarization).

        Returns:
            A SummarizationResult with the modified groups and summary metadata.
        """
        if not groups:
            return SummarizationResult(groups=[], summary_text=None, summarization_input=None)

        # Determine which groups are protected
        first_group: list[list[Message]] = []
        if self._preserve_first and len(groups) > 0:
            first_group = [groups[0]]
            remaining_groups = groups[1:]
        else:
            remaining_groups = groups

        split_point = max(0, len(remaining_groups) - preserve_recent)
        to_summarize = remaining_groups[:split_point]
        to_keep = remaining_groups[split_point:]

        if not to_summarize:
            return SummarizationResult(groups=groups, summary_text=self._current_summary, summarization_input=None)

        # Build content to summarize
        if self._current_summary is not None and 0 < self._summarized_up_to <= len(to_summarize):
            # Delta summarization: only new groups since last summary
            delta = to_summarize[self._summarized_up_to :]
            if not delta:
                # No new groups to summarize — return existing summary + recent
                summary_group = [
                    Message(
                        role="user",
                        content=_SUMMARY_PREFIX + self._current_summary,
                    )
                ]
                result_groups = [*first_group, summary_group, *to_keep]
                return SummarizationResult(
                    groups=result_groups, summary_text=self._current_summary, summarization_input=None
                )
            delta_messages = flatten_groups(delta)
            content = (
                f"Previous summary:\n{self._current_summary}\n\n"
                f"New messages to incorporate:\n{_format_messages(delta_messages)}"
            )
        else:
            # Full summarization
            content = _format_messages(flatten_groups(to_summarize))

        response = await self._llm_client.generate(
            system_prompt=self._summary_prompt,
            messages=[Message(role="user", content=content)],
        )

        self._current_summary = response.content or ""
        self._summarized_up_to = len(to_summarize)

        summary_group = [
            Message(
                role="user",
                content=_SUMMARY_PREFIX + self._current_summary,
            )
        ]
        result_groups = [*first_group, summary_group, *to_keep]
        return SummarizationResult(
            groups=result_groups, summary_text=self._current_summary, summarization_input=content
        )

    def reset(self) -> None:
        """Reset summarization state. Clears accumulated summary for a fresh run."""
        self._current_summary = None
        self._summarized_up_to = 0


def _format_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        prefix = msg.role.upper()
        if msg.content:
            parts.append(f"{prefix}: {msg.content}")
        if msg.tool_calls:
            parts.extend(f"{prefix} [tool_call]: {tc.name}({tc.arguments})" for tc in msg.tool_calls)
    return "\n".join(parts)
