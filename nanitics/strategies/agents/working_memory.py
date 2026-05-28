from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkingMemory(Protocol):
    """Protocol for an in-run structured scratchpad.

    Working memory stores structured content organized into named sections.
    The agent writes content with ``## Section`` headers, and each section
    is stored independently. Content persists across agent steps within a
    single run but is not persisted across runs.

    Implementations must support full replacement (``write``), partial
    updates (``update``), and reading the current state (``read``).

    **For:** single-agent in-run progress tracking — synthesized findings,
    open questions, decisions the agent wants to carry forward across its
    own steps within one ``Agent.run`` call.

    **Not for:** state that must survive process restarts (use
    ``LongTermStore`` or ``SemanticStore``), similarity-based recall (use
    ``SemanticStore`` or ``EpisodeStore``), multi-agent coordination (use
    ``SharedMemory`` or ``Blackboard``), or replay of the agent's own
    prior assistant turns (use ``ThreadStore`` — behavioral continuity,
    not information continuity).

    Module layout note: this protocol lives in
    ``nanitics.strategies.agents`` rather than ``nanitics.capabilities.memory``
    because ``ReActAgent`` consumes it natively (it's part of the agent
    contract, not an external capability). It is re-exported from
    ``nanitics.capabilities.memory`` so consumers can import it alongside
    the other memory protocols; the canonical location remains
    ``strategies/agents/``.
    """

    def read(self) -> str | None:
        """Read the current working memory content.

        Returns:
            Formatted string with all sections, or None if empty.
        """
        ...

    def write(self, content: str) -> None:
        """Replace all working memory content.

        Parses ``## Section`` headers from the content and stores each
        section independently. Any existing sections not present in the
        new content are removed.

        Args:
            content: Full working memory content with ``## Section`` headers.
        """
        ...

    def update(self, updates: dict[str, str]) -> None:
        """Merge specific sections without replacing the entire memory.

        Args:
            updates: Mapping of section name to section content. Existing
                sections not in this dict are preserved.
        """
        ...

    def clear(self) -> None:
        """Remove all sections from working memory."""
        ...

    def reset(self) -> None:
        """Reset working memory to its initial empty state."""
        ...


_WORKING_MEMORY_INSTRUCTIONS = (
    "You have a working memory scratchpad. Its current contents appear in the "
    "[Working Memory] section of the conversation. To update it, include a "
    "<working_memory> block in your response containing the complete updated "
    "contents. The block replaces your entire working memory — omitted sections "
    "are lost, so include everything you want to keep. Include a "
    "<working_memory> block after every step where you receive tool results.\n\n"
    "Your scratchpad is a curated extract, not a transcript. Record only what "
    "you'll need in later steps — synthesized findings, decisions, and open "
    "questions. Use terse structured entries (bullets, key-value pairs) rather "
    "than prose. No redundancy across sections. As understanding develops, "
    "compress and merge rather than append. The conversation history has the "
    "raw details."
)


class WorkingMemoryContributor:
    """System prompt contributor that teaches the agent how to use working memory.

    Adds instructions explaining the ``<working_memory>`` block format, the
    full-replacement semantics, and best practices for concise structured entries.
    """

    def system_prompt_section(self) -> tuple[str, str]:
        return ("working_memory", _WORKING_MEMORY_INSTRUCTIONS)
