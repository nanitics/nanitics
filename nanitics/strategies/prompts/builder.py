from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanitics.infrastructure.llm.protocol import SystemPromptSection


@runtime_checkable
class SystemPromptContributor(Protocol):
    """Protocol for components that contribute sections to the system prompt.

    Implementations return ``(section_name, content)`` for a cacheable
    section (the default), ``(section_name, content, cacheable)`` to
    control cache eligibility, or ``None`` to opt out. Many SDK features
    (working memory, planning, episodic memory) implement this protocol
    to inject instructions automatically.
    """

    def system_prompt_section(self) -> tuple[str, str] | tuple[str, str, bool] | None:
        """Return the section, optionally with a ``cacheable`` flag, or None."""
        ...


class SystemPromptBuilder:
    """Composes a system prompt from named sections.

    Sections are stored by name (later additions overwrite earlier ones
    with the same name) and joined with double newlines in insertion
    order. Empty sections are skipped.
    """

    def __init__(self) -> None:
        self._sections: dict[str, tuple[str, bool]] = {}

    def add_section(self, name: str, content: str, cacheable: bool = True) -> SystemPromptBuilder:
        """Add or replace a named section. Returns self for chaining."""
        self._sections[name] = (content, cacheable)
        return self

    def remove_section(self, name: str) -> SystemPromptBuilder:
        """Remove a section by name. No-op if the section doesn't exist."""
        self._sections.pop(name, None)
        return self

    def has_section(self, name: str) -> bool:
        """Check whether a section with the given name exists."""
        return name in self._sections

    def build(self) -> str:
        """Build the final system prompt string."""
        parts = []
        for content, _ in self._sections.values():
            stripped = content.strip()
            if stripped:
                parts.append(stripped)
        return "\n\n".join(parts)

    def build_sections(self) -> list[SystemPromptSection]:
        """Build structured sections with caching metadata."""
        result = []
        for content, cacheable in self._sections.values():
            stripped = content.strip()
            if stripped:
                result.append(SystemPromptSection(content=stripped, cacheable=cacheable))
        return result
