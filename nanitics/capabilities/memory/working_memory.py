from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

from nanitics.capabilities.memory.context_provider import (
    ContextContent,
)
from nanitics.core.agents.working_memory import (
    WorkingMemory,
    WorkingMemoryContributor,
)
from nanitics.infrastructure.llm.protocol import Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import WorkingMemoryReadEvent

# Re-export from Core
__all__ = [
    "InMemoryWorkingMemory",
    "WorkingMemory",
    "WorkingMemoryContributor",
    "WorkingMemoryProvider",
]


class InMemoryWorkingMemory:
    """In-memory implementation of the ``WorkingMemory`` protocol.

    Stores sections in an ``OrderedDict`` preserving insertion order.
    Content written with ``## Section`` headers is parsed into independent
    sections. Reading returns all sections formatted with headers.
    """

    def __init__(self) -> None:
        self._sections: OrderedDict[str, str] = OrderedDict()

    def read(self) -> str | None:
        if not self._sections:
            return None
        parts = ["[Working Memory]"]
        for section, content in self._sections.items():
            parts.append(f"## {section}")
            parts.append(content)
            parts.append("")
        return "\n".join(parts).rstrip()

    def write(self, content: str) -> None:
        self._sections.clear()
        current_section: str | None = None
        current_lines: list[str] = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section is not None:
                    self._sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[3:].strip()
                current_lines = []
            elif current_section is not None:
                current_lines.append(line)

        if current_section is not None:
            self._sections[current_section] = "\n".join(current_lines).strip()

    def update(self, updates: dict[str, str]) -> None:
        for section, content in updates.items():
            self._sections[section] = content

    def clear(self) -> None:
        self._sections.clear()

    def reset(self) -> None:
        self.clear()


class WorkingMemoryProvider:
    """Context provider that injects working memory into the LLM context.

    Reads the current working memory state and returns it as a protected,
    high-priority context block. Emits a ``WorkingMemoryReadEvent`` when
    content is read.

    Args:
        memory: The working memory store to read from.
        emitter: Optional event emitter for observability. Captured at
            construction time — prefer ``emitter_provider`` for any
            provider attached to an agent, so the emitter follows the
            agent's per-task bound emitter under delegation and
            concurrent sharing.
        emitter_provider: Callback returning the current emitter.
            Resolves through the owning agent's per-task emitter when
            wired from :class:`~nanitics.core.agents.base.Agent`.
            Overrides ``emitter`` when both are supplied. When an agent
            receives this provider via ``context_providers`` and no
            provider is set, the agent auto-wires
            ``emitter_provider=lambda: agent._emitter`` so memory events
            carry the correct per-task trace lineage.
    """

    def __init__(
        self,
        memory: WorkingMemory,
        emitter: EventEmitter | None = None,
        *,
        emitter_provider: Callable[[], EventEmitter | None] | None = None,
    ) -> None:
        self._memory = memory
        self._static_emitter = emitter
        self._emitter_provider: Callable[[], EventEmitter | None] | None = emitter_provider

    @property
    def _emitter(self) -> EventEmitter | None:
        """Emitter used for trace events.

        Resolves through ``emitter_provider`` when set (so the provider
        follows its owning agent's per-task bound emitter); otherwise
        the static emitter passed at construction.
        """
        if self._emitter_provider is not None:
            return self._emitter_provider()
        return self._static_emitter

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        content = self._memory.read()
        if content is None:
            return None
        emitter = self._emitter
        if emitter is not None:
            token_count = max(1, len(content) // 4)
            emitter.emit(
                WorkingMemoryReadEvent(
                    trace_id=emitter.trace_id,
                    span_id=emitter.span_id,
                    parent_span_id=emitter.parent_span_id,
                    content=content,
                    token_count=token_count,
                )
            )
        return ContextContent(content=content, priority=0, protected=True, provider_name="working_memory")
