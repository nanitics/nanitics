from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from nanitics.core.agents.base import AgentResult


class HandoffPayload(BaseModel):
    """Structured data model for context passed between agents during handoff.

    Contains task state, findings, decisions, open questions, and artifacts.
    Call ``render()`` to produce a markdown document suitable for the
    receiving agent's context.

    Attributes:
        task_state: Current state of the work being handed off.
        findings: Key findings discovered so far.
        decisions: Decisions already made.
        open_questions: Unresolved questions for the next agent.
        artifacts: Named artifacts (outlines, drafts, code snippets).
        metadata: Arbitrary metadata.
    """

    model_config = ConfigDict(frozen=True)

    task_state: str
    findings: list[str] = []
    decisions: list[str] = []
    open_questions: list[str] = []
    artifacts: dict[str, str] = {}
    metadata: dict[str, Any] = {}

    def render(self) -> str:
        """Render the payload as a markdown document with sections for each field."""
        sections: list[str] = ["## Handoff Context"]

        sections.append(f"\n### Task State\n{self.task_state}")

        if self.findings:
            items = "\n".join(f"- {f}" for f in self.findings)
            sections.append(f"\n### Findings\n{items}")

        if self.decisions:
            items = "\n".join(f"- {d}" for d in self.decisions)
            sections.append(f"\n### Decisions\n{items}")

        if self.open_questions:
            items = "\n".join(f"- {q}" for q in self.open_questions)
            sections.append(f"\n### Open Questions\n{items}")

        if self.artifacts:
            parts: list[str] = []
            for name, content in self.artifacts.items():
                parts.append(f"#### {name}\n{content}")
            sections.append("\n### Artifacts\n" + "\n".join(parts))

        return "\n".join(sections)


class HandoffTransfer:
    """A ContextTransferStrategy that builds and renders a HandoffPayload.

    Uses a builder function to construct a ``HandoffPayload`` from the
    agent result, then renders it as markdown.

    Args:
        builder: Callable that creates a ``HandoffPayload`` from an
            ``AgentResult``.
    """

    def __init__(self, builder: Callable[[AgentResult], HandoffPayload]) -> None:
        self._builder = builder

    async def extract(self, result: AgentResult) -> str:
        payload = self._builder(result)
        return payload.render()


def handoff_sender_instructions(payload_fields: list[str] | None = None) -> str:
    """Generate system prompt instructions for an agent producing a handoff.

    Tells the agent to structure its output with the specified fields
    so it can be parsed into a ``HandoffPayload``.

    Args:
        payload_fields: Fields to include. Defaults to task_state, findings,
            decisions, open_questions, and artifacts.
    """
    fields = payload_fields or [
        "task_state",
        "findings",
        "decisions",
        "open_questions",
        "artifacts",
    ]
    field_list = "\n".join(f"- **{f}**" for f in fields)
    return (
        "When your work is complete, produce a structured handoff summary "
        "containing the following sections:\n"
        f"{field_list}\n\n"
        "Be concise but thorough. Include only information relevant to the next agent."
    )


def handoff_receiver_instructions() -> str:
    """Generate system prompt instructions for an agent receiving a handoff.

    Tells the agent to expect structured handoff context and to build
    on the findings, decisions, and artifacts provided.
    """
    return (
        "You are receiving a handoff from a previous agent. "
        "Your input contains a structured handoff context with these possible sections: "
        "Task State, Findings, Decisions, Open Questions, and Artifacts. "
        "Use this context to continue the work. "
        "Address any open questions and build on the findings and decisions provided."
    )
