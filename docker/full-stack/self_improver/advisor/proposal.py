"""Proposal data model for the advisory optimization system.

Specialist agents emit :class:`Proposal` values; the ranking primitive composes
them into a ranked list; the renderer turns them into JSON and Markdown.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ProposalSeverity(StrEnum):
    """Severity of an advisor proposal.

    ``critical`` proposals are the first-class ordering axis in
    :func:`self_improver.advisor.rank_proposals` — they precede every ``warning``
    regardless of score, which in turn precede every ``observation``.
    """

    CRITICAL = "critical"
    WARNING = "warning"
    OBSERVATION = "observation"


class ProposalCategory(StrEnum):
    """Category of the concern a proposal addresses.

    The enum intentionally covers both the advisor-specific dimensions
    its specialists emit (``prompts``, ``tool-descriptions``,
    ``coordination-patterns``, ``agent-strategy``, ``iteration-budgets``)
    and the broader ``run-analyst`` taxonomy (``application-logic``,
    ``configuration``, ``sdk``, ``evaluation``, ``observability``) so
    the shared rubric corpus can be consumed by both consumers without
    schema divergence.
    """

    PROMPTS = "prompts"
    TOOL_DESCRIPTIONS = "tool-descriptions"
    COORDINATION_PATTERNS = "coordination-patterns"
    AGENT_STRATEGY = "agent-strategy"
    ITERATION_BUDGETS = "iteration-budgets"
    APPLICATION_LOGIC = "application-logic"
    CONFIGURATION = "configuration"
    SDK = "sdk"
    EVALUATION = "evaluation"
    OBSERVABILITY = "observability"


class RubricSource(StrEnum):
    """Origin of the rubric that produced a proposal.

    The advisor ranks builtin and adopter-custom proposals on equal terms; this
    label is metadata for attribution and debugging, not an ordering input.
    """

    BUILTIN = "builtin"
    CUSTOM = "custom"


class EvidenceReference(BaseModel):
    """A pointer into the analyzed trace that supports a proposal.

    The advisor cites, it does not rehydrate — ``event_index`` plus ``event_type``
    and a short ``excerpt`` is enough for a human reader to locate the full
    event without inflating every proposal with a nested event snapshot.
    """

    model_config = ConfigDict(frozen=True)

    event_index: int = Field(ge=0, description="0-based index into the analyzed trace.")
    event_type: str = Field(min_length=1, description="The BaseEvent discriminator value of the cited event.")
    excerpt: str = Field(min_length=1, description="Short verbatim slice the specialist deemed salient.")


class Proposal(BaseModel):
    """A single ranked, evidence-cited improvement suggestion.

    Attributes:
        rubric_id: Globally unique id of the rubric that produced this proposal.
        rubric_source: ``builtin`` or ``custom`` — set by the advisor based on
            which rubric fired, never inferred from adopter-controlled data.
        severity: First-class ordering axis used by :func:`rank_proposals`.
        category: Taxonomy category shared with the ``run-analyst`` agent.
        target_dimension: Which specialist owns this proposal — matches the
            rubric's ``target_dimension`` field.
        headline: One-line summary shown in the Markdown report.
        detail: Markdown-safe paragraph(s) expanding on the headline.
        evidence: Trace citations backing the proposal; may be empty for
            structural proposals (e.g., a missing tool description).
        suggested_action: Actionable next-step prose the adopter can act on.
        ranking_score: Specialist-assigned float in ``[0.0, 1.0]`` used as the
            within-severity ordering tie-breaker.
    """

    model_config = ConfigDict(frozen=True)

    rubric_id: str = Field(min_length=1)
    rubric_source: RubricSource
    severity: ProposalSeverity
    category: ProposalCategory
    target_dimension: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    suggested_action: str = Field(min_length=1)
    ranking_score: float = Field(ge=0.0, le=1.0)
