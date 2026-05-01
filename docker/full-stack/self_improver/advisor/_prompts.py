"""Prompt builder for the advisor specialists.

Internal module — not part of the public advisor surface. The builder is
pure: given the same rubrics and trace events, it produces the same
string. No LLM calls, no file I/O beyond inspecting the already-loaded
rubric bodies.
"""

from __future__ import annotations

import json

from nanitics.infrastructure.observability.events import TraceEvent
from self_improver.advisor.rubric import Rubric

_TASK_FRAMING = (
    "Analyze the supplied trace against the rubrics provided in your system "
    "prompt. For each rubric that fires on this trace, emit one Proposal. "
    "Return a JSON object matching the SpecialistProposals schema: "
    '{"proposals": [...]}. '
    "Every proposal must cite specific trace evidence (event_index, event_type, "
    "and a short verbatim excerpt). Use the rubric's id, severity, category, "
    "and target_dimension verbatim — do not invent category or target_dimension "
    "values that the rubric does not declare. Score each proposal in [0.0, 1.0] "
    "where 1.0 reflects the proposal the adopter should act on first within its "
    "severity. If no rubric fires, return an empty proposals list."
)


def _render_role_description(target_dimension: str) -> str:
    """Render the specialist's role preamble for ``target_dimension``."""
    return (
        f"You are the Nanitics advisor specialist for the '{target_dimension}' "
        f"dimension of a multi-agent application. You read a single trace "
        f"captured from a Nanitics run and produce rubric-grounded proposals "
        f"that describe specific, evidence-cited improvements to the "
        f"application's {target_dimension}.\n\n"
        f"Emit a JSON object conforming to the SpecialistProposals schema: a "
        f"single key 'proposals' whose value is a list of Proposal objects. "
        f"Each Proposal requires: rubric_id, rubric_source, severity, "
        f"category, target_dimension, headline, detail, evidence (list of "
        f"EvidenceReference), suggested_action, and ranking_score.\n\n"
        f"Evidence citations: each entry carries event_index (0-based into "
        f"the trace), event_type (the BaseEvent discriminator, e.g. 'agent.start' "
        f"or 'multi_agent.delegation'), and excerpt (a short verbatim slice).\n\n"
        f"ranking_score semantics: a float in [0.0, 1.0] where higher means "
        f"the adopter should act on this proposal earlier within its severity "
        f"bucket. Severity (critical > warning > observation) is the primary "
        f"ordering axis — the score only sorts within a severity.\n\n"
        f"The rubric's category and target_dimension are fixed by the rubric "
        f"itself; do not emit categories outside the rubric you cite."
    )


def _render_rubric_block(rubrics: list[Rubric]) -> str:
    """Concatenate rubric bodies into a deterministic text block."""
    if not rubrics:
        return "## Rubrics\n\n(No rubrics loaded for this specialist.)"

    ordered = sorted(rubrics, key=lambda r: r.id)
    parts = [
        "## Rubrics",
        *(
            f"### Rubric id: {rubric.id}\n"
            f"- severity: {rubric.severity.value}\n"
            f"- category: {rubric.category.value}\n"
            f"- target_dimension: {rubric.target_dimension}\n"
            f"- source: {rubric.source.value}\n\n"
            f"{rubric.body.strip()}"
            for rubric in ordered
        ),
    ]
    return "\n\n".join(parts)


def _render_trace_block(trace_events: list[TraceEvent]) -> str:
    """Render the trace events as a deterministic text block.

    Each event is emitted as ``[index] event_type`` followed by the event's
    JSON body (``model_dump(mode="json")`` via Pydantic). Deterministic under
    identical inputs — serialization is sorted-key-free but Pydantic's
    ``model_dump`` preserves field declaration order, and the list order
    preserves the input iteration order.
    """
    parts = ["## Trace events"]
    if not trace_events:
        parts.append("(trace contains no events)")
        return "\n\n".join(parts)

    for index, event in enumerate(trace_events):
        body = event.model_dump(mode="json")
        parts.append(f"[{index}] {event.event_type}\n{json.dumps(body, indent=2, ensure_ascii=False, default=str)}")
    return "\n\n".join(parts)


def build_shared_block(trace_events: list[TraceEvent]) -> str:
    """Build the per-trace, dimension-agnostic block (task framing + trace).

    Identical across all specialists for a given trace, so it forms the
    cacheable prefix when the prompt is split into structured sections.
    """
    return "\n\n".join((_TASK_FRAMING, _render_trace_block(trace_events)))


def build_dimension_block(target_dimension: str, rubrics: list[Rubric]) -> str:
    """Build the per-dimension block (role description + filtered rubrics).

    Differs across specialists, so it must follow the cacheable prefix
    when the prompt is split into structured sections.
    """
    scoped_rubrics = [r for r in rubrics if r.target_dimension == target_dimension]
    return "\n\n".join((_render_role_description(target_dimension), _render_rubric_block(scoped_rubrics)))


def build_specialist_system_prompt(
    target_dimension: str,
    rubrics: list[Rubric],
    trace_events: list[TraceEvent],
) -> str:
    """Build the specialist's system prompt as a single deterministic string.

    Composes two blocks in order:

    1. **Shared** — task framing followed by the full trace. Identical
       across specialists for a given trace; sits in the cacheable prefix.
    2. **Dimension-specific** — role description and the rubrics filtered
       to ``target_dimension``. Differs per specialist; sits after the
       cacheable prefix so cache reads can be reused across specialists.

    Args:
        target_dimension: The specialist's target dimension (``"prompts"``,
            ``"tool_descriptions"``, ``"coordination_patterns"``, etc.).
            The specialist only receives rubrics whose
            :attr:`Rubric.target_dimension` matches this argument.
        rubrics: Rubrics already loaded via
            :func:`self_improver.advisor.load_rubrics`. Filtered internally by
            ``target_dimension``.
        trace_events: The already-loaded trace events. Typically produced
            by :func:`self_improver.advisor.load_trace`.

    Returns:
        The combined system prompt string, with sections separated by blank
        lines.
    """
    return "\n\n".join(
        (
            build_shared_block(trace_events),
            build_dimension_block(target_dimension, rubrics),
        )
    )


__all__ = [
    "build_dimension_block",
    "build_shared_block",
    "build_specialist_system_prompt",
]
