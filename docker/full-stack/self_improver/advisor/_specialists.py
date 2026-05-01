"""Specialist-agent factory for the advisor runtime.

Internal module — not part of the public surface. Each specialist is a
:class:`ReasoningAgent` whose ``output_schema`` is :class:`SpecialistProposals`,
constraining the LLM call to emit structured proposal JSON. No tools, no
ReAct loop — the specialist performs a single reasoning pass over its
rubrics and the trace.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from nanitics.core.agents.reasoning import ReasoningAgent
from nanitics.infrastructure.llm.protocol import LLMClient
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import TraceEvent
from self_improver.advisor._prompts import build_dimension_block, build_shared_block
from self_improver.advisor.proposal import Proposal
from self_improver.advisor.rubric import Rubric

# Launch target dimensions — the three specialists shipping at launch.
# Rubrics whose ``target_dimension`` matches none of these are silently
# excluded so the deferred dimensions can be added later without changing
# the rubric file format.
_LAUNCH_TARGET_DIMENSIONS: tuple[str, ...] = (
    "prompts",
    "tool_descriptions",
    "coordination_patterns",
)

_SPECIALIST_TASK = (
    "Analyze the supplied trace against the rubrics in your system prompt and emit the SpecialistProposals JSON object."
)


class SpecialistProposals(BaseModel):
    """Structured-output contract for a specialist's single LLM call.

    Used as ``output_schema`` on the specialist :class:`ReasoningAgent`.
    Each specialist's ``proposals`` list feeds directly into the advisor's
    ranked output.

    Not public: lives under the advisor's internal namespace because the
    aggregated ``list[Proposal]`` is what adopters consume — the
    per-specialist wrapper is an implementation detail.
    """

    model_config = ConfigDict(frozen=True)

    proposals: list[Proposal]


def _specialist_name(target_dimension: str) -> str:
    """Produce a stable, trace-friendly agent name for a specialist."""
    return f"advisor-{target_dimension}"


class _SharedContextContributor:
    """Contribute the trace + task framing as a cacheable section.

    Identical content across all specialists for a given trace, so it
    sits in the cacheable prefix that cross-specialist cache reads share
    when ``AnthropicLLMClient`` runs with ``enable_caching=True``.
    """

    def __init__(self, trace_events: list[TraceEvent]) -> None:
        self._content = build_shared_block(trace_events)

    def system_prompt_section(self) -> tuple[str, str, bool]:
        return ("advisor_shared", self._content, True)


class _DimensionContextContributor:
    """Contribute the role description + dimension-filtered rubrics.

    Differs per specialist, so it must follow the cacheable prefix —
    marking it ``cacheable=False`` keeps the cross-specialist cache key
    stable on the shared prefix above.
    """

    def __init__(self, target_dimension: str, rubrics: list[Rubric]) -> None:
        self._content = build_dimension_block(target_dimension, rubrics)

    def system_prompt_section(self) -> tuple[str, str, bool]:
        return ("advisor_dimension", self._content, False)


def build_specialist(
    *,
    target_dimension: str,
    rubrics: list[Rubric],
    trace_events: list[TraceEvent],
    llm_client: LLMClient,
    emitter: EventEmitter,
) -> ReasoningAgent:
    """Build one specialist agent for a single ``target_dimension``.

    The specialist emits :class:`SpecialistProposals` via its
    ``output_schema``. Rubrics are filtered by
    :attr:`Rubric.target_dimension` before being baked into the system
    prompt — the specialist only sees rubrics it is responsible for.

    The system prompt is supplied via two contributors: a cacheable
    shared block (task framing + trace) and a non-cacheable dimension
    block (role + rubrics). When the LLM client supports prompt caching
    (``AnthropicLLMClient`` with ``enable_caching=True``) the shared
    prefix is reused as a cache read across specialists analysing the
    same trace.

    Args:
        target_dimension: The specialist's domain (e.g. ``"prompts"``).
        rubrics: Full rubric list (mixed dimensions). Filtered internally.
        trace_events: Events already loaded via
            :func:`self_improver.advisor.load_trace`.
        llm_client: The LLM client used for this specialist's single pass.
        emitter: Event emitter shared with the caller for observability.

    Returns:
        A configured :class:`ReasoningAgent`.
    """
    return ReasoningAgent(
        name=_specialist_name(target_dimension),
        llm_client=llm_client,
        emitter=emitter,
        system_prompt="",
        prompt_contributors=[
            _SharedContextContributor(trace_events),
            _DimensionContextContributor(target_dimension, rubrics),
        ],
        output_schema=SpecialistProposals,
    )


def build_all_specialists(
    *,
    rubrics: list[Rubric],
    trace_events: list[TraceEvent],
    llm_client: LLMClient,
    emitter: EventEmitter,
) -> list[ReasoningAgent]:
    """Build one specialist per launch target dimension.

    Iterates :data:`_LAUNCH_TARGET_DIMENSIONS`. Rubrics whose
    ``target_dimension`` has no launched specialist (``agent_strategy``,
    ``iteration_budgets``) are silently excluded.

    Args:
        rubrics: Full rubric list; partitioned by ``target_dimension``.
        trace_events: Events already loaded.
        llm_client: Shared LLM client for every specialist.
        emitter: Shared emitter; each specialist uses it for observability.

    Returns:
        One :class:`ReasoningAgent` per launch target dimension, in the
        order declared by :data:`_LAUNCH_TARGET_DIMENSIONS`.
    """
    return [
        build_specialist(
            target_dimension=target_dimension,
            rubrics=rubrics,
            trace_events=trace_events,
            llm_client=llm_client,
            emitter=emitter,
        )
        for target_dimension in _LAUNCH_TARGET_DIMENSIONS
    ]


async def run_specialist(
    *,
    target_dimension: str,
    rubrics: list[Rubric],
    trace_events: list[TraceEvent],
    llm_client: LLMClient,
    emitter: EventEmitter,
) -> list[Proposal]:
    """Run one specialist to completion and return its proposal list.

    Builds the specialist, invokes it against the static specialist task,
    and returns the parsed :class:`SpecialistProposals.proposals`. A
    specialist whose run completed without producing a parsed payload is
    a contract violation — the advisor wires ``output_schema=SpecialistProposals``
    so the agent framework must supply a parsed result. Propagate, do not
    mask.
    """
    specialist = build_specialist(
        target_dimension=target_dimension,
        rubrics=rubrics,
        trace_events=trace_events,
        llm_client=llm_client,
        emitter=emitter,
    )
    result = await specialist.run(_SPECIALIST_TASK)
    parsed = result.parsed
    if not isinstance(parsed, SpecialistProposals):
        raise RuntimeError(f"Specialist {specialist.name!r} completed without a parsed SpecialistProposals payload.")
    return list(parsed.proposals)


__all__ = [
    "SpecialistProposals",
    "build_all_specialists",
    "build_specialist",
    "run_specialist",
]
