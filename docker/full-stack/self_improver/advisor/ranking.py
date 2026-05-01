"""Ranking primitive for advisor proposals.

:func:`rank_proposals` is a pure function — no side effects, no LLM calls, no
file I/O. It sorts already-scored proposals by severity (primary),
``ranking_score`` descending within severity, and ``rubric_id`` ASCII-ascending
as tie-break. Computing the ``ranking_score`` itself is the specialists'
concern; this primitive only composes.
"""

from __future__ import annotations

from collections.abc import Iterable

from self_improver.advisor.proposal import Proposal, ProposalSeverity

# Lower sort key sorts first — critical before warning before observation.
_SEVERITY_ORDER: dict[ProposalSeverity, int] = {
    ProposalSeverity.CRITICAL: 0,
    ProposalSeverity.WARNING: 1,
    ProposalSeverity.OBSERVATION: 2,
}


def rank_proposals(proposals: Iterable[Proposal]) -> list[Proposal]:
    """Return ``proposals`` sorted into advisor display order.

    Ordering rules, in priority order:

    1. **Severity.** ``critical`` before ``warning`` before ``observation``.
    2. **Ranking score.** Within a severity bucket, higher ``ranking_score``
       sorts earlier (descending).
    3. **Rubric id.** ASCII-ascending on ``rubric_id`` as tie-break.

    Ordering is deterministic under identical inputs. Rubric source
    (builtin vs. custom) does NOT influence ordering — builtin and
    adopter-custom rubrics are treated as equals.

    Args:
        proposals: Any iterable of :class:`Proposal`. Consumed once.

    Returns:
        A new list containing the same proposals in display order. The input
        iterable is not mutated; an empty iterable produces an empty list.
    """
    return sorted(
        proposals,
        key=lambda p: (_SEVERITY_ORDER[p.severity], -p.ranking_score, p.rubric_id),
    )


__all__ = ["rank_proposals"]
