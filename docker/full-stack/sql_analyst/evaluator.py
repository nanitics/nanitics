"""Value-level ground-truth evaluator.

Implements the :class:`~nanitics.OutputEvaluator` protocol against a
canonical :class:`~sql_analyst.questions.SampleQuestion`. The agent's
final output is parsed as a JSON envelope
(``{"answer": <value>, "sql": <str>, "rowcount": <int>}``) and its
``answer`` field is fed to the question's
:class:`~sql_analyst.questions.ExpectedAnswer`.``compare``:

- ``True`` → :class:`~nanitics.EvaluationVerdict` ``ACCEPT``.
- ``False`` → ``REVISE`` with the value-level feedback the comparator
  returns (e.g., ``"expected scalar 37, got 42"``).

The evaluator never calls an LLM. Value equality (or numeric
tolerance for scalars) is the only gate — this is deliberately not an
LLM-as-judge or a SQL-shape comparator.
"""

from __future__ import annotations

import json
from typing import Any

from nanitics import EvaluationContext, EvaluationResult, EvaluationVerdict
from sql_analyst.questions import SampleQuestion

_EVALUATOR_NAME = "sql_analyst_ground_truth"
_ENVELOPE_FEEDBACK = (
    "Protocol violation: your final response must be a single JSON object "
    'matching {"answer": <value>, "sql": <str>, "rowcount": <int>}. '
    "Re-run and return only that JSON — no prose, no code fences."
)


class GroundTruthEvaluator:
    """Evaluator that compares the agent's answer to a known-good value.

    Args:
        question: The :class:`SampleQuestion` whose ``expected`` field
            holds the canonical answer.
        max_revisions: Maximum revision attempts the caller is willing
            to extend after a ``REVISE``. When the output is
            unparseable *and* no revisions remain, the verdict flips
            from ``REVISE`` to ``REJECT`` so the
            :class:`~nanitics.Supervisor` escalates cleanly.
    """

    def __init__(self, question: SampleQuestion, *, max_revisions: int = 2) -> None:
        if max_revisions < 0:
            raise ValueError("max_revisions must be >= 0")
        self._question = question
        self._max_revisions = max_revisions

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        parsed = _try_parse_envelope(output)
        if parsed is None or "answer" not in parsed:
            if self._max_revisions <= 0:
                return EvaluationResult(
                    verdict=EvaluationVerdict.REJECT,
                    evaluator_name=_EVALUATOR_NAME,
                    feedback=_ENVELOPE_FEEDBACK,
                )
            return EvaluationResult(
                verdict=EvaluationVerdict.REVISE,
                evaluator_name=_EVALUATOR_NAME,
                feedback=_ENVELOPE_FEEDBACK,
            )

        passed, comparator_feedback = self._question.expected.compare(parsed["answer"])
        if passed:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                evaluator_name=_EVALUATOR_NAME,
            )
        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            evaluator_name=_EVALUATOR_NAME,
            feedback=comparator_feedback,
        )


def _try_parse_envelope(output: str) -> dict[str, Any] | None:
    """Parse *output* as a JSON object. Return ``None`` on failure."""
    if not isinstance(output, str):
        return None
    text = output.strip()
    if not text:
        return None
    # Tolerate an accidental ```json fence — a common LLM slip that is
    # otherwise a perfectly valid envelope. Stripping it here is a
    # mild UX favor, not a semantic concession.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
