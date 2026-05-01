from __future__ import annotations

from nanitics.capabilities.evaluation.protocol import (
    EvaluationCheck,
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)


class ProgrammaticEvaluator:
    """Rule-based evaluator using predicate functions.

    Runs all checks against the output string. Returns ACCEPT (score 1.0)
    if every check passes, or REVISE (score 0.0) with aggregated feedback
    from all failing checks. Never returns REJECT.
    """

    def __init__(
        self,
        checks: list[EvaluationCheck],
        max_revisions: int = 1,
    ) -> None:
        """Initialize the programmatic evaluator.

        Args:
            checks: Predicate checks to run against the output.
            max_revisions: Maximum revision attempts before the agent gives up.
        """
        if max_revisions < 0:
            raise ValueError(f"max_revisions must be non-negative, got {max_revisions}")
        self._checks = checks
        self._max_revisions = max_revisions

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        failed = [check for check in self._checks if not check.check(output)]

        if not failed:
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=1.0,
                evaluator_name="programmatic",
            )

        feedback = "\n".join(f"- {check.name}: {check.feedback}" for check in failed)
        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            score=0.0,
            feedback=feedback,
            evaluator_name="programmatic",
        )
