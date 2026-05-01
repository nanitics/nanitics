from __future__ import annotations

from nanitics.capabilities.evaluation.protocol import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
    OutputEvaluator,
)


class CompositeEvaluator:
    """Chains multiple evaluators with short-circuit behavior.

    Runs evaluators in order. Returns immediately on the first non-ACCEPT
    verdict. If all evaluators accept, returns the last evaluator's result.
    Use this to run cheap checks first (programmatic) and expensive checks
    (LLM) only when the cheap ones pass.
    """

    def __init__(
        self,
        evaluators: list[OutputEvaluator],
        max_revisions: int = 1,
    ) -> None:
        """Initialize the composite evaluator.

        Args:
            evaluators: Evaluators to run in order. Place cheaper evaluators first.
            max_revisions: Maximum revision attempts. Set this on the composite,
                not on individual evaluators — the agent reads this value.
        """
        if max_revisions < 0:
            raise ValueError(f"max_revisions must be non-negative, got {max_revisions}")
        self._evaluators = evaluators
        self._max_revisions = max_revisions

    @property
    def max_revisions(self) -> int:
        return self._max_revisions

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        last_result: EvaluationResult | None = None

        for evaluator in self._evaluators:
            result = await evaluator.evaluate(output, context)
            last_result = result

            if result.verdict != EvaluationVerdict.ACCEPT:
                return result

        if last_result is not None:
            return last_result

        return EvaluationResult(
            verdict=EvaluationVerdict.ACCEPT,
            score=None,
            evaluator_name="composite",
        )
