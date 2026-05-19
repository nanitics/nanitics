"""Deterministic structural evaluators for tree-search validation.

These evaluators are NOT mocks. They implement the real ``OutputEvaluator``
protocol, return valid ``EvaluationResult`` instances, and are invoked by
the SDK's real evaluation loop against real-LLM-generated node bodies.

What makes them deterministic is the *verdict policy*: instead of
string-matching the LLM's content (the approach the replaced evaluators
took, which produced probabilistic REJECT / ACCEPT / REVISE paths under
LLM variance), these evaluators derive their verdict from structural
properties available on ``EvaluationContext`` — ``depth`` and a per-run
counter of children seen at each parent depth. The LLM still drives node
body generation; only the scoring / accept-reject policy is structural.

Intended for tree-search agents (``LATSAgent``, ``TreeOfThoughtAgent``)
only. Non-tree agents do not populate ``EvaluationContext.depth``; see
the null-safety note on each class.
"""

from __future__ import annotations

from nanitics.evaluation import (
    EvaluationContext,
    EvaluationResult,
    EvaluationVerdict,
)


class DepthPrioritizedEvaluator:
    """Structural evaluator: REJECT first child per depth, ACCEPT deep terminals.

    Gating rule — applied in order for every ``evaluate`` call:

    1. Increment the per-depth children-seen counter for ``context.depth``.
    2. If this was the first child seen at ``context.depth`` AND
       ``context.depth > 0`` (never REJECT the root), return
       ``REJECT`` with score ``0.0``. This guarantees at least one
       ``TreeSearchNodePrunedEvent`` fires per run as soon as
       ``branching_factor >= 2`` — the first child at each parent depth
       is pruned while subsequent siblings survive and carry the search
       forward.
    3. Else if ``context.depth >= accept_depth``, return ``ACCEPT`` with
       score ``depth / max_depth`` (strictly increasing in depth, so
       deeper accepted terminals outscore shallower ones — giving an
       unambiguous argmax for the agent's best-node selector).
    4. Otherwise, return ``REVISE`` with score ``0.5`` (intermediate
       nodes remain expandable).

    State is per-instance (``_children_seen_by_parent_depth``). Construct
    a fresh evaluator per agent run — a parametrized test that builds a
    fresh agent per case therefore stays clean.

    Null safety. ``context.depth`` is ``None`` for non-tree agents; this
    class coerces a ``None`` depth to ``REVISE`` rather than raising, but
    the intended use is exclusively tree-search agents where depth is
    populated.

    Attributes:
        max_revisions: Always ``0``. Search agents do not use the
            revision loop; this is a protocol requirement.
    """

    max_revisions: int = 0

    def __init__(self, *, accept_depth: int) -> None:
        """Build a structural evaluator.

        Args:
            accept_depth: Depth at which a node becomes eligible for
                ``ACCEPT``. Nodes at lower depth that are not the first
                child at their depth receive ``REVISE``.
        """
        self._accept_depth = accept_depth
        self._children_seen_by_parent_depth: dict[int, int] = {}

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        """Apply the structural gating rule. See class docstring for policy."""
        del output  # Verdict is structural; LLM output drives only the node body.

        depth = context.depth
        if depth is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.REVISE,
                score=0.5,
                evaluator_name="depth_prioritized",
            )

        count = self._children_seen_by_parent_depth.get(depth, 0) + 1
        self._children_seen_by_parent_depth[depth] = count

        if count == 1 and depth > 0:
            return EvaluationResult(
                verdict=EvaluationVerdict.REJECT,
                score=0.0,
                evaluator_name="depth_prioritized",
            )

        if depth >= self._accept_depth:
            denom = context.max_depth if context.max_depth else depth
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=float(depth) / float(denom),
                evaluator_name="depth_prioritized",
            )

        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            score=0.5,
            evaluator_name="depth_prioritized",
        )


class LengthScoredEvaluator:
    """Structural evaluator with output-length-based ACCEPT score.

    Same REJECT / REVISE policy as ``DepthPrioritizedEvaluator`` — the
    first child at each parent depth is rejected, intermediate depths
    REVISE — but the ``ACCEPT`` score is ``min(1.0, len(output) / score_cap)``
    rather than depth-keyed.

    Use this for ``SearchStrategy.BEST_FIRST`` in ToT, where the
    strategy-specific assertion requires score differentiation across
    siblings generated at the same depth. A pure depth-keyed score ties
    every same-depth ACCEPT and collapses ``BEST_FIRST`` to BFS-like
    behaviour by insertion-order tie-break.

    Null safety. ``context.depth`` is ``None`` for non-tree agents; this
    class coerces a ``None`` depth to ``REVISE`` rather than raising.

    Attributes:
        max_revisions: Always ``0``. Search agents do not use the
            revision loop; this is a protocol requirement.
    """

    max_revisions: int = 0

    def __init__(self, *, accept_depth: int, score_cap: float = 800.0) -> None:
        """Build a length-scored structural evaluator.

        Args:
            accept_depth: Depth at which a node becomes eligible for
                ``ACCEPT``.
            score_cap: Denominator for the length-based score. Outputs
                at or above this length receive the max score of ``1.0``.
        """
        self._accept_depth = accept_depth
        self._score_cap = score_cap
        self._children_seen_by_parent_depth: dict[int, int] = {}

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        """Apply the length-scored structural gating rule."""
        depth = context.depth
        if depth is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.REVISE,
                score=0.5,
                evaluator_name="length_scored",
            )

        count = self._children_seen_by_parent_depth.get(depth, 0) + 1
        self._children_seen_by_parent_depth[depth] = count

        if count == 1 and depth > 0:
            return EvaluationResult(
                verdict=EvaluationVerdict.REJECT,
                score=0.0,
                evaluator_name="length_scored",
            )

        if depth >= self._accept_depth:
            score = min(1.0, len(output) / self._score_cap)
            return EvaluationResult(
                verdict=EvaluationVerdict.ACCEPT,
                score=score,
                evaluator_name="length_scored",
            )

        return EvaluationResult(
            verdict=EvaluationVerdict.REVISE,
            score=0.5,
            evaluator_name="length_scored",
        )


class RejectTerminalEvaluator:
    """Structural evaluator: REJECT every candidate at or below accept depth.

    Gating rule — applied in order for every ``evaluate`` call:

    1. If ``context.depth`` is ``None`` (non-tree-agent caller), return
       ``REVISE`` score ``0.5`` (null-safety parity with siblings).
    2. If ``context.depth == 0``, return ``REVISE`` (the root is never
       REJECTed — it is created directly from the input task, not via
       ``evaluate``; the guard is included only so a misuse does not
       destroy the tree).
    3. If ``context.depth < accept_depth``, return ``REVISE`` score
       ``0.5`` (intermediate nodes remain expandable, mirroring
       ``DepthPrioritizedEvaluator`` on that band).
    4. If ``context.depth >= accept_depth``, return **``REJECT`` score
       ``0.0`` unconditionally**. This guarantees no node ever becomes
       an accepted terminal — even when the LLM marks
       ``is_complete=True``, the SDK prunes the node before it enters
       ``accepted_terminal_ids`` (see ``tree_of_thought.py`` around
       line 381).

    Purpose. This evaluator is the lever for exercising the
    no-accepted-terminal fallback branch in
    ``TreeOfThoughtAgent._select_best_node`` (tree_of_thought.py:434-437):
    the agent finishes with ``accepted_terminal_ids`` empty, no terminal
    nodes survive pruning, and ``_select_best_node`` must fall back to
    the highest-scoring non-root, non-pruned node.

    Null safety. ``context.depth`` is ``None`` for non-tree agents; this
    class coerces a ``None`` depth to ``REVISE`` rather than raising.

    Attributes:
        max_revisions: Always ``0``. Search agents do not use the
            revision loop; this is a protocol requirement.
    """

    max_revisions: int = 0

    def __init__(self, *, accept_depth: int) -> None:
        """Build a reject-at-terminal structural evaluator.

        Args:
            accept_depth: Depth at or above which every node receives
                ``REJECT``. Shape parity with siblings — the argument is
                retained even though the policy never emits ``ACCEPT``,
                so the constructor signature is interchangeable across
                evaluator choices in parametrized tests.
        """
        self._accept_depth = accept_depth

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult:
        """Apply the reject-at-terminal gating rule. See class docstring."""
        del output  # Verdict is structural; LLM output drives only the node body.

        depth = context.depth
        if depth is None:
            return EvaluationResult(
                verdict=EvaluationVerdict.REVISE,
                score=0.5,
                evaluator_name="reject_terminal",
            )

        if depth == 0:
            return EvaluationResult(
                verdict=EvaluationVerdict.REVISE,
                score=0.5,
                evaluator_name="reject_terminal",
            )

        if depth < self._accept_depth:
            return EvaluationResult(
                verdict=EvaluationVerdict.REVISE,
                score=0.5,
                evaluator_name="reject_terminal",
            )

        return EvaluationResult(
            verdict=EvaluationVerdict.REJECT,
            score=0.0,
            evaluator_name="reject_terminal",
        )
