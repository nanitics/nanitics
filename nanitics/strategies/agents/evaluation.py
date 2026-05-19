from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from nanitics.infrastructure.llm.protocol import ContentBlock, Message


class EvaluationVerdict(StrEnum):
    """Verdict produced by an evaluator after assessing agent output.

    Determines what happens next in the evaluation loop:
    ACCEPT ends the loop and returns the output.
    REVISE re-enters the agent loop with feedback (if revisions remain).
    REJECT terminates immediately with ``evaluation_failed``.
    """

    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"
    EVALUATOR_ERROR = "evaluator_error"


class EvaluationResult(BaseModel):
    """Result of evaluating agent output.

    Attributes:
        verdict: Accept, revise, or reject decision.
        score: Optional quality score (0.0–1.0). Informational only;
            the verdict drives the accept/revise/reject decision.
        feedback: Guidance appended to the conversation when revising.
        evaluator_name: Identifies which evaluator produced this result.
        error_detail: Description of the underlying error when verdict
            is ``EVALUATOR_ERROR``. None for other verdicts.
    """

    model_config = ConfigDict(frozen=True)

    verdict: EvaluationVerdict
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    feedback: str | None = None
    evaluator_name: str
    error_detail: str | None = None


class EvaluationContext(BaseModel):
    """Context available to evaluators when assessing output.

    Attributes:
        messages: Full conversation history accumulated during the run.
        task_input: The original task string passed to ``agent.run()``.
        depth: Current node depth in a tree search (None for non-tree agents).
        max_depth: Maximum allowed depth in a tree search.
        trajectory_length: Number of nodes from root to current node (LATS).
        total_nodes_explored: Total nodes explored so far in the tree search.
    """

    model_config = ConfigDict(frozen=True)

    messages: list[Message]
    task_input: str | list[ContentBlock]
    depth: int | None = None
    max_depth: int | None = None
    trajectory_length: int | None = None
    total_nodes_explored: int | None = None


class EvaluationCheck(BaseModel):
    """A single predicate check used by ``ProgrammaticEvaluator``.

    Attributes:
        name: Human-readable name for this check (appears in feedback).
        check: Predicate function that takes the output string and returns
            True if the check passes.
        feedback: Message included in the revision prompt when this check fails.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    check: Callable[[str], bool]
    feedback: str


@runtime_checkable
class OutputEvaluator(Protocol):
    """Protocol for output quality evaluators.

    Implement this to create custom evaluation logic. The agent calls
    ``evaluate()`` after producing final output (no more tool calls).
    The ``max_revisions`` property tells the agent how many revision
    attempts to allow before giving up.
    """

    @property
    def max_revisions(self) -> int: ...

    async def evaluate(self, output: str, context: EvaluationContext) -> EvaluationResult: ...
