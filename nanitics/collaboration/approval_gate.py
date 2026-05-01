from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputProvider,
    HumanInputRequest,
    HumanInputType,
)
from nanitics.composition.orchestration.protocol import StepResult
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)


class ApprovalGate:
    """Workflow step that pauses execution for human approval of a draft artifact
    produced by a previous step.

    Used between steps in a workflow to gate progression on human review.
    Supports approve, modify, revise, and reject decisions. For tool-call
    approval, see :class:`ApprovalWrappedTool`; for agent-initiated approval
    requests, see :func:`create_request_approval_tool`.

    Args:
        provider: Handles delivering the request to a human.
        emitter: Event emitter for observability.
        prompt: Static string or callable that generates the prompt from step input.
        name: Step name for identification in workflows.
        allow_revision: Whether to signal that revision is allowed in request metadata.
        context: Additional context string or callable for the human reviewer.
        run_id: Optional run identifier, included in the request for correlation.
        agent_name: Name of the agent whose output the gate reviews. Flows onto
            both the emitted :class:`HumanInputRequest` and
            :class:`HumanInputRequestEvent` so adopters can filter HITL events
            by producer agent. ``None`` when the adopter does not wire it (the
            gate does not fall back to its own ``name``).
    """

    def __init__(
        self,
        provider: HumanInputProvider,
        emitter: EventEmitter | None = None,
        prompt: str | Callable[[Any], str] = "Approve proceeding?",
        name: str = "approval_gate",
        allow_revision: bool = False,
        context: str | Callable[[Any], str] | None = None,
        run_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self._provider = provider
        self._emitter = emitter
        self._prompt = prompt
        self._name = name
        self._allow_revision = allow_revision
        self._context = context
        self._run_id = run_id
        self._agent_name = agent_name

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, input: Any, *, revision_count: int = 0) -> StepResult:
        """Request human approval and return a StepResult based on the decision.

        The gate derives a deterministic ``request_id`` as
        ``f"{run_id}:{name}:{revision_count}"`` so a logical approval keeps
        the same identity across suspend/resume re-execution, while a
        revision loop progresses through a stable sequence of slots
        (``:0``, ``:1``, ``:2``, …). ``RevisionGate`` passes its ``attempt``
        counter in via the keyword.

        Args:
            input: The value to present to the human reviewer.
            revision_count: The revision-slot index (``0`` on the initial
                pass, incremented by ``RevisionGate`` on each REVISE loop).

        Returns:
            StepResult with output and metadata reflecting the human's decision.

        Raises:
            ValueError: If ``run_id`` is ``None`` — HITL identity across
                resume requires a stable run identifier.
        """
        if self._run_id is None:
            raise ValueError("ApprovalGate requires run_id for stable request identity across resume")
        request_id = f"{self._run_id}:{self._name}:{revision_count}"
        prompt_text = self._prompt(input) if callable(self._prompt) else self._prompt
        context_text = self._context(input) if callable(self._context) else self._context

        metadata: dict[str, Any] = {"step_name": self._name}
        if self._allow_revision:
            metadata["allow_revision"] = True

        request = HumanInputRequest(
            request_id=request_id,
            run_id=self._run_id,
            request_type=HumanInputType.APPROVAL,
            prompt=prompt_text,
            context=context_text,
            metadata=metadata,
            agent_name=self._agent_name,
        )

        if self._emitter is not None:
            self._emitter.emit(
                HumanInputRequestEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    request_id=request_id,
                    request_type=HumanInputType.APPROVAL.value,
                    prompt=prompt_text,
                    context=context_text,
                    agent_name=self._agent_name,
                    metadata=metadata,
                )
            )

        start = time.monotonic()
        response = await self._provider.request_input(request)
        wait_ms = int((time.monotonic() - start) * 1000)

        if self._emitter is not None:
            self._emitter.emit(
                HumanInputResponseEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    request_id=request_id,
                    decision=response.decision.value,
                    has_content=response.content is not None,
                    wait_duration_ms=wait_ms,
                )
            )

        if response.decision == HumanDecision.APPROVE:
            return StepResult(output=input)

        if response.decision == HumanDecision.OVERRIDE:
            output = response.content if response.content is not None else input
            return StepResult(output=output, metadata={"modified": True})

        if response.decision == HumanDecision.REVISE:
            return StepResult(
                output=None,
                metadata={
                    "revision_requested": True,
                    "feedback": response.content or "",
                },
            )

        reason = response.content or "Rejected by human"
        return StepResult(
            output=None,
            metadata={"rejected": True, "reason": reason},
        )
