from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class HumanInputType(StrEnum):
    """Type of human input being requested.

    Two values: ``APPROVAL`` for approval-semantic requests (gate-level or
    tool-level), ``QUESTION`` for question-semantic requests.
    """

    APPROVAL = "approval"
    QUESTION = "question"


class HumanDecision(StrEnum):
    """Decision made by a human in response to a HITL request."""

    APPROVE = "approve"
    REJECT = "reject"
    OVERRIDE = "override"
    ANSWER = "answer"
    ESCALATE = "escalate"
    REVISE = "revise"


class HumanInputRequest(BaseModel):
    """Immutable request for human input.

    Describes what the agent needs from a human: approval for an action,
    an answer to a question, or review of a plan.

    Attributes:
        request_id: Unique identifier (auto-generated UUID).
        run_id: Associates the request with a specific agent run.
        request_type: The kind of input needed.
        prompt: What the agent is asking the human.
        context: Additional context to help the human decide.
        options: Suggested choices for question-type requests.
        metadata: Arbitrary data (tool name, parameters, etc.).
        agent_name: Which agent made the request.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str | None = None
    request_type: HumanInputType
    prompt: str
    context: str | None = None
    options: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None


class HumanInputResponse(BaseModel):
    """Immutable response from a human.

    Attributes:
        request_id: Matches the original request.
        decision: The human's decision.
        content: Human's message — reason, answer, or instructions.
        metadata: Additional data (e.g., ``modified_params`` for tool modification).
        responded_at: When the response was created (UTC).
    """

    model_config = ConfigDict(frozen=True)

    request_id: str
    decision: HumanDecision
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    responded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class HumanInputProvider(Protocol):
    """Protocol for providing human input to agents.

    Implementations handle the mechanics of delivering requests to humans
    and collecting responses — via CLI prompts, API endpoints, UI dialogs, etc.
    """

    async def request_input(self, request: HumanInputRequest) -> HumanInputResponse:
        """Send a request to a human and return their response."""
        ...


class CallbackHumanInputProvider:
    """HumanInputProvider backed by a sync or async callback.

    Useful for testing (auto-approve) and simple integrations (CLI prompts).

    Args:
        callback: Function that receives a request and returns a response.
            May be sync or async.
    """

    def __init__(
        self,
        callback: Callable[
            [HumanInputRequest],
            HumanInputResponse | Awaitable[HumanInputResponse],
        ],
    ) -> None:
        self._callback = callback

    async def request_input(self, request: HumanInputRequest) -> HumanInputResponse:
        result = self._callback(request)
        if inspect.isawaitable(result):
            return await result
        return result
