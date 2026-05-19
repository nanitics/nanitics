from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    DebateArgumentEvent,
    DebateCompleteEvent,
    DebateResolutionEvent,
    DebateStartEvent,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.base import Agent

# --- Models ---


class Argument(BaseModel):
    """A single argument in a debate.

    Attributes:
        round: Round number this argument was made in.
        agent_name: Name of the debating agent.
        position: The position being argued.
        content: The argument content.
    """

    model_config = ConfigDict(frozen=True)

    round: int
    agent_name: str
    position: str
    content: str


class DebateResolution(BaseModel):
    """The judge's verdict after evaluating a debate.

    Attributes:
        winner: Name of the winning agent, or ``None`` if no clear winner.
        reasoning: The judge's reasoning for the verdict.
        synthesis: Synthesis of the best arguments from both sides.
    """

    model_config = ConfigDict(frozen=True)

    winner: str | None
    reasoning: str
    synthesis: str


class Debater(BaseModel):
    """Pairs an agent with its assigned position in a debate.

    Attributes:
        agent: The agent that will argue the position.
        position: The position to argue (e.g. "for microservices").
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    agent: Agent
    position: str


class DebateResult(BaseModel):
    """Outcome of a debate.

    Attributes:
        resolution: The judge's verdict.
        transcript: All arguments in chronological order.
        rounds_completed: Number of debate rounds.
        termination_reason: Why the debate ended.
    """

    model_config = ConfigDict(frozen=True)

    resolution: DebateResolution
    transcript: list[Argument]
    rounds_completed: int
    termination_reason: str


# --- Protocols ---


@runtime_checkable
class ResolutionStrategy(Protocol):
    """Protocol for strategies that evaluate a debate transcript and produce a verdict."""

    async def resolve(self, transcript: list[Argument], task: str) -> DebateResolution: ...


# --- Resolution Strategies ---


class JudgeResolution:
    """Resolve a debate using a separate judge agent.

    The judge receives the full transcript and produces a free-form
    verdict. The output is used for both reasoning and synthesis.

    Args:
        judge: Agent to act as the debate judge.
    """

    def __init__(self, judge: Agent) -> None:
        self._judge = judge

    async def resolve(self, transcript: list[Argument], task: str) -> DebateResolution:
        formatted = _format_transcript(transcript)
        prompt = (
            f"You are judging a debate on: {task}\n\nHere is the full transcript:\n\n"
            f"{formatted}\n\nEvaluate the arguments and provide your judgment."
        )
        result = await self._judge.run(prompt)
        output = result.output or ""
        return DebateResolution(
            winner=None,
            reasoning=output,
            synthesis=output,
        )


class _JudgeVerdictSchema(BaseModel):
    winner: str
    reasoning: str
    synthesis: str


class LLMJudgeResolution:
    """Resolve a debate using an LLM with structured output.

    Produces a typed verdict with winner, reasoning, and synthesis.
    Falls back to unstructured output if structured parsing fails.

    Args:
        llm_client: LLM client for judging.
        criteria: Optional evaluation criteria to guide the judge.
    """

    def __init__(self, llm_client: LLMClient, criteria: str | None = None) -> None:
        self._llm_client = llm_client
        self._criteria = criteria

    async def resolve(self, transcript: list[Argument], task: str) -> DebateResolution:
        formatted = _format_transcript(transcript)
        criteria_section = f"\n\nEvaluation criteria: {self._criteria}" if self._criteria else ""
        prompt = (
            f"You are judging a debate on: {task}\n\n"
            f"Here is the full transcript:\n\n{formatted}\n\n"
            "Evaluate the arguments based on argument quality, logical coherence, and evidence handling."
            f"{criteria_section}\n\n"
            "Determine the winner, provide your reasoning, and synthesize the best arguments."
        )
        response = await self._llm_client.generate(
            system_prompt="You are an impartial debate judge.",
            messages=[Message(role="user", content=prompt)],
            output_schema=_JudgeVerdictSchema,
        )
        if response.parsed is not None:
            verdict: _JudgeVerdictSchema = cast(_JudgeVerdictSchema, response.parsed)
            return DebateResolution(
                winner=verdict.winner,
                reasoning=verdict.reasoning,
                synthesis=verdict.synthesis,
            )
        # Fallback if structured output fails
        content = response.content or ""
        return DebateResolution(
            winner=None,
            reasoning=content,
            synthesis=content,
        )


# --- Transcript Formatting ---


def _format_transcript(transcript: list[Argument]) -> str:
    if not transcript:
        return ""
    rounds: dict[int, list[Argument]] = {}
    for arg in transcript:
        rounds.setdefault(arg.round, []).append(arg)
    parts: list[str] = []
    for round_num in sorted(rounds):
        parts.append(f"Round {round_num}:")
        parts.extend(f"[{arg.position} - {arg.agent_name}]: {arg.content}" for arg in rounds[round_num])
        parts.append("")
    return "\n".join(parts).rstrip()


# --- Debate Controller ---


class Debate:
    """Structured adversarial reasoning between agents.

    Each debater argues an assigned position across multiple rounds.
    After all rounds, a resolution strategy evaluates the transcript
    and produces a verdict.

    Args:
        debaters: At least 2 debaters with assigned positions.
        emitter: Event emitter for debate events.
        resolution: Strategy for evaluating the debate and producing a verdict.
        max_rounds: Number of debate rounds.
        cancellation_token: Cancellation signal.

    Raises:
        ValueError: If fewer than 2 debaters are provided.
    """

    def __init__(
        self,
        *,
        debaters: list[Debater],
        emitter: EventEmitter,
        resolution: ResolutionStrategy,
        max_rounds: int = 3,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        if len(debaters) < 2:
            raise ValueError("Debate requires at least 2 debaters")
        self._debaters = debaters
        self._emitter = emitter
        self._resolution = resolution
        self._max_rounds = max_rounds
        self._cancellation_token = cancellation_token

    async def run(self, task: str) -> DebateResult:
        """Execute the debate.

        Runs all rounds sequentially, with each debater producing one
        argument per round. After completion, the resolution strategy
        evaluates the full transcript.

        Args:
            task: The topic or question to debate.

        Returns:
            A ``DebateResult`` with the verdict and full transcript.
        """
        strategy_name = type(self._resolution).__name__

        self._emitter.emit(
            DebateStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                task=task,
                debater_names=[d.agent.name for d in self._debaters],
                positions={d.agent.name: d.position for d in self._debaters},
                max_rounds=self._max_rounds,
                resolution_strategy=strategy_name,
            )
        )

        transcript: list[Argument] = []

        for round_num in range(1, self._max_rounds + 1):
            for debater in self._debaters:
                if round_num == 1:
                    formatted_task = (
                        f"You are arguing the **{debater.position}** position on: {task}. "
                        f"Present your opening argument."
                    )
                else:
                    formatted_transcript = _format_transcript(transcript)
                    formatted_task = (
                        f"You are arguing the **{debater.position}** position. "
                        f"Here is the debate so far:\n\n{formatted_transcript}\n\n"
                        "Respond to the opposing arguments. "
                        "Defend your position, address counterpoints, and strengthen your case."
                    )

                result = await debater.agent.bind(self._emitter).run(formatted_task)
                argument = Argument(
                    round=round_num,
                    agent_name=debater.agent.name,
                    position=debater.position,
                    content=result.output or "",
                )
                transcript.append(argument)

                self._emitter.emit(
                    DebateArgumentEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        round=round_num,
                        agent_name=debater.agent.name,
                        position=debater.position,
                        argument=argument.content,
                    )
                )

        resolution = await self._resolution.resolve(transcript, task)

        self._emitter.emit(
            DebateResolutionEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                winner=resolution.winner,
                reasoning=resolution.reasoning,
                rounds_completed=self._max_rounds,
            )
        )

        self._emitter.emit(
            DebateCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                winner=resolution.winner,
                rounds_completed=self._max_rounds,
                total_arguments=len(transcript),
                termination_reason="max_rounds",
            )
        )

        return DebateResult(
            resolution=resolution,
            transcript=transcript,
            rounds_completed=self._max_rounds,
            termination_reason="max_rounds",
        )
