from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.core.agents.base import AgentInput, AgentResult
from nanitics.core.agents.context import ContextManagement, ContextProvider
from nanitics.core.agents.errors import ErrorHandling
from nanitics.core.agents.evaluation import (
    EvaluationContext,
    EvaluationVerdict,
    OutputEvaluator,
)
from nanitics.core.agents.react import ReActAgent
from nanitics.core.prompts.builder import SystemPromptBuilder
from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import EvaluationEvent
from nanitics.safety.cancellation import CancellationToken


class FinalOutputStrategy(StrEnum):
    """How the orchestrator produces its final output.

    - ``SYNTHESIZE`` (default): the coordinator LLM produces a final
      text turn that composes the specialists' results into a single
      response. Preserves the documented orchestrator contract.
    - ``RELAY_LAST``: the orchestrator returns the most recent
      specialist ``tool_result`` content verbatim as
      ``AgentResult.output``. The coordinator's final text turn — if
      any — is discarded. Choose this when one specialist produces the
      actual deliverable and a coordinator summary would only compress
      or paraphrase it.
    """

    SYNTHESIZE = "synthesize"
    RELAY_LAST = "relay_last"


def orchestrator_prompt_section(specialists: list[AgentTool]) -> tuple[str, str]:
    """Build the orchestration system prompt section listing specialists and strategy.

    Returns a (section_name, section_content) tuple suitable for use with
    ``SystemPromptBuilder.add_section``. The content lists each specialist's
    name and description, followed by a delegation strategy.

    Args:
        specialists: AgentTool instances representing available specialist agents.

    Returns:
        Tuple of ("Orchestration", formatted_content).
    """
    specialist_lines = [f"- **{s.schema.name}**: {s.schema.description}" for s in specialists]
    specialist_listing = "\n".join(specialist_lines)

    content = (
        f"## Available Specialists\n"
        f"{specialist_listing}\n\n"
        f"## Strategy\n"
        f"1. Analyze the task and identify what needs to be done.\n"
        f"2. Break complex tasks into subtasks matched to specialist capabilities.\n"
        f"3. Delegate by calling specialist tools with clear, specific task descriptions.\n"
        f"4. Subtasks that are independent can be delegated in the same step (parallel tool calls).\n"
        f"5. After receiving specialist results, assess completeness and quality; "
        f"if a specialist's result is insufficient, re-delegate with refined instructions.\n"
        f"6. Combine the specialists' findings into the response, preserving their "
        f"substance rather than compressing them."
    )
    return ("Orchestration", content)


class _OrchestratorAgent(ReActAgent):
    """``ReActAgent`` subclass that applies the orchestrator's final-output policy.

    Under ``FinalOutputStrategy.SYNTHESIZE`` (the default) this behaves
    identically to ``ReActAgent``. Under ``FinalOutputStrategy.RELAY_LAST``
    the post-run output is replaced by the most recent
    ``tool_result`` content in the conversation — the coordinator's
    synthesis text (if any) is discarded.

    When a ``RELAY_LAST`` orchestrator carries an ``OutputEvaluator``,
    the evaluator is removed from the inner ``ReActAgent`` loop (which
    would otherwise evaluate the discarded synthesis text) and re-run
    here against the relayed content. ``ACCEPT`` keeps the run's
    ``termination_reason``; ``REVISE`` is non-actionable without a
    coordinator LLM turn and downgrades the result to
    ``evaluation_skipped`` with an emitted event; ``REJECT`` marks the
    run ``evaluation_failed``.
    """

    def __init__(
        self,
        *,
        final_output_strategy: FinalOutputStrategy,
        relay_evaluator: OutputEvaluator | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._final_output_strategy = final_output_strategy
        self._relay_evaluator = relay_evaluator

    async def _execute(self, input: AgentInput) -> AgentResult:
        result = await super()._execute(input)
        if self._final_output_strategy is FinalOutputStrategy.SYNTHESIZE:
            return result
        return await self._apply_relay(result, input)

    async def _apply_relay(self, result: AgentResult, task_input: AgentInput) -> AgentResult:
        relayed_output = self._extract_last_tool_result(result.messages)
        if relayed_output is None:
            # Fallback: the coordinator answered directly without delegating,
            # or the run terminated before any tool_result was produced.
            # Keep the existing output — it is what SYNTHESIZE would have
            # produced in the same shape.
            return result

        termination_reason = result.termination_reason
        if self._relay_evaluator is not None and termination_reason == "complete":
            eval_result = await self._relay_evaluator.evaluate(
                relayed_output,
                EvaluationContext(messages=result.messages, task_input=task_input),
            )
            self._emitter.emit(
                EvaluationEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    evaluator_name=eval_result.evaluator_name,
                    verdict=eval_result.verdict.value,
                    score=eval_result.score,
                    feedback=eval_result.feedback,
                    revision_attempt=0,
                )
            )
            if eval_result.verdict == EvaluationVerdict.REVISE:
                # REVISE is non-actionable under RELAY_LAST (there is no
                # coordinator LLM turn to act on the feedback). Downgrade
                # to evaluation_skipped rather than looping or rewriting.
                termination_reason = "evaluation_skipped"
            elif eval_result.verdict == EvaluationVerdict.REJECT:
                termination_reason = "evaluation_failed"
            elif eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
                termination_reason = "evaluation_skipped"

        return result.model_copy(
            update={
                "output": relayed_output,
                "termination_reason": termination_reason,
            }
        )

    @staticmethod
    def _extract_last_tool_result(messages: list[Message]) -> str | None:
        """Return the content of the most recent ``tool_result`` message.

        ``ToolResult.content`` is always a string — the str branch covers
        the full range of inputs produced by the tool registry. Non-str
        ``tool_result`` content is not generated by any in-tree code
        path and, if it somehow appears, callers fall back to the
        coordinator's synthesis output instead of inventing a string.
        """
        for msg in reversed(messages):
            if msg.role == "tool_result" and isinstance(msg.content, str):
                return msg.content
        return None


def create_orchestrator(
    *,
    name: str,
    llm_client: LLMClient,
    emitter: EventEmitter,
    specialists: list[AgentTool],
    system_prompt: str | None = None,
    max_iterations: int = 15,
    cancellation_token: CancellationToken | None = None,
    error_handler: ErrorHandling | None = None,
    context_manager: ContextManagement | None = None,
    context_providers: list[ContextProvider] | None = None,
    output_evaluator: OutputEvaluator | None = None,
    output_schema: type[BaseModel] | None = None,
    final_output_strategy: FinalOutputStrategy = FinalOutputStrategy.SYNTHESIZE,
) -> ReActAgent:
    """Create a ReActAgent configured as an orchestrator for specialist agents.

    The orchestrator analyzes incoming tasks, delegates subtasks to the
    appropriate specialists (provided as ``AgentTool`` instances), and —
    by default — synthesizes their results into a coherent response.

    If no ``system_prompt`` is provided, one is generated automatically
    listing available specialists and a delegation strategy.

    The ``final_output_strategy`` controls how the coordinator's final
    output is produced:

    - ``FinalOutputStrategy.SYNTHESIZE`` (default): the coordinator LLM
      composes a final text turn that combines the specialists' results.
    - ``FinalOutputStrategy.RELAY_LAST``: the orchestrator returns the
      most recent specialist ``tool_result`` content verbatim, skipping
      a coordinator synthesis rewrite. Incompatible with
      ``output_schema`` — schema-constrained output is itself a
      synthesis step and requires the coordinator's final LLM turn.

    Args:
        name: Name for the orchestrator agent.
        llm_client: LLM client for orchestration reasoning.
        emitter: Event emitter for tracing.
        specialists: Specialist agents wrapped as ``AgentTool`` instances.
        system_prompt: Optional custom system prompt. When provided, the
            auto-generated specialist listing is skipped.
        max_iterations: Maximum orchestrator loop iterations.
        cancellation_token: Shared cancellation signal, propagated to all
            specialists.
        error_handler: Error handling configuration.
        context_manager: Context management configuration.
        context_providers: Additional context providers.
        output_evaluator: Output quality evaluator. Under ``RELAY_LAST``
            the evaluator runs against the relayed specialist output;
            ``REVISE`` is non-actionable and downgrades the result to
            ``evaluation_skipped``.
        output_schema: Pydantic model for structured output. Rejected
            when combined with ``RELAY_LAST``.
        final_output_strategy: How the coordinator's final output is
            produced. See ``FinalOutputStrategy``.

    Returns:
        A configured orchestrator agent with specialist tools attached.

    Raises:
        ValueError: If ``final_output_strategy`` is ``RELAY_LAST`` and an
            ``output_schema`` is also provided.
    """
    if final_output_strategy is FinalOutputStrategy.RELAY_LAST and output_schema is not None:
        raise ValueError(
            "RELAY_LAST is incompatible with output_schema; schema-constrained output requires a synthesis turn."
        )

    if system_prompt is None:
        builder = SystemPromptBuilder()
        builder.add_section(
            "Role",
            "You are an orchestrating agent that coordinates specialist agents to accomplish complex tasks.",
        )
        section_name, section_content = orchestrator_prompt_section(specialists)
        builder.add_section(section_name, section_content)
        resolved_prompt = builder.build()
    else:
        resolved_prompt = system_prompt

    for specialist in specialists:
        specialist._caller_name = name
        if cancellation_token is not None:
            specialist.cancellation_token = cancellation_token

    # Under RELAY_LAST, the inner ReActAgent loop must not evaluate the
    # coordinator's discarded synthesis text. The orchestrator wrapper
    # re-runs the evaluator against the relayed specialist content.
    inner_evaluator: OutputEvaluator | None
    relay_evaluator: OutputEvaluator | None
    if final_output_strategy is FinalOutputStrategy.RELAY_LAST:
        inner_evaluator = None
        relay_evaluator = output_evaluator
    else:
        inner_evaluator = output_evaluator
        relay_evaluator = None

    return _OrchestratorAgent(
        final_output_strategy=final_output_strategy,
        relay_evaluator=relay_evaluator,
        name=name,
        llm_client=llm_client,
        emitter=emitter,
        system_prompt=resolved_prompt,
        tools=specialists,
        max_iterations=max_iterations,
        cancellation_token=cancellation_token,
        error_handler=error_handler,
        context_manager=context_manager,
        context_providers=context_providers,
        output_evaluator=inner_evaluator,
        output_schema=output_schema,
    )
