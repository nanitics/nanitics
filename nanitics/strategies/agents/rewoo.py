from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from nanitics.infrastructure.llm.protocol import LLMClient, Message
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    PlanCreatedEvent,
    PlanStepDetail,
    PlanStepUpdatedEvent,
    ToolInfo,
    Usage,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.context import ContextManagement, ContextProvider
from nanitics.strategies.agents.evaluation import EvaluationVerdict, OutputEvaluator
from nanitics.strategies.prompts.builder import SystemPromptContributor
from nanitics.strategies.tools import Tool, ToolRegistry

from .base import Agent, AgentInput, AgentResult, _input_to_text

if TYPE_CHECKING:
    from nanitics.capabilities.planning.store import PlanStore

_VARIABLE_PATTERN = re.compile(r"#(\d+)")


class ReWOOStep(BaseModel):
    """A single step in a ReWOO execution plan.

    Attributes:
        step_number: Unique identifier for this step (1-indexed).
        description: What this step does.
        tool_name: Which tool to call.
        arguments: Tool arguments. May contain ``#N`` variable
            references to earlier step results.
        depends_on: Step numbers this step depends on.
    """

    model_config = ConfigDict(frozen=True)

    step_number: int
    description: str
    tool_name: str
    arguments: dict[str, Any]
    depends_on: list[int]


class ReWOOPlan(BaseModel):
    """Structured plan produced by the ReWOO planner phase.

    Attributes:
        steps: Ordered list of execution steps with dependencies.
    """

    model_config = ConfigDict(frozen=True)

    steps: list[ReWOOStep]


def _substitute_variables(arguments: dict[str, Any], variable_map: dict[int, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str):

            def _replace(match: re.Match[str]) -> str:
                step_num = int(match.group(1))
                return variable_map.get(step_num, f"[Step #{step_num} failed or not available]")

            result[key] = _VARIABLE_PATTERN.sub(_replace, value)
        else:
            result[key] = value
    return result


def _build_execution_levels(steps: list[ReWOOStep]) -> list[list[ReWOOStep]]:
    step_map = {s.step_number: s for s in steps}
    assigned: dict[int, int] = {}
    levels: list[list[ReWOOStep]] = []

    def _get_level(step_number: int, visited: set[int]) -> int:
        if step_number in assigned:
            return assigned[step_number]
        if step_number in visited:
            raise ValueError(f"Circular dependency detected involving step #{step_number}")
        visited.add(step_number)
        step = step_map[step_number]
        if not step.depends_on:
            level = 0
        else:
            level = max(_get_level(dep, visited) for dep in step.depends_on) + 1
        assigned[step_number] = level
        visited.discard(step_number)
        return level

    for step in steps:
        _get_level(step.step_number, set())

    max_level = max(assigned.values()) if assigned else 0
    levels = [[] for _ in range(max_level + 1)]
    for step in steps:
        levels[assigned[step.step_number]].append(step)

    return levels


_PLANNER_INSTRUCTIONS = """Create a complete execution plan for the task. Each step must specify a tool call.

Available tools:
{tool_descriptions}

Rules:
- Each step has a step number (#1, #2, #3, ...) that identifies its result
- Later steps can reference earlier results using #N notation in arguments
- List which step numbers each step depends on
- Include only tool invocation steps — no reasoning or analysis steps
- Order steps to satisfy dependencies
- All steps needed to answer the task must be included"""


class ReWOOAgent(Agent):
    """Agent that plans all steps upfront, then executes without re-reasoning.

    Implements the ReWOO (Reasoning Without Observation) pattern with three
    phases:

    1. **Planner** — LLM generates a structured ``ReWOOPlan`` specifying
       tool calls and their dependencies.
    2. **Worker** — Executes tool calls in dependency order. Independent
       steps within the same dependency level run in parallel.
    3. **Solver** — LLM synthesizes all tool results into a final answer.

    This minimizes LLM calls (typically 2–3 regardless of tool count) at
    the cost of not being able to adapt the plan based on intermediate
    results.

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model to use.
        emitter: Event emitter for observability.
        system_prompt: Base system prompt text.
        tools: Tools available for execution.
        plan_store: Persistent store for the generated plan.
        max_observation_length: Truncate tool results beyond this length.
        cancellation_token: External cancellation signal.
        output_evaluator: Quality gate for the solver's final answer.
        context_manager: Context window management.
        context_providers: Inject context before each LLM call.
        prompt_contributors: Additional system prompt sections.
        output_schema: Pydantic model for structured solver output. When
            provided, the solver LLM call is constrained to produce JSON
            matching this schema, mirroring the structured-output surface
            on ``ReasoningAgent`` / ``ReActAgent``. ``None`` by default —
            the solver produces free-text output.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        tools: Sequence[Tool],
        plan_store: PlanStore,
        max_observation_length: int = 5000,
        cancellation_token: CancellationToken | None = None,
        output_evaluator: OutputEvaluator | None = None,
        context_manager: ContextManagement | None = None,
        context_providers: list[ContextProvider] | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            llm_client=llm_client,
            emitter=emitter,
            system_prompt=system_prompt,
            cancellation_token=cancellation_token,
            context_manager=context_manager,
            context_providers=context_providers,
            output_evaluator=output_evaluator,
            prompt_contributors=prompt_contributors,
        )
        self._tool_registry = ToolRegistry(emitter_provider=lambda: self._emitter)
        self._tool_registry.register_all(tools)
        self._plan_store = plan_store
        self._max_observation_length = max_observation_length
        self._output_schema = output_schema

    def _agent_type(self) -> str:
        return "rewoo"

    def _active_capabilities(self) -> list[str]:
        caps = super()._active_capabilities()
        caps.append("planning")
        caps.append("tool_use")
        return caps

    def _get_tools_available(self) -> list[str]:
        return [s.name for s in self._tool_registry.list_schemas()]

    def _get_tool_schemas(self) -> list[ToolInfo]:
        return [
            ToolInfo(
                name=s.name,
                description=s.description,
                requires_approval=s.requires_approval,
            )
            for s in self._tool_registry.list_schemas()
        ]

    def _build_planner_prompt(self) -> str:
        schemas = self._tool_registry.list_schemas()
        tool_lines: list[str] = []
        for s in schemas:
            tool_lines.append(f"- {s.name}: {s.description}")
            tool_lines.append(f"  Parameters: {s.parameters}")
        tool_descriptions = "\n".join(tool_lines)
        instructions = _PLANNER_INSTRUCTIONS.format(tool_descriptions=tool_descriptions)
        return f"{self._system_prompt}\n\n{instructions}"

    def _build_solver_input(self, input: AgentInput, plan: ReWOOPlan, variable_map: dict[int, str]) -> str:
        task_text = _input_to_text(input)
        lines = [f"Task: {task_text}", "", "Plan and Observations:"]
        for step in plan.steps:
            lines.append(f"Step #{step.step_number}: {step.description} (tool: {step.tool_name})")
            observation = variable_map.get(step.step_number, "[No result]")
            lines.append(f"Observation: {observation}")
            lines.append("")
        lines.append("Synthesize these results into a comprehensive answer to the original task.")
        lines.append("Note any steps that failed and their impact on the completeness of the answer.")
        return "\n".join(lines)

    async def _execute(self, input: AgentInput) -> AgentResult:
        from nanitics.capabilities.planning.models import Plan, PlanStatus, PlanStep, StepStatus

        usages: list[Usage] = []
        all_messages: list[Message] = []
        step_number = 0

        # --- Phase 1: Planner ---
        with self._emitter.span("planner"):
            planner_prompt = self._build_planner_prompt()
            original_system_prompt = self._system_prompt
            self._system_prompt = planner_prompt

            planner_messages = [Message(role="user", content=input)]
            response = await self._call_llm(planner_messages, output_schema=ReWOOPlan)
            usages.append(response.usage)

            self._system_prompt = original_system_prompt

            rewoo_plan: ReWOOPlan = cast(ReWOOPlan, response.parsed)
            all_messages.append(Message(role="user", content=input))
            all_messages.append(Message(role="assistant", content=response.content))

            # Map to Plan and store
            plan_steps = [
                PlanStep(
                    description=f"#{s.step_number}: {s.description}",
                    metadata={"tool": s.tool_name, "args": s.arguments, "variable": f"#{s.step_number}"},
                )
                for s in rewoo_plan.steps
            ]
            plan = Plan(name=f"{self._name}-plan", steps=plan_steps)
            await self._plan_store.save(plan)

            levels = _build_execution_levels(rewoo_plan.steps)
            level_map = {}
            for level_idx, level_steps in enumerate(levels):
                for s in level_steps:
                    level_map[s.step_number] = level_idx

            step_details = [
                PlanStepDetail(
                    step_id=plan_steps[s.step_number - 1].id,
                    description=s.description,
                    metadata={
                        "tool": s.tool_name,
                        "args": s.arguments,
                        "variable": f"#{s.step_number}",
                        "depends_on": s.depends_on,
                        "execution_level": level_map[s.step_number],
                    },
                )
                for s in rewoo_plan.steps
            ]

            self._emitter.emit(
                PlanCreatedEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    plan_id=plan.id,
                    plan_name=plan.name,
                    step_count=len(plan_steps),
                    goal_count=0,
                    steps=step_details,
                )
            )

            step_number += 1
            self._emit_step(step_number, artifact=rewoo_plan.model_dump())

        # --- Phase 2: Worker ---
        variable_map: dict[int, str] = {}
        with self._emitter.span("worker"):
            for level in levels:
                if self._is_cancelled:
                    self._emit_safety_cancellation(step_number)
                    break

                async def _execute_step(step: ReWOOStep) -> None:
                    resolved_args = _substitute_variables(step.arguments, variable_map)
                    from nanitics.infrastructure.llm.protocol import ToolCall as LLMToolCall

                    tool_call = LLMToolCall(
                        id=f"rewoo-step-{step.step_number}",
                        name=step.tool_name,
                        arguments=dict(resolved_args.items()),
                    )

                    plan_step = plan_steps[step.step_number - 1]
                    try:
                        with self._emitter.span(f"step-{step.step_number}"):
                            tool_result = await self._tool_registry.dispatch(tool_call)
                            result_text = tool_result.content[: self._max_observation_length]
                            variable_map[step.step_number] = result_text

                            updated_step = plan_step.model_copy(
                                update={"status": StepStatus.completed, "result": result_text}
                            )
                            plan_steps[step.step_number - 1] = updated_step
                    except Exception as e:
                        error_text = str(e)[: self._max_observation_length]
                        variable_map[step.step_number] = f"[ERROR] {error_text}"

                        self._emit_error(e, step_number=step.step_number)

                        updated_step = plan_step.model_copy(update={"status": StepStatus.failed, "result": error_text})
                        plan_steps[step.step_number - 1] = updated_step

                    self._emitter.emit(
                        PlanStepUpdatedEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            plan_id=plan.id,
                            step_id=plan_steps[step.step_number - 1].id,
                            step_description=plan_steps[step.step_number - 1].description,
                            previous_status=StepStatus.not_started,
                            new_status=str(plan_steps[step.step_number - 1].status),
                            has_result=plan_steps[step.step_number - 1].result is not None,
                        )
                    )

                await asyncio.gather(*[_execute_step(s) for s in level])

                for s in level:
                    step_number += 1
                    obs = variable_map.get(s.step_number, "[No result]")
                    self._emit_step(step_number, action=s.tool_name, observation=obs)

            # Update plan in store
            all_completed = all(ps.status == StepStatus.completed for ps in plan_steps)
            final_status = PlanStatus.completed if all_completed else PlanStatus.active
            updated_plan = plan.model_copy(
                update={
                    "steps": plan_steps,
                    "status": final_status,
                }
            )
            await self._plan_store.update(updated_plan)

        # --- Phase 3: Solver ---
        with self._emitter.span("solver"):
            solver_input = self._build_solver_input(input, rewoo_plan, variable_map)
            solver_messages = [Message(role="user", content=solver_input)]
            response = await self._call_llm(solver_messages, output_schema=self._output_schema)
            usages.append(response.usage)

            output = response.content
            all_messages.append(Message(role="user", content=solver_input))
            all_messages.append(Message(role="assistant", content=output))

            termination_reason = "complete"

            if self._output_evaluator is not None:
                revision_count = 0
                max_revisions = self._output_evaluator.max_revisions

                if self._is_truncated(response):
                    from nanitics.strategies.agents.evaluation import EvaluationResult

                    eval_result = EvaluationResult(
                        verdict=EvaluationVerdict.REVISE,
                        score=None,
                        feedback=self._TRUNCATION_FEEDBACK,
                        evaluator_name="truncation",
                    )
                    self._emit_truncation_events(revision_count, max_revisions)
                else:
                    eval_result = await self._evaluate_output(
                        output or "",
                        input,
                        all_messages,
                        revision_count,
                    )

                while eval_result.verdict == EvaluationVerdict.REVISE and revision_count < max_revisions:
                    solver_messages.append(Message(role="assistant", content=output))
                    solver_messages.append(Message(role="user", content=eval_result.feedback or ""))
                    if eval_result.evaluator_name != "truncation":
                        self._emit_evaluation_revision(
                            eval_result.feedback or "",
                            revision_count,
                            max_revisions,
                        )
                    revision_count += 1

                    response = await self._call_llm(solver_messages, output_schema=self._output_schema)
                    usages.append(response.usage)
                    output = response.content
                    all_messages.append(Message(role="user", content=eval_result.feedback or ""))
                    all_messages.append(Message(role="assistant", content=output))

                    if self._is_truncated(response):
                        eval_result = EvaluationResult(
                            verdict=EvaluationVerdict.REVISE,
                            score=None,
                            feedback=self._TRUNCATION_FEEDBACK,
                            evaluator_name="truncation",
                        )
                        self._emit_truncation_events(revision_count, max_revisions)
                    else:
                        eval_result = await self._evaluate_output(
                            output or "",
                            input,
                            all_messages,
                            revision_count,
                        )

                if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
                    termination_reason = "evaluation_skipped"
                elif eval_result.verdict != EvaluationVerdict.ACCEPT:
                    termination_reason = "evaluation_failed"

            step_number += 1
            self._emit_step(
                step_number,
                thought=response.reasoning_text,
                artifact=response.parsed.model_dump() if response.parsed else None,
            )

        total_steps = 1 + len(rewoo_plan.steps) + 1  # planner + worker steps + solver
        return AgentResult(
            output=output,
            parsed=response.parsed,
            total_steps=total_steps,
            termination_reason=termination_reason,
            messages=all_messages,
            usage=self._aggregate_usage(usages),
        )
