from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from nanitics.composition.threads.store import ThreadLocks, ThreadStore
from nanitics.infrastructure.errors import AgentIterationLimitError, AgentToolCallLimitError
from nanitics.infrastructure.llm.protocol import LLMClient, Message, ToolCall
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    ErrorCorrectionEvent,
    ErrorDegradationEvent,
    ExecutionResumedEvent,
    ToolInfo,
    Usage,
    WorkingMemoryUpdateEvent,
)
from nanitics.safety.cancellable_dispatch import RunCancelled, run_cancellable
from nanitics.safety.cancellation import CancellationToken
from nanitics.safety.iteration_limits import IterationLimiter, ToolCallLimiter
from nanitics.strategies.agents.context import ContextManagement, ContextProvider
from nanitics.strategies.agents.errors import ErrorHandling
from nanitics.strategies.agents.evaluation import (
    EvaluationResult,
    EvaluationVerdict,
    OutputEvaluator,
)
from nanitics.strategies.agents.parsing import (
    parse_working_memory_update,
    strip_working_memory_block,
)
from nanitics.strategies.agents.working_memory import WorkingMemory, WorkingMemoryContributor
from nanitics.strategies.prompts.builder import SystemPromptContributor
from nanitics.strategies.tools import Tool, ToolRegistry

from .base import Agent, AgentInput, AgentResult


class ReActAgent(Agent):
    """Agent that interleaves reasoning and action in a loop.

    The ReAct (Reasoning + Acting) pattern is the default agent type. On each
    step the agent sends the conversation to the LLM, which either returns a
    final text answer or requests one or more tool calls. Tool results are
    appended to the conversation and the loop continues.

    Supports working memory, output evaluation with revision, error handling
    with self-correction, context management, and durable execution
    (checkpoint/resume).

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model to use.
        emitter: Event emitter for observability.
        system_prompt: Base system prompt text.
        tools: Tools available to the agent. Required — pass an empty
            sequence for a tool-less ReAct loop (reasoning only).
        max_iterations: Maximum reasoning steps before forced termination
            (default: 10).
        max_tool_calls: Maximum total tool calls across all reasoning steps.
            When ``None`` (default), no tool call limit is applied.
        cancellation_token: External cancellation signal.
        error_handler: Error recovery strategy.
        context_manager: Context window management.
        context_providers: Inject context before each LLM call.
        working_memory: Structured scratchpad the agent can read and
            update across steps.
        output_evaluator: Quality gate for the final output.
        prompt_contributors: Additional system prompt sections.
        tool_state: Per-run state dict injected into tool execution via
            ``ToolContext``.
        output_schema: Pydantic model for structured output. When
            provided, an additional LLM call is made after the tool-use
            loop to produce schema-constrained JSON.
        initial_messages: Optional prior conversation messages to prepend
            before the current user input. Enables multi-turn conversations
            where history is loaded from an external store.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        tools: Sequence[Tool],
        max_iterations: int = 10,
        max_tool_calls: int | None = None,
        cancellation_token: CancellationToken | None = None,
        error_handler: ErrorHandling | None = None,
        context_manager: ContextManagement | None = None,
        context_providers: list[ContextProvider] | None = None,
        working_memory: WorkingMemory | None = None,
        output_evaluator: OutputEvaluator | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
        tool_state: dict[str, Any] | None = None,
        streaming: bool = False,
        output_schema: type[BaseModel] | None = None,
        initial_messages: list[Message] | None = None,
        run_id: str | None = None,
        thread_store: ThreadStore | None = None,
        thread_locks: ThreadLocks | None = None,
    ) -> None:
        if run_id is not None:
            tool_state = dict(tool_state) if tool_state else {}
            tool_state.setdefault("run_id", run_id)
        contributors: list[SystemPromptContributor] = []
        if working_memory is not None:
            contributors.append(WorkingMemoryContributor())
        if prompt_contributors is not None:
            contributors.extend(prompt_contributors)
        super().__init__(
            name=name,
            llm_client=llm_client,
            emitter=emitter,
            system_prompt=system_prompt,
            cancellation_token=cancellation_token,
            error_handler=error_handler,
            context_manager=context_manager,
            context_providers=context_providers,
            output_evaluator=output_evaluator,
            prompt_contributors=contributors if contributors else None,
            streaming=streaming,
            thread_store=thread_store,
            thread_locks=thread_locks,
        )
        self._tool_registry = ToolRegistry(
            tool_state=tool_state,
            emitter_provider=lambda: self._emitter,
        )
        self._tool_registry.register_all(tools)
        self._limiter = IterationLimiter(max_iterations)
        self._tool_call_limiter = ToolCallLimiter(max_tool_calls) if max_tool_calls is not None else None
        self._working_memory = working_memory
        self._output_schema = output_schema
        self._initial_messages = list(initial_messages) if initial_messages else None

    @property
    def supports_dynamic_tools(self) -> bool:
        return True

    def _agent_type(self) -> str:
        return "react"

    def _active_capabilities(self) -> list[str]:
        caps = super()._active_capabilities()
        caps.append("tool_use")
        if self._output_schema is not None:
            caps.append("structured_output")
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

    def add_tools(self, tools: Sequence[Tool]) -> list[str]:
        """Register additional tools, skipping any already registered."""
        added: list[str] = []
        for tool in tools:
            name = tool.schema.name
            if not self._tool_registry.has(name):
                self._tool_registry.register(tool)
                added.append(name)
        return added

    def remove_tools(self, names: Sequence[str]) -> None:
        """Unregister tools by name."""
        for name in names:
            self._tool_registry.unregister(name)

    def update_tool_state(self, key: str, value: Any) -> None:
        self._tool_registry.update_state(key, value)

    async def _execute(self, input: AgentInput, *, thread_key: str | None = None) -> AgentResult:
        tool_schemas = self._tool_registry.list_schemas()
        available_tools = [s.name for s in tool_schemas]

        # --- Resume path ---
        if self._resume_state is not None:
            return await self._execute_resume(input, tool_schemas, available_tools)

        # --- Normal path ---
        self._limiter.reset()
        self._error_handler.reset()
        if self._tool_call_limiter is not None:
            self._tool_call_limiter.reset()
        if self._context_manager is not None:
            self._context_manager.reset()
        if self._working_memory is not None:
            self._working_memory.reset()
        messages: list[Message] = []
        if self._initial_messages:
            messages.extend(self._initial_messages)
        messages.extend(await self._load_thread_prefix(thread_key))
        messages.append(Message(role="user", content=input))
        usages: list[Usage] = []

        return await self._run_loop(
            task_input=input,
            messages=messages,
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            step_number=0,
            revision_count=0,
            usages=usages,
        )

    async def _run_loop(
        self,
        *,
        task_input: AgentInput,
        messages: list[Message],
        tool_schemas: list[Any],
        available_tools: list[str],
        step_number: int,
        revision_count: int,
        usages: list[Usage],
    ) -> AgentResult:
        output: str | None = None
        parsed = None
        termination_reason = "complete"

        while True:  # Outer loop: handles REVISE re-entry for structured output
            termination_reason = "complete"

            while True:  # Tool loop
                if self._is_cancelled:
                    self._emit_safety_cancellation(step_number)
                    termination_reason = "cancelled"
                    break

                try:
                    self._limiter.step()
                except AgentIterationLimitError:
                    self._emit_safety_iteration_limit(
                        self._limiter.current_iteration,
                        self._limiter.max_iterations,
                        step_number,
                    )
                    termination_reason = "iteration_limit"
                    break

                step_number += 1

                with self._emitter.span(f"step-{step_number}"):
                    response = await self._call_llm(
                        messages,
                        tools=tool_schemas if tool_schemas else None,
                    )
                    usages.append(response.usage)

                    assistant_content = response.content
                    if self._working_memory is not None and assistant_content:
                        wm_update = parse_working_memory_update(assistant_content)
                        if wm_update is not None:
                            previous = self._working_memory.read()
                            self._working_memory.write(wm_update)
                            self._emitter.emit(
                                WorkingMemoryUpdateEvent(
                                    trace_id=self._emitter.trace_id,
                                    span_id=self._emitter.span_id,
                                    parent_span_id=self._emitter.parent_span_id,
                                    previous_content=previous,
                                    new_content=self._working_memory.read() or wm_update,
                                    source="llm_output",
                                )
                            )
                            assistant_content = strip_working_memory_block(assistant_content)
                            if not assistant_content:
                                assistant_content = "[Working memory updated]"

                    if not response.tool_calls:
                        if self._output_evaluator is not None and self._output_schema is None:
                            max_revisions = self._output_evaluator.max_revisions

                            if self._is_truncated(response):
                                if revision_count < max_revisions:
                                    messages.append(Message(role="assistant", content=assistant_content))
                                    messages.append(Message(role="user", content=self._TRUNCATION_FEEDBACK))
                                    self._emit_truncation_events(revision_count, max_revisions)
                                    revision_count += 1
                                    continue
                                termination_reason = "evaluation_failed"
                                self._emit_evaluation_exhausted(
                                    evaluator_name="truncation",
                                    verdict="revise",
                                    revision_count=revision_count,
                                    max_revisions=max_revisions,
                                    feedback=self._TRUNCATION_FEEDBACK,
                                )
                                output = assistant_content
                                messages.append(Message(role="assistant", content=assistant_content))
                                self._emit_step(
                                    step_number,
                                    thought=response.reasoning_text,
                                    observation=response.content or None,
                                )
                                break

                            eval_result = await self._evaluate_output(
                                assistant_content or "",
                                task_input,
                                messages,
                                revision_count,
                            )
                            if eval_result.verdict == EvaluationVerdict.REVISE and revision_count < max_revisions:
                                messages.append(Message(role="assistant", content=assistant_content))
                                messages.append(Message(role="user", content=eval_result.feedback or ""))
                                self._emit_evaluation_revision(
                                    eval_result.feedback or "",
                                    revision_count,
                                    max_revisions,
                                )
                                revision_count += 1
                                continue
                            if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
                                termination_reason = "evaluation_skipped"
                            elif eval_result.verdict != EvaluationVerdict.ACCEPT:
                                termination_reason = "evaluation_failed"
                                self._emit_evaluation_exhausted(
                                    evaluator_name=eval_result.evaluator_name,
                                    verdict=eval_result.verdict.value,
                                    revision_count=revision_count,
                                    max_revisions=max_revisions,
                                    feedback=eval_result.feedback,
                                )

                        output = assistant_content
                        messages.append(Message(role="assistant", content=assistant_content))
                        self._emit_step(
                            step_number,
                            thought=response.reasoning_text,
                            observation=response.content or None,
                        )
                        break

                    messages.append(
                        Message(
                            role="assistant",
                            content=assistant_content,
                            tool_calls=response.tool_calls,
                        )
                    )

                    try:
                        await self._dispatch_tool_batch(
                            response.tool_calls,
                            messages,
                            available_tools,
                            step_number,
                            revision_count,
                            usages,
                        )
                    except RunCancelled as exc:
                        self._emit_safety_cancellation(exc.step_number or step_number)
                        termination_reason = "cancelled"
                        break

                    if self._tool_call_limiter is not None:
                        try:
                            self._tool_call_limiter.step(len(response.tool_calls))
                        except AgentToolCallLimitError:
                            self._emit_safety_tool_call_limit(
                                self._tool_call_limiter.current_tool_calls,
                                self._tool_call_limiter.max_tool_calls,
                                step_number,
                            )
                            action = ", ".join(tc.name for tc in response.tool_calls)
                            observation = self._format_observations(response.tool_calls, messages)
                            self._emit_step(
                                step_number,
                                thought=response.reasoning_text,
                                action=action,
                                observation=observation,
                            )
                            termination_reason = "tool_call_limit"
                            break

                    action = ", ".join(tc.name for tc in response.tool_calls)
                    observation = self._format_observations(response.tool_calls, messages)
                    self._emit_step(
                        step_number,
                        thought=response.reasoning_text,
                        action=action,
                        observation=observation,
                    )

            # --- Structured final call ---
            if self._output_schema is not None and termination_reason == "complete":
                step_number += 1
                with self._emitter.span(f"step-{step_number}"):
                    messages.append(
                        Message(role="user", content="Based on your analysis, produce the final structured output.")
                    )
                    structured_response = await self._call_llm(messages, output_schema=self._output_schema)
                    usages.append(structured_response.usage)
                    output = structured_response.content
                    parsed = structured_response.parsed
                    messages.append(Message(role="assistant", content=output))

                    if self._output_evaluator is not None:
                        max_revisions = self._output_evaluator.max_revisions

                        if self._is_truncated(structured_response):
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
                                task_input,
                                messages,
                                revision_count,
                            )

                        if eval_result.verdict == EvaluationVerdict.REVISE and revision_count < max_revisions:
                            messages.append(Message(role="user", content=eval_result.feedback or ""))
                            if eval_result.evaluator_name != "truncation":
                                self._emit_evaluation_revision(
                                    eval_result.feedback or "",
                                    revision_count,
                                    max_revisions,
                                )
                            revision_count += 1
                            self._emit_step(
                                step_number,
                                thought=structured_response.reasoning_text,
                                artifact=parsed.model_dump() if parsed else None,
                            )
                            continue  # Re-enter outer loop → tool loop

                        if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
                            termination_reason = "evaluation_skipped"
                        elif eval_result.verdict != EvaluationVerdict.ACCEPT:
                            termination_reason = "evaluation_failed"
                            self._emit_evaluation_exhausted(
                                evaluator_name=eval_result.evaluator_name,
                                verdict=eval_result.verdict.value,
                                revision_count=revision_count,
                                max_revisions=max_revisions,
                                feedback=eval_result.feedback,
                            )

                    self._emit_step(
                        step_number,
                        thought=structured_response.reasoning_text,
                        artifact=parsed.model_dump() if parsed else None,
                    )

            break  # Exit outer loop

        return AgentResult(
            output=output,
            parsed=parsed,
            total_steps=step_number,
            termination_reason=termination_reason,
            messages=messages,
            usage=self._aggregate_usage(usages),
        )

    async def _dispatch_tool_batch(
        self,
        tool_calls: list[ToolCall],
        messages: list[Message],
        available_tools: list[str],
        step_number: int,
        revision_count: int,
        usages: list[Usage],
        *,
        start_index: int = 0,
    ) -> None:
        """Dispatch tool calls, catching SuspendExecution to build checkpoint state."""
        from nanitics.composition.durability.suspension import SuspendExecution

        tool_attempts: dict[int, int] = {}
        completed_tool_results: dict[int, dict[str, Any]] = {}

        for i, tool_call in enumerate(tool_calls):
            if i < start_index:
                continue
            tool_attempts.setdefault(i, 0)
            try:
                result = await run_cancellable(
                    self._tool_registry.dispatch(tool_call),
                    self._cancellation_token,
                    tool_name=tool_call.name,
                    step_number=step_number,
                )
                # Propagate ``ToolResult.metadata`` onto ``Message.metadata`` so
                # application code that inspects the conversation (e.g.
                # ``TruncationPolicy`` reading ``metadata['protected']``) sees
                # what the tool surfaced. ``or None`` normalises the empty-dict
                # default to ``None`` — preserving the convention that
                # ``Message.metadata is None`` means "no metadata."
                tr_metadata = result.metadata or None
                messages.append(
                    Message(
                        role="tool_result",
                        content=result.content,
                        tool_call_id=tool_call.id,
                        metadata=tr_metadata,
                    )
                )
                completed_tool_results[i] = {
                    "content": result.content,
                    "tool_call_id": tool_call.id,
                    "metadata": tr_metadata,
                }
            except SuspendExecution as exc:
                exc.checkpoint_data = self._build_checkpoint_state(
                    messages=messages,
                    step_number=step_number,
                    revision_count=revision_count,
                    usages=usages,
                    tool_calls=tool_calls,
                    completed_tool_results=completed_tool_results,
                    suspended_tool_index=i,
                )
                raise
            except RunCancelled:
                # Cancellation is a control-flow signal — surface it
                # past the error-handler so the agent loop converts it
                # to ``termination_reason="cancelled"``.
                raise
            except Exception as e:
                correction = self._error_handler.handle_tool_error(e, tool_attempts[i], available_tools)
                tool_attempts[i] += 1
                if correction is not None:
                    # No ``ToolResult`` is available on the correction path —
                    # the correction string is synthesised by the error
                    # handler. ``metadata`` is only round-tripped from
                    # successful ``ToolResult`` returns; here it stays
                    # ``None`` (the default) by design.
                    messages.append(
                        Message(
                            role="tool_result",
                            content=correction,
                            tool_call_id=tool_call.id,
                        )
                    )
                    completed_tool_results[i] = {
                        "content": correction,
                        "tool_call_id": tool_call.id,
                    }
                    self._emitter.emit(
                        ErrorCorrectionEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            correction_prompt=correction,
                            attempt=tool_attempts[i],
                            max_attempts=self._error_handler.max_corrections,
                        )
                    )
                elif self._error_handler.should_degrade(e, tool_attempts[i]):
                    degradation_msg = self._error_handler.format_degradation_message(e)
                    # No ``ToolResult`` on the degradation path either — the
                    # degradation string is synthesised by the error handler.
                    # ``metadata`` stays ``None`` for the same reason as the
                    # correction branch above.
                    messages.append(
                        Message(
                            role="tool_result",
                            content=degradation_msg,
                            tool_call_id=tool_call.id,
                        )
                    )
                    completed_tool_results[i] = {
                        "content": degradation_msg,
                        "tool_call_id": tool_call.id,
                    }
                    self._emitter.emit(
                        ErrorDegradationEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            degradation_message=degradation_msg,
                        )
                    )
                else:
                    raise

    def _build_checkpoint_state(
        self,
        *,
        messages: list[Message],
        step_number: int,
        revision_count: int,
        usages: list[Usage],
        tool_calls: list[ToolCall],
        completed_tool_results: dict[int, dict[str, Any]],
        suspended_tool_index: int,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "agent_type": "react",
            "messages": [m.model_dump() for m in messages],
            "step_number": step_number,
            "revision_count": revision_count,
            "working_memory": (self._working_memory.read() if self._working_memory is not None else None),
            "usages": [u.model_dump() for u in usages],
            "limiter_count": self._limiter.current_iteration,
            "tool_call_limiter_count": (
                self._tool_call_limiter.current_tool_calls if self._tool_call_limiter is not None else 0
            ),
            "error_handler_state": {
                "total_corrections": self._error_handler.total_corrections,
            },
            "tool_calls": [tc.model_dump() for tc in tool_calls],
            "completed_tool_results": {str(k): v for k, v in completed_tool_results.items()},
            "suspended_tool_index": suspended_tool_index,
        }
        return state

    async def _execute_resume(
        self,
        input: AgentInput,
        tool_schemas: list[Any],
        available_tools: list[str],
    ) -> AgentResult:
        """Resume agent execution from a checkpoint."""
        state = self._resume_state
        assert state is not None
        self._resume_state = None

        # Restore state
        messages = [Message(**m) for m in state["messages"]]
        step_number: int = state["step_number"]
        revision_count: int = state["revision_count"]
        usages = [Usage(**u) for u in state["usages"]]

        # Restore component state
        self._limiter.restore(state["limiter_count"])
        if self._tool_call_limiter is not None and "tool_call_limiter_count" in state:
            self._tool_call_limiter.restore(state["tool_call_limiter_count"])
        self._error_handler.restore(state["error_handler_state"]["total_corrections"])
        if self._working_memory is not None and state["working_memory"] is not None:
            self._working_memory.write(state["working_memory"])

        # Restore tool batch position
        tool_calls = [ToolCall(**tc) for tc in state["tool_calls"]]
        completed_tool_results: dict[int, dict[str, Any]] = {
            int(k): v for k, v in state["completed_tool_results"].items()
        }
        suspended_tool_index: int = state["suspended_tool_index"]

        self._emitter.emit(
            ExecutionResumedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                checkpoint_id="",
                suspension_id="",
                resumed_from_step=f"step-{step_number}",
            )
        )

        # Inject stored tool_result messages for completed tool calls (0..k-1).
        # ``tr.get("metadata")`` (not ``tr["metadata"]``) keeps pre-Phase-3
        # checkpoints loadable — those entries don't carry the ``"metadata"``
        # key and restore as ``metadata=None``.
        for idx in sorted(completed_tool_results.keys()):
            if idx < suspended_tool_index:
                tr = completed_tool_results[idx]
                messages.append(
                    Message(
                        role="tool_result",
                        content=tr["content"],
                        tool_call_id=tr["tool_call_id"],
                        metadata=tr.get("metadata"),
                    )
                )

        # Re-execute from the suspended tool call onward
        await self._dispatch_tool_batch(
            tool_calls,
            messages,
            available_tools,
            step_number,
            revision_count,
            usages,
            start_index=suspended_tool_index,
        )

        # Emit step event for the completed batch
        action = ", ".join(tc.name for tc in tool_calls)
        observation = self._format_observations(tool_calls, messages)
        self._emit_step(
            step_number,
            thought=None,
            action=action,
            observation=observation,
        )

        # Continue the main loop
        return await self._run_loop(
            task_input=input,
            messages=messages,
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            step_number=step_number,
            revision_count=revision_count,
            usages=usages,
        )

    @staticmethod
    def _format_observations(tool_calls: list[ToolCall], messages: list[Message]) -> str:
        """Extract observations from the most recent tool_result messages."""
        n = len(tool_calls)
        recent_results = messages[-n:]
        parts: list[str] = []
        for tc, msg in zip(tool_calls, recent_results, strict=False):
            if msg.role == "tool_result":
                parts.append(f"{tc.name}: {msg.content}")
        return "\n".join(parts)
