from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from nanitics.composition.threads.store import ThreadLocks, ThreadStore
from nanitics.capabilities.context.token_counter import EstimateTokenCounter
from nanitics.capabilities.context.tool_result import ToolResultPolicy
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
from nanitics.strategies.tools import FunctionTool, Tool, ToolRegistry, tool

from .base import Agent, AgentInput, AgentResult

# Capability-aware environment guidance used when a human-input channel
# (a tool with ``schema.human_channel``) is present. It tells the model to
# reach for that channel rather than assuming, and not to end on a bare
# question. Topology-neutral: it does not assert the result is one-way, since
# a ReActAgent used as a sub-agent feeds a parent that can react. When no
# human channel is present, ``environment_guidance`` stays ``None`` and the
# base class uses its standard autonomous-operation text unchanged.
_HUMAN_CHANNEL_ENVIRONMENT_GUIDANCE = (
    "You operate autonomously rather than as a conversational chatbot. When a "
    "decision or information requires a person, call `ask_human` rather than "
    "assuming; otherwise proceed on reasonable assumptions and state them. End "
    "your turn with a complete result, not a question."
)

# Reserved tool name for the explicit-completion terminal. Auto-registered
# when ``require_explicit_finish=True``; reserved against consumer collision
# only in that mode.
_FINISH_TOOL_NAME = "finish"


@dataclass(frozen=True)
class _EvalDecision:
    """Outcome of running a candidate output through the evaluation gate.

    ``revise`` true means the caller should append ``feedback`` and continue the
    loop. Otherwise the candidate is terminal: ``reason_override`` is a
    non-``None`` ``termination_reason`` (``"evaluation_failed"`` /
    ``"evaluation_skipped"``) when the evaluator forced one, or ``None`` for a
    clean accept (the caller keeps its own default reason).
    """

    revise: bool
    feedback: str | None = None
    reason_override: str | None = None


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
        tool_result_policy: Bounds the size of individual tool results
            before they enter the message list. Applied at the
            :class:`~nanitics.strategies.tools.ToolRegistry` dispatch
            seam. Defaults to ``None``.
        context_providers: Inject context before each LLM call.
        working_memory: Structured scratchpad the agent can read and
            update across steps.
        output_evaluator: Quality gate for the final output.
        prompt_contributors: Additional system prompt sections.
        tool_state: Per-run state dict injected into tool execution via
            ``ToolContext``.
        output_schema: Pydantic model for structured output. When
            provided, an additional LLM call is made after the tool-use
            loop to produce schema-constrained JSON. If a tool marked
            ``return_direct`` fires, that synthesis call is skipped:
            ``termination_reason`` is ``"return_direct"``, ``output`` is the
            tool's ``ToolResult.content``, and ``parsed`` is ``None``
            regardless of ``output_schema``. A ``return_direct`` tool that
            needs to hand back structured data puts it in
            ``ToolResult.metadata``, which round-trips onto the
            ``tool_result`` ``Message.metadata`` (read from the last
            ``tool_result`` message in ``messages``) and is never sent to
            the LLM.
        initial_messages: Optional prior conversation messages to prepend
            before the current user input. Enables multi-turn conversations
            where history is loaded from an external store.
        require_explicit_finish: Opt-in completion mode (default ``False``).
            When ``True``, the run terminates only via a typed terminal
            action: a ``finish`` tool is auto-registered, and a no-tool-call
            turn no longer ends the run — instead the loop appends the
            assistant text plus a nudge and continues, bounded by
            ``max_iterations``. ``finish(result: str)`` (or the
            ``output_schema`` shape when set) delivers the run's output with
            ``termination_reason="finished"``; its result is subject to the
            ``output_evaluator`` gate just as a bare-text answer is in default
            mode. Use for autonomous, one-way-output agents where a clarifying
            question must route through ``ask_human`` rather than silently
            ending the run; leave ``False`` for conversational agents whose
            bare-text turns are caught by a host loop. Raises ``ValueError`` if
            a tool named ``finish`` is already registered.
        suspend_on_budget: Opt-in resumable exhaustion (default ``False``).
            When ``True`` *and* the run executes under step-level durability
            (a checkpoint sink is attached, e.g. via ``DurableRun(...,
            step_checkpoints=True)``), hitting ``max_iterations`` /
            ``max_tool_calls`` parks the run as a ``"budget_exhausted"``
            suspension — checkpointing the full conversation and surfacing the
            last assistant turn — instead of ending it. A host resumes it
            through
            :meth:`~nanitics.composition.durability.resume.ResumeService.continue_run`
            with the agent rebuilt on a larger budget, and the run picks up the
            same ReAct loop where it left off. Without a sink (no durability)
            this is inert: exhaustion ends the run normally with
            ``termination_reason="iteration_limit"`` / ``"tool_call_limit"``.
            Leave ``False`` to keep exhaustion terminal.
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
        tool_result_policy: ToolResultPolicy | None = None,
        context_providers: list[ContextProvider] | None = None,
        working_memory: WorkingMemory | None = None,
        output_evaluator: OutputEvaluator | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
        tool_state: dict[str, Any] | None = None,
        streaming: bool = False,
        output_schema: type[BaseModel] | None = None,
        initial_messages: list[Message] | None = None,
        require_explicit_finish: bool = False,
        suspend_on_budget: bool = False,
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
        # Capability-aware environment guidance: only when a human-input
        # channel is present does the standard "make reasonable assumptions"
        # text contradict the tool surface, so only then is it replaced.
        # Agents with no human channel keep the base class's default text
        # byte-for-byte.
        environment_guidance = (
            _HUMAN_CHANNEL_ENVIRONMENT_GUIDANCE if any(t.schema.human_channel for t in tools) else None
        )
        super().__init__(
            name=name,
            llm_client=llm_client,
            emitter=emitter,
            system_prompt=system_prompt,
            environment_guidance=environment_guidance,
            cancellation_token=cancellation_token,
            error_handler=error_handler,
            context_manager=context_manager,
            tool_result_policy=tool_result_policy,
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
            tool_result_policy=tool_result_policy,
            token_counter=EstimateTokenCounter() if tool_result_policy is not None else None,
        )
        self._tool_registry.register_all(tools)
        self._limiter = IterationLimiter(max_iterations)
        self._tool_call_limiter = ToolCallLimiter(max_tool_calls) if max_tool_calls is not None else None
        self._working_memory = working_memory
        self._output_schema = output_schema
        self._initial_messages = list(initial_messages) if initial_messages else None
        self._require_explicit_finish = require_explicit_finish
        self._suspend_on_budget = suspend_on_budget
        if require_explicit_finish:
            if self._tool_registry.has(_FINISH_TOOL_NAME):
                raise ValueError(
                    f"Tool name {_FINISH_TOOL_NAME!r} is reserved when "
                    "require_explicit_finish=True. Rename the conflicting tool, "
                    "or disable require_explicit_finish."
                )
            self._tool_registry.register(self._build_finish_tool())

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
        for new_tool in tools:
            name = new_tool.schema.name
            if not self._tool_registry.has(name):
                self._tool_registry.register(new_tool)
                added.append(name)
        return added

    def remove_tools(self, names: Sequence[str]) -> None:
        """Unregister tools by name."""
        for name in names:
            self._tool_registry.unregister(name)

    def update_tool_state(self, key: str, value: Any) -> None:
        self._tool_registry.update_state(key, value)

    def _build_finish_tool(self) -> FunctionTool:
        """Build the auto-registered ``finish`` terminal tool.

        With an ``output_schema``, ``finish`` takes the schema's fields
        directly, so structured output is committed as the terminal action
        and no separate synthesis call is made. Otherwise it takes a single
        ``result`` string. The loop recognises the call by name, runs the
        result through the output-evaluation gate, and ends the run with
        ``termination_reason="finished"``.
        """
        if self._output_schema is not None:
            schema = self._output_schema

            async def finish_structured(**fields: Any) -> str:
                return schema(**fields).model_dump_json()

            return FunctionTool(
                fn=finish_structured,
                name=_FINISH_TOOL_NAME,
                description=(
                    "Deliver your final result and end the run. Fill in the "
                    "result fields with your complete answer. Calling this is "
                    "the only way to deliver a result; a plain message reaches "
                    "no one."
                ),
                parameters_model=schema,
            )

        @tool(
            name=_FINISH_TOOL_NAME,
            description=(
                "Deliver your final result and end the run. Pass your complete "
                "answer as `result`. Calling this is the only way to deliver a "
                "result; a plain message reaches no one."
            ),
        )
        async def finish(result: str) -> str:
            return result

        return finish

    def _human_channel_present(self) -> bool:
        """Whether a human-input channel tool is currently registered.

        Read dynamically from the registry so tools added via
        :meth:`add_tools` after construction are reflected in the
        explicit-finish nudge.
        """
        return any(s.human_channel for s in self._tool_registry.list_schemas())

    def _explicit_finish_nudge(self) -> str:
        """The user message appended after a bare-text turn in explicit mode.

        Capability-aware: offers ``ask_human`` only when a human channel is
        registered.
        """
        if self._human_channel_present():
            return (
                "You ended your turn without delivering a result. Call "
                "`finish(result=…)` to deliver your result, or `ask_human(…)` "
                "to ask the recipient a question. A plain message is not "
                "delivered to anyone — finishing and asking are the only two "
                "ways to end your turn."
            )
        return (
            "You ended your turn without delivering a result. Call "
            "`finish(result=…)` to deliver your result. A plain message is not "
            "delivered to anyone — finishing is the only way to end your turn. "
            "You cannot reach a person; if information is missing, make a "
            "reasonable assumption, state it, and finish."
        )

    def _finish_outcome(self, tool_calls: list[ToolCall]) -> tuple[str, BaseModel | None] | None:
        """Detect a successful ``finish`` call in a completed batch.

        Returns ``(output, parsed)`` when the batch contains a ``finish`` call
        whose arguments are valid (which is exactly when its dispatch
        succeeded — the ``finish`` function cannot fail otherwise). Returns
        ``None`` when explicit-finish is off, no ``finish`` was called, or the
        ``finish`` call's arguments were invalid (its dispatch produced an
        error correction, so the run must continue rather than terminate).
        """
        if not self._require_explicit_finish:
            return None
        finish_call = next((tc for tc in tool_calls if tc.name == _FINISH_TOOL_NAME), None)
        if finish_call is None:
            return None
        if self._output_schema is not None:
            try:
                parsed = self._output_schema(**finish_call.arguments)
            except ValidationError:
                return None
            return parsed.model_dump_json(), parsed
        result = finish_call.arguments.get("result")
        if result is None:
            return None
        return str(result), None

    async def _evaluate_candidate(
        self,
        candidate: str,
        task_input: AgentInput,
        messages: list[Message],
        revision_count: int,
        response: Any,
    ) -> _EvalDecision:
        """Run a candidate output through the output-evaluation gate.

        Shared by the bare-text terminal (default mode) and the ``finish``
        terminal (explicit mode). Emits the same truncation / revision /
        exhaustion events as the inline logic it replaces, and performs no
        message mutation — the caller appends the assistant message and any
        feedback, so the two call sites can shape the conversation differently.
        Assumes an ``output_evaluator`` is configured; callers gate on that.
        """
        assert self._output_evaluator is not None
        max_revisions = self._output_evaluator.max_revisions

        if self._is_truncated(response):
            if revision_count < max_revisions:
                self._emit_truncation_events(revision_count, max_revisions)
                return _EvalDecision(revise=True, feedback=self._TRUNCATION_FEEDBACK)
            self._emit_evaluation_exhausted(
                evaluator_name="truncation",
                verdict="revise",
                revision_count=revision_count,
                max_revisions=max_revisions,
                feedback=self._TRUNCATION_FEEDBACK,
            )
            return _EvalDecision(revise=False, reason_override="evaluation_failed")

        eval_result = await self._evaluate_output(candidate or "", task_input, messages, revision_count)
        if eval_result.verdict == EvaluationVerdict.REVISE and revision_count < max_revisions:
            self._emit_evaluation_revision(eval_result.feedback or "", revision_count, max_revisions)
            return _EvalDecision(revise=True, feedback=eval_result.feedback or "")
        if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
            return _EvalDecision(revise=False, reason_override="evaluation_skipped")
        if eval_result.verdict != EvaluationVerdict.ACCEPT:
            self._emit_evaluation_exhausted(
                evaluator_name=eval_result.evaluator_name,
                verdict=eval_result.verdict.value,
                revision_count=revision_count,
                max_revisions=max_revisions,
                feedback=eval_result.feedback,
            )
            return _EvalDecision(revise=False, reason_override="evaluation_failed")
        return _EvalDecision(revise=False)

    async def _execute(self, input: AgentInput, *, thread_key: str | None = None) -> AgentResult:
        tool_schemas = self._tool_registry.list_schemas()
        available_tools = [s.name for s in tool_schemas]

        # --- Resume path ---
        # Discriminate on the state shape, not a reason field: a HITL
        # suspension snapshot carries ``suspended_tool_index`` (it suspended
        # mid-batch) and resumes mid-batch via ``_execute_resume``; a step
        # snapshot has no ``suspended_tool_index`` (the batch completed) and
        # continues from message history via ``_execute_crash_resume``.
        if self._resume_state is not None:
            if "suspended_tool_index" in self._resume_state:
                return await self._execute_resume(input, tool_schemas, available_tools)
            return await self._execute_crash_resume(input, tool_schemas, available_tools)

        # --- Normal path ---
        self._limiter.reset()
        self._error_handler.reset()
        if self._tool_call_limiter is not None:
            self._tool_call_limiter.reset()
        if self._context_manager is not None:
            self._context_manager.reset()
        if self._tool_result_policy is not None:
            self._tool_result_policy.reset()
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
        pending_return_direct: tuple[int, str] | None = None,
    ) -> AgentResult:
        output: str | None = None
        parsed = None
        termination_reason = "complete"

        while True:  # Outer loop: handles REVISE re-entry for structured output
            termination_reason = "complete"

            while True:  # Tool loop
                # A return_direct tool that fired in a batch dispatched before
                # this loop was (re-)entered — only the resume path supplies
                # this, when the suspended batch's terminal call is a
                # return_direct tool. Apply it once, before any LLM call.
                if pending_return_direct is not None:
                    output = pending_return_direct[1]
                    termination_reason = "return_direct"
                    pending_return_direct = None
                    break

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
                    self._maybe_suspend_on_budget(
                        messages=messages,
                        step_number=step_number,
                        revision_count=revision_count,
                        usages=usages,
                        reason="iteration_limit",
                    )
                    termination_reason = "iteration_limit"
                    break

                step_number += 1

                with self._emitter.span(f"step-{step_number}"):
                    try:
                        # Race the whole LLM call — generate *and* its internal
                        # retry backoff — against the token, so a cancel during
                        # a transient-error backoff sleep stops immediately
                        # rather than waiting out the delay.
                        response = await run_cancellable(
                            self._call_llm(
                                messages,
                                tools=tool_schemas if tool_schemas else None,
                            ),
                            self._cancellation_token,
                            step_number=step_number,
                        )
                    except RunCancelled as exc:
                        self._emit_safety_cancellation(exc.step_number or step_number)
                        return self._cancelled_result(
                            messages=messages,
                            step_number=step_number,
                            usages=usages,
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
                        # Explicit-finish mode: a bare-text turn delivers
                        # nothing. Append the text plus a capability-aware nudge
                        # and continue — the model must call ``finish`` (or
                        # ``ask_human``) to end the run. The ``max_iterations``
                        # limiter at the top of this loop is the backstop, so a
                        # model that never picks a terminal ends "iteration_limit"
                        # rather than looping forever.
                        if self._require_explicit_finish:
                            messages.append(Message(role="assistant", content=assistant_content))
                            messages.append(Message(role="user", content=self._explicit_finish_nudge()))
                            self._emit_step(
                                step_number,
                                thought=response.reasoning_text,
                                observation=response.content or None,
                            )
                            continue

                        decision = _EvalDecision(revise=False)
                        if self._output_evaluator is not None and self._output_schema is None:
                            decision = await self._evaluate_candidate(
                                assistant_content or "",
                                task_input,
                                messages,
                                revision_count,
                                response,
                            )
                        if decision.revise:
                            messages.append(Message(role="assistant", content=assistant_content))
                            messages.append(Message(role="user", content=decision.feedback or ""))
                            revision_count += 1
                            continue
                        if decision.reason_override is not None:
                            termination_reason = decision.reason_override

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
                        return_direct_hit = await self._dispatch_tool_batch(
                            response.tool_calls,
                            messages,
                            available_tools,
                            step_number,
                            revision_count,
                            usages,
                        )
                    except RunCancelled as exc:
                        self._emit_safety_cancellation(exc.step_number or step_number)
                        return self._cancelled_result(
                            messages=messages,
                            step_number=step_number,
                            usages=usages,
                        )

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
                            self._maybe_suspend_on_budget(
                                messages=messages,
                                step_number=step_number,
                                revision_count=revision_count,
                                usages=usages,
                                reason="tool_call_limit",
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

                    # Step-level durability: hand the sink a completed-batch
                    # snapshot so an interrupted run resumes from this point
                    # without re-firing the tools just completed (they replay
                    # from ``messages``). No-op when no sink is attached
                    # (step_checkpoints disabled). The in-flight batch at a
                    # crash is the one-step replay window — co-called side
                    # effects in *that* batch may repeat on resume.
                    await self._checkpoint_completed_batch(
                        messages=messages,
                        step_number=step_number,
                        revision_count=revision_count,
                        usages=usages,
                    )

                    # Explicit finish: the model called ``finish`` in this
                    # batch. Its result is the run's output, gated by the
                    # output evaluator exactly as a bare-text answer is in
                    # default mode. ``finish`` takes precedence over a
                    # co-batched return_direct tool (checked first). On REVISE
                    # the loop continues — the finish tool_result is already in
                    # ``messages``, so appending the feedback prompts the model
                    # to finish again. Otherwise the run ends "finished" (or the
                    # evaluator's forced reason). The output_schema synthesis
                    # call below is skipped because termination_reason is no
                    # longer "complete".
                    finish_outcome = self._finish_outcome(response.tool_calls)
                    if finish_outcome is not None:
                        finish_output, finish_parsed = finish_outcome
                        decision = _EvalDecision(revise=False)
                        if self._output_evaluator is not None:
                            decision = await self._evaluate_candidate(
                                finish_output,
                                task_input,
                                messages,
                                revision_count,
                                response,
                            )
                        if decision.revise:
                            messages.append(Message(role="user", content=decision.feedback or ""))
                            revision_count += 1
                            continue
                        output = finish_output
                        parsed = finish_parsed
                        termination_reason = decision.reason_override or "finished"
                        break

                    # A return_direct tool fired in this batch: end the run on
                    # its result, skipping the closing LLM turn. The whole batch
                    # already ran (co-called side effects fired); the lowest-index
                    # return_direct call wins. The output_schema guard below sees
                    # termination_reason != "complete" and skips structured
                    # synthesis, and the output_evaluator paths are never reached,
                    # so both are bypassed without extra branching.
                    if return_direct_hit is not None:
                        output = return_direct_hit[1]
                        termination_reason = "return_direct"
                        break

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

    def _cancelled_result(
        self,
        *,
        messages: list[Message],
        step_number: int,
        usages: list[Usage],
    ) -> AgentResult:
        """Build the terminal result for a cooperatively cancelled run.

        Shared by the normal loop's tool-dispatch ``except RunCancelled``
        and :meth:`_execute_resume` so a cancellation concludes with
        ``termination_reason="cancelled"`` on either path — instead of the
        signal escaping ``ResumeService.resume`` on the resume path alone.
        ``output``/``parsed`` are ``None``: a cancelled run produced no
        final answer.
        """
        return AgentResult(
            output=None,
            parsed=None,
            total_steps=step_number,
            termination_reason="cancelled",
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
        return_direct_hit: tuple[int, str] | None = None,
    ) -> tuple[int, str] | None:
        """Dispatch tool calls, catching SuspendExecution to build checkpoint state.

        Returns the ``(index, content)`` of the first call in batch order whose
        tool has ``schema.return_direct`` set and executed successfully, or
        ``None`` if no such call fired. The whole batch always runs to
        completion — detection records the hit but never breaks early, so
        co-called tools' side effects fire regardless. ``return_direct_hit``
        carries a hit found by an earlier (pre-suspension) dispatch of the same
        batch into a resumed dispatch, so a return_direct call that ran before
        the suspension point still wins on the lowest index.
        """
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
                # Record the first successful return_direct call in batch order.
                # Only successful ToolResults qualify — the error-correction and
                # degradation branches below synthesise a string with no
                # ToolResult, so a tool that raised never terminates the run.
                if return_direct_hit is None and self._tool_registry.get(tool_call.name).schema.return_direct:
                    return_direct_hit = (i, result.content)
            except SuspendExecution as exc:
                exc.checkpoint_data = self._build_checkpoint_state(
                    messages=messages,
                    step_number=step_number,
                    revision_count=revision_count,
                    usages=usages,
                    tool_calls=tool_calls,
                    completed_tool_results=completed_tool_results,
                    suspended_tool_index=i,
                    return_direct_hit=return_direct_hit,
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

        return return_direct_hit

    def _maybe_suspend_on_budget(
        self,
        *,
        messages: list[Message],
        step_number: int,
        revision_count: int,
        usages: list[Usage],
        reason: str,
    ) -> None:
        """Park the run as a resumable ``budget_exhausted`` suspension, or no-op.

        Called at an ``iteration_limit`` / ``tool_call_limit`` break. Suspends
        only when ``suspend_on_budget`` is on *and* a checkpoint sink is attached
        — i.e. the run executes under step-level durability, so the suspension
        can actually be persisted and continued. Otherwise it returns and the
        caller ends the run normally with the budget ``termination_reason``.

        The suspension carries a completed-step snapshot (the same shape a
        per-tool-batch cursor uses, so resume re-enters via
        :meth:`_execute_crash_resume`) and the run's last assistant turn, so a
        host can show the partial work. The orchestrator that contains this
        agent persists it under ``agent_checkpoint`` and re-enters the agent on
        :meth:`~nanitics.composition.durability.resume.ResumeService.continue_run`.
        """
        if not self._suspend_on_budget or self._checkpoint_sink is None:
            return

        from uuid import uuid4

        from nanitics.composition.durability.models import SuspensionInfo
        from nanitics.composition.durability.suspension import SuspendExecution

        checkpoint_data = self._completed_step_state(
            messages=messages,
            step_number=step_number,
            revision_count=revision_count,
            usages=usages,
        )
        if reason == "iteration_limit":
            # ``IterationLimiter.step`` increments past the ceiling *before*
            # raising, so ``current_iteration`` is ``max_iterations + 1`` — a
            # step that never ran an LLM call. Persist the completed count so a
            # continue with a larger ceiling has real headroom, and a continue
            # with the *same* ceiling cleanly re-parks (rather than tripping the
            # ``restore`` bound). Tool-call counts are left as-is: those calls
            # actually executed, so the continue budget must exceed them.
            checkpoint_data["limiter_count"] = self._limiter.max_iterations
        last_assistant_text = next(
            (
                m.content
                for m in reversed(messages)
                if m.role == "assistant" and isinstance(m.content, str) and m.content
            ),
            None,
        )
        suspension_info = SuspensionInfo(
            suspension_id=str(uuid4()),
            suspension_type="budget_exhausted",
            request_id="",
            request_type=reason,
            prompt="",
            agent_name=self._name,
            last_assistant_text=last_assistant_text,
        )
        raise SuspendExecution(suspension_info=suspension_info, checkpoint_data=checkpoint_data)

    def _completed_step_state(
        self,
        *,
        messages: list[Message],
        step_number: int,
        revision_count: int,
        usages: list[Usage],
    ) -> dict[str, Any]:
        """Snapshot the agent's position after a *completed* tool batch.

        The shared subset of :meth:`_build_checkpoint_state` that describes a
        clean, replayable position — message history (which already holds the
        completed batch's ``tool_result`` messages), counters, working memory,
        and limiter state. It omits the suspended-batch fields
        (``suspended_tool_index`` / ``completed_tool_results`` / ``tool_calls`` /
        ``return_direct_hit``): the batch is done, so resume re-enters the loop
        fresh from ``messages`` rather than re-dispatching a partial batch. The
        absence of ``suspended_tool_index`` is also the discriminator the resume
        path uses to choose the crash-resume route over :meth:`_execute_resume`.
        """
        return {
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
        }

    async def _checkpoint_completed_batch(
        self,
        *,
        messages: list[Message],
        step_number: int,
        revision_count: int,
        usages: list[Usage],
    ) -> None:
        """Hand the sink a completed-batch snapshot, if a sink is attached.

        No-op unless step-level durability injected a sink. In ReAct a
        reasoning turn produces exactly one tool batch, so the ``tool_call``
        and ``agent_turn`` cadences coincide at this batch boundary; the
        cadence selects the journal ``step_kind`` label. The agent passes the
        agent-relative step-path tail (``turn#n``, derived from the deterministic
        ``step_number`` so the key is stable across resumes); the sink prepends
        the orchestration prefix and owns the store write.
        """
        if self._checkpoint_sink is None:
            return
        snapshot = self._completed_step_state(
            messages=messages,
            step_number=step_number,
            revision_count=revision_count,
            usages=usages,
        )
        await self._checkpoint_sink.save_step(
            step_path=f"turn#{step_number}",
            step_kind=self._checkpoint_cadence,
            state=snapshot,
        )

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
        return_direct_hit: tuple[int, str] | None = None,
    ) -> dict[str, Any]:
        state = self._completed_step_state(
            messages=messages,
            step_number=step_number,
            revision_count=revision_count,
            usages=usages,
        )
        state.update(
            {
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "completed_tool_results": {str(k): v for k, v in completed_tool_results.items()},
                "suspended_tool_index": suspended_tool_index,
                # A return_direct call that ran before the suspension point.
                # Stored as a ``[index, content]`` list (JSON has no tuples) so
                # it survives the checkpoint and still wins on the lowest index
                # after resume.
                "return_direct_hit": list(return_direct_hit) if return_direct_hit is not None else None,
            }
        )
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
        # A return_direct call that fired before the suspension point, restored
        # as a tuple. ``.get`` keeps pre-return_direct checkpoints loadable.
        stored_hit = state.get("return_direct_hit")
        pre_suspend_hit: tuple[int, str] | None = (
            (int(stored_hit[0]), str(stored_hit[1])) if stored_hit is not None else None
        )

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

        # Re-execute from the suspended tool call onward. A pre-suspension
        # return_direct hit is seeded so it still wins on the lowest index over
        # any return_direct call in the re-dispatched tail. A cooperative
        # cancellation here concludes "cancelled" through the same contract as
        # the normal loop, rather than escaping ``ResumeService.resume``.
        try:
            return_direct_hit = await self._dispatch_tool_batch(
                tool_calls,
                messages,
                available_tools,
                step_number,
                revision_count,
                usages,
                start_index=suspended_tool_index,
                return_direct_hit=pre_suspend_hit,
            )
        except RunCancelled as exc:
            self._emit_safety_cancellation(exc.step_number or step_number)
            return self._cancelled_result(
                messages=messages,
                step_number=step_number,
                usages=usages,
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

        # Continue the main loop. A return_direct hit terminates through the
        # same one-shot path at the top of ``_run_loop``'s tool loop — no
        # second copy of the termination logic.
        return await self._run_loop(
            task_input=input,
            messages=messages,
            tool_schemas=tool_schemas,
            available_tools=available_tools,
            pending_return_direct=return_direct_hit,
            step_number=step_number,
            revision_count=revision_count,
            usages=usages,
        )

    async def _execute_crash_resume(
        self,
        input: AgentInput,
        tool_schemas: list[Any],
        available_tools: list[str],
    ) -> AgentResult:
        """Resume from a completed-batch (step-durability) snapshot.

        Distinct from :meth:`_execute_resume`, which resumes a *suspended*
        batch mid-dispatch. Here the last checkpointed batch completed, so its
        ``tool_result`` messages are already in ``state["messages"]``: restoring
        the message history and counters and re-entering ``_run_loop`` fresh
        replays those completed tools from history (never re-firing them). Only
        the in-flight batch at crash time — never reached by a sink write — can
        repeat, which is the one-step replay window of the durability contract.
        """
        state = self._resume_state
        assert state is not None
        self._resume_state = None

        messages = [Message(**m) for m in state["messages"]]
        step_number: int = state["step_number"]
        revision_count: int = state["revision_count"]
        usages = [Usage(**u) for u in state["usages"]]

        self._limiter.restore(state["limiter_count"])
        if self._tool_call_limiter is not None and "tool_call_limiter_count" in state:
            self._tool_call_limiter.restore(state["tool_call_limiter_count"])
        self._error_handler.restore(state["error_handler_state"]["total_corrections"])
        if self._working_memory is not None and state["working_memory"] is not None:
            self._working_memory.write(state["working_memory"])

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

        # If the checkpointed batch already contained a successful ``finish``
        # call, the run had reached its terminal an instant before the crash.
        # Conclude "finished" directly from history rather than re-entering the
        # loop and re-dispatching ``finish`` (which the durability contract's
        # one-step replay would otherwise do). The evaluator gate is not re-run
        # here: it ran (or was about to) at original finish time and its verdict
        # is not checkpointed, so on resume the checkpointed finish is taken as
        # terminal — consistent with the documented one-step replay window.
        last_batch = next(
            (m for m in reversed(messages) if m.role == "assistant" and m.tool_calls),
            None,
        )
        if last_batch is not None:
            finish_outcome = self._finish_outcome(last_batch.tool_calls or [])
            if finish_outcome is not None:
                finish_output, finish_parsed = finish_outcome
                return AgentResult(
                    output=finish_output,
                    parsed=finish_parsed,
                    total_steps=step_number,
                    termination_reason="finished",
                    messages=messages,
                    usage=self._aggregate_usage(usages),
                )

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
