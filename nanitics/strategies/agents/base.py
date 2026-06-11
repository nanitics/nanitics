from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field

from nanitics.capabilities.context.tool_result import ToolResultPolicy
from nanitics.capabilities.errors.handler import ErrorHandler
from nanitics.strategies.agents.bound import BoundAgent, RunContext
from nanitics.strategies.agents.context import ContextContent, ContextManagement, ContextProvider
from nanitics.strategies.agents.errors import ErrorHandling

if TYPE_CHECKING:
    from nanitics.composition.durability.store import CheckpointCadence, StepCheckpointSink
    from nanitics.composition.threads.store import ThreadLocks, ThreadStore
    from nanitics.strategies.tools import Tool
from nanitics.infrastructure.errors import LLMSchemaViolationError, NaniticsError
from nanitics.infrastructure.llm.protocol import (
    ContentBlock,
    LLMClient,
    LLMResponse,
    Message,
    TextContentBlock,
    ToolSchema,
)
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    AgentCompleteEvent,
    AgentErrorEvent,
    AgentStartEvent,
    AgentStepEvent,
    ContextAssemblyEvent,
    ContextContribution,
    ErrorCorrectionEvent,
    EvaluationEvent,
    EvaluationExhaustedEvent,
    EvaluationRevisionEvent,
    ExecutionSuspendedEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    LLMTokenEvent,
    SafetyCancellationEvent,
    SafetyIterationLimitEvent,
    SafetyToolCallLimitEvent,
    ToolInfo,
    Usage,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.evaluation import (
    EvaluationContext,
    EvaluationResult,
    OutputEvaluator,
)
from nanitics.strategies.prompts.builder import SystemPromptBuilder, SystemPromptContributor

AgentInput = str | list[ContentBlock]


def _input_to_text(input: AgentInput) -> str:
    if isinstance(input, str):
        return input
    return " ".join(b.text for b in input if isinstance(b, TextContentBlock))


def _render_context_wrapper(content: ContextContent) -> str:
    """Render a ``ContextContent`` into the canonical ``<nanitics:context>`` wrapper string.

    The wrapper is the single load-bearing signal the LLM uses to
    distinguish SDK-injected context from user speech. See
    :meth:`Agent._inject_context` for the authoritative spec; this
    helper is the single implementation of that spec.
    """
    protected_attr = "true" if content.protected else "false"
    return (
        f'<nanitics:context provider="{content.provider_name}" '
        f'priority="{content.priority}" protected="{protected_attr}">\n'
        f"{content.content}\n"
        f"</nanitics:context>"
    )


class AgentResult(PydanticBaseModel):
    """Immutable result of an agent run.

    Attributes:
        output: The agent's final text response, or ``None`` if the agent
            did not produce one (e.g., cancelled before completion).
        parsed: Parsed structured output when ``output_schema`` was
            provided to the agent. ``None`` otherwise. Excluded from
            serialization.
        total_steps: Number of reasoning steps the agent executed.
        termination_reason: Why the agent stopped — ``"complete"``,
            ``"iteration_limit"``, ``"cancelled"``, ``"evaluation_failed"``,
            or ``"return_direct"`` (a tool marked ``return_direct`` ended the
            run on its result). Consumers may switch on this value. On
            ``"return_direct"``, ``parsed`` is ``None`` even when
            ``output_schema`` was configured, and structured terminal data is
            read from the last ``tool_result`` message's ``metadata`` in
            ``messages``.
        messages: Full conversation history including user input, assistant
            responses, and tool results.
        usage: Aggregated token usage across all LLM calls in the run.
    """

    model_config = ConfigDict(frozen=True)

    output: str | None
    parsed: PydanticBaseModel | None = Field(default=None, exclude=True)
    total_steps: int
    termination_reason: str
    messages: list[Message]
    usage: Usage
    thread_key: str | None = None


class Agent(ABC):
    """Abstract base class for all agent types.

    Provides the common infrastructure for LLM interaction, tool dispatch,
    context injection, output evaluation, and event emission. Subclasses
    implement ``_execute()`` to define the agent's reasoning loop.

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model to use for generation.
        emitter: Receives all events emitted during execution.
        system_prompt: Base system prompt text. Combined with any
            ``prompt_contributors`` sections via ``SystemPromptBuilder``.
        cancellation_token: External signal to cancel the agent mid-run.
        error_handler: Strategy for recovering from LLM and tool errors.
        context_manager: Manages the context window (truncation,
            summarization) when conversations grow long.
        tool_result_policy: Bounds the size of individual tool results
            before they enter the message list. Applied at the
            :class:`~nanitics.strategies.tools.ToolRegistry` dispatch
            seam in tool-bearing subclasses. Defaults to ``None``
            (no policy applied). Symmetric to ``context_manager`` but
            for the tool-result side of the message list.
        context_providers: Inject additional context before each LLM call.
        output_evaluator: Quality gate that can trigger output revision.
        prompt_contributors: Components that add sections to the system
            prompt.
        thread_store: Persists per-thread :class:`Message` prefixes
            across :meth:`run` calls keyed by ``thread_key``. When
            ``None``, ``thread_key`` is accepted by :meth:`run` but no
            prefix is loaded and no append happens — the agent's
            configuration decides whether persistence occurs, so
            wrapping code can pass ``thread_key`` unconditionally.
        thread_locks: Per-key serialization for concurrent :meth:`run`
            calls. Defaults to a fresh per-agent
            :class:`~nanitics.composition.threads.ThreadLocks`. Pass a
            shared instance to coordinate threads across multiple agent
            instances.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        cancellation_token: CancellationToken | None = None,
        error_handler: ErrorHandling | None = None,
        context_manager: ContextManagement | None = None,
        tool_result_policy: ToolResultPolicy | None = None,
        context_providers: list[ContextProvider] | None = None,
        output_evaluator: OutputEvaluator | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
        streaming: bool = False,
        thread_store: ThreadStore | None = None,
        thread_locks: ThreadLocks | None = None,
    ) -> None:
        self._name = name
        self._llm_client = llm_client
        self._default_emitter = emitter
        # Per-instance ``ContextVar`` holding the emitter active in the
        # current asyncio task. ``BoundAgent.run`` → ``_run_with_context``
        # sets this for the duration of a bound call. Because it is a
        # ``ContextVar``, a ``set`` in task A is not visible in task B, so
        # a shared agent can serve multiple concurrent bound runs without
        # attribute-level races.
        self._emitter_var: ContextVar[EventEmitter] = ContextVar(f"agent_emitter_{id(self)}")
        builder = SystemPromptBuilder()
        builder.add_section("base", system_prompt)
        builder.add_section(
            "environment",
            "You operate autonomously rather than as a conversational "
            "chatbot. Make reasonable assumptions when information is "
            "incomplete and state them explicitly.",
        )
        for contributor in prompt_contributors or []:
            section = contributor.system_prompt_section()
            if section is None:
                continue
            name, content, *rest = section
            builder.add_section(name, content, cacheable=rest[0] if rest else True)
        self._system_prompt = builder.build()
        # Structured sections travel to the LLM client alongside the flat
        # system prompt. Cache-aware clients (``AnthropicLLMClient`` today)
        # honour the per-section ``cacheable`` flag set by each
        # contributor; cache-blind clients ignore the parameter. The flat
        # ``self._system_prompt`` remains the event-surface form used by
        # ``LLMRequestEvent.system_prompt`` and by providers that do not
        # accept structured sections.
        self._system_prompt_sections = builder.build_sections()
        self._cancellation_token = cancellation_token
        self._error_handler: ErrorHandling = error_handler if error_handler is not None else ErrorHandler()
        self._context_manager = context_manager
        self._tool_result_policy = tool_result_policy
        self._context_providers = context_providers
        self._output_evaluator = output_evaluator
        self._streaming = streaming
        self._thread_store = thread_store
        if thread_locks is None:
            from nanitics.composition.threads.store import ThreadLocks as _ThreadLocks

            self._thread_locks: ThreadLocks = _ThreadLocks()
        else:
            self._thread_locks = thread_locks
        self._resume_state: dict[str, Any] | None = None
        # Optional checkpoint sink for agent-internal step-level durability.
        # ``None`` (the default) means the agent never checkpoints between
        # tool batches — byte-for-byte today's behaviour. The orchestration
        # layer injects a sink via ``_set_checkpoint_sink`` only when
        # step-level durability is enabled. See
        # :class:`~nanitics.composition.durability.store.StepCheckpointSink`.
        self._checkpoint_sink: StepCheckpointSink | None = None
        self._checkpoint_cadence: CheckpointCadence = "tool_call"
        # Auto-wire agent-owned emitter-caching capabilities to the agent's
        # per-task emitter. Any attached context provider or output
        # evaluator that exposes an unset ``_emitter_provider`` attribute
        # receives a callback resolving through ``self._emitter`` — so
        # their trace events follow the current task's bound emitter
        # under delegation and concurrent sharing, mirroring the
        # ``ToolRegistry`` wiring in tool-bearing subclasses.
        for capability in [*(context_providers or []), output_evaluator]:
            if capability is None:
                continue
            if getattr(capability, "_emitter_provider", "unset") is None:
                # Duck-typed: any capability exposing a settable
                # ``_emitter_provider`` attribute (currently the memory
                # providers and ``LLMEvaluator``) auto-inherits the
                # agent's per-task emitter. Capabilities without the
                # attribute are skipped by the ``"unset"`` sentinel.
                capability._emitter_provider = lambda: self._emitter  # type: ignore[union-attr]

    @property
    def name(self) -> str:
        """The agent's name, used in events and traces."""
        return self._name

    @property
    def cancellation_token(self) -> CancellationToken | None:
        """The cancellation token, if one was provided."""
        return self._cancellation_token

    def set_cancellation_token(self, token: CancellationToken) -> None:
        """Attach or replace the cancellation token for this agent."""
        self._cancellation_token = token

    @property
    def _emitter(self) -> EventEmitter:
        """The emitter active for the current asyncio task.

        Resolves to the per-call child emitter when the agent is running
        under a :class:`BoundAgent`; otherwise to the default emitter
        supplied at construction. Backed by a per-instance
        ``ContextVar`` so concurrent bound runs of a shared agent each
        observe their own emitter without mutating ``self``.
        """
        return self._emitter_var.get(self._default_emitter)

    def bind(self, parent_emitter: EventEmitter) -> BoundAgent:
        """Return a non-mutating per-invocation binding of this agent.

        Creates a child emitter from ``parent_emitter`` **in the calling
        asyncio task** so that ``InMemoryEmitter``'s span-stack
        ``ContextVar`` is set in the correct task context. The returned
        :class:`BoundAgent` handle drives ``run`` under that child
        emitter without touching ``self`` — safe to call concurrently
        from independent tasks on a shared agent.

        Use wherever an agent participates in another agent's trace
        (``AgentTool``, ``Handoff``, ``ReflexionAgent``) or is shared
        across a concurrent orchestrator (``MapReduce``, ``Parallel``,
        ``DAG``, ``Broadcast``, ``Consensus``, ``Blackboard``,
        ``Bidding``).
        """
        return BoundAgent(self, RunContext(emitter=parent_emitter.create_child()))

    async def _run_with_context(
        self, input: AgentInput, ctx: RunContext, *, thread_key: str | None = None
    ) -> AgentResult:
        """Run the agent under a per-invocation :class:`RunContext`.

        Installs ``ctx.emitter`` on the per-task ``ContextVar`` for the
        duration of the call, then delegates to :meth:`run`. Does not
        mutate ``self`` — concurrent callers in distinct tasks each
        observe their own ``ctx`` through the task-local ``ContextVar``.
        """
        token = self._emitter_var.set(ctx.emitter)
        try:
            return await self.run(input, thread_key=thread_key)
        finally:
            self._emitter_var.reset(token)

    def _set_resume_state(self, state: dict[str, Any]) -> None:
        self._resume_state = state

    def _set_checkpoint_sink(
        self,
        sink: StepCheckpointSink,
        *,
        cadence: CheckpointCadence = "tool_call",
    ) -> None:
        """Attach a step-checkpoint sink for agent-internal durability.

        Symmetric to :meth:`_set_resume_state`. Injected by the orchestration
        layer (via ``_BoundAgentStep``) only when step-level durability is
        enabled, so a tool-bearing subclass can hand a completed-step snapshot
        to the sink after each completed step. Default (sink never set) leaves
        behaviour unchanged.
        """
        self._checkpoint_sink = sink
        self._checkpoint_cadence = cadence

    async def _load_thread_prefix(self, thread_key: str | None) -> list[Message]:
        """Load the :class:`Message` prefix for a thread.

        Returns ``[]`` when ``thread_key`` is ``None`` or when no
        :class:`~nanitics.composition.threads.ThreadStore` is configured
        on the agent. Subclass ``_execute`` methods opt in by calling
        this; the returned messages slot between
        :attr:`_initial_messages` (when present) and the new user input.

        The messages are real :class:`Message` objects intended for
        direct splicing into the per-run message list — they are NOT
        wrapped in the ``<nanitics:context>`` envelope that
        :meth:`_inject_context` applies to ``ContextProvider``
        contributions. See ``temp/sdk-thread-identity/design-rationale.md``
        §4 for the rationale.
        """
        if thread_key is None or self._thread_store is None:
            return []
        return await self._thread_store.load(thread_key)

    async def run(self, input: AgentInput, *, thread_key: str | None = None) -> AgentResult:
        """Execute the agent on the given task.

        Runs the agent's reasoning loop within a trace span. Emits start,
        complete, and error events. Propagates ``SuspendExecution`` for
        durable execution (human-in-the-loop suspension).

        Args:
            input: The task or question for the agent to work on.
                Accepts a plain string or a list of content blocks
                for multimodal input (text + images).
            thread_key: Opaque key identifying the conversation thread
                this run continues. When set, the configured
                :class:`~nanitics.composition.threads.ThreadStore`'s
                prefix for the key is loaded before ``_execute`` and the
                run's new messages are appended on successful
                completion. Concurrent same-key runs raise
                :class:`~nanitics.infrastructure.errors.ThreadInUseError`.
                When no ``thread_store`` is configured the key is
                accepted but no prefix is loaded and no append happens.

        Returns:
            The result of the run including output, step count, messages,
            token usage, and the ``thread_key`` (echoed back for
            correlation, ``None`` when the run had no thread).

        Raises:
            SuspendExecution: When the agent suspends for human input.
                On a thread-keyed run, the loaded prefix and key are
                snapshotted into the suspension's ``checkpoint_data`` so
                ``_execute_resume`` can use a frozen view of the thread
                without re-consulting the live store.
            ThreadInUseError: When another ``run`` is already in flight
                for the same ``thread_key`` on this agent (or any agent
                sharing the same :class:`ThreadLocks` instance).
        """
        saved_key, saved_prefix = self._peek_resume_thread_context()
        effective_key = thread_key if thread_key is not None else saved_key
        if effective_key is None:
            return await self._run_in_span(input, thread_key=None, prefix=[])
        async with self._thread_locks.hold(effective_key):
            if saved_prefix is not None:
                prefix = saved_prefix
            else:
                prefix = await self._load_thread_prefix(effective_key)
            return await self._run_in_span(input, thread_key=effective_key, prefix=prefix)

    def _peek_resume_thread_context(self) -> tuple[str | None, list[Message] | None]:
        """Inspect ``_resume_state`` for a saved thread context.

        Returns ``(thread_key, frozen_prefix)``. Both are ``None`` when
        the resume state is absent or carries no thread information. The
        peek does not consume ``_resume_state`` — the subclass's
        ``_execute_resume`` still owns full restoration of the run.
        """
        if self._resume_state is None:
            return None, None
        key = self._resume_state.get("thread_key")
        prefix_data = self._resume_state.get("thread_prefix")
        if prefix_data is None:
            return key, None
        return key, [Message(**m) for m in prefix_data]

    async def _run_in_span(self, input: AgentInput, *, thread_key: str | None, prefix: list[Message]) -> AgentResult:
        from nanitics.composition.durability.suspension import SuspendExecution

        with self._emitter.span(self._name):
            try:
                self._emit_start(
                    _input_to_text(input),
                    self._get_tools_available(),
                    self._get_tool_schemas(),
                    thread_key=thread_key,
                    replayed_message_count=len(prefix),
                )
                result = await self._execute(input, thread_key=thread_key)
                if thread_key is not None:
                    result = result.model_copy(update={"thread_key": thread_key})
                    if self._thread_store is not None:
                        new_messages = self._new_messages_after_prefix(result.messages, prefix)
                        await self._thread_store.append(thread_key, new_messages)
                self._emit_complete(result)
                return result
            except SuspendExecution as exc:
                if thread_key is not None:
                    state = exc.checkpoint_data if exc.checkpoint_data is not None else {}
                    state["thread_key"] = thread_key
                    state["thread_prefix"] = [m.model_dump() for m in prefix]
                    exc.checkpoint_data = state
                self._emitter.emit(
                    ExecutionSuspendedEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        suspension_id=exc.suspension_info.suspension_id,
                        suspension_type="hitl",
                        checkpoint_id="",
                        agent_name=self._name,
                    )
                )
                raise
            except Exception as e:
                self._emit_error(e)
                raise

    def _new_messages_after_prefix(self, run_messages: list[Message], prefix: list[Message]) -> list[Message]:
        """Slice ``run_messages`` to the messages produced during this run.

        Returns ``run_messages[len(_initial_messages) + len(prefix):]``
        — the messages added after the static seed and the dynamic
        thread prefix. Only these advance the thread on append; the seed
        and the prefix are not re-appended.
        """
        initial = getattr(self, "_initial_messages", None) or []
        return list(run_messages[len(initial) + len(prefix) :])

    @abstractmethod
    async def _execute(self, input: AgentInput, *, thread_key: str | None = None) -> AgentResult: ...

    @abstractmethod
    def _agent_type(self) -> str:
        """Return the agent type identifier (e.g. ``'react'``, ``'codeact'``)."""
        ...

    def _active_capabilities(self) -> list[str]:
        """Detect active capabilities from the agent's configuration.

        Subclasses should call ``super()`` and extend the returned list
        with type-specific capabilities.
        """
        caps: list[str] = []
        if self._output_evaluator is not None:
            caps.append("evaluation")
        if self._context_manager is not None:
            caps.append("context_management")
        if self._error_handler.max_corrections > 0:
            caps.append("error_handling")
        if self._cancellation_token is not None:
            caps.append("cancellation")
        if self._streaming:
            caps.append("streaming")
        # Detect memory types from context providers
        if self._context_providers:
            from nanitics.capabilities.memory.episodic import EpisodicMemoryProvider
            from nanitics.capabilities.memory.working_memory import WorkingMemoryProvider

            for provider in self._context_providers:
                if isinstance(provider, WorkingMemoryProvider):
                    caps.append("working_memory")
                elif isinstance(provider, EpisodicMemoryProvider):
                    caps.append("episodic_memory")
        return caps

    @property
    def supports_dynamic_tools(self) -> bool:
        """Whether this agent supports dynamic tool injection.

        Only agents with a reactive tool loop (e.g. ``ReActAgent``) should
        override this to return ``True``.
        """
        return False

    def add_tools(self, tools: Sequence[Tool]) -> list[str]:
        """Add tools to this agent at runtime.

        Returns the names of tools that were actually added. The base
        implementation returns an empty list (no tool support).
        """
        return []

    def remove_tools(self, names: Sequence[str]) -> None:  # noqa: B027
        """Remove previously added tools by name.

        No-op in the base implementation.
        """

    def update_tool_state(self, key: str, value: Any) -> None:
        """Update a single key in the agent's tool state.

        The updated value is available to all tools via
        ``ToolContext.state`` on subsequent invocations.

        Only supported by agents with a tool registry (e.g.
        ``ReActAgent``, ``LATSAgent``). Raises ``NotImplementedError``
        on agents without tool support.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support tool state")

    def _get_tools_available(self) -> list[str]:
        return []

    def _get_tool_schemas(self) -> list[ToolInfo]:
        return []

    async def _inject_context(self, messages: list[Message]) -> list[Message]:
        """Collect ``ContextProvider`` contributions and splice them into ``messages``.

        Each non-``None`` ``ContextContent`` returned by a provider is
        materialised as a single ``Message(role="user", …)`` whose
        ``content`` wraps the provider's raw string in a namespaced,
        XML-like delimiter that signals "SDK-injected context" to the
        LLM — distinct from a human user turn. The wrapper is the one
        load-bearing signal the LLM uses to distinguish provider output
        from user speech; without it, Anthropic-backed agents in
        particular tend to treat the injected text as untrusted external
        data and refuse to reference it.

        **Wire shape (the canonical spec).** For a single contribution
        ``ContextContent(content="[Working Memory]\\n- item", priority=0,
        protected=True, provider_name="working_memory")``, the rendered
        message content is:

        ```
        <nanitics:context provider="working_memory" priority="0" protected="true">
        [Working Memory]
        - item
        </nanitics:context>
        ```

        Attribute rendering rules:

        - ``provider`` — ``ContextContent.provider_name`` verbatim. The
          empty default is preserved as ``provider=""``; the SDK does
          not substitute a placeholder.
        - ``priority`` — decimal string of the integer (e.g. ``"0"``,
          ``"10"``, ``"-5"``).
        - ``protected`` — ``"true"`` when ``protected is True``,
          ``"false"`` otherwise. Lowercase, no other values.

        Body rendering rules:

        - ``ContextContent.content`` appears verbatim between the tags.
          No stripping, escaping, or re-indentation — the provider's
          pre-existing human-readable label (``[Working Memory]``,
          ``[Past Experiences]``, ``[Current Plan: …]``) stays inside
          the body untouched.
        - A single ``\\n`` separates the opening tag from the body and
          a single ``\\n`` separates the body from the closing tag.

        Multiple contributions each render into their own wrapper — they
        are not merged. The priority sort runs first (ascending, lower
        priority rendered earlier), and wrappers are emitted in sorted
        order. ``ContextAssemblyEvent.contributions[].content`` carries
        the rendered wrapped string (the same bytes the LLM sees), so
        trace renderings stay faithful to the wire shape.

        Insertion point and the assistant/tool_result-pair guard are
        unchanged: the wrappers splice in immediately before the latest
        contiguous user/tool_result run, pulled one step earlier if that
        would split an ``assistant(tool_use)`` → ``tool_result`` pair.
        """
        if not self._context_providers:
            return messages

        results: list[ContextContent] = []
        for provider in self._context_providers:
            result = await provider.provide(messages)
            if result is not None:
                results.append(result)

        if not results:
            return messages

        rendered_by_id = {id(r): _render_context_wrapper(r) for r in results}

        self._emitter.emit(
            ContextAssemblyEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                contributions=[
                    ContextContribution(
                        provider_name=r.provider_name,
                        content_length=len(rendered_by_id[id(r)]),
                        priority=r.priority,
                        protected=r.protected,
                        content=rendered_by_id[id(r)],
                    )
                    for r in results
                ],
                total_injected=len(results),
            )
        )

        results.sort(key=lambda r: r.priority)

        injected = [
            Message(
                role="user",
                content=rendered_by_id[id(r)],
                metadata={"protected": r.protected},
            )
            for r in results
        ]

        # Find insertion point: before the most recent turn's messages.
        # Walk backwards to find where the last user/tool_result sequence starts.
        insert_at = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role in ("user", "tool_result"):
                insert_at = i
            else:
                break

        # Don't insert between an assistant(tool_use) and its tool_results
        if (
            insert_at > 0
            and insert_at < len(messages)
            and messages[insert_at].role == "tool_result"
            and messages[insert_at - 1].role == "assistant"
            and messages[insert_at - 1].tool_calls
        ):
            insert_at -= 1

        return messages[:insert_at] + injected + messages[insert_at:]

    async def _call_llm(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        output_schema: type[PydanticBaseModel] | None = None,
    ) -> LLMResponse:
        managed_messages = await self._inject_context(messages)
        if self._context_manager is not None:
            managed_messages = await self._context_manager.prepare(
                self._system_prompt, managed_messages, tools, self._emitter
            )

        self._emitter.emit(
            LLMRequestEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                model_name=self._llm_client.model or "",
                system_prompt=self._system_prompt,
                messages=[m.model_dump() for m in managed_messages],
                tools=[t.model_dump() for t in tools] if tools else None,
                output_schema=(output_schema.model_json_schema() if output_schema else None),
            )
        )

        start = time.perf_counter()

        def _on_token(token: str) -> None:
            self._emitter.emit(
                LLMTokenEvent(
                    trace_id=self._emitter.trace_id,
                    span_id=self._emitter.span_id,
                    parent_span_id=self._emitter.parent_span_id,
                    token=token,
                    agent_name=self._name,
                )
            )

        async def _generate() -> LLMResponse:
            return await self._llm_client.generate(
                system_prompt=self._system_prompt,
                messages=managed_messages,
                tools=tools,
                output_schema=output_schema,
                on_token=_on_token if self._streaming else None,
                system_prompt_sections=self._system_prompt_sections,
            )

        try:
            response = await _generate()
        except LLMSchemaViolationError as e:
            correction_attempt = 0
            last_error = e
            while True:
                correction = self._error_handler.handle_llm_correction(last_error, correction_attempt)
                if correction is None:
                    raise last_error from None
                self._emitter.emit(
                    ErrorCorrectionEvent(
                        trace_id=self._emitter.trace_id,
                        span_id=self._emitter.span_id,
                        parent_span_id=self._emitter.parent_span_id,
                        error_type=type(last_error).__name__,
                        error_message=str(last_error),
                        correction_prompt=correction,
                        attempt=correction_attempt + 1,
                        max_attempts=self._error_handler.max_corrections,
                    )
                )
                managed_messages.append(Message(role="assistant", content=last_error.received or ""))
                managed_messages.append(Message(role="user", content=correction))
                correction_attempt += 1
                try:
                    response = await _generate()
                    break
                except LLMSchemaViolationError as retry_e:
                    last_error = retry_e
        except Exception as e:
            response = cast(LLMResponse, await self._error_handler.handle_llm_error(e, _generate, self._emitter))

        duration_ms = (time.perf_counter() - start) * 1000

        self._emitter.emit(
            LLMResponseEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                model_name=response.model,
                content=response.content,
                tool_calls=[tc.model_dump() for tc in response.tool_calls] if response.tool_calls else None,
                usage=response.usage,
                duration_ms=duration_ms,
            )
        )

        return response

    @property
    def _is_cancelled(self) -> bool:
        if self._cancellation_token is None:
            return False
        return self._cancellation_token.is_cancelled

    def _emit_start(
        self,
        task_input: str,
        tools_available: list[str],
        tool_schemas: list[ToolInfo] | None = None,
        *,
        thread_key: str | None = None,
        replayed_message_count: int = 0,
    ) -> None:
        model_name = getattr(self._llm_client, "model", None)
        self._emitter.emit(
            AgentStartEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                agent_name=self._name,
                task_input=task_input,
                tools_available=tools_available,
                tool_schemas=tool_schemas or [],
                agent_type=self._agent_type(),
                capabilities=self._active_capabilities(),
                model_name=model_name,
                thread_key=thread_key,
                replayed_message_count=replayed_message_count,
            )
        )

    def _emit_step(
        self,
        step_number: int,
        *,
        thought: str | None = None,
        action: str | None = None,
        observation: str | None = None,
        artifact: dict[str, Any] | None = None,
    ) -> None:
        self._emitter.emit(
            AgentStepEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                agent_name=self._name,
                step_number=step_number,
                thought=thought,
                action=action,
                observation=observation,
                artifact=artifact,
            )
        )

    def _emit_complete(self, result: AgentResult) -> None:
        self._emitter.emit(
            AgentCompleteEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                agent_name=self._name,
                output=result.output,
                total_steps=result.total_steps,
                termination_reason=result.termination_reason,
            )
        )

    def _emit_error(self, error: Exception, step_number: int | None = None) -> None:
        if isinstance(error, NaniticsError):
            error_metadata: dict[str, Any] = error.to_dict()
        else:
            error_metadata = {"message": str(error)}

        self._emitter.emit(
            AgentErrorEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                agent_name=self._name,
                error_type=type(error).__name__,
                error_message=str(error),
                error_metadata=error_metadata,
                step_number=step_number,
            )
        )

    def _emit_safety_iteration_limit(self, current_iteration: int, max_iterations: int, step_number: int) -> None:
        self._emitter.emit(
            SafetyIterationLimitEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                agent_name=self._name,
                current_iteration=current_iteration,
                max_iterations=max_iterations,
                step_number=step_number,
            )
        )

    def _emit_safety_tool_call_limit(self, current_tool_calls: int, max_tool_calls: int, step_number: int) -> None:
        self._emitter.emit(
            SafetyToolCallLimitEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                agent_name=self._name,
                current_tool_calls=current_tool_calls,
                max_tool_calls=max_tool_calls,
                step_number=step_number,
            )
        )

    def _emit_safety_cancellation(self, step_number: int | None = None) -> None:
        self._emitter.emit(
            SafetyCancellationEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                agent_name=self._name,
                step_number=step_number,
            )
        )

    def _emit_evaluation_exhausted(
        self,
        evaluator_name: str,
        verdict: str,
        revision_count: int,
        max_revisions: int,
        feedback: str | None,
    ) -> None:
        self._emitter.emit(
            EvaluationExhaustedEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                evaluator_name=evaluator_name,
                verdict=verdict,
                revision_count=revision_count,
                max_revisions=max_revisions,
                feedback=feedback,
            )
        )

    async def _evaluate_output(
        self,
        output: str,
        task_input: AgentInput,
        messages: list[Message],
        revision_attempt: int,
        *,
        depth: int | None = None,
        max_depth: int | None = None,
        trajectory_length: int | None = None,
        total_nodes_explored: int | None = None,
    ) -> EvaluationResult:
        assert self._output_evaluator is not None
        context = EvaluationContext(
            messages=messages,
            task_input=task_input,
            depth=depth,
            max_depth=max_depth,
            trajectory_length=trajectory_length,
            total_nodes_explored=total_nodes_explored,
        )
        result = await self._output_evaluator.evaluate(output, context)
        self._emitter.emit(
            EvaluationEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                evaluator_name=result.evaluator_name,
                verdict=result.verdict.value,
                score=result.score,
                feedback=result.feedback,
                revision_attempt=revision_attempt,
            )
        )
        return result

    def _emit_evaluation_revision(
        self,
        feedback: str,
        revision_attempt: int,
        max_revisions: int,
    ) -> None:
        self._emitter.emit(
            EvaluationRevisionEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                feedback=feedback,
                revision_attempt=revision_attempt,
                max_revisions=max_revisions,
            )
        )

    _TRUNCATION_FEEDBACK = (
        "Your response was cut off because it exceeded the maximum output length. "
        "Rewrite your response more concisely to fit within the output limit. "
        "Think about how to convey the essential information in a more concise way. "
    )

    def _is_truncated(self, response: LLMResponse) -> bool:
        return response.stop_reason == "max_tokens"

    def _emit_truncation_events(
        self,
        revision_attempt: int,
        max_revisions: int,
    ) -> None:
        self._emitter.emit(
            EvaluationEvent(
                trace_id=self._emitter.trace_id,
                span_id=self._emitter.span_id,
                parent_span_id=self._emitter.parent_span_id,
                evaluator_name="truncation",
                verdict="revise",
                score=None,
                feedback=self._TRUNCATION_FEEDBACK,
                revision_attempt=revision_attempt,
            )
        )
        self._emit_evaluation_revision(
            self._TRUNCATION_FEEDBACK,
            revision_attempt,
            max_revisions,
        )

    def _aggregate_usage(self, usages: list[Usage]) -> Usage:
        return Usage(
            input_tokens=sum(u.input_tokens for u in usages),
            output_tokens=sum(u.output_tokens for u in usages),
        )
