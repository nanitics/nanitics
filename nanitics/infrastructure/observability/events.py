from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, computed_field


class Usage(BaseModel):
    """Token usage statistics from an LLM call.

    ``total_tokens`` is derived from ``input_tokens + output_tokens`` and
    is therefore exposed as a read-only computed field: it cannot be
    passed to the constructor and is ignored if present in inputs to
    ``model_validate``. It still appears in ``model_dump()`` output so
    downstream consumers (trace stores, observatory UI, OTLP exporters)
    see the same wire shape.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseEvent(BaseModel):
    """Base class for all trace events.

    Every event carries trace context (trace_id, span_id, parent_span_id),
    a unique event_id, a UTC timestamp, and an event_type discriminator.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str


# --- Agent Lifecycle ---


class ToolInfo(BaseModel):
    """Metadata about a tool available to an agent."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    requires_approval: bool = False


class AgentStartEvent(BaseEvent):
    """Emitted when an agent begins execution."""

    event_type: Literal["agent.start"] = "agent.start"
    agent_name: str
    task_input: str
    model_name: str | None = None
    tools_available: list[str]
    tool_schemas: list[ToolInfo] = Field(default_factory=list)
    agent_type: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class AgentStepEvent(BaseEvent):
    """Emitted at each iteration of the agent loop.

    Each call to the LLM inside an agent's execution produces one
    ``agent.step`` event. Agents that call the LLM once (e.g.
    ``ReasoningAgent`` without an evaluator) emit one event; agents
    that loop (``ReActAgent``, ``CodeActAgent``, ``ReWOOAgent``,
    ``ReasoningAgent`` with revisions) emit one per iteration.

    Field contracts:

    ``thought`` — free-text reasoning from the model on this step, if
    any. Populated from ``LLMResponse.reasoning_text``. **Do not
    populate** with structured output, parsed JSON, or final content —
    use ``artifact`` for structured output and
    ``LLMResponseEvent.content`` for the full response body.

    ``action`` — what the agent did this step. Tool name(s) for
    tool-using agents (comma-joined when the step dispatches multiple
    calls); concatenated code blocks for ``CodeActAgent``; ``None`` for
    agents that do not act on this step (``ReasoningAgent``, ``ReWOO``
    planner, step-marker emissions on tree-search agents).

    ``observation`` — what the agent observed. Formatted tool results
    for tool-using agents; code execution output for ``CodeActAgent``;
    on a terminal no-tool-calls step (``ReActAgent`` or ``CodeActAgent``
    exit with a final answer and no further tool dispatch), the model's
    final content — the "what the step produced" observation that
    parallels tool-result content on tool-using steps. ``None`` when
    nothing was observed and no final content was produced.

    ``artifact`` — structured per-step output. Producers call
    ``model.model_dump()`` on the Pydantic model representing the
    step's structured output (plan, verdict, evaluation, parsed
    output). Consumers type-assert on the agent type and parse.
    """

    event_type: Literal["agent.step"] = "agent.step"
    agent_name: str
    step_number: int
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    artifact: dict[str, Any] | None = None


class AgentCompleteEvent(BaseEvent):
    """Emitted when an agent finishes successfully."""

    event_type: Literal["agent.complete"] = "agent.complete"
    agent_name: str
    output: str | None = None
    total_steps: int
    termination_reason: str


class AgentErrorEvent(BaseEvent):
    """Emitted when an agent encounters a fatal error."""

    event_type: Literal["agent.error"] = "agent.error"
    agent_name: str
    error_type: str
    error_message: str
    error_metadata: dict[str, Any]
    step_number: int | None = None


# --- LLM Calls ---


class LLMRequestEvent(BaseEvent):
    """Emitted before sending a request to the LLM."""

    event_type: Literal["llm.request"] = "llm.request"
    model_name: str
    system_prompt: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    output_schema: dict[str, Any] | None = None
    label: str | None = None


class LLMTokenEvent(BaseEvent):
    """Emitted for each text delta during streaming LLM generation."""

    event_type: Literal["llm.token"] = "llm.token"
    token: str
    agent_name: str


class LLMResponseEvent(BaseEvent):
    """Emitted after receiving a response from the LLM."""

    event_type: Literal["llm.response"] = "llm.response"
    model_name: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: Usage
    duration_ms: float
    label: str | None = None


# --- Tool Calls ---


class ToolInvokeEvent(BaseEvent):
    """Emitted when the registry dispatches a tool that actually executes.

    The underlying ``execute()`` call has begun (or, for
    parameter-validation failures, was attempted and rejected
    pre-execute). Wrappers that short-circuit before the inner tool
    runs — e.g.
    :class:`~nanitics.collaboration.approval_wrapped.ApprovalWrappedTool`
    on the reject path — cause this event to be suppressed; see
    :attr:`~nanitics.strategies.tools.protocol.ToolResult.executed`.
    """

    event_type: Literal["tool.invoke"] = "tool.invoke"
    tool_call_id: str
    tool_name: str
    parameters: dict[str, Any]


class ToolResultEvent(BaseEvent):
    """Emitted as the paired result for every :class:`ToolInvokeEvent`.

    Present on the success, error, timeout, and parameter-validation
    paths — whenever the tool actually executed or was attempted and
    rejected pre-execute. Never emitted when a wrapper short-circuits
    execution; in that case neither this event nor its paired
    :class:`ToolInvokeEvent` is emitted. See
    :attr:`~nanitics.strategies.tools.protocol.ToolResult.executed`.
    """

    event_type: Literal["tool.result"] = "tool.result"
    tool_call_id: str
    tool_name: str
    result: str | None = None
    error: str | None = None
    success: bool
    duration_ms: float


# --- Error Recovery ---


class ErrorRetryEvent(BaseEvent):
    """Emitted before retrying a failed operation."""

    event_type: Literal["error.retry"] = "error.retry"
    error_type: str
    error_message: str
    attempt: int
    max_attempts: int
    delay_ms: float
    category: str


class ErrorCorrectionEvent(BaseEvent):
    """Emitted when injecting a correction prompt for the LLM to self-correct."""

    event_type: Literal["error.correction"] = "error.correction"
    error_type: str
    error_message: str
    correction_prompt: str
    attempt: int
    max_attempts: int


class ErrorDegradationEvent(BaseEvent):
    """Emitted when falling back to degraded behavior after exhausting retries."""

    event_type: Literal["error.degradation"] = "error.degradation"
    error_type: str
    error_message: str
    degradation_message: str


# --- Context Management ---


class RemovedMessageInfo(BaseModel):
    """Information about a message removed during context truncation."""

    model_config = ConfigDict(frozen=True)

    role: str
    original_index: int
    content: str | list[Any]


class ContextTruncationEvent(BaseEvent):
    """Emitted after truncating conversation history to fit the context window."""

    event_type: Literal["context.truncation"] = "context.truncation"
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    removed_messages: list[RemovedMessageInfo] = Field(default_factory=list)


class ContextSummarizationEvent(BaseEvent):
    """Emitted after summarizing conversation history to reduce token usage."""

    event_type: Literal["context.summarization"] = "context.summarization"
    messages_summarized: int
    summary_tokens: int
    original_tokens: int
    summary_text: str
    summarization_input: str


class ContextContribution(BaseModel):
    """Information about a single context provider's contribution during assembly."""

    model_config = ConfigDict(frozen=True)

    provider_name: str
    content_length: int
    priority: int
    protected: bool
    content: str


class ContextAssemblyEvent(BaseEvent):
    """Emitted after assembling context from all registered providers."""

    event_type: Literal["context.assembly"] = "context.assembly"
    contributions: list[ContextContribution]
    total_injected: int


# --- Working Memory ---


class WorkingMemoryReadEvent(BaseEvent):
    """Emitted when reading working memory for context injection."""

    event_type: Literal["memory.working.read"] = "memory.working.read"
    content: str | None
    token_count: int


class WorkingMemoryUpdateEvent(BaseEvent):
    """Emitted when working memory content is updated."""

    event_type: Literal["memory.working.update"] = "memory.working.update"
    previous_content: str | None
    new_content: str
    source: str


# --- Long-Term Memory ---


class LongTermStoreEvent(BaseEvent):
    """Emitted when storing a key-value pair in long-term memory."""

    event_type: Literal["memory.longterm.store"] = "memory.longterm.store"
    key: str
    value: str
    namespace: str | None


class LongTermRetrieveEvent(BaseEvent):
    """Emitted when retrieving a value by key from long-term memory."""

    event_type: Literal["memory.longterm.retrieve"] = "memory.longterm.retrieve"
    key: str
    namespace: str | None
    found: bool
    value: str | None


class LongTermDeleteEvent(BaseEvent):
    """Emitted when deleting a key from long-term memory."""

    event_type: Literal["memory.longterm.delete"] = "memory.longterm.delete"
    key: str
    namespace: str | None


class LongTermListEvent(BaseEvent):
    """Emitted when listing all keys in long-term memory."""

    event_type: Literal["memory.longterm.list"] = "memory.longterm.list"
    namespace: str | None
    keys: list[str]


# --- Semantic Memory ---


class SemanticStoreEvent(BaseEvent):
    """Emitted when storing content with embeddings in semantic memory."""

    event_type: Literal["memory.semantic.store"] = "memory.semantic.store"
    content: str
    entry_id: str
    namespace: str | None


class SemanticSearchEvent(BaseEvent):
    """Emitted when searching semantic memory by similarity."""

    event_type: Literal["memory.semantic.search"] = "memory.semantic.search"
    query: str
    results_count: int
    top_score: float | None
    namespace: str | None


class SemanticDeleteEvent(BaseEvent):
    """Emitted when deleting an entry from semantic memory."""

    event_type: Literal["memory.semantic.delete"] = "memory.semantic.delete"
    entry_id: str
    namespace: str | None


# --- Episodic Memory ---


class EpisodeRecordEvent(BaseEvent):
    """Emitted when recording a new episode in episodic memory."""

    event_type: Literal["memory.episode.record"] = "memory.episode.record"
    episode_id: str
    situation: str
    outcome: str
    has_reflection: bool
    namespace: str | None


class EpisodeRecallEvent(BaseEvent):
    """Emitted when recalling past episodes by similarity."""

    event_type: Literal["memory.episode.recall"] = "memory.episode.recall"
    query: str
    results_count: int
    top_score: float | None
    namespace: str | None


class EpisodeForgetEvent(BaseEvent):
    """Emitted when forgetting an episode from episodic memory."""

    event_type: Literal["memory.episode.forget"] = "memory.episode.forget"
    episode_id: str
    namespace: str | None


# --- Evaluation ---


class EvaluationEvent(BaseEvent):
    """Emitted after evaluating agent output quality."""

    event_type: Literal["evaluation.result"] = "evaluation.result"
    evaluator_name: str
    verdict: str
    score: float | None = None
    feedback: str | None = None
    revision_attempt: int


class EvaluationExhaustedEvent(BaseEvent):
    """Emitted when evaluation revision budget is exhausted and non-passing output is served."""

    event_type: Literal["evaluation.exhausted"] = "evaluation.exhausted"
    evaluator_name: str
    verdict: str
    revision_count: int
    max_revisions: int
    feedback: str | None = None


class EvaluationRevisionEvent(BaseEvent):
    """Emitted when requesting an output revision based on evaluation feedback."""

    event_type: Literal["evaluation.revision"] = "evaluation.revision"
    feedback: str
    revision_attempt: int
    max_revisions: int


# --- Reflection ---


class ReflectionGeneratedEvent(BaseEvent):
    """Emitted when a ReflexionAgent generates a self-reflection."""

    event_type: Literal["reflection.generated"] = "reflection.generated"
    attempt_number: int
    max_attempts: int
    reflection_text: str
    evaluation_feedback: str | None
    episode_id: str


# --- Spans ---


class SpanStartEvent(BaseEvent):
    """Emitted when entering a named span."""

    event_type: Literal["span.start"] = "span.start"
    name: str


class SpanEndEvent(BaseEvent):
    """Emitted when exiting a named span, with duration."""

    event_type: Literal["span.end"] = "span.end"
    name: str
    duration_ms: float


# --- Workflow Orchestration ---


class WorkflowStartEvent(BaseEvent):
    """Emitted when a workflow begins execution."""

    event_type: Literal["workflow.start"] = "workflow.start"
    workflow_name: str
    workflow_type: str
    step_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowStepDefinition(BaseModel):
    """Describes a single step in a workflow's structure."""

    model_config = ConfigDict(frozen=True)

    name: str
    step_type: str  # "agent", "workflow", "function", "custom"
    index: int | None = None
    depends_on: list[str] = Field(default_factory=list)
    parallel_group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowStructureEvent(BaseEvent):
    """Emitted at workflow start to describe the complete step structure."""

    event_type: Literal["workflow.structure"] = "workflow.structure"
    workflow_name: str
    workflow_type: str
    steps: list[WorkflowStepDefinition]


class WorkflowStepCompleteEvent(BaseEvent):
    """Emitted when a workflow step finishes."""

    event_type: Literal["workflow.step.complete"] = "workflow.step.complete"
    workflow_name: str
    step_name: str
    step_index: int
    step_duration_ms: int | None = None
    step_output: str | None = None


class WorkflowCompleteEvent(BaseEvent):
    """Emitted when a workflow finishes successfully."""

    event_type: Literal["workflow.complete"] = "workflow.complete"
    workflow_name: str
    workflow_type: str
    total_steps_executed: int


class WorkflowErrorEvent(BaseEvent):
    """Emitted when a workflow encounters an error."""

    event_type: Literal["workflow.error"] = "workflow.error"
    workflow_name: str
    workflow_type: str
    error_type: str
    error_message: str
    failed_step: str | None = None


# --- Multi-Agent ---


class DelegationEvent(BaseEvent):
    """Emitted when an agent delegates a task to another agent."""

    event_type: Literal["multi_agent.delegation"] = "multi_agent.delegation"
    caller_agent: str
    delegate_agent: str
    task: str
    transfer_strategy: str


class HandoffEvent(BaseEvent):
    """Emitted during a structured handoff between agents."""

    event_type: Literal["multi_agent.handoff"] = "multi_agent.handoff"
    from_agent: str
    to_agent: str
    payload_fields: list[str]
    payload_size: int


class SupervisionEvent(BaseEvent):
    """Emitted when a supervisor intervenes on an agent's run."""

    event_type: Literal["multi_agent.supervision"] = "multi_agent.supervision"
    supervised_agent: str
    action: str
    trigger_name: str
    feedback: str | None = None
    reassigned_to: str | None = None
    attempt: int


# --- Human-in-the-Loop ---


class HumanInputRequestEvent(BaseEvent):
    """Emitted when an agent requests human input."""

    event_type: Literal["hitl.request"] = "hitl.request"
    request_id: str
    request_type: str
    prompt: str
    context: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanInputResponseEvent(BaseEvent):
    """Emitted when a human responds to an input request."""

    event_type: Literal["hitl.response"] = "hitl.response"
    request_id: str
    decision: str
    has_content: bool
    wait_duration_ms: int


# --- Revision ---


class RevisionStartEvent(BaseEvent):
    """Emitted when a revision workflow begins."""

    event_type: Literal["revision.start"] = "revision.start"
    step_name: str
    worker_count: int
    max_revisions: int


class RevisionAttemptEvent(BaseEvent):
    """Emitted at each attempt within a revision workflow."""

    event_type: Literal["revision.attempt"] = "revision.attempt"
    step_name: str
    attempt_number: int
    feedback: str


class RevisionCompleteEvent(BaseEvent):
    """Emitted when a revision workflow finishes."""

    event_type: Literal["revision.complete"] = "revision.complete"
    step_name: str
    total_attempts: int
    final_decision: str


# --- Durable Execution ---


class ExecutionSuspendedEvent(BaseEvent):
    """Emitted when execution suspends, e.g. for HITL or a timeout."""

    event_type: Literal["execution.suspended"] = "execution.suspended"
    suspension_id: str
    suspension_type: str
    checkpoint_id: str
    step_name: str | None = None
    agent_name: str | None = None


class ExecutionResumedEvent(BaseEvent):
    """Emitted when execution resumes from a checkpoint."""

    event_type: Literal["execution.resumed"] = "execution.resumed"
    checkpoint_id: str
    suspension_id: str
    resumed_from_step: str | None = None


class CheckpointSavedEvent(BaseEvent):
    """Emitted when a checkpoint is persisted."""

    event_type: Literal["checkpoint.saved"] = "checkpoint.saved"
    checkpoint_id: str
    checkpoint_type: str
    run_id: str


# --- Planning ---


class PlanStepDetail(BaseModel, frozen=True):
    """Detail of a single plan step, used in plan creation events."""

    step_id: str
    description: str
    metadata: dict[str, Any]


class PlanCreatedEvent(BaseEvent):
    """Emitted when an agent creates a new plan."""

    event_type: Literal["planning.plan.created"] = "planning.plan.created"
    plan_id: str
    plan_name: str
    step_count: int
    goal_count: int
    namespace: str | None = None
    steps: list[PlanStepDetail] = Field(default_factory=list)


class PlanStepUpdatedEvent(BaseEvent):
    """Emitted when a plan step's status changes."""

    event_type: Literal["planning.step.updated"] = "planning.step.updated"
    plan_id: str
    step_id: str
    step_description: str
    previous_status: str
    new_status: str
    has_result: bool


class PlanRevisedEvent(BaseEvent):
    """Emitted when an agent revises its plan."""

    event_type: Literal["planning.plan.revised"] = "planning.plan.revised"
    plan_id: str
    steps_before: int
    steps_after: int
    steps_preserved: int
    revision_reason: str


class GoalStatusChangedEvent(BaseEvent):
    """Emitted when a goal's status changes."""

    event_type: Literal["planning.goal.status_changed"] = "planning.goal.status_changed"
    plan_id: str
    goal_id: str
    goal_description: str
    previous_status: str
    new_status: str


# --- Code Execution Events ---


class CodeExecutionEvent(BaseEvent):
    """Emitted when a CodeActAgent submits code for execution."""

    event_type: Literal["code.execution"] = "code.execution"
    agent_name: str
    code: str
    step_number: int


class CodeExecutionResultEvent(BaseEvent):
    """Emitted after code execution completes in a sandbox."""

    event_type: Literal["code.execution.result"] = "code.execution.result"
    agent_name: str
    stdout: str
    stderr: str
    return_value: str | None = None
    success: bool
    error: str | None = None
    duration_ms: float
    step_number: int


# --- Tree Search Events ---


class TreeSearchNodeCreatedEvent(BaseEvent):
    """Emitted when a new node is added to the search tree."""

    event_type: Literal["tree_search.node.created"] = "tree_search.node.created"
    node_id: str
    parent_id: str | None
    depth: int
    content: str
    node_type: str  # "thought" or "action"
    action: str | None = None
    observation: str | None = None
    is_terminal: bool = False
    is_failed: bool = False
    error_message: str | None = None
    terminal_suppressed: bool | None = None


class TreeSearchNodeEvaluatedEvent(BaseEvent):
    """Emitted when a search tree node receives an evaluation score."""

    event_type: Literal["tree_search.node.evaluated"] = "tree_search.node.evaluated"
    node_id: str
    score: float
    is_terminal: bool


class TreeSearchNodePrunedEvent(BaseEvent):
    """Emitted when a search tree node is pruned."""

    event_type: Literal["tree_search.node.pruned"] = "tree_search.node.pruned"
    node_id: str
    reason: str


class TreeSearchCompleteEvent(BaseEvent):
    """Emitted when a tree search process finishes."""

    event_type: Literal["tree_search.complete"] = "tree_search.complete"
    total_nodes: int
    max_depth_reached: int
    selected_node_id: str
    termination_reason: str
    search_strategy: str
    accepted_count: int | None = None


class MCTSIterationEvent(BaseEvent):
    """Emitted at each Monte Carlo Tree Search iteration (used by LATSAgent)."""

    event_type: Literal["mcts.iteration"] = "mcts.iteration"
    iteration_number: int
    selected_node_id: str
    selection_path: list[str]
    expanded_count: int
    best_value_so_far: float
    node_values: dict[str, float] = Field(default_factory=dict)


class MCTSBackpropagationEvent(BaseEvent):
    """Emitted when a value is backpropagated through the MCTS tree."""

    event_type: Literal["mcts.backpropagation"] = "mcts.backpropagation"
    propagated_value: float
    path_length: int
    updated_node_ids: list[str]


# --- Shared Memory ---


class SharedMemoryWriteEvent(BaseEvent):
    """Emitted when writing an entry to shared memory."""

    event_type: Literal["memory.shared.write"] = "memory.shared.write"
    entry_id: str
    author: str
    content: str
    scope: str | None = None
    entry_count: int


class SharedMemoryReadEvent(BaseEvent):
    """Emitted when reading entries from shared memory."""

    event_type: Literal["memory.shared.read"] = "memory.shared.read"
    scope: str | None = None
    author_filter: str | None = None
    entries_returned: int


class SharedMemorySupersededEvent(BaseEvent):
    """Emitted when an entry in shared memory is superseded by a new one."""

    event_type: Literal["memory.shared.supersede"] = "memory.shared.supersede"
    original_entry_id: str
    new_entry_id: str
    author: str
    content: str
    scope: str | None = None


class SharedMemoryRetractEvent(BaseEvent):
    """Emitted when an entry is retracted from shared memory."""

    event_type: Literal["memory.shared.retract"] = "memory.shared.retract"
    entry_id: str
    author: str
    reason: str
    scope: str | None = None


# --- Blackboard ---


class BlackboardRoundEntry(BaseModel):
    """A single contribution within a blackboard round."""

    model_config = ConfigDict(frozen=True)

    operation: Literal["write", "supersede", "retract"]
    author: str
    content: str = ""
    scope: str | None = None
    entry_id: str
    original_entry_id: str | None = None
    retract_reason: str | None = None


class BlackboardStartEvent(BaseEvent):
    """Emitted when a blackboard coordination pattern begins."""

    event_type: Literal["blackboard.start"] = "blackboard.start"
    task: str
    agent_names: list[str]
    control_strategy: str
    max_rounds: int


class BlackboardRoundEvent(BaseEvent):
    """Emitted at each round of blackboard execution."""

    event_type: Literal["blackboard.round"] = "blackboard.round"
    round_number: int
    agents_activated: list[str]
    contributions: int
    total_contributions: int
    round_entries: list[BlackboardRoundEntry] = Field(default_factory=list)


class BlackboardCompleteEvent(BaseEvent):
    """Emitted when a blackboard coordination pattern finishes."""

    event_type: Literal["blackboard.complete"] = "blackboard.complete"
    rounds_completed: int
    termination_reason: str
    total_contributions: int
    agent_contributions: dict[str, int]


# --- Broadcast ---


class BroadcastStartEvent(BaseEvent):
    """Emitted when broadcasting a task to multiple agents."""

    event_type: Literal["multi_agent.broadcast.start"] = "multi_agent.broadcast.start"
    task: str
    agent_names: list[str]
    response_strategy: str


class BroadcastResponseEvent(BaseEvent):
    """Emitted when an agent responds to a broadcast."""

    event_type: Literal["multi_agent.broadcast.response"] = "multi_agent.broadcast.response"
    agent_name: str
    output: str
    steps: int
    error: str | None = None


class BroadcastCompleteEvent(BaseEvent):
    """Emitted when all broadcast responses have been collected and aggregated."""

    event_type: Literal["multi_agent.broadcast.complete"] = "multi_agent.broadcast.complete"
    total_agents: int
    responses_collected: int
    response_strategy: str
    aggregated_output: str
    failures: int = 0


# --- Bidding ---


class BiddingStartEvent(BaseEvent):
    """Emitted when a bidding auction starts."""

    event_type: Literal["multi_agent.bidding.start"] = "multi_agent.bidding.start"
    task: str
    participant_names: list[str]


class BidReceivedEvent(BaseEvent):
    """Emitted when an agent submits a bid."""

    event_type: Literal["multi_agent.bidding.bid"] = "multi_agent.bidding.bid"
    agent_name: str
    confidence: float
    reasoning: str
    estimated_cost: float | None = None
    error: str | None = None


class BidAllocatedEvent(BaseEvent):
    """Emitted when a bid is allocated to a winner or all bids are rejected."""

    event_type: Literal["multi_agent.bidding.allocated"] = "multi_agent.bidding.allocated"
    winner: str | None
    confidence: float | None
    total_bids: int
    rejection_reason: str | None = None


class BiddingCompleteEvent(BaseEvent):
    """Emitted when the bidding process finishes."""

    event_type: Literal["multi_agent.bidding.complete"] = "multi_agent.bidding.complete"
    winner: str | None
    total_participants: int
    allocated: bool


# --- Judge routing ---


class JudgeRoutingStartEvent(BaseEvent):
    """Emitted when comparative-judgment routing starts."""

    event_type: Literal["multi_agent.judge_routing.start"] = "multi_agent.judge_routing.start"
    task: str
    participant_names: list[str]


class JudgeRankingEvent(BaseEvent):
    """Emitted once per candidate in the judge's ranking."""

    event_type: Literal["multi_agent.judge_routing.ranking"] = "multi_agent.judge_routing.ranking"
    agent_name: str
    rank: int
    confidence: float
    reasoning: str
    estimated_cost: float | None = None


class JudgeAllocatedEvent(BaseEvent):
    """Emitted when the judge selects a winner or rejects all candidates."""

    event_type: Literal["multi_agent.judge_routing.allocated"] = "multi_agent.judge_routing.allocated"
    winner: str | None
    confidence: float | None
    total_candidates: int
    rejection_reason: str | None = None


class JudgeRoutingCompleteEvent(BaseEvent):
    """Emitted when the judge-routing process finishes."""

    event_type: Literal["multi_agent.judge_routing.complete"] = "multi_agent.judge_routing.complete"
    winner: str | None
    total_participants: int
    allocated: bool
    judge_error: str | None = None


# --- Debate ---


class DebateStartEvent(BaseEvent):
    """Emitted when a debate between agents begins."""

    event_type: Literal["multi_agent.debate.start"] = "multi_agent.debate.start"
    task: str
    debater_names: list[str]
    positions: dict[str, str]
    max_rounds: int
    resolution_strategy: str


class DebateArgumentEvent(BaseEvent):
    """Emitted when an agent makes an argument during a debate."""

    event_type: Literal["multi_agent.debate.argument"] = "multi_agent.debate.argument"
    round: int
    agent_name: str
    position: str
    argument: str


class DebateResolutionEvent(BaseEvent):
    """Emitted when a judge resolves a debate."""

    event_type: Literal["multi_agent.debate.resolution"] = "multi_agent.debate.resolution"
    winner: str | None
    reasoning: str
    rounds_completed: int


class DebateCompleteEvent(BaseEvent):
    """Emitted when a debate finishes."""

    event_type: Literal["multi_agent.debate.complete"] = "multi_agent.debate.complete"
    winner: str | None
    rounds_completed: int
    total_arguments: int
    termination_reason: str


# --- Consensus ---


class ConsensusStartEvent(BaseEvent):
    """Emitted when a consensus process starts."""

    event_type: Literal["multi_agent.consensus.start"] = "multi_agent.consensus.start"
    task: str
    agent_names: list[str]
    strategy: str
    deliberation_enabled: bool


class ConsensusVoteEvent(BaseEvent):
    """Emitted when an agent casts a vote during consensus."""

    event_type: Literal["multi_agent.consensus.vote"] = "multi_agent.consensus.vote"
    agent_name: str
    output: str
    round: int
    error: str | None = None


class ConsensusAgreementEvent(BaseEvent):
    """Emitted after measuring agreement level in a consensus round."""

    event_type: Literal["multi_agent.consensus.agreement"] = "multi_agent.consensus.agreement"
    round: int
    agreement_level: float
    converged: bool


class ConsensusCompleteEvent(BaseEvent):
    """Emitted when the consensus process finishes."""

    event_type: Literal["multi_agent.consensus.complete"] = "multi_agent.consensus.complete"
    strategy: str
    rounds_completed: int
    final_agreement: float
    agents_participated: int
    termination_reason: str


# --- Peer Network ---


class PeerNetworkStartEvent(BaseEvent):
    """Emitted when a peer network starts execution."""

    event_type: Literal["multi_agent.peer.start"] = "multi_agent.peer.start"
    task: str
    entry_agent: str
    peer_names: list[str]
    peer_descriptions: dict[str, str]
    max_invocations: int


class PeerConsultationEvent(BaseEvent):
    """Emitted when one peer consults another in the network."""

    event_type: Literal["multi_agent.peer.consultation"] = "multi_agent.peer.consultation"
    from_agent: str
    to_agent: str
    message: str
    consultation_number: int
    remaining_budget: int


class PeerNetworkCompleteEvent(BaseEvent):
    """Emitted when a peer network finishes execution."""

    event_type: Literal["multi_agent.peer.complete"] = "multi_agent.peer.complete"
    entry_agent: str
    total_consultations: int
    invocations_used: int
    agents_consulted: list[str]
    termination_reason: str


# --- Message Bus ---


class MessageBusStartEvent(BaseEvent):
    """Emitted when a message bus starts processing."""

    event_type: Literal["multi_agent.bus.start"] = "multi_agent.bus.start"
    seed_topics: list[str]
    seed_count: int
    subscriber_count: int
    subscriptions: dict[str, list[str]]
    max_messages: int
    max_depth: int


class MessagePublishedEvent(BaseEvent):
    """Emitted when a message is published to the bus."""

    event_type: Literal["multi_agent.bus.published"] = "multi_agent.bus.published"
    message_id: str
    topic: str
    author: str
    content: str
    depth: int
    parent_message_id: str | None = None


class MessageDeliveredEvent(BaseEvent):
    """Emitted when a message is delivered to and processed by a subscriber."""

    event_type: Literal["multi_agent.bus.delivered"] = "multi_agent.bus.delivered"
    message_id: str
    topic: str
    agent_name: str
    output: str
    steps: int
    messages_published: int
    error: str | None = None


class MessageBusCompleteEvent(BaseEvent):
    """Emitted when the message bus finishes processing all messages."""

    event_type: Literal["multi_agent.bus.complete"] = "multi_agent.bus.complete"
    total_messages: int
    total_executions: int
    failed_executions: int = 0
    max_depth_reached: int
    termination_reason: str
    agent_execution_counts: dict[str, int]


# --- Model Routing ---


class ModelRoutingEvent(BaseEvent):
    """Emitted when ``RoutingLLMClient`` selects a backend for a request."""

    event_type: Literal["model.routing"] = "model.routing"
    strategy_name: str
    selected_key: str
    available_keys: list[str]


# --- Run Lifecycle ---


class RunStartEvent(BaseEvent):
    """Emitted when a workflow run begins."""

    event_type: Literal["run.start"] = "run.start"
    run_id: str
    workflow_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunCompleteEvent(BaseEvent):
    """Emitted when a workflow run completes successfully."""

    event_type: Literal["run.complete"] = "run.complete"
    run_id: str
    duration_ms: int


class RunFailedEvent(BaseEvent):
    """Emitted when a workflow run fails."""

    event_type: Literal["run.failed"] = "run.failed"
    run_id: str
    error_type: str
    error_message: str


class RunSuspendedEvent(BaseEvent):
    """Emitted when a workflow run is suspended."""

    event_type: Literal["run.suspended"] = "run.suspended"
    run_id: str
    suspension_id: str


# --- Safety ---


class SafetyIterationLimitEvent(BaseEvent):
    """Emitted when an agent's iteration limit is reached and the agent terminates."""

    event_type: Literal["safety.iteration_limit"] = "safety.iteration_limit"
    agent_name: str
    current_iteration: int
    max_iterations: int
    step_number: int


class SafetyToolCallLimitEvent(BaseEvent):
    """Emitted when an agent's tool call limit is reached and the tool loop terminates."""

    event_type: Literal["safety.tool_call_limit"] = "safety.tool_call_limit"
    agent_name: str
    current_tool_calls: int
    max_tool_calls: int
    step_number: int


class SafetyCancellationEvent(BaseEvent):
    """Emitted when an agent detects a cancellation signal and terminates."""

    event_type: Literal["safety.cancellation"] = "safety.cancellation"
    agent_name: str
    step_number: int | None = None


# --- Service Errors ---


class ServiceErrorEvent(BaseEvent):
    """Emitted when an external service call fails.

    Tools emit this event to signal service-level failures (rate limits,
    auth errors, timeouts) so they can be surfaced to the user in real-time
    while the agent continues to reason about the error via ToolResult.
    """

    event_type: Literal["service.error"] = "service.error"
    service_name: str
    error_type: str  # rate_limit | auth_error | timeout | unavailable
    message: str
    tool_name: str | None = None


# --- Discriminated Union ---


def _get_event_type(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("event_type", ""))
    return str(getattr(v, "event_type", ""))


TraceEvent = Annotated[
    Annotated[AgentStartEvent, Tag("agent.start")]
    | Annotated[AgentStepEvent, Tag("agent.step")]
    | Annotated[AgentCompleteEvent, Tag("agent.complete")]
    | Annotated[AgentErrorEvent, Tag("agent.error")]
    | Annotated[LLMRequestEvent, Tag("llm.request")]
    | Annotated[LLMTokenEvent, Tag("llm.token")]
    | Annotated[LLMResponseEvent, Tag("llm.response")]
    | Annotated[ToolInvokeEvent, Tag("tool.invoke")]
    | Annotated[ToolResultEvent, Tag("tool.result")]
    | Annotated[ErrorRetryEvent, Tag("error.retry")]
    | Annotated[ErrorCorrectionEvent, Tag("error.correction")]
    | Annotated[ErrorDegradationEvent, Tag("error.degradation")]
    | Annotated[ContextTruncationEvent, Tag("context.truncation")]
    | Annotated[ContextSummarizationEvent, Tag("context.summarization")]
    | Annotated[ContextAssemblyEvent, Tag("context.assembly")]
    | Annotated[WorkingMemoryReadEvent, Tag("memory.working.read")]
    | Annotated[WorkingMemoryUpdateEvent, Tag("memory.working.update")]
    | Annotated[LongTermStoreEvent, Tag("memory.longterm.store")]
    | Annotated[LongTermRetrieveEvent, Tag("memory.longterm.retrieve")]
    | Annotated[LongTermDeleteEvent, Tag("memory.longterm.delete")]
    | Annotated[LongTermListEvent, Tag("memory.longterm.list")]
    | Annotated[SemanticStoreEvent, Tag("memory.semantic.store")]
    | Annotated[SemanticSearchEvent, Tag("memory.semantic.search")]
    | Annotated[SemanticDeleteEvent, Tag("memory.semantic.delete")]
    | Annotated[EpisodeRecordEvent, Tag("memory.episode.record")]
    | Annotated[EpisodeRecallEvent, Tag("memory.episode.recall")]
    | Annotated[EpisodeForgetEvent, Tag("memory.episode.forget")]
    | Annotated[EvaluationEvent, Tag("evaluation.result")]
    | Annotated[EvaluationExhaustedEvent, Tag("evaluation.exhausted")]
    | Annotated[EvaluationRevisionEvent, Tag("evaluation.revision")]
    | Annotated[SpanStartEvent, Tag("span.start")]
    | Annotated[SpanEndEvent, Tag("span.end")]
    | Annotated[WorkflowStartEvent, Tag("workflow.start")]
    | Annotated[WorkflowStructureEvent, Tag("workflow.structure")]
    | Annotated[WorkflowStepCompleteEvent, Tag("workflow.step.complete")]
    | Annotated[WorkflowCompleteEvent, Tag("workflow.complete")]
    | Annotated[WorkflowErrorEvent, Tag("workflow.error")]
    | Annotated[DelegationEvent, Tag("multi_agent.delegation")]
    | Annotated[HandoffEvent, Tag("multi_agent.handoff")]
    | Annotated[SupervisionEvent, Tag("multi_agent.supervision")]
    | Annotated[HumanInputRequestEvent, Tag("hitl.request")]
    | Annotated[HumanInputResponseEvent, Tag("hitl.response")]
    | Annotated[RevisionStartEvent, Tag("revision.start")]
    | Annotated[RevisionAttemptEvent, Tag("revision.attempt")]
    | Annotated[RevisionCompleteEvent, Tag("revision.complete")]
    | Annotated[ExecutionSuspendedEvent, Tag("execution.suspended")]
    | Annotated[ExecutionResumedEvent, Tag("execution.resumed")]
    | Annotated[CheckpointSavedEvent, Tag("checkpoint.saved")]
    | Annotated[PlanCreatedEvent, Tag("planning.plan.created")]
    | Annotated[PlanStepUpdatedEvent, Tag("planning.step.updated")]
    | Annotated[PlanRevisedEvent, Tag("planning.plan.revised")]
    | Annotated[GoalStatusChangedEvent, Tag("planning.goal.status_changed")]
    | Annotated[CodeExecutionEvent, Tag("code.execution")]
    | Annotated[CodeExecutionResultEvent, Tag("code.execution.result")]
    | Annotated[ReflectionGeneratedEvent, Tag("reflection.generated")]
    | Annotated[TreeSearchNodeCreatedEvent, Tag("tree_search.node.created")]
    | Annotated[TreeSearchNodeEvaluatedEvent, Tag("tree_search.node.evaluated")]
    | Annotated[TreeSearchNodePrunedEvent, Tag("tree_search.node.pruned")]
    | Annotated[TreeSearchCompleteEvent, Tag("tree_search.complete")]
    | Annotated[MCTSIterationEvent, Tag("mcts.iteration")]
    | Annotated[MCTSBackpropagationEvent, Tag("mcts.backpropagation")]
    | Annotated[SharedMemoryWriteEvent, Tag("memory.shared.write")]
    | Annotated[SharedMemoryReadEvent, Tag("memory.shared.read")]
    | Annotated[SharedMemorySupersededEvent, Tag("memory.shared.supersede")]
    | Annotated[SharedMemoryRetractEvent, Tag("memory.shared.retract")]
    | Annotated[BlackboardStartEvent, Tag("blackboard.start")]
    | Annotated[BlackboardRoundEvent, Tag("blackboard.round")]
    | Annotated[BlackboardCompleteEvent, Tag("blackboard.complete")]
    | Annotated[BroadcastStartEvent, Tag("multi_agent.broadcast.start")]
    | Annotated[BroadcastResponseEvent, Tag("multi_agent.broadcast.response")]
    | Annotated[BroadcastCompleteEvent, Tag("multi_agent.broadcast.complete")]
    | Annotated[BiddingStartEvent, Tag("multi_agent.bidding.start")]
    | Annotated[BidReceivedEvent, Tag("multi_agent.bidding.bid")]
    | Annotated[BidAllocatedEvent, Tag("multi_agent.bidding.allocated")]
    | Annotated[BiddingCompleteEvent, Tag("multi_agent.bidding.complete")]
    | Annotated[JudgeRoutingStartEvent, Tag("multi_agent.judge_routing.start")]
    | Annotated[JudgeRankingEvent, Tag("multi_agent.judge_routing.ranking")]
    | Annotated[JudgeAllocatedEvent, Tag("multi_agent.judge_routing.allocated")]
    | Annotated[JudgeRoutingCompleteEvent, Tag("multi_agent.judge_routing.complete")]
    | Annotated[DebateStartEvent, Tag("multi_agent.debate.start")]
    | Annotated[DebateArgumentEvent, Tag("multi_agent.debate.argument")]
    | Annotated[DebateResolutionEvent, Tag("multi_agent.debate.resolution")]
    | Annotated[DebateCompleteEvent, Tag("multi_agent.debate.complete")]
    | Annotated[ConsensusStartEvent, Tag("multi_agent.consensus.start")]
    | Annotated[ConsensusVoteEvent, Tag("multi_agent.consensus.vote")]
    | Annotated[ConsensusAgreementEvent, Tag("multi_agent.consensus.agreement")]
    | Annotated[ConsensusCompleteEvent, Tag("multi_agent.consensus.complete")]
    | Annotated[PeerNetworkStartEvent, Tag("multi_agent.peer.start")]
    | Annotated[PeerConsultationEvent, Tag("multi_agent.peer.consultation")]
    | Annotated[PeerNetworkCompleteEvent, Tag("multi_agent.peer.complete")]
    | Annotated[MessageBusStartEvent, Tag("multi_agent.bus.start")]
    | Annotated[MessagePublishedEvent, Tag("multi_agent.bus.published")]
    | Annotated[MessageDeliveredEvent, Tag("multi_agent.bus.delivered")]
    | Annotated[MessageBusCompleteEvent, Tag("multi_agent.bus.complete")]
    | Annotated[ModelRoutingEvent, Tag("model.routing")]
    | Annotated[RunStartEvent, Tag("run.start")]
    | Annotated[RunCompleteEvent, Tag("run.complete")]
    | Annotated[RunFailedEvent, Tag("run.failed")]
    | Annotated[RunSuspendedEvent, Tag("run.suspended")]
    | Annotated[SafetyIterationLimitEvent, Tag("safety.iteration_limit")]
    | Annotated[SafetyToolCallLimitEvent, Tag("safety.tool_call_limit")]
    | Annotated[SafetyCancellationEvent, Tag("safety.cancellation")]
    | Annotated[ServiceErrorEvent, Tag("service.error")],
    Discriminator(_get_event_type),
]
