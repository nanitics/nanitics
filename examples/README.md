# SDK Examples

Runnable examples demonstrating each component of the Nanitics SDK. Every
example uses `MockLLMClient` for deterministic, API-key-free execution
(the `providers/` section is the one exception — the real-LLM quickstart
is gated on credentials).

Examples are grouped by theme. The reading order within each section is
top-to-bottom: the first example in a theme is the gentlest starting
point, and later ones build on it.

## Running

These examples live in the repo, not in the installed wheel, so clone
first (`git clone …/nanitics`) and then run one directly:

```bash
uv run python examples/agents/react_agent.py
```

`uv run` resolves the project's dev dependencies automatically; if you
prefer a plain virtualenv with `pip install -e .[dev]`, drop the `uv run`
prefix.

Run all examples as tests (they all use mocks, so this is fast and
deterministic):

```bash
uv run pytest tests/test_examples.py -v
```

## Tools

Tool construction, composition, and the primitives that surround them —
registries, system-prompt contributors, events, web search, MCP, sandbox
and code execution.

| Example | Description | Guide |
|---|---|---|
| [tools/tool_basics.py](tools/tool_basics.py) | `@tool` decorator, `ToolResult`, Pydantic validation, `Tool` protocol, `ToolRegistry`, `ToolContext`, error handling | [Tools](../docs/guides/tools.md) |
| [tools/multi_tool_package.py](tools/multi_tool_package.py) | Factory returning `((tool_a, tool_b), state_dict)`; sibling tools sharing per-run state through `ToolContext.state`; agent-side `tool_state` plumbing | [Tools](../docs/guides/tools.md#multi-tool-packages-with-shared-state) |
| [tools/system_prompt_builder.py](tools/system_prompt_builder.py) | `SystemPromptBuilder`, `SystemPromptContributor`, section assembly, agent prompt composition | [Core Concepts](../docs/guides/core-concepts.md) |
| [tools/event_emitter.py](tools/event_emitter.py) | `InMemoryEmitter`, event type hierarchy, spans, listeners, child emitters, memory capping, event levels | [Observability](../docs/guides/observability.md) |
| [tools/web_search_tool.py](tools/web_search_tool.py) | `create_web_search_tool` (Tavily/Brave), `respx`-intercepted HTTPS, `ReActAgent` integration, `ToolInvokeEvent`/`ToolResultEvent` | [Tools](../docs/guides/tools.md) |
| [tools/conditional_tool.py](tools/conditional_tool.py) | `ConditionalTool`, `is_enabled` state-driven schema visibility, `ToolRegistry` filtering | [Tools](../docs/guides/tools.md) |
| [tools/mcp_tools.py](tools/mcp_tools.py) | `MCPClient`, in-process `FastMCP` server, `ReActAgent` over MCP-backed tools, multi-server `name_prefix` | [Tools](../docs/guides/tools.md) |
| [tools/http_file_tools.py](tools/http_file_tools.py) | `create_http_tool`, `create_file_read_tool`, UTF-8/base64 via `metadata.encoding`, `ReActAgent` integration | [Tools](../docs/guides/tools.md) |
| [tools/sandbox.py](tools/sandbox.py) | `SandboxConfig` security boundaries, `ExecutionResult`, `MockSandbox` lifecycle, deterministic testing | [Safety](../docs/guides/safety.md) |
| [tools/code_execution_tool.py](tools/code_execution_tool.py) | `create_code_execution_tool` over `Sandbox` protocol, `MockSandbox` / `DockerSandbox`, failure surfacing | [Tools](../docs/guides/tools.md) |

## Observability

Traces, collectors, and the instrumented client for observing non-agent
LLM calls.

| Example | Description | Guide |
|---|---|---|
| [observability/trace_collection.py](observability/trace_collection.py) | `InMemoryTraceStore`, `InMemoryPersistentTraceStore`, `TraceCollector`, `TraceQuery`, hierarchy, SSE streaming | [Observability](../docs/guides/observability.md) |
| [observability/instrumented_client.py](observability/instrumented_client.py) | `InstrumentedLLMClient`, non-agent LLM tracing, `label` partitioning, `LLMRequestEvent`/`LLMResponseEvent` | [Observability](../docs/guides/observability.md) |
| [observability/redaction_hook.py](observability/redaction_hook.py) | `RedactionHook` protocol, `TraceCollector`/`TracedExecutor` wire-ins, un-redacted emitter vs. redacted store/SSE, fail-closed exception semantics | [Trace Surface Hygiene](../docs/guides/observability.md#trace-surface-hygiene) |

## Agents

The agent types — `ReActAgent` as the baseline, then reasoning,
reflexion, ReWOO, CodeAct, LATS, and tree-of-thought specialisations.
Multimodal input is also here since it's an agent-capability concern.

| Example | Description | Guide |
|---|---|---|
| [agents/react_agent.py](agents/react_agent.py) | `ReActAgent` with tools, multi-turn conversation, `AgentResult` inspection, event tracing | [Getting Started](../docs/guides/getting-started.md) |
| [agents/reasoning_agent.py](agents/reasoning_agent.py) | `ReasoningAgent`, single-call reasoning, structured output with Pydantic, evaluation-driven revision | [Agent Types](../docs/guides/agent-types.md) |
| [agents/dispatch_over_structured_output.py](agents/dispatch_over_structured_output.py) | `ReasoningAgent` with `output_schema`, pure-Python dispatch over typed output, `await`-chain orchestration — the pre-pattern check before reaching for multi-agent or workflow primitives | [Multi-Agent Foundations](../docs/guides/multi-agent-foundations.md#pattern-progression) |
| [agents/refusal_as_output.py](agents/refusal_as_output.py) | Deterministic routing onto a `Literal` typology, `ReasoningAgent` with `output_schema=RefusalRationaleDraft`, assembly into a typed `RefusalToDraft` — application-layer composition on top of `ReasoningAgent` | [Agent Types](../docs/guides/agent-types.md#refusal-as-output) |
| [agents/reflexion_agent.py](agents/reflexion_agent.py) | `ReflexionAgent`, evaluate-reflect-retry loop, `ProgrammaticEvaluator`, episodic memory | [Agent Types](../docs/guides/agent-types.md) |
| [agents/rewoo_agent.py](agents/rewoo_agent.py) | `ReWOOAgent` — plan-first execution with variable substitution, parallel steps, plan persistence | [Agent Types](../docs/guides/agent-types.md) |
| [agents/codeact_agent.py](agents/codeact_agent.py) | `CodeActAgent`, `MockSandbox`, code execution loop, self-correction, tool bridge | [Agent Types](../docs/guides/agent-types.md) |
| [agents/lats_agent.py](agents/lats_agent.py) | `LATSAgent` — MCTS tree search, UCB1 selection, evaluation-guided pruning, backpropagation | [Agent Types](../docs/guides/agent-types.md) |
| [agents/tree_of_thought.py](agents/tree_of_thought.py) | `TreeOfThoughtAgent` — branching candidate generation, BFS/DFS/BEST_FIRST, pruning | [Agent Types](../docs/guides/agent-types.md) |
| [agents/multimodal_input.py](agents/multimodal_input.py) | `ReasoningAgent` multimodal input, `TextContentBlock`, `ImageContentBlock`, base64 encoding | [Agent Types](../docs/guides/agent-types.md) |

## Memory

Working, episodic, long-term, semantic, shared, and persistent
Postgres-backed semantic memory.

| Example | Description | Guide |
|---|---|---|
| [memory/working_memory.py](memory/working_memory.py) | `InMemoryWorkingMemory`, `WorkingMemoryContributor`, `WorkingMemoryProvider`, agent integration | [Memory](../docs/guides/memory.md) |
| [memory/episodic_memory.py](memory/episodic_memory.py) | `InMemoryEpisodeStore`, `extract_episode`, `RecallFilters`, pruning, `create_episodic_memory_tools` | [Memory](../docs/guides/memory.md) |
| [memory/long_term_memory.py](memory/long_term_memory.py) | `InMemoryLongTermStore`, `create_long_term_memory_tools`, namespace isolation, multi-run persistence | [Memory](../docs/guides/memory.md) |
| [memory/semantic_memory.py](memory/semantic_memory.py) | `InMemorySemanticStore`, `MockEmbeddingClient`, similarity ranking, namespace isolation, `SemanticMemoryProvider`, `SemanticMemoryContributor` | [Memory](../docs/guides/memory.md) |
| [memory/shared_memory.py](memory/shared_memory.py) | `InMemorySharedMemory`, entry lifecycle, two-agent coordination through shared state | [Memory](../docs/guides/memory.md#shared-memory) |
| [memory/persistent_semantic_memory.py](memory/persistent_semantic_memory.py) | `PostgresSemanticStore` + pgvector as drop-in for `InMemorySemanticStore`, CRUD, namespace isolation | [Memory](../docs/guides/memory.md) |

## Context

Context assembly, truncation, and summarization.

| Example | Description | Guide |
|---|---|---|
| [context/context_management.py](context/context_management.py) | `ContextManager`, `EstimateTokenCounter`, `TruncationPolicy`, `SummarizationPolicy`, message grouping | [Context Management](../docs/guides/context-management.md) |

## Control

Safety primitives: iteration limits, cancellation, error handling.

| Example | Description | Guide |
|---|---|---|
| [control/iteration_limits.py](control/iteration_limits.py) | `IterationLimiter`, `AgentIterationLimitError`, `SafetyIterationLimitEvent` | [Safety](../docs/guides/safety.md) |
| [control/cancellation.py](control/cancellation.py) | `CancellationToken`, cooperative cancellation, tool-triggered cancellation, `SafetyCancellationEvent` | [Safety](../docs/guides/safety.md) |
| [control/error_handling.py](control/error_handling.py) | `classify_error`, `ErrorHandler`, correction prompts, graceful degradation | [Error Handling](../docs/guides/error-handling.md) |

## Evaluation

Output evaluators and revision loops.

| Example | Description | Guide |
|---|---|---|
| [evaluation/evaluation.py](evaluation/evaluation.py) | `ProgrammaticEvaluator`, `LLMEvaluator`, `CompositeEvaluator`, custom `OutputEvaluator`, revision loops | [Evaluation](../docs/guides/evaluation.md) |

## Planning

Plan/goal primitives and plan-to-workflow conversion.

| Example | Description | Guide |
|---|---|---|
| [planning/planning.py](planning/planning.py) | `PlanningCapability`, `InMemoryPlanStore`, agent-driven `create_plan`, plan revision | [Planning](../docs/guides/planning.md) |
| [planning/planning_goals.py](planning/planning_goals.py) | `Goal`, `GoalStatus`, `create_goal`/`update_goal` tools, `GoalSatisfactionEvaluator`, hierarchies | [Planning](../docs/guides/planning.md) |

## Multi-agent

Foundations (agent-as-tool, handoff, context transfer, broadcast,
message bus, peer network) then coordination (supervisor, orchestrator,
bidding, blackboard, debate, consensus). Read in order; later examples
assume the earlier primitives.

| Example | Description | Guide |
|---|---|---|
| [multi_agent/agent_tool.py](multi_agent/agent_tool.py) | `AgentTool`, wrapping agent as tool, schema inspection, `DelegationEvent` tracing | [Multi-Agent Foundations](../docs/guides/multi-agent-foundations.md#agent-as-tool) |
| [multi_agent/handoff.py](multi_agent/handoff.py) | `HandoffPayload`, `HandoffTransfer`, `HandoffStep`, `create_handoff_chain`, `HandoffEvent` | [Multi-Agent Foundations](../docs/guides/multi-agent-foundations.md#handoff-protocol) |
| [multi_agent/context_transfer.py](multi_agent/context_transfer.py) | `RawOutputTransfer`, `TrajectoryTransfer`, `SummaryTransfer`, `CustomTransfer` — strategy comparison | [Multi-Agent Foundations](../docs/guides/multi-agent-foundations.md#context-transfer) |
| [multi_agent/broadcast.py](multi_agent/broadcast.py) | `Broadcast`, `CollectAll`, `SelectBest`, `FilterResponses`, `CapabilityFilter`, `AgentFailure` | [Multi-Agent Foundations](../docs/guides/multi-agent-foundations.md#broadcast) |
| [multi_agent/message_bus.py](multi_agent/message_bus.py) | `MessageBus`, `BusMessage`, `TopicSubscription`, `MessageFilter`, `MaxMessagesTermination` | [Multi-Agent Foundations](../docs/guides/multi-agent-foundations.md#message-bus) |
| [multi_agent/peer_network.py](multi_agent/peer_network.py) | `PeerNetwork`, `PeerSpec`, `PeerBudgetExceededError`, transitive chains | [Multi-Agent Foundations](../docs/guides/multi-agent-foundations.md#peer-network) |
| [multi_agent/supervisor.py](multi_agent/supervisor.py) | `Supervisor`, `PredicateTrigger`, `QualityTrigger`, `BudgetTrigger`, accept/retry/reassign/escalate | [Multi-Agent Coordination](../docs/guides/multi-agent-coordination.md#supervisor) |
| [multi_agent/orchestrator.py](multi_agent/orchestrator.py) | `create_orchestrator`, `orchestrator_prompt_section`, specialist `AgentTool` delegation, `FinalOutputStrategy.RELAY_LAST` relay mode | [Multi-Agent Coordination](../docs/guides/multi-agent-coordination.md#orchestrator) |
| [multi_agent/bidding.py](multi_agent/bidding.py) | `Bidding` auction, `FixedBidGenerator`, `LLMBidGenerator`, allocation strategies, calibration-anchor template, `HighestConfidence(tiebreaker=...)` chain | [Multi-Agent Coordination](../docs/guides/multi-agent-coordination.md#bidding) |
| [multi_agent/judge_router.py](multi_agent/judge_router.py) | `JudgeRouter` comparative-judgment routing, `DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE`, below-threshold rejection, judge-routing event family, `Bidding` vs `JudgeRouter` trace comparison | [Multi-Agent Coordination](../docs/guides/multi-agent-coordination.md#judge-routed-allocation) |
| [multi_agent/blackboard.py](multi_agent/blackboard.py) | `Blackboard`, `ScheduledControl`/`PrioritizedControl`/`OpportunisticControl`, termination | [Multi-Agent Coordination](../docs/guides/multi-agent-coordination.md#blackboard) |
| [multi_agent/debate.py](multi_agent/debate.py) | `Debate`, `Debater`, `JudgeResolution`, `LLMJudgeResolution` with criteria | [Multi-Agent Coordination](../docs/guides/multi-agent-coordination.md#debate) |
| [multi_agent/consensus.py](multi_agent/consensus.py) | `Consensus`, `MajorityVoting`/`WeightedVoting`/`BestOfN`, `DeliberationConfig` | [Multi-Agent Coordination](../docs/guides/multi-agent-coordination.md#consensus) |

## Workflows

Compositional workflows: sequential, parallel, DAG, loop, map/reduce,
conditional, and plan-to-workflow.

| Example | Description | Guide |
|---|---|---|
| [workflows/sequential_pipeline.py](workflows/sequential_pipeline.py) | `Sequential` with `FunctionStep`/`AgentStep`, `Pipeline` with `Stage` type contracts | [Orchestration](../docs/guides/orchestration.md) |
| [workflows/parallel.py](workflows/parallel.py) | `Parallel`, default and custom aggregators, `FailurePolicy` | [Orchestration](../docs/guides/orchestration.md) |
| [workflows/dag.py](workflows/dag.py) | `DAG`, `DAGNode`, diamond execution, input routing, cycle/dangling-ref validation, `BEST_EFFORT` | [Orchestration](../docs/guides/orchestration.md) |
| [workflows/loop.py](workflows/loop.py) | `Loop` with condition callback, iteration safety net, async termination | [Orchestration](../docs/guides/orchestration.md) |
| [workflows/map_reduce.py](workflows/map_reduce.py) | `MapReduce`, structural splitting, concurrency, `FailurePolicy`, async splitter/reducer | [Orchestration](../docs/guides/orchestration.md) |
| [workflows/conditional.py](workflows/conditional.py) | `Conditional`, sync/async router, branch selection, default fallback | [Orchestration](../docs/guides/orchestration.md) |
| [workflows/plan_to_workflow.py](workflows/plan_to_workflow.py) | `plan_to_workflow`, `TaskPlan`, `TaskNode` — auto-selecting Parallel/Sequential/DAG | [Orchestration](../docs/guides/orchestration.md) |

## Human-in-the-Loop

Approval gates, revision gates, approval-wrapped tools, agent-initiated
HITL, and async HTTP-style HITL. Durable HITL lives under `durability/`.

| Example | Description | Guide |
|---|---|---|
| [hitl/approval_gate.py](hitl/approval_gate.py) | `ApprovalGate`, `CallbackHumanInputProvider`, APPROVE/REJECT/MODIFY/REVISE, dynamic prompts | [Human-in-the-Loop](../docs/guides/human-in-the-loop.md) |
| [hitl/revision_gate.py](hitl/revision_gate.py) | `RevisionGate`, approve/revise/reject, feedback injection, max revisions | [Human-in-the-Loop](../docs/guides/human-in-the-loop.md) |
| [hitl/approval_wrapped_tool.py](hitl/approval_wrapped_tool.py) | `ApprovalWrappedTool`, schema preservation, approve/reject/modify, parameter modification | [Human-in-the-Loop](../docs/guides/human-in-the-loop.md) |
| [hitl/hitl_tools.py](hitl/hitl_tools.py) | `create_request_approval_tool`, `create_ask_human_tool`, `create_hitl_tools`, agent-initiated | [Human-in-the-Loop](../docs/guides/human-in-the-loop.md) |
| [hitl/async_hitl.py](hitl/async_hitl.py) | `AsyncHumanInputProvider`, async `Future` resolution, HTTP-integration producer/consumer | [Human-in-the-Loop](../docs/guides/human-in-the-loop.md) |

## Providers

LLM client adapters: routing, LiteLLM, OpenAI, and the real-LLM
Anthropic quickstart.

| Example | Description | Guide |
|---|---|---|
| [providers/llm_routing.py](providers/llm_routing.py) | `RoutingLLMClient`, `RuleBasedRouting`, `CostBudgetRouting`, custom `RoutingStrategy` | [Core Concepts](../docs/guides/core-concepts.md) |
| [providers/litellm_adapter.py](providers/litellm_adapter.py) | `LiteLLMClient`, 100+ providers behind one `LLMClient`, `RoutingLLMClient` fallback | [Core Concepts](../docs/guides/core-concepts.md) |
| [providers/openai_client.py](providers/openai_client.py) | `OpenAILLMClient` as drop-in provider, `MockLLMClient` hermetic, `RoutingLLMClient` cross-provider | [Core Concepts](../docs/guides/core-concepts.md) |
| [providers/real_llm_quickstart.py](providers/real_llm_quickstart.py) | `AnthropicLLMClient` live quickstart, `ANTHROPIC_API_KEY` gated, printed trace, token cost | [Getting Started](../docs/guides/getting-started.md) |

## Durability

Checkpoint suspension and the canonical durable resume service.

| Example | Description | Guide |
|---|---|---|
| [durability/checkpoint_suspension.py](durability/checkpoint_suspension.py) | `InMemoryCheckpointStore`, `RunCheckpoint`, `SuspendExecution`, `DurableHumanInputProvider` suspend/resume | [Orchestration](../docs/guides/orchestration.md) |
| [durability/durable_resume_service.py](durability/durable_resume_service.py) | `DurableRun`, `SuspendedRun`, `ResumeService`, `ResumeContext` — canonical durable HITL with out-of-process handoff and nested suspension | [Human-in-the-Loop](../docs/guides/human-in-the-loop.md) |
