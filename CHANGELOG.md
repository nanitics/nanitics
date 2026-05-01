# Changelog

All notable changes to the Nanitics SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-01

### Added

#### Agent strategies
- Seven built-in agent strategies: ReAct, Reasoning, Reflexion, ReWOO, CodeAct, LATS, and Tree of Thought.
- `SystemPromptBuilder`, `SystemPromptContributor`, and `SystemPromptSection` for composing dynamic system prompts from independent contributors.

#### Multi-agent orchestration
- Workflow composition patterns: `Pipeline`, `Sequential`, `Parallel`, `Loop`, `Conditional`, `MapReduce`, and `DAG`.
- Handoff patterns (`HandoffStep`, `HandoffTransfer`, `HandoffPayload`) for explicit agent-to-agent delegation.
- `AgentTool` for composing any agent as a callable tool inside another agent.
- `MessageBus` / `TopicSubscription` publish-subscribe coordination and `PeerNetwork` / `PeerSpec` for decentralised peer coordination.

#### Multi-agent coordination strategies
- `Supervisor` with configurable `SupervisionTrigger` and `SupervisionAction` for hierarchical oversight.
- `Blackboard` shared-state coordination with predicate and composite termination conditions.
- `Debate` and `Debater` for adversarial multi-agent deliberation.
- `Consensus` for aggregated multi-agent agreement.
- `Bidding` with `FixedBidGenerator` and `LLMBidGenerator` for calibrated task allocation.
- `Broadcast` for fan-out coordination.
- `create_orchestrator` for dynamic role assignment.
- `JudgeRouter` with `LLMJudgeResolution` for comparative output selection.

#### Tool system
- `@tool` decorator and `FunctionTool` with automatic JSON schema generation.
- `ConditionalTool` for runtime conditional tool routing.
- `ToolRegistry` for multi-tool management.
- `MCPClient`, `MCPTool`, `MCPStdioParameters` for Model Context Protocol integration.
- Built-in tools: code execution, file reading, HTTP requests, and web search (available via optional extras).

#### Memory
- Four complementary memory layers: working memory (session-scoped), episodic memory (experience storage), long-term memory (persistent knowledge), and semantic memory (embedding-based retrieval).
- `SharedMemory` and `SharedEntry` for inter-agent state.
- Automatic tool generation for all memory types via `create_*_memory_tools` helpers.
- `InMemoryEpisodicStore`, `InMemorySemanticStore`, and `PostgresSemanticStore` storage backends.

#### Planning
- `UpfrontPlanContributor`, `AdaptivePlanningContributor`, `DecompositionContributor` for agent-driven planning.
- `Plan`, `PlanStep`, `TaskPlan`, `TaskNode`, `Goal` data models with status tracking.
- `GoalSatisfactionEvaluator` and `PlanAdherenceEvaluator` for automated plan quality checks.
- `PlanStore` / `InMemoryPlanStore` for plan persistence.
- `create_planning_tools` for exposing planning to agents.

#### Evaluation
- `LLMEvaluator`, `ProgrammaticEvaluator`, `CompositeEvaluator` for flexible output quality checks.
- `EvaluationCheck`, `EvaluationResult`, `EvaluationVerdict`, `OutputEvaluator`, `EvaluationContext`.
- `EvaluationEvent` and `EvaluationRevisionEvent` for observable evaluation in traces.

#### Human-in-the-loop (HITL)
- `ApprovalGate` and `ApprovalWrappedTool` for tool-level human approval.
- `AsyncHumanInputProvider`, `CallbackHumanInputProvider`, `DurableHumanInputProvider` (survives restarts).
- `create_ask_human_tool` and `create_request_approval_tool` for agent-facing HITL tools.
- `RevisionGate` for structured revision loops.
- `InMemoryHitlRequestStore` and `PostgresHitlRequestStore` for durable request persistence.
- `SuspendExecution`, `SuspendedRun`, `CheckpointStore` for long-running workflow suspension and resumption.

#### Observability and tracing
- `EventEmitter` / `InMemoryEmitter` with a rich event taxonomy: `LLMRequestEvent`, `LLMResponseEvent`, `ToolInvokeEvent`, `ToolResultEvent`, `AgentStepEvent`, `SupervisionEvent`, `EvaluationEvent`, and more.
- `TraceStore` interface with `InMemoryTraceStore`, `InMemoryPersistentTraceStore`, and `PostgresTraceStore`.
- `TraceQuery`, `TraceCollector`, `TraceSummary`, `TraceSummaryStats` for structured trace analysis.
- `TraceLevel` hierarchy for trace verbosity control.
- `TracedExecutor` for instrumented agent execution.
- `ResumeService` for resuming runs from checkpoints.
- Embedded Observatory UI: `create_observatory_router` mounts a full-stack React trace browser into any FastAPI application.

#### LLM clients and routing
- First-party clients: `AnthropicLLMClient`, `OpenAILLMClient`, `MistralLLMClient`, `LiteLLMClient`.
- `MockLLMClient` and `MockEmbeddingClient` for deterministic, API-key-free tests.
- `RoutingLLMClient` with `RuleBasedRouting` and `CostBudgetRouting` strategies.
- `InstrumentedLLMClient` for custom observability hooks.
- `EmbeddingClient` interface with `VoyageEmbeddingClient`.

#### Context management
- `ContextManager`, `TokenCounter`, `EstimateTokenCounter` for token budget enforcement.
- `TruncationPolicy`, `SummarizationPolicy`, `MessageGrouper` for message window management.
- `ContextProvider` / `ContextContent` for dynamic context injection.

#### Error handling and resilience
- `ErrorClassifier`, `ErrorCategory`, `classify_error` for structured error taxonomy.
- `RetryPolicy`, `ErrorHandler` for automatic recovery.
- `format_correction_prompt` for LLM-driven error correction.

#### Safety and execution constraints
- `IterationLimiter` and `ToolCallLimiter` for preventing runaway loops.
- `CancellationToken` for graceful shutdown.
- `DockerSandbox` and `MockSandbox` for sandboxed code execution.

#### Advanced composition primitives
- Allocation strategies: `AllEligible`, `EligibilityFilter`, `CapabilityFilter`.
- Aggregation strategies: `MajorityVoting`, `WeightedVoting`, `HighestConfidence`, `LowestCost`, `SelectBest`, `CollectAll`.
- Control strategies: `ScheduledControl`, `PrioritizedControl`, `OpportunisticControl`.
- Termination conditions: `MaxExecutionsTermination`, `MaxMessagesTermination`, `MaxRoundsTermination`.
- Trigger types: `BudgetTrigger`, `QualityTrigger`, `PredicateTrigger`.
- Durable run primitives: `DurableRun`, `RunCheckpoint`, `ResumeContext`, `ResumeResult`.

[Unreleased]: https://github.com/nanitics/nanitics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nanitics/nanitics/releases/tag/v0.1.0
