# Changelog

All notable changes to the Nanitics SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Flattened the public surface into top-level subpackages.** The flat `nanitics.*` re-export surface (~286 names) is removed; every public symbol now lives at a predictable subpackage path. New top-level subpackages: `nanitics.memory`, `nanitics.evaluation`, `nanitics.planning`, `nanitics.context`, `nanitics.errors`, `nanitics.tracing`, `nanitics.hitl`. Existing subpackages (`strategies`, `specialized`, `composition`, `collaboration`, `safety`, `tools`, `patterns`, `infrastructure`) continue as before. Top-level `nanitics` now exposes only `__version__`. Adopters with `from nanitics import X` need to update imports to the relevant subpackage (`from nanitics.strategies import Agent`, `from nanitics.memory import SemanticStore`, `from nanitics.errors import NaniticsError`, etc.); per the no-backward-compatibility rule, no shim is provided. The leaked `nanitics.deprecated` re-export (`typing_extensions.deprecated`) is removed; future deprecation policy is documented separately. Closes F-01, F-02, F-03, F-04.
- **Renamed `nanitics.experimental` to `nanitics.specialized`.** The subpackage that hosts specialized agent strategies (`ReWOOAgent`, `ReflexionAgent`, `TreeOfThoughtAgent`, `LATSAgent`), long-tail workflows (`Loop`, `Conditional`, `MapReduce`, `Pipeline`), coordination patterns (`Bidding`, `Debate`, `Consensus`), reactive/peer topologies (`MessageBus`, `PeerNetwork`), hierarchical-decomposition planning, `ConditionalTool`, and `MistralLLMClient` moves from `nanitics.experimental` to `nanitics.specialized`. The previous name signalled adoption guidance but read as maturity; `specialized` matches the package's actual partitioning axis. The flat `nanitics.*` re-export surface is unchanged — only the subpackage path and the package docstring change. Adopters importing from `nanitics.experimental.*` directly need to rewrite to `nanitics.specialized.*`. Per the no-backward-compatibility rule, no shim is provided.
- **Renamed `nanitics.core` to `nanitics.strategies`.** The subpackage that hosts the long-tail agent strategies (`ReActAgent`, `ReWOOAgent`, `ReflexionAgent`, `LATSAgent`, `TreeOfThoughtAgent`, `CodeActAgent`, `ReasoningAgent`) plus the foundational `Agent` / `Tool` / `SystemPromptBuilder` primitives moves from `nanitics.core` to `nanitics.strategies`. The flat `nanitics.*` re-export surface is unchanged — only the subpackage path changes. Adopters importing from `nanitics.core.*` directly need to rewrite to `nanitics.strategies.*`. Per the no-backward-compatibility rule, no shim is provided.

### Documentation

- Reverted the Propodeum-client framing introduced in #47. The README tagline, `docs/index.md` opener, `docs/vision.md` opener, and `pyproject.toml` `description` field are restored to SDK-focused positioning. The README "Why Nanitics?" first bullet returns to "Composable primitives, not a framework." The README "About" section now carries a single neutral attribution line. Nanitics is the SDK; Propodeum is the maintainer, not the audience. No code, public-API surface, or behavior changes.

## [0.3.0] - 2026-05-12

### Changed

- **Public API surface split into three namespaces.** Top-level `nanitics` is now curated to primitives and load-bearing compositions for building most agentic systems (286 symbols); `nanitics.patterns` exposes named compositions over the core primitives — `create_orchestrator`, the structured handoff stack (9 symbols); `nanitics.experimental` exposes specialized primitives that are structurally distinct but niche — `ReWOOAgent`, `ReflexionAgent`, `TreeOfThoughtAgent`, `LATSAgent`, the `Loop`/`Conditional`/`MapReduce`/`Pipeline` workflows, `Bidding`/`Debate`/`Consensus`, `MessageBus`, `PeerNetwork`, hierarchical-decomposition planning, `MistralLLMClient`, `ConditionalTool` (68 symbols). Nothing was removed — every symbol from the previous surface is reachable in one of the three namespaces. **The `experimental` and `patterns` namespaces signal adoption guidance, not maturity** — every symbol there is part of the v1.0 surface and supported. Examples, tests, validation scripts, and guides updated to match.

#### Migration

Adopters with `from nanitics import X` for moved symbols need to update imports. The mapping by area:

| Area | Symbols | New import |
|---|---|---|
| Orchestrator factory | `create_orchestrator`, `orchestrator_prompt_section`, `FinalOutputStrategy` | `from nanitics.patterns import ...` |
| Structured handoff stack | `HandoffPayload`, `HandoffStep`, `HandoffTransfer`, `create_handoff_chain`, `handoff_sender_instructions`, `handoff_receiver_instructions` | `from nanitics.patterns import ...` |
| Specialized agent strategies | `ReWOOAgent`, `ReflexionAgent`, `TreeOfThoughtAgent`, `LATSAgent` (+ `ReWOOPlan`, `ReWOOStep`, `SearchStrategy`, `ActionNode`, `ThoughtNode`) | `from nanitics.experimental import ...` |
| Long-tail workflows | `Loop`, `Conditional`, `MapReduce`, `Pipeline`, `Stage`, `PipelineContractError` | `from nanitics.experimental import ...` |
| Coordination long-tail | `Bidding` (+ allocation strategies, `BidGenerator`, `Bid`, `BiddingResult`, `FixedBidGenerator`, `LLMBidGenerator`, `DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE`); `Debate`, `Debater`, `Argument`, `DebateResolution`, `DebateResult`, `ResolutionStrategy`, `JudgeResolution`, `LLMJudgeResolution`; `Consensus`, `ConsensusAggregation`, `ConsensusResponse`, `ConsensusResult`, `DeliberationConfig`, `AggregationStrategy`, `MajorityVoting`, `WeightedVoting`, `BestOfN` | `from nanitics.experimental import ...` |
| Reactive / peer topologies | `MessageBus`, `PeerNetwork` (+ `BusMessage`, `BusState`, `TopicSubscription`, `MessageFilter`, `MessageHistoryProvider`, `MessageBusContributor`, `MessageBusResult`, termination conditions, `PeerSpec`, `PeerBudgetExceededError`, `AgentExecution`, `FailedExecution`, `create_bus_tools`) | `from nanitics.experimental import ...` |
| Hierarchical-decomposition planning | `TaskPlan`, `TaskNode`, `DecompositionContributor`, `plan_to_workflow` | `from nanitics.experimental import ...` |
| Niche tools | `ConditionalTool` | `from nanitics.experimental import ConditionalTool` |
| LLM clients | `MistralLLMClient` (`LiteLLMClient` in core covers Mistral too) | `from nanitics.experimental import MistralLLMClient` |

`Blackboard`, `Supervisor`, `JudgeRouter`, `BiddableAgent` (shared with `JudgeRouter`), `Broadcast`, `AgentTool`, and the four context-transfer strategies remain at the top level — they are core primitives.

### Documentation

- README, `docs/index.md`, and `docs/vision.md` repositioned around the depth-tier framing. Nanitics is now described as the Python SDK behind [Propodeum](https://propodeum.com)'s production client engagements. The README "Why Nanitics?" section is rewritten around ownability, traceability, and real-services validation; a new "About" section cross-links to propodeum.com. The `pyproject.toml` `description` field is updated to "The Python SDK for production agents." No code, public-API surface, or behavior changes.
- `docs/guides/diagnosing-agent-issues.md` refactored from a six-layer ordered ladder into an eight-domain map plus an SDK-domain check. The previous "walk top-to-bottom, fix at the highest layer" framing was self-contradictory (the misdiagnosis examples recommended deeper-layer fixes before surface ones). The new framing names domains as categories and introduces a symptom-to-domain routing table as the front door for triage. Substantive additions: multi-agent coordination promoted to its own domain (per Cemri et al.'s MAST taxonomy, where 36.9% of multi-agent failures live in inter-agent dynamics); the deterministic-code layer broadened to cover both "wrong layer of decision-making" and "buggy plumbing around the agent"; cascading failures named as a cross-cutting concern with "trace backwards to the first wrong step" as the operational discipline; a diagnostic-process header (read trace, classify symptom, route to domain, fix at root); references to external taxonomies (MAST, Hamel Husain's evaluation methodology). No code or public-API surface changes.
- New `docs/missing-patterns.md` capturing what was considered and rejected for v1.0 inclusion during the API-surface audit, plus the surface-decision notes for `BiddableAgent` / template constants / planning-track / Mistral splits.
- Namespace-tiering callouts added to `docs/guides/agent-types.md`, `orchestration.md`, `multi-agent-foundations.md`, `multi-agent-coordination.md`, `planning.md`, the README, and `docs/index.md` so readers know where each named symbol lives.

## [0.2.1] - 2026-05-07

### Fixed

- `InMemoryEmitter.create_child()` now seeds the child emitter's stack with the parent's current `span_id`, so the first span opened by a bound child agent (`Agent.bind(parent)`, `AgentTool` delegation, `ReflexionAgent` inner attempts, workflow children) parents under the calling agent's current span instead of a synthetic root UUID nothing else in the trace ever names. Previously, `PersistentTraceStore` saw those spans as orphans and Observatory's tree builder hoisted every composite-agent subtree to the run root. Three `tests/test_event_emitter.py` cases that pinned the old (buggy) semantics are rewritten under the corrected contract; `examples/tools/event_emitter.py` Section 5 is updated accordingly.

### Changed

- `examples/homepage.py`: tool description, mock LLM responses, and search-corpus snippets reworded for legibility when reading the captured trace in Observatory; jittered LLM/tool delays added so the rendered trace shows realistic per-span durations rather than every span clocking 0ms. Bracket counts pinned by `tests/test_homepage_trace_shape.py` and the delegation task string are unchanged; the website-snippet portion (system prompts, evaluator definition, agent constructors) is untouched.

## [0.2.0] - 2026-05-06

### Added

- `TracedExecutor.execute` accepts an optional caller-supplied `run_id` keyword argument; when omitted, a UUID is generated as before. Lets HTTP-boundary callers allocate a `run_id` before scheduling — e.g. `POST` returns `202` with the id and an SSE stream resumes on that id. The persistence layer (`register_run`) remains the authority on uniqueness.
- `ToolResult.metadata` propagates onto the constructed `tool_result` `Message.metadata`, making application-side metadata available to downstream context-management policies (e.g. `TruncationPolicy.protected`) without a post-processing pass. Wired in `ReActAgent` (success path; persisted across resume) and `LATS` (`ActionNode.metadata`). The LLM-strip guarantee is preserved at every provider serializer. `CodeActAgent` observation messages are intentionally unaffected and pinned by regression test.

### Changed

- Default `ErrorClassifier` classifies any `ToolError` subclass as `CORRECTABLE` via base-class default; previously only `ToolParameterError`, `ToolExecutionError`, and `ToolNotFoundError` were correctable and everything else fell through to `FATAL`. App-defined `ToolError` subclasses now route through the correction loop without per-class registration.
- `ToolTimeoutError` is now classified as `RETRYABLE` (its docstring promised this; no clause matched it before).
- `ToolCallLimiter` accepts `max_tool_calls=0` and rejects only negative values. The post-dispatch check already enforced "no tool calls" on the first positive `step()`.
- `AnthropicLLMClient` raises `ValueError` (not `LLMProviderError`) when constructed without an API key — construction-time misconfiguration is no longer surfaced as a service error. Live-API error paths are unchanged.

### Fixed

- `validation/conftest.py` no longer provisions a `pgvector/pgvector:pg16` testcontainer eagerly when `POSTGRES_URL` is unset. Provisioning is deferred to `pytest_collection_modifyitems` and gated on a new `pytest.mark.postgres` marker (applied by `requires_postgres`). Runs that don't touch Postgres pay no Docker cost; previously, validation meta-tests in CI hit Docker Hub anonymous-pull rate limits unnecessarily.
- Dependabot litellm ignore now uses a PEP 440 range; the `uv` ecosystem rejects the `1.83.x` shorthand and the broken syntax was flagging every Dependabot PR.

### Security

- Observatory router uses `Path.is_relative_to()` for the assets-directory containment check (the CodeQL-recognized idiom for path-injection taint). Functionally equivalent to the previous string-prefix check.
- `/readyz` no longer includes the underlying exception text in its JSON response; the fixed `"trace store probe failed"` detail combined with the existing `"store": "error"` field is sufficient operator signal. The exception belongs in container logs, not in a public readiness endpoint. Resolves CodeQL `py/stack-trace-exposure`.

### Documentation

- `docs/guides/tools.md`: dispatch-over-structured-output pre-pattern; multi-tool packages; `ToolResult.metadata` round-trip; structured tool errors as a named pattern.
- `docs/guides/memory.md`: long-term store-event scope; namespace as tier separator.
- `docs/guides/observability.md`: Lifespan-Scoped Singleton-Emitter Listener.
- `docs/guides/security.md`: Trust-Boundary Value Objects.
- `docs/guides/agent-types.md`: Structured Output (`Literal` typology and refusal-as-output); `ReActAgent(tools=[])` → `ReasoningAgent` pointer.
- `docs/guides/error-handling.md`: `ToolError` hierarchy diagram now lists `ToolTimeoutError`.

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
