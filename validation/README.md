# Validation Suite

Real-service validation scripts for the Nanitics SDK. This tree is
maintainer-facing tooling — it is not shipped to end users, not
discovered by the root `pytest`, and not part of the public docs.

## What this is

Validation scripts are **pass/fail acceptance tests** that verify the
SDK does what it should do against real services. Each script exercises
a primitive's distinguishing capability and asserts against it.

- Runs SDK components against **real** LLM providers, embedding
  providers, and optionally Postgres and Docker.
- Exports a machine-readable JSON trace per run to `validation/traces/`.
- Complements (does not replace) the mock-based test suite under `tests/`.

## What this is NOT

Validation scripts do not evaluate reasoning quality, measure
behavioral change over time, or produce scored outputs. That belongs
in the evals layer. A run that surfaces a behavioral observation which
is neither a bug nor a script error is an input to evals — not a
reason to relax an assertion, expand the script, or leave it red.

## Running

```
just validate                                               # Full suite
just validate-quick                                         # Scripts tagged @pytest.mark.quick
just validate fail-fast                                     # Stop on the first failing script
just validate from=validation/memory/episodic_memory.py     # Start from this script onward (sorted)
just validate fail-fast from=validation/memory/episodic_memory.py
just validate validation/smoke/smoke.py                     # Run a single script
just validate validation/ -- -k smoke                       # Forward flags to pytest after `--`
```

Options follow the recipe name. Both conventional (`--fail-fast`,
`--from=PATH`) and bare-kebab (`fail-fast`, `from=PATH`) forms are
accepted.

If `ANTHROPIC_API_KEY` is unset, the whole suite hard-skips with a
single summary line. Provider-specific scripts skip individually when
their credentials are missing.

## Authoring a new script

Copy `smoke/smoke.py` as a template. Every new script must:

1. Use `make_llm_client(...)` from `validation.helpers` — never
   hand-wire API keys.
2. Accept the `traced_emitter: InMemoryEmitter` fixture on the test
   function and wire it into the agent (`emitter=traced_emitter`). The
   fixture auto-saves a trace on teardown — even when the test fails
   — to `validation/traces/<test-name-without-test-prefix>.json`.
3. State acceptance criteria as executable assertions:
   - `assert_trace_contains` for trace-shape claims.
   - `assert_result_satisfies` for fuzzy output claims (LLM-as-judge).
4. Tag with `@pytest.mark.quick` if smoke-weight.

When reviewing a new script, check that it exercises a distinguishing
capability, that retries and error surfaces are asserted (not
swallowed), and that mock imports from `nanitics.*` are absent.

## Scripts

Scripts are grouped by theme. Within a theme, the order below reflects
the recommended reading order.

### Smoke

- `smoke/smoke.py` — framework smoke test (ReAct, one tool, quick).

### Tools

- `tools/tool_execution.py` — tool schema population under a real LLM.
- `tools/conditional_tool.py` — `ConditionalTool` schema visibility under a real LLM.
- `tools/mcp_tools.py` — MCP-discovered tools invoked through a real LLM.
- `tools/reference_tools.py` — four reference tools (file_read, http_request, web_search, code_execution) exercised in sections.

### Agents

- `agents/react_agent.py` — multi-tool ReAct validation (quick).
- `agents/reasoning_agent.py` — structured output via `output_schema` (quick).
- `agents/reflexion_agent.py` — evaluate-reflect-retry loop.
- `agents/rewoo_agent.py` — plan-first, then execute, then synthesize (quick).
- `agents/codeact_agent.py` — code generation with sandboxed execution, self-correction, tool bridge, and state persistence (requires Docker).
- `agents/lats_agent.py` — distinguishing LATS capabilities: re-selection, backpropagation, pruning, evaluator-guided selection.
- `agents/tree_of_thought.py` — strategy coverage across BFS / DFS / BEST_FIRST with strategy-specific expansion predicates.
- `agents/tree_of_thought_termination.py` — termination-condition coverage for tree-of-thought.

### Control

- `control/iteration_limits.py` — iteration limit fires under a designed-to-loop scenario with a real LLM (quick).
- `control/cancellation.py` — `CancellationToken` cooperative cancellation under a real LLM.
- `control/error_handling.py` — `ErrorHandler` correction prompt drives a real-LLM retry of a flaky tool.
- `control/error_classifier.py` — error classifier coverage under a real LLM.

### Context

- `context/context_management.py` — truncation/summarization under real token pressure.
- `context/message_grouper.py` — message-grouping behaviour under real LLM outputs.

### Evaluation

- `evaluation/evaluation.py` — evaluators driving revision loops under a real LLM.

### Planning

- `planning/planning_contributors.py` — planning prompt-section contributors under a real LLM.
- `planning/planning_evaluators.py` — planning-evaluator behaviour under a real LLM.

### Memory

- `memory/episodic_memory.py` — real Voyage embeddings for episodic recall.
- `memory/long_term_memory.py` — long-term memory CRUD and recall under real LLM.
- `memory/semantic_memory.py` — real Voyage embeddings for semantic search, namespace-filtered tools, and `SemanticMemoryProvider` automatic context injection.
- `memory/shared_memory.py` — shared-memory coordination under real LLMs.
- `memory/persistent_semantic_memory.py` — `PostgresSemanticStore` + pgvector end-to-end.

### Threads

- `threads/thread_identity_replay.py` — `ThreadStore` + `thread_key` behavioral continuity across two real-LLM runs; verifies replayed prior assistant turns are treated as the model's own work (quick).

### Multi-agent

- `multi_agent/agent_tool.py` — caller `ReActAgent` delegates to specialist via `AgentTool` (quick).
- `multi_agent/handoff.py` — `HandoffTransfer` carries context between two real agents (quick).
- `multi_agent/broadcast.py` — `Broadcast` fan-out under real LLMs.
- `multi_agent/message_bus.py` — `MessageBus` pub/sub routing under real LLMs.
- `multi_agent/peer_network.py` — `PeerNetwork` consultation under real LLMs.
- `multi_agent/supervisor.py` — `Supervisor` triggers retry on a real-LLM agent's first attempt.
- `multi_agent/orchestrator.py` — `create_orchestrator` decomposes a task across two real specialists.
- `multi_agent/bidding.py` — `Bidding` allocation under real-LLM bid generation.
- `multi_agent/judge_router.py` — `JudgeRouter` comparative-judgment routing under a real LLM, with calibration-anchor template injection (quick).
- `multi_agent/blackboard.py` — `Blackboard` control strategies under real LLMs.
- `multi_agent/debate.py` — `Debate` with two real debaters and a real-LLM judge produces a verdict.
- `multi_agent/consensus.py` — `Consensus` with three real voters and `MajorityVoting` reaches agreement.

### Workflows

- `workflows/sequential.py` — two-stage pipeline of real agents.
- `workflows/parallel.py` — parallel branches of real agents.
- `workflows/dag.py` — diamond DAG of real agents.
- `workflows/loop.py` — `Loop` termination under real LLM outputs.
- `workflows/map_reduce.py` — map-reduce of real agents over a three-item split.
- `workflows/conditional.py` — `Conditional` router under real LLMs.
- `workflows/pipeline.py` — `Pipeline` stage contracts under real LLMs.

### Human-in-the-Loop

- `hitl/approval_gate.py` — `ApprovalGate` with `CallbackHumanInputProvider` gates a real-LLM-produced output (quick).
- `hitl/revision_gate.py` — `RevisionGate` lifecycle under real LLMs.
- `hitl/approval_wrapped_tool.py` — `ApprovalWrappedTool` under real LLMs.
- `hitl/hitl_tools.py` — agent-initiated HITL tools under real LLMs.
- `hitl/async_hitl.py` — `AsyncHumanInputProvider` under real LLMs.

### Providers

- `providers/llm_routing.py` — `RoutingLLMClient` dispatch across real providers.
- `providers/anthropic_prompt_caching.py` — Anthropic prompt caching hit on a repeated system prompt (quick).
- `providers/litellm_adapter.py` — `LiteLLMClient` against a real provider.
- `providers/openai_client.py` — `OpenAILLMClient` end-to-end.

### Durability

- `durability/checkpoint_suspension.py` — checkpoint suspend/resume under real LLMs.
- `durability/durable_hitl.py` — `DurableHumanInputProvider` + `PostgresHitlRequestStore` persistence across simulated process restart.
- `durability/durable_resume_service.py` — `DurableRun` + `ResumeService` end-to-end against real Postgres: suspend, save response, resume, assert final output.
- `durability/postgres_checkpoint_store.py` — `PostgresCheckpointStore` save/load/delete round-trip against real Postgres; verifies most-recent-wins ordering and composite-index tie-break.

### Observability

- `observability/observatory.py` — observatory trace ingest end-to-end.
- `observability/postgres_trace_store.py` — Postgres trace store round-trip.

## What NOT to do

- Do not import `MockLLMClient`, `MockEmbeddingClient`, or `MockSandbox`. Validation uses real services by design. Mock-only workflows belong in `tests/` or `examples/`.
- Do not swallow exceptions. Real errors must surface so infrastructure problems are visible.
- Do not hand-wire API keys. `make_llm_client` resolves env vars and raises `ValueError` with install-extra guidance when a key is missing.
- Do not assert on exact output strings. Use `assert_result_satisfies(output, criteria)` for fuzzy criteria the judge can evaluate.

## Trace format

See `validation/helpers/trace.py` for the on-disk schema. The top of
each trace file has a `summary` block (event count, token totals, tool
calls, iterations) for at-a-glance review.

## See also

- `DEVELOPMENT.md` § "Validation suite" — user-facing entry point.
