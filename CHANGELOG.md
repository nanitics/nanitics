# Changelog

All notable changes to the Nanitics SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-06-01

### Added

- **Generalized tool copy: `FunctionTool.replace` / `AgentTool.replace`.** A
  single method derives a copy of a tool with selected schema metadata
  overridden, leaving everything else unchanged. Arguments left unset keep
  their current values, so `tool.replace(return_direct=True)` flips one flag
  and `tool.replace(name="…", description="…")` relabels. `FunctionTool.replace`
  accepts `name`, `description`, `return_direct`, `requires_approval`, and
  `timeout_seconds`; `AgentTool.replace` accepts `name`, `description`, and
  `return_direct`. Only schema metadata is overridable — the wrapped function,
  its parameter schema, and `ToolContext` injection are preserved, so a copy
  cannot desync its validation from its implementation. To change those, build
  a new tool.

### Deprecated

- **`FunctionTool.with_return_direct` / `AgentTool.with_return_direct`.**
  Superseded by the general `replace`; `with_return_direct(value)` is now a thin
  alias for `replace(return_direct=value)` and emits a `DeprecationWarning`.
  Migrate to `tool.replace(return_direct=…)`. Removed in 1.0.

## [0.8.0] - 2026-06-01

### Added

- **Derive a `return_direct` variant of a tool.** `FunctionTool` and
  `AgentTool` gain `with_return_direct(value=True)`, returning a copy of the
  tool with `return_direct` set and everything else preserved: the wrapped
  function, the parameter schema, `ToolContext` injection, and the other
  SDK-side `ToolSchema` flags (`requires_approval`, `timeout_seconds`). Define
  a write tool once with `@tool`, keep its closing LLM turn in an interactive
  caller, and derive a tool-terminating variant for a headless caller without
  duplicating the definition or reaching into private attributes. Additive and
  non-breaking.

## [0.7.0] - 2026-06-01

### Added

- **Tool-terminated runs (`return_direct`).** A tool can now end a
  `ReActAgent` run on its own result instead of forcing one more LLM turn to
  produce a closing message. Mark a tool `return_direct=True` — on the
  `tool()` decorator, `FunctionTool`, or `AgentTool` — and when the model
  calls it, the loop runs the whole tool batch (co-called tools' side effects
  still fire), then ends on the first `return_direct` call in batch order,
  using that call's `ToolResult.content` as `output` with
  `termination_reason="return_direct"`. The closing LLM generation is skipped.
  This is the pattern other frameworks call `return_direct`; it removes the
  wasted generation a headless caller pays when an agent's terminal action is
  a tool call (e.g. a delegate whose proposal is read straight from the tool
  result). The flag is SDK-side only and is never serialized to any LLM
  provider, matching `requires_approval` and `timeout_seconds`. Resume-safe: a
  `return_direct` call reached across a human-in-the-loop suspension (in either
  order relative to the suspending tool) terminates correctly on resume.
  - When `output_schema` is also set, the structured-synthesis call is
    likewise skipped: `output` is the tool's content and `parsed` is `None`.
    A `return_direct` tool that needs to hand back structured data puts it in
    `ToolResult.metadata`, which round-trips onto the `tool_result`
    `Message.metadata` (read from the last `tool_result` message in
    `messages`) and is never sent to the LLM.
  - `output_evaluator` is bypassed on a `return_direct` termination — there is
    no free-text output to gate.

## [0.6.0] - 2026-05-29

### Added

- **Nested workflows resume at their own suspension point.** A `Workflow`
  nested inside another workflow via `WorkflowStep` (e.g. a `Sequential`
  inside a `Conditional` branch) previously re-executed from the top on
  resume: the parent could not thread the resume checkpoint into the
  suspended child, so a nested `Conditional`'s router fired again on every
  resume (an extra LLM call, and unstable under a non-deterministic router),
  and any step before the suspension point in the nested workflow re-ran.
  Each orchestrator now surfaces its suspension state up through
  `SuspendExecution.orchestration_state`, the parent embeds it under a new
  recursive `nested_checkpoint` key in the single persisted checkpoint, and
  on resume the parent reconstructs the child checkpoint and re-enters the
  nested workflow exactly where it suspended. Applies uniformly across
  `Sequential`, `Conditional`, `Parallel`, `DAG`, `Loop`, `Pipeline`, and
  `MapReduce`. The leaf agent-resume path (`agent_checkpoint`) is unchanged;
  `agent_checkpoint` and `nested_checkpoint` are mutually exclusive per
  suspended step.

### Changed

- **`CHECKPOINT_SCHEMA_VERSION` bumped `2` → `3`.** The recursive
  `nested_checkpoint` frame changes the checkpoint `state` shape. There is no
  in-place migration: a v2 checkpoint loaded by v3 code raises
  `CheckpointVersionError` (existing behavior). **Operational note:** drain
  any in-flight suspended runs before upgrading — a run suspended under 0.5.x
  cannot be resumed under 0.6.0. Non-nested checkpoints are otherwise
  unaffected; only the schema constant and the optional new key differ.
- **`SuspendExecution` gains an optional `orchestration_state` field**
  (default `None`). Additive; existing constructors are unaffected. It
  carries the suspending workflow frame's checkpoint state up to its parent
  during nested suspension.

## [0.5.2] - 2026-05-29

### Added

- **`step_metadata` on `WorkflowStepCompleteEvent` — full `StepResult.metadata`
  surfaced to observers.** Workflow orchestrators (`Sequential`, `Parallel`,
  `DAG`, `Pipeline`, `MapReduce`, `Loop`, `Conditional`) previously emitted
  only `step_output` on step completion, dropping `StepResult.metadata`.
  Consumers that needed structured side-info — `final_output`,
  `termination_reason`, `total_steps`, etc. — had to thread a side-channel
  callback through the step. The event now carries a new
  `step_metadata: dict[str, Any]` field populated from `result.metadata` at
  every emission site, mirroring the in-memory contract. A field validator
  coerces values to a JSON-safe shape via
  `pydantic_core.to_jsonable_python` with `repr()` as the fallback, so
  sinks can rely on the event round-tripping through `model_dump_json()`.
  Strictly additive: the field defaults to `{}` and no existing consumer
  changes behaviour.

## [0.5.1] - 2026-05-28

### Added

- **`ToolResultPolicy` — bounded tool output, symmetric to `ContextManagement`.**
  New protocol `ToolResultPolicy` (under `nanitics.context`) with three
  default implementations: `ErrorOnLargeToolResult` (recommended default —
  raises `ToolResultTooLargeError`, surfaced through the agent's
  error-handling capability as a correction prompt for the LLM),
  `TruncateToolResult` (opt-in head/tail slice with marker), and
  `SummarizeToolResult` (opt-in LLM summarization with truncate-on-failure
  fallback). The policy hooks at a single seam in `ToolRegistry.dispatch`,
  applied after a tool's `execute()` returns and before the result enters
  the message list. Composes orthogonally with `ContextManager`: this
  layer bounds individual tool results; `ContextManager` bounds the total
  message list. `Agent`, `ReActAgent`, `LATSAgent`, `ReWOOAgent`, and
  `CodeActAgent` each gain a `tool_result_policy: ToolResultPolicy | None = None`
  kwarg (default `None` — strictly additive, no behaviour change unless
  wired). New `ToolResultTooLargeError(ToolError)` under `nanitics.errors`
  and `ToolResultPolicyAppliedEvent` under `nanitics.tracing` for
  observability (action discriminator: `"truncated"` / `"summarized"` /
  `"errored"`). When reaching for a policy, start with
  `ErrorOnLargeToolResult` — surfacing the failure to the LLM is more
  actionable than silent data loss. See [docs/guides/context-management.md](docs/guides/context-management.md#tool-result-policies)
  and [examples/context/tool_result_policy.py](examples/context/tool_result_policy.py).

## [0.5.0] - 2026-05-28

### Added

- **Per-step and per-workflow-run token usage propagation.** `StepResult`
  gained a first-class, typed `usage: Usage | None` field populated by
  `AgentStep`, `_BoundAgentStep`, `_BoundHandoffStep`, and `HandoffStep`.
  `Sequential` and `Pipeline` runners now attach an aggregated `usage` to
  their returned `StepResult` (the sum across every sub-step's `usage`,
  with `None` only when every sub-step contributed `None`); recursive
  aggregation through `WorkflowStep` follows by construction. The
  cancellation-mid-flight branch returns the partial aggregate of what
  was actually spent; resume from a checkpoint reconstructs sub-step
  usages from checkpoint state (pre-bump checkpoints with no `"usage"`
  key resume cleanly, contributing `None`). `SupervisionResult` gained a
  non-optional `usage: Usage` field — the sum across **every** attempt
  the `Supervisor` ran during a single `supervise()` call (distinct from
  `result.usage`, the final attempt only). Checkpoint
  `CHECKPOINT_SCHEMA_VERSION` bumped from 1 to 2 to carry the new
  `"usage"` entry in `state["completed_results"][name]` for both
  Sequential and Pipeline. Loop, MapReduce, DAG, and Conditional remain
  out of scope — they continue to return `StepResult.usage = None` and
  will be addressed in a Phase 2 follow-up.

### Deprecations

- **Deprecated:** `StepResult.metadata["usage"]` dict mirror written by
  `AgentStep`, `_BoundAgentStep`, `_BoundHandoffStep`, and `HandoffStep`.
  Use `StepResult.usage` (the typed field) instead. Removed in 1.0.0. See
  [docs/migrations/step-result-usage.md](docs/migrations/step-result-usage.md).

- **Discoverability pass for the threads/memory substrate.** Closes
  the threads/memory maturation arc shipped over PRs #83–#87 with the
  examples and migration content a new consumer searches for first.
  The substrate distinction stays the same: `WorkingMemory`,
  `LongTermStore`, `SemanticStore`, `EpisodeStore`, and `SharedMemory`
  are *information continuity* — content the agent reads through a
  provider-injected, namespaced `<nanitics:context>` envelope and
  reasons over as external data; `thread_key` + `ThreadStore` is
  *behavioral continuity* — the agent's own prior assistant turns,
  tool calls, and tool results replayed unwrapped into the next run's
  message list. Five new runnable, mocked examples make the substrate
  searchable from `examples/`:
  `examples/multi_agent/threads_drafter_critic_pipeline.py` (the same
  drafter at positions 0 and 2 in `create_handoff_chain` via
  `thread_keys`), `examples/multi_agent/threads_persistent_peers.py`
  (`PeerSpec.thread_key` across sequential `network.run` calls),
  `examples/multi_agent/threads_repeated_agent_tool.py` (one
  `AgentTool(thread_key=...)` dispatched twice in one outer run),
  `examples/multi_agent/threads_orchestrator_stateful_specialist.py`
  (a stateful specialist threaded through `create_orchestrator`), and
  `examples/memory/working_memory_vs_threads.py` (the haiku-revision
  scenario run twice — once via journaling into `WorkingMemory`, once
  via `thread_key` — with assertions on the `<nanitics:context
  provider="working_memory">` wrapper in the former and its absence in
  the latter). A new migration guide at
  [`docs/guides/migrating-from-working-memory-workaround.md`](docs/guides/migrating-from-working-memory-workaround.md)
  walks consumers off the `WorkingMemory`-as-fake-transcript pattern
  symptom-first; the substrate-comparison recipe inline in
  `docs/guides/memory.md` § "Behavioral Continuity" shows the two
  surfaces composing on a single agent. Per-construct behavioral-
  continuity recipes already live inline in
  `docs/guides/multi-agent-foundations.md` § "Behavioral Continuity
  in Multi-Agent Patterns" and are not duplicated. No new SDK code in
  this pass.

- **Multi-agent thread propagation.** Every multi-agent construct now
  routes `thread_key` to the agents it owns, so behavioral continuity
  composes through delegation, pipelines, peer networks, blackboards,
  fan-out, and post-execution monitoring. Per-construct API:
  `AgentTool(..., thread_key="…")` and `HandoffStep(..., thread_key="…")`
  forward the key on every execute; `create_handoff_chain` accepts a
  `thread_keys: list[str | None]` parallel to `agents` so the same
  agent appearing twice can share a thread (e.g.,
  drafter→critic→drafter). `AgentStep` (workflow adapter) gains the
  same shape, and the `_BoundHandoffStep` / `_BoundAgentStep` workflow
  wrappers now forward the key. `PeerSpec` gains `thread_key`;
  `PeerNetwork` accepts `thread_store` and wires it into every peer's
  `ReActAgent`; `PeerNetwork.run(..., thread_key=…)` is an entry-agent
  override. Per-peer-identity is the default scoping (per-pair /
  per-network deferred until consumers report a need). `Blackboard`,
  `Broadcast`, `Consensus`, `Bidding`, `JudgeRouter`, and `Supervisor`
  accept `thread_keys: dict[str, str]` (agent-name → key) with
  construction-time validation against unknown agent names.
  `Supervisor`'s RETRY appends to the supervisee's thread (so the
  feedback-augmented retry sees the prior attempt as natural
  conversation history); REASSIGN switches to the new agent's own
  thread or runs stateless. `TopicSubscription.thread_key` carries the
  subscriber's behavioral continuity through a `MessageBus`,
  orthogonal to `MessageHistoryProvider` (which conveys bus-topic
  history). `Debater.thread_key` carries per-debater continuity across
  rounds. Every key is opt-in; default behavior is unchanged when
  unset. Recipes in `docs/guides/multi-agent-foundations.md`.

- **Thread identity primitive — `thread_key` + `ThreadStore`.** A new
  substrate for behavioral continuity: `Agent.run(input, thread_key=...)`
  loads a per-thread `Message` prefix from a configured
  `ThreadStore` before `_execute` and appends the run's new messages on
  successful completion, so a subsequent run on the same key sees prior
  assistant turns, tool calls, and tool results as its own conversation
  history. New public surface under `nanitics.composition`:
  `ThreadStore` protocol (`load` / `append` / `clear`),
  `InMemoryThreadStore` reference implementation, `ThreadLocks`
  in-process per-key serialization. Concurrent same-key runs raise
  `ThreadInUseError` (also re-exported from `nanitics.errors`);
  different-key runs proceed in parallel. `Agent.__init__` gains
  `thread_store` and `thread_locks` keyword parameters; `Agent.run`
  gains a `thread_key` keyword. `AgentResult.thread_key` echoes the
  active key for downstream correlation; `AgentStartEvent` gains
  `thread_key` and `replayed_message_count` fields (additive, defaults
  preserved so existing consumers ingest unchanged). Replayed messages
  bypass the `<nanitics:context>` wrapper by design — the model treats
  them as its own prior turns, not as injected context. On suspend
  inside a thread-keyed run, the prefix is snapshotted into
  `RunCheckpoint.state` and the resume uses that frozen view rather
  than re-consulting the live store. `ReActAgent._execute` opts in;
  other agent subtypes accept `thread_key` for signature uniformity
  but do not replay the prefix in this phase. No default trimming or
  compaction ships with this primitive — consumers configure
  `ContextManagement` at the LLM-call boundary or wrap the store.

- **`MCPAuthError` distinguishes 401/403 from generic MCP transport
  failures.** A new exception class under
  `nanitics.infrastructure.errors` (also re-exported from
  `nanitics.errors`), subclassing `LLMProviderError`, raised by
  `MCPClient.__aenter__` and `MCPTool.execute` when the underlying SSE or
  Streamable HTTP transport surfaces a 401 or 403. Carries
  `status_code: int` and the raw `www_authenticate: str | None` header
  value; the SDK does not parse the header. Detection walks `__cause__`
  and `__context__` chains for an `httpx.HTTPStatusError` (bounded at
  10 hops). Stdio is unaffected. Existing `except LLMProviderError`
  catches still fire; the change is purely additive.

- **`PostgresCheckpointStore` ships the second half of durable HITL on
  Postgres.** A persistent `CheckpointStore` implementation backed by
  `asyncpg`, mirroring `PostgresHitlRequestStore`. Stores each
  `RunCheckpoint` as a single JSONB blob keyed by `checkpoint_id`,
  with indexed `run_id` and `created_at` columns; `load(run_id)`
  returns the most recent checkpoint with a deterministic
  `(created_at DESC, checkpoint_id DESC)` tie-break under
  same-microsecond writes. Schema applied once via
  `get_checkpoint_schema_sql()` — no migration framework, no FK to
  `hitl_requests` (the link is logical via `suspension_info.request_id`
  inside the blob, so adopters can deploy either store standalone).
  `delete()` and `delete_for_run()` are silent on missing rows,
  matching `InMemoryCheckpointStore` semantics. Lazy-imported from
  `nanitics.composition` and `nanitics.composition.durability` under
  the existing `postgres` extra; adopters who do not install the extra
  see `None` for both symbols.

- **`RunRecord.parent_run_id` models hierarchical specialist runs.** A
  new nullable `parent_run_id: str | None` field on `RunRecord` links a
  child run to the run that dispatched it. The Postgres v3 migration
  adds the column with a self-referencing FK and `ON DELETE CASCADE`,
  plus a partial index on non-null values. `register_run` gains a
  keyword-only `parent_run_id=None`; `list_runs` and `count_runs` gain
  a three-state filter (`_UNSET` default → no filter; `None` → top-level
  only; `str` → children of that parent). The SDK does not enforce
  `trace_id` parity between parent and child — the caller decides
  whether they share a trace. `delete_run` cascades to children at the
  database layer.

- **`MCPClient.stdio(errlog=...)` redirects the child process's stderr.**
  New keyword-only parameter forwards to upstream `_stdio_client(errlog=…)`
  when supplied; the upstream `sys.stderr` default is preserved when
  omitted. Stdio-only — the SSE and Streamable HTTP transports have no
  child process. The caller owns the stream's lifetime.

- **`PostgresTraceStore.ensure_schema()` raises on sibling-store
  schema conflicts.** When the version row reports a baseline that
  must have created the configured `runs_table`, but
  `information_schema` reports the table is missing,
  `ensure_schema()` now raises `RuntimeError` with a message naming
  both `table_name` and `runs_table`. The usual cause is another
  store instance sharing `table_name` and using a different
  `runs_table` configuration; the SDK refuses to silently re-run
  migrations under a different table name.

- **Observatory is drop-in: the prebuilt SPA ships inside the Python
  wheel.** `mount_observatory(app, store, prefix="/observatory")` now
  attaches the JSON API and the embedded UI in one line, with no
  `static_dir` argument and no frontend toolchain at the call site. The
  Vite-built SPA lives under `nanitics/observatory/ui_assets/`
  (`.gitignore`d; populated by `just observatory-build`) and is
  force-included into the wheel via `[tool.hatch.build.targets.wheel]
  artifacts`. The release workflow builds the bundle before `uv build`
  so every published wheel carries it.

- **Observatory router split into API and UI surfaces.**
  `create_observatory_api_router(store)` returns the JSON data endpoints
  (runs, span tree, agents, workflows, events, SSE stream).
  `create_observatory_ui_router(*, static_dir=None)` returns the SPA
  catch-all (`/` + `/assets/{path}`). `mount_observatory` is the
  convenience helper that mounts both under one prefix. The split is
  the seam consumers attach different middleware to (session auth on
  the UI, bearer-token on the API) without forking the router.

- **SPA picks up its mount prefix at request time.** The UI router
  rewrites a `<script id="nanitics-observatory-base">` tag in
  `index.html` to set `window.__NANITICS_OBSERVATORY_BASE__` to the
  live mount prefix (JSON-encoded, so a hostile prefix cannot break out
  of the literal). The same prebuilt bundle works at `/observatory`,
  `/api/observatory`, `/admin/runs`, or any other path — no rebuild
  needed. `ObservatoryClient` defaults to that global when constructed
  with no argument, so the embedded SPA needs no explicit base URL.

- **Version pin between UI and API by construction.** Because the SPA
  ships *inside the same wheel that ships the router*, the UI in
  `0.X.0` always speaks the `0.X.0` API. The `@nanitics/observatory`
  npm package is now reframed as the deliberate escape hatch for
  embedders (custom React app shells) and customizers (custom
  `agentViewRegistry` / `panelRegistry` entries) — including the case
  where a consumer explicitly wants to run a newer UI against an older
  API while rolling out a migration.

- **`@nanitics/observatory` publishes via npm OIDC Trusted Publishing.** The publish workflow now authenticates to npm using OpenID Connect tokens issued by GitHub Actions instead of a long-lived `NPM_TOKEN` secret. No tokens are stored in the repo; the npm package settings name `nanitics/nanitics` + `publish-observatory.yml` as the only trusted publisher for `@nanitics/observatory`, and the package access policy is set to "require 2FA and disallow tokens" so even a compromised bypass-2FA token cannot publish. Bumps the workflow's Node version to 22 and upgrades npm to a release that supports trusted publishing (≥11.5.1).

- **`@nanitics/observatory` published to npm.** The React components for embedding the Observatory (run list, run detail, event timeline, registries, hooks) now ship as a public scoped package on the npm registry. Downstream apps can `npm install @nanitics/observatory` instead of consuming the package as a filesystem dependency. The package ships a precompiled ESM bundle, TypeScript declarations, and a precompiled stylesheet at `@nanitics/observatory/styles.css`. Peer deps: `react@^19.2.6`, `react-dom@^19.2.6`. Publishing is automated by `.github/workflows/publish-observatory.yml` (triggered by `observatory-v*` tags) with npm provenance signing.

### Removed

- **`nanitics.observatory.create_observatory_router`.** Removed in
  favor of the API/UI split above. Adopters migrate as follows:
  `app.include_router(create_observatory_router(store, static_dir=DIR), prefix="/observatory")`
  → `mount_observatory(app, store, prefix="/observatory")` (no
  `static_dir` needed; the wheel ships the bundle). Per the
  no-backwards-compatibility rule, no shim is provided.

- **Committed `observatory/dist-embed/` bundle.** The pre-built SPA is
  no longer checked into the repo; it's a build artifact at
  `nanitics/observatory/ui_assets/` (`.gitignore`d) produced by
  `just observatory-build`. The `dist-embed/` path is gone everywhere
  — `Dockerfile`s no longer copy it, `static_dir=UI_DIR` wiring is no
  longer needed in adopter apps, and the docker compose images now
  pick the bundle up automatically via the SDK install.

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
