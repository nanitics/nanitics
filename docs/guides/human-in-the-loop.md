# Human-in-the-Loop

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Not every decision should be left to an agent. The human-in-the-loop (HITL) system lets humans participate in agent execution — approving actions, answering questions, reviewing plans, and revising output. This ranges from lightweight confirmation prompts to full durable execution where a workflow suspends, persists its state, and resumes after a human responds hours later.

## When to Use HITL

**Use HITL when** agents take high-stakes actions (deleting data, sending emails, spending money), when compliance requires human approval, when agents need clarification they can't resolve independently, or when output quality requires human review before delivery.

**Skip HITL when** the workflow is fully automated, human latency is unacceptable, or the agent's actions are low-risk and easily reversible.

## Choosing a Mechanism

The SDK provides four HITL mechanisms at different points on the autonomy spectrum:

| Level | Mechanism | Who Initiates | Example |
|-------|-----------|---------------|---------|
| Agent-initiated | HITL tools (`request_approval`, `ask_human`) | Agent decides when to involve human | Agent asks for clarification mid-task |
| Developer-mandated | `ApprovalWrappedTool` | Always triggers before tool execution | All database writes require approval |
| Workflow-level | `ApprovalGate`, `RevisionGate` | Triggers between workflow steps | Review draft before publishing |
| Durable | `DurableHumanInputProvider` | Suspends execution entirely | Human reviews offline, process resumes later |

**Use HITL tools** when the agent should judge whether human input is needed. The agent treats them like any other tool — calling them when the situation warrants it. Best for advisory interactions where the agent needs clarification or wants confirmation on uncertain decisions.

**Use `ApprovalWrappedTool`** when every invocation of a specific tool must be approved, regardless of context. The developer makes the policy decision, not the agent. Best for dangerous operations (deletes, payments, external mutations) where you want a hard gate on execution.

**Use `ApprovalGate`** when you need a checkpoint between workflow steps — review an output before it flows to the next step. Best for sequential workflows where intermediate results need human validation.

**Use `RevisionGate`** when you want iterative refinement — the human can request changes, and workers re-run with feedback until the output is approved. Best for content generation, plan review, or any workflow where the first draft rarely satisfies.

**Use `DurableHumanInputProvider`** when the human might take minutes or hours to respond. This suspends execution, persists state, and resumes in a new process after the response arrives. Required for production workflows where the process can't block indefinitely.

These mechanisms compose: a workflow can use `ApprovalWrappedTool` on individual tools *and* an `ApprovalGate` between steps, with the whole thing backed by `DurableHumanInputProvider` for process survival.

A quick heuristic: if the human involvement is about **what the agent does** (tool calls), use wrapping or HITL tools. If it's about **what the agent produces** (outputs between steps), use gates. If the response might not come within the process lifetime, add durability.

## Core Abstraction: HumanInputProvider

Every HITL interaction flows through the `HumanInputProvider` protocol. It accepts a `HumanInputRequest` describing what the agent needs (request type, prompt, context, options) and returns a `HumanInputResponse` carrying the human's decision and content. All mechanisms — tools, wrapped tools, gates — use this protocol, so swapping the underlying provider changes how the human interacts without modifying any agent or workflow code.

The SDK ships three implementations:

- **`CallbackHumanInputProvider`** — wraps a sync or async callback. Best for testing and CLI tools where the response is immediate.
- **`AsyncHumanInputProvider`** — suspends on an `asyncio.Future` and exposes a `resolve()` method for external callers (HTTP endpoints, WebSocket handlers). Best for real-time UIs and API integrations where the process stays alive. The `resolve()` method is async — store persistence is awaited rather than fire-and-forget, so failures propagate to the caller.
- **`DurableHumanInputProvider`** — persists the request via a `HitlRequestStore` and raises `SuspendExecution`. Best for long-running workflows. See [Durable Execution](#durable-execution).

| Aspect | `CallbackHumanInputProvider` | `AsyncHumanInputProvider` | `DurableHumanInputProvider` |
|--------|------------------------------|---------------------------|---------------------------|
| Blocking mechanism | Callback function | `asyncio.Future` | Raises `SuspendExecution` |
| Resolution | Callback returns response | External `resolve()` call | Stored response looked up on re-execution |
| Process survival | No | No (in-memory futures) | Yes (persisted state) |
| Best for | Testing, CLI | HTTP APIs, real-time UIs | Long-running workflows |

> **See also:** `HumanInputProvider`, `HumanInputRequest`, `HumanInputResponse` docstrings for field details. [`examples/hitl/async_hitl.py`](../../examples/hitl/async_hitl.py) — `AsyncHumanInputProvider` end-to-end: async `Future` resolution, HTTP-integration producer/consumer shape, `InMemoryHitlRequestStore` round-trip, and `asyncio.wait_for` timeout handling.

## HITL Tools

HITL tools let the agent decide when to involve a human. The agent calls them like any other tool — the SDK handles the request/response flow through the configured `HumanInputProvider`.

The tool descriptions explicitly tell the LLM that these are the only way to communicate with the user — text responses from the agent are not visible to the human. Two tools are available:

- **`request_approval`** — the agent asks for approval before proceeding with an action. Returns a formatted string describing the human's decision.
- **`ask_human`** — the agent asks a question when it needs clarification. Supports optional suggested choices.

Create them individually with `create_ask_human_tool` / `create_request_approval_tool`, or together with `create_hitl_tools`. Both tools emit `HumanInputRequestEvent` and `HumanInputResponseEvent` for observability.

The agent-initiated nature is both the strength and the risk: a well-prompted agent knows when to ask, but a poorly-prompted one may never call the tools or may over-rely on them. If you need a guarantee that approval happens, use `ApprovalWrappedTool` instead.

> **See also:** [examples/hitl/hitl_tools.py](../../examples/hitl/hitl_tools.py)

## ApprovalWrappedTool

`ApprovalWrappedTool` wraps an existing tool so that every invocation requires human approval before execution. Unlike HITL tools where the agent chooses when to ask, this is mandatory — the developer decides which tools are gated.

When the agent calls a wrapped tool, the human sees the tool name, description, and proposed parameters. They can approve (tool executes normally), override the parameters, or reject (tool returns a rejection message without executing). The wrapped tool's schema has `requires_approval=True` set, signaling to the agent that approval is built in.

Only wrap tools with significant consequences. Wrapping every tool creates excessive interruptions that degrade the agent's ability to work autonomously.

> **See also:** [examples/hitl/approval_wrapped_tool.py](../../examples/hitl/approval_wrapped_tool.py)

## ApprovalGate

`ApprovalGate` is a workflow step that pauses execution for human review. Insert it between steps in a `Sequential` workflow — for example, between a drafting step and a publishing step.

The gate presents the step input to the human with a configurable prompt (static string or callable that receives the input). The human can make one of four decisions:

- **Approve** — passes the input through unchanged to the next step
- **Override** — substitutes the human's content for the original input
- **Revise** — signals that workers should re-run with feedback (used by `RevisionGate`)
- **Reject** — halts the workflow with a rejection marker

Not all decisions apply in every context. In single-action approvals (gates, wrapped tools), all four are available. In composite approvals (multiple proposed actions), per-action decisions are limited to approve, reject, and override — revise applies at the proposal level ("request changes") rather than per action, to avoid intra-proposal dependency tracking.

The `prompt` and `context` parameters accept callables, enabling dynamic prompts based on the content being reviewed — useful for showing previews or summaries of the output under review.

Pass the optional `agent_name` kwarg when the gate reviews output produced by a named agent (e.g. `ApprovalGate(..., agent_name="drafter")`). The value flows onto both the emitted `HumanInputRequest` and `HumanInputRequestEvent`, so adopters can filter HITL events by producer agent. When the producer is a `FunctionStep` or a plain value rather than a named agent, leave it unset — the gate will not fall back to its own name. See the `ApprovalGate` docstring for the full field contract.

> **See also:** [examples/hitl/approval_gate.py](../../examples/hitl/approval_gate.py)

## RevisionGate

`RevisionGate` combines worker steps with an `ApprovalGate` in a revision loop. Workers produce output, the gate asks for review, and if the human requests revision, workers re-run with the feedback appended. This repeats until approval, rejection, or `max_revisions` is reached.

The revision loop works as follows:

1. Workers run on the input and produce output
2. The `ApprovalGate` presents output to the human
3. On **approve**, the output is returned. On **reject**, the workflow halts.
4. On **revise**, the previous output and feedback are appended to the original input, and workers re-run with instructions to change only what the feedback requests
5. If `max_revisions` is exceeded, the gate auto-rejects

Multiple workers run in parallel — their outputs are collected into a dict keyed by step name. A single worker's output is passed directly. An optional `on_output` callback can intercept worker output before the gate for transformation or side effects (receives `output`, `attempt`, and `feedback`; return a value to replace the output, or `None` to keep the original).

Always set `max_revisions` to a reasonable bound. Without a limit, a reviewer who keeps requesting changes creates an expensive loop of LLM calls that may never converge.

> **See also:** [examples/hitl/revision_gate.py](../../examples/hitl/revision_gate.py)

## Durable Execution

In-process HITL works when the human responds within seconds. But when a human might take minutes or hours — reviewing a document, approving a deployment — the process can't keep a connection open waiting. Durable execution solves this by persisting execution state, suspending the process, and resuming later.

The durable execution flow involves three components:

1. **`DurableHumanInputProvider`** — on every call, looks up any stored response in the `HitlRequestStore` by `request_id`; if present, returns it directly (the resume path). Otherwise persists the request and raises `SuspendExecution`. The provider holds no in-process state — the store is the single source of truth.

2. **`HitlRequestStore`** — protocol for persisting pending requests so an external system (API, UI) can display them and collect responses. `InMemoryHitlRequestStore` is provided for testing; implement the protocol with a database for production. Pending requests are those saved without a matching response.

3. **`CheckpointStore`** — persists workflow state (completed steps, current position) so execution can resume in a new process. Without checkpoints, the workflow would re-run all steps from the beginning. See [Orchestration — Resumable Workflows](orchestration.md#resumable-workflows).

The execution cycle is:

1. Run the workflow normally
2. When HITL is needed, `DurableHumanInputProvider` persists the request and raises `SuspendExecution`
3. The workflow orchestrator catches the signal and saves a checkpoint
4. An external system (API endpoint, UI, CLI) displays pending requests and collects the response
5. The response is saved to the `HitlRequestStore`
6. The workflow resumes from the checkpoint; on re-execution the tool call re-computes the same deterministic `request_id` and the provider finds the stored response

`SuspendExecution` inherits from `BaseException` (not `Exception`) so it propagates cleanly through `except Exception` blocks in tool execution, agent loops, and orchestrators without being accidentally caught.

> **See also:** [examples/durability/checkpoint_suspension.py](../../examples/durability/checkpoint_suspension.py), `DurableHumanInputProvider` and `HitlRequestStore` docstrings for implementation details.

### Durable resume service

The three-component / six-step cycle above is the machinery. Most applications don't hand-roll it — they use two cooperating SDK types that wrap the cycle end-to-end:

- **`DurableRun`** — wraps an `Agent` or `Workflow` plus the stores. Its `start(input)` executes the runnable and converts a `SuspendExecution` into a `SuspendedRun` — a plain, JSON-serializable payload (`run_id`, `suspension_info`, `pending_request`, `checkpoint_id`) you ship across a process boundary. When wrapping a `Workflow` (rather than an `Agent`), the workflow itself must be constructed with a `checkpoint_store` matching the one passed to `DurableRun`; the `Agent` branch wires this automatically, the `Workflow` branch does not. See [`validation/durability/durable_resume_service.py`](https://github.com/nanitics/nanitics/blob/main/validation/durability/durable_resume_service.py) for the production pattern.
- **`ResumeService`** — constructed once per process with the stores and a `factory(ctx: ResumeContext) -> DurableRun`. Every inbound `HumanInputResponse` goes through `service.resume(run_id, response)`: the service loads the checkpoint, validates the response, persists it, invokes the factory to reconstruct the run, and drives it forward.

What the caller writes:

<!-- verify: skip — illustrative sketch combining suspend-side and resume-side fragments (typically different processes); `workflow`, `hitl_store`, `checkpoint_store`, `build_workflow`, `task`, `run_id`, `response`, `ship_payload_to_external_system` are caller-supplied and the top-level `await`s run inside an async context -->
```python
from nanitics.composition import DurableRun, ResumeContext, ResumeService, SuspendedRun

# Suspend side
durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)
outcome = await durable.start(task)
if isinstance(outcome, SuspendedRun):
    ship_payload_to_external_system(outcome)

# Resume side (typically in a different process)
def factory(ctx: ResumeContext) -> DurableRun:
    workflow = build_workflow(ctx.run_id, ctx.hitl_store, ctx.checkpoint_store)
    return DurableRun(workflow, hitl_store=ctx.hitl_store, checkpoint_store=ctx.checkpoint_store)

service = ResumeService(hitl_store=hitl_store, checkpoint_store=checkpoint_store, factory=factory)
result = await service.resume(run_id, response)
```

Both `DurableRun.start` and `ResumeService.resume` return `ResumeResult | SuspendedRun` — completion or continuation. Neither raises `SuspendExecution`: every suspension is a value. A run that suspends multiple times (two approval gates, or a gate followed by a tool-level `ask_human`) goes through the same `service.resume(...)` call each time; the returned `SuspendedRun` carries a fresh `pending_request.request_id` for the next round.

`ResumeService.resume` validates that `response.request_id` matches the checkpoint's pending `suspension_info.request_id` — a mismatch raises `ValueError` rather than silently saving the response to the wrong slot. Concurrency on the same `run_id` is the caller's concern (use an idempotency key at the transport layer); the SDK does not lock.

> **See also:** [examples/durability/durable_resume_service.py](../../examples/durability/durable_resume_service.py), `DurableRun` and `ResumeService` docstrings.

### Nested workflows

A workflow nested inside another workflow (via `WorkflowStep` — e.g. a `Sequential` inside a `Conditional` branch, or a `Parallel` branch) that suspends on a human gate resumes at its own suspension point, not from the top of the nested workflow. The parent orchestrator threads the resume checkpoint into the suspended child, so a nested `Conditional`'s router is not re-invoked on resume and no step before the suspension point in the nested workflow re-runs. This works to arbitrary nesting depth and across every orchestrator (`Sequential`, `Conditional`, `Parallel`, `DAG`, `Loop`, `Pipeline`, `MapReduce`) — only the top-level workflow needs a `checkpoint_store`; nested workflows surface their state up into the single persisted checkpoint automatically.

> **Migration (0.6.0):** the checkpoint schema bumped from `2` to `3` to carry nested-workflow state. There is no in-place migration — a checkpoint persisted by 0.5.x raises `CheckpointVersionError` when resumed under 0.6.0. Drain any in-flight suspended runs before upgrading.

## Testing HITL Flows

`CallbackHumanInputProvider` is the primary tool for testing. Pass a lambda or function that returns the desired `HumanInputResponse` for each request — auto-approve, auto-reject, or a conditional function that inspects the request and responds accordingly.

For production durable HITL on Postgres, pair `PostgresHitlRequestStore` (`nanitics.hitl`) with `PostgresCheckpointStore` (`nanitics.composition`). Apply both schemas once at deploy via `get_hitl_schema_sql()` and `get_checkpoint_schema_sql()` and pass the resulting stores to `DurableRun` / `ResumeService`. The schemas are independent — no foreign keys between the two tables — so an adopter can deploy either store standalone.

For testing durable flows, `InMemoryHitlRequestStore` and `InMemoryCheckpointStore` provide in-memory implementations of the persistence protocols. Simulate the full suspend/resume cycle by catching `SuspendExecution`, saving a response to the store, and resuming from the checkpoint — the re-executed tool call finds the response via its deterministic `request_id`.

## Events

All HITL interactions emit events through the `EventEmitter`. These events are essential for building UIs that show pending requests, tracking approval latency, and auditing human decisions.

| Event | When |
|-------|------|
| `HumanInputRequestEvent` | A request is sent to the human |
| `HumanInputResponseEvent` | A response is received (includes wait duration) |
| `RevisionStartEvent` | A `RevisionGate` begins its loop |
| `RevisionAttemptEvent` | A revision iteration starts |
| `RevisionCompleteEvent` | A `RevisionGate` finishes |
| `ExecutionSuspendedEvent` | A workflow suspends for durable HITL |
| `ExecutionResumedEvent` | A workflow resumes from a checkpoint |

The `HumanInputResponseEvent` includes the wait duration between request and response — useful for measuring human response times and identifying bottlenecks in approval workflows.

All four approval surfaces — `ApprovalGate`, `RevisionGate` (via the composed `ApprovalGate`), `ApprovalWrappedTool`, and `create_request_approval_tool` — emit `HumanInputRequestEvent.request_type == "approval"`. There is no separate `"plan_review"` value: an adopter filtering by `request_type == "approval"` sees every gate-, wrapper-, and tool-level approval. `ask_human` emits `request_type == "question"`; those are the only two live values on `HumanInputType`.

## Pitfalls

**Constructing agents without a `run_id`.** HITL tools derive `request_id` from `{run_id}:{tool_call_id}` — both are required for stable identity across suspend/resume. An agent run without a `run_id` raises `ValueError` at the first HITL tool dispatch. For gates, `ApprovalGate` requires `run_id` at construction for the same reason.

**Using `CallbackHumanInputProvider` for slow interactions.** Callbacks block the event loop while waiting. For human interactions that take more than a few seconds, use `AsyncHumanInputProvider` (in-process) or `DurableHumanInputProvider` (cross-process).

**Not setting `run_id` on the store.** Without `run_id`, requests are saved with `run_id=None` and `get_pending_requests()` can't find them. Pass `run_id` when constructing the store — it applies as a fallback to requests that don't have one set.

**Wrapping too many tools with `ApprovalWrappedTool`.** Every wrapped tool interrupts the agent's flow. Only wrap tools with significant consequences — not every API call needs approval.

**Unbounded revision loops.** Always set a reasonable `max_revisions` on `RevisionGate`. Without a limit, a reviewer who keeps requesting changes creates an expensive loop of LLM calls.

**Mixing durability with non-durable providers.** If a workflow uses `DurableHumanInputProvider` for gates but `CallbackHumanInputProvider` for agent-level HITL tools, the agent-level interactions won't survive process restarts. Use the same durable provider throughout, or accept that only gate-level interactions are durable.

**`SuspendExecution` caught by broad exception handlers.** Since `SuspendExecution` inherits from `BaseException`, it propagates through `except Exception`. But custom middleware or wrappers that catch `BaseException` will swallow it — ensure nothing between the provider and the orchestrator catches `BaseException` without re-raising.
