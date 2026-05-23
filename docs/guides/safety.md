# Safety

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Agents run in loops, call external services, and — with `CodeActAgent` — execute arbitrary code. Without constraints, a malfunctioning agent can loop forever, consume unbounded tokens, or run dangerous code. The safety system provides three layers of protection: iteration limits, cooperative cancellation, and sandboxed code execution.

## Safety Mechanism Comparison

| Mechanism | What it prevents | Scope | Failure mode |
|-----------|-----------------|-------|-------------|
| Iteration limits | Infinite loops, unbounded token spend | Per-agent step counting | Raises `AgentIterationLimitError` |
| Tool call limits | Unbounded tool usage within allowed steps | Per-agent cumulative tool call counting | Raises `AgentToolCallLimitError` |
| Cancellation | Runaway agents in production (API timeouts, user abort) | External signal — checked between steps and raced against the in-flight tool/sandbox await | Agent returns `AgentResult(termination_reason="cancelled")`; an in-flight tool call is interrupted, not waited out |
| Sandboxing | Dangerous code execution on host | Code execution environments (`CodeActAgent`) | Code runs isolated; failures contained |

## When to Use Each

**Always set iteration limits** in production. An agent without a limit can loop indefinitely, burning tokens and time. Set limits higher than you expect the agent to need, but low enough to catch infinite loops. An agent that legitimately needs 200 steps probably needs a different architecture.

**Use tool call limits** when an agent can request multiple tool calls per step. Iteration limits alone don't prevent an agent from making dozens of tool calls within a few steps. Tool call limits provide a tighter bound on external resource consumption.

**Always provide a cancellation token** when the caller needs the ability to stop execution externally — API servers, UIs, or orchestrators that enforce timeouts.

**Use sandboxes** whenever an agent generates and executes code (`CodeActAgent`). Never run LLM-generated code in the host process.

## Iteration Limiter

`IterationLimiter` caps the number of steps an agent can take before raising `AgentIterationLimitError`. Loop-based agents (`ReActAgent`, `CodeActAgent`) create one internally from their `max_iterations` parameter. Other agent types use bounding mechanisms appropriate to their architecture.

> **See also:** [examples/control/iteration_limits.py](../../examples/control/iteration_limits.py)

## Tool Call Limiter

`ToolCallLimiter` caps the cumulative number of tool calls across all agent steps. Unlike `IterationLimiter` which increments by 1 per step, `ToolCallLimiter.step(count)` accepts the batch size since one LLM response can request multiple tool calls.

`ReActAgent` creates one internally when `max_tool_calls` is set. The limiter checks *after* tool dispatch — the batch that exceeds the limit completes fully, and the agent stops before the next iteration. This avoids malformed message sequences from partially-executed tool batches.

When both `max_iterations` and `max_tool_calls` are set, whichever limit is reached first terminates the loop. The `termination_reason` on the result indicates which limit triggered (`"iteration_limit"` or `"tool_call_limit"`).

> **See also:** [examples/control/iteration_limits.py](../../examples/control/iteration_limits.py)

### Bounding Parameters by Agent Type

| Agent Type | Parameter(s) | Default | Mechanism |
|------------|-------------|---------|----------|
| ReActAgent | `max_iterations` | 10 | `IterationLimiter` — caps reasoning/action steps |
| ReActAgent | `max_tool_calls` | None | `ToolCallLimiter` — caps cumulative tool calls across all steps |
| CodeActAgent | `max_iterations` | 10 | `IterationLimiter` — caps code generation/execution cycles |
| ReflexionAgent | `max_attempts` | 3 | Retry counter — limits reflection/retry cycles |
| TreeOfThoughtAgent | `max_depth`, `max_nodes` | 5, 50 | Tree search bounds — limits depth and total nodes explored |
| LATSAgent | `max_iterations`, `max_depth` | 20, 10 | Manual counting — limits search iterations and tree depth |
| ReWOOAgent | (none) | — | Plan-then-execute: bounded by plan size |
| ReasoningAgent | (none) | — | Single LLM call — bounded by design |

## Composition and nested agents

Per-agent iteration and tool-call limits bound a single agent. When agents delegate — via `AgentTool`, handoff chains, or workflows that wrap agents — budgets compose multiplicatively, not additively. An outer agent with `max_iterations=10` that uses a delegated agent (also `max_iterations=10`) as a tool is bounded at 100 LLM calls, not 20. Add another layer and you reach 1,000.

**Set tighter limits on delegated agents.** A helper invoked from a tool context rarely needs ten steps of reasoning — consider `max_iterations=3` or less for agents that appear as an `AgentTool`.

**Use `PeerNetwork` when agents consult each other.** `PeerNetwork` ships with a shared `InvocationBudget` (`max_invocations`, default 50) that caps total peer consultations across the whole network — the built-in primitive for bounded multi-agent coordination.

**The SDK does not detect delegation cycles.** If Agent A uses Agent B as a tool and Agent B uses Agent A as a tool, per-agent iteration limits are the only backstop. Prefer `PeerNetwork` over circular `AgentTool` graphs.

A total-run budget spanning nested agents is not currently a primitive — per-agent limits and `PeerNetwork.max_invocations` are the available tools.

## Cancellation Token

`CancellationToken` provides cooperative cancellation — an external signal that tells the agent to stop gracefully. The token is thread-safe and can be triggered from any thread (e.g., an API timeout handler). Cancellation is irreversible once signalled.

Cancellation is **checked between agent steps and raced against the in-flight tool or sandbox await**. When `cancel()` fires during a tool call, the underlying coroutine is cancelled and the agent exits its loop with `AgentResult.termination_reason="cancelled"`. The tool-authoring contract is unchanged — tools do not need to accept a cancellation parameter; the agent loop is responsible for honoring the token. A long-running LLM request is still not interrupted directly; pair cancellation with the provider's `request_timeout` (and, for adopter-authored tools that use `create_http_tool`, its `request_timeout` argument) for the LLM-call path.

Internally the helper that races the token against an awaitable raises `RunCancelled` (re-exported from `nanitics.errors`). Application code normally observes the structured `AgentResult` instead of catching the exception — `RunCancelled` is an internal control-flow signal that the agent loop converts to the public `termination_reason="cancelled"` outcome.

> **See also:** [examples/control/cancellation.py](../../examples/control/cancellation.py)

## Sandbox

`CodeActAgent` generates and executes arbitrary code — that code must run in isolation. The `Sandbox` protocol defines the interface: start an environment, execute code, reset state, and clean up. All implementations support `async with` for automatic resource cleanup.

`DockerSandbox` is the production implementation. It runs a persistent Python process inside a Docker container with a hardened security posture (read-only filesystem, privilege restrictions, resource limits, network isolation by default). It also supports a tool bridge that lets sandboxed code call agent tools across the isolation boundary. Requires `pip install nanitics[code_execution]`.

`MockSandbox` returns predefined `ExecutionResult` responses in sequence — use it for testing agents that require a sandbox without needing Docker.

For the honest-limits posture — what the Docker container blocks and what it does not — see [Security § DockerSandbox honest limits](security.md#dockersandbox-honest-limits).

> **See also:** [examples/tools/sandbox.py](../../examples/tools/sandbox.py)

## Pitfalls

**Setting iteration limits too low.** A limit of 3 on a ReAct agent gives it almost no room to reason. If the agent consistently hits limits, increase the limit or simplify the task.

**Confusing iteration limits and tool call limits.** Iteration limits count reasoning steps. Tool call limits count individual tool invocations across all steps. An agent with `max_iterations=5` and `max_tool_calls=10` can take at most 5 steps, but also no more than 10 total tool calls across those steps.

**Assuming cancellation is immediate.** Cancellation interrupts an in-flight tool or sandbox call and is also checked between steps, but the active LLM request is not interrupted — the run stops once the current LLM call returns. Pair the token with the provider's `request_timeout` if you need to bound that hop.

**Forgetting sandbox cleanup.** Always use `async with` or explicitly call `cleanup()`. Docker containers persist until cleaned up, which can leak resources.

**Running CodeActAgent without a sandbox.** LLM-generated code can do anything the host process can do. Always provide a `Sandbox` implementation.
