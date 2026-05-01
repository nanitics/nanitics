# Agent Types

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Choosing the right agent type is the most important architectural decision. Each type implements a different reasoning strategy — how the agent thinks, plans, and acts. This guide helps you pick the right one and understand the trade-offs.

For constructor signatures, parameters, and data models, see the docstrings in the source code.

## Decision Guide

Start here. Match your task requirements to the right agent type.

**Does the task need external actions (tool calls)?**
- No → `ReasoningAgent` (single LLM call, optional structured output)
- Yes → continue below

**Is the task well-defined with known steps?**
- Yes, and you want to minimize LLM calls → `ReWOOAgent` (plan first, execute without re-reasoning)
- No, the agent needs to adapt → continue below

**Is code the natural expression of the action?**
- Yes → `CodeActAgent` (write and execute Python in a sandbox)
- No → continue below

**Will this task be repeated, and should the agent learn from mistakes?**
- Yes → `ReflexionAgent` (wraps an inner agent with cross-run learning)
- No → continue below

**Does the problem have a single clear solution path?**
- Yes → `ReActAgent` (the default — interleaved reasoning and action)
- No, multiple approaches should be explored → continue below

**How hard is the problem?**
- Moderately complex, explore alternatives → `TreeOfThoughtAgent` (branching evaluation-guided search)
- Extremely complex, needs backtracking and reward signals → `LATSAgent` (MCTS-based tree search)

### Comparison Table

| Agent Type | Tool Support | LLM Calls | Planning | Learning | Cost | Best For |
|---|---|---|---|---|---|---|
| `ReActAgent` | Yes | Medium (per step) | Implicit | No | Medium | Most tasks |
| `ReasoningAgent` | No | 1 (+ revisions) | No | No | Low | Extraction, classification |
| `ReWOOAgent` | Yes | 2–3 (plan + solve) | Explicit upfront | No | Low–Medium | Well-defined workflows |
| `ReflexionAgent` | Via inner agent | High (multi-attempt) | Inner agent decides | Yes (episodic) | High | Repeated similar tasks |
| `CodeActAgent` | Via code | Medium (per step) | Implicit | No | Medium | Data analysis, computation |
| `TreeOfThoughtAgent` | No | High (branching) | Branching paths | No | High | Creative, multi-path problems |
| `LATSAgent` | Yes | Very High | MCTS tree search | Yes (episodic) | Very High | Hardest problems |

## ReActAgent

The default agent type. On each step, the LLM either calls a tool or produces a final text answer. The loop continues until the LLM responds without tool calls, the iteration limit is reached, or the agent is cancelled.

Key capabilities that inform when to choose ReAct:

- **Working memory** — attach a `WorkingMemory` instance and the agent maintains structured state across steps by parsing `<working_memory>` blocks from LLM output. See [Memory](memory.md).
- **Output evaluation** — attach an `OutputEvaluator` and the agent self-revises its final answer until accepted or `max_revisions` exhausted. When combined with `output_schema`, evaluation runs on the structured output.
- **Error self-correction** — with an `ErrorHandler`, tool errors are fed back to the LLM as correction prompts rather than crashing the run.
- **Structured output** — set `output_schema` to a Pydantic `BaseModel` subclass. After the tool-use loop, one additional schema-constrained LLM call produces typed JSON in `result.parsed`.
- **Durable execution** — supports checkpoint/resume for human-in-the-loop suspension. See [Human-in-the-Loop](human-in-the-loop.md).
- **Dynamic tool injection** — the only agent type compatible with Blackboard, MessageBus, and PeerNetwork coordination patterns.

**When to use:** Most tasks. If you're unsure which agent type to pick, start with ReAct.

**When not to use:** Pure data extraction (no tools needed), tasks where you need to minimize LLM calls (consider ReWOO), tasks where code is the natural action modality (consider CodeAct).

> **See also:** [examples/agents/react_agent.py](../../examples/agents/react_agent.py)

## ReasoningAgent

Single LLM call with no tool loop. Optionally returns structured output via a Pydantic model schema. With an `output_evaluator` attached, the agent will re-prompt the LLM for revisions if the evaluator returns `REVISE`, making it multi-step despite being fundamentally a single-call agent.

**When to use:** Data extraction, classification, structured analysis — any task requiring no external actions.

**When not to use:** Tasks requiring tools, multi-step reasoning, or adaptive behavior.

> **See also:** [examples/agents/reasoning_agent.py](../../examples/agents/reasoning_agent.py)

## ReWOOAgent

**Re**asoning **W**ith**O**ut **O**bservation. Plans all steps upfront, executes tools without re-reasoning, then synthesizes results in three distinct phases:

1. **Planner** — LLM generates a structured plan with steps, tool calls, and dependencies
2. **Worker** — executes tool calls in dependency order, parallelizing independent steps
3. **Solver** — LLM synthesizes all observations into a final answer

Steps can reference earlier results using `#N` variable substitution (e.g., `{"text": "#1"}` substitutes the result of step 1). Steps are grouped into execution levels by declared dependencies and run in parallel within each level. The plan is persisted in a `PlanStore` and updated with step results and statuses as execution proceeds. The solver phase supports output evaluation with revision.

The key trade-off: ReWOO minimizes LLM calls (typically 2–3 total regardless of tool count) but commits to a plan upfront. If a tool result should change the strategy, ReWOO can't adapt — it will execute the remaining plan regardless.

**When to use:** Well-defined tasks with predictable tool call sequences where you want to minimize LLM cost.

**When not to use:** Tasks requiring adaptive reasoning — if tool results should change the strategy, use ReAct instead.

> **See also:** [examples/agents/rewoo_agent.py](../../examples/agents/rewoo_agent.py)

## ReflexionAgent

Wraps an inner agent with a retry-and-reflect loop. After each attempt, an evaluator checks the output. If it fails, the agent generates a reflection (analysis of what went wrong), stores it as an episode, and tries again.

The execution loop:

1. Run the inner agent on the task
2. Evaluate the output with the provided evaluator
3. If accepted → store a success episode and return
4. If not accepted and attempts remain → generate a reflection, store as episode, retry from step 1
5. After `max_attempts` → return the last result with `termination_reason="evaluation_failed"`

Reflections are generated by calling the LLM with a specialized prompt that includes the task, the failed output, evaluation feedback, and tools used.

Each attempt is recorded as an episode. The episode store can be shared across runs, enabling cross-run learning on similar tasks. For reflections to be injected into the inner agent *within the same run* (across retry attempts), the inner agent must have an `EpisodicMemoryProvider` context provider configured to read from the same `episode_store`.

The inner agent can be any agent type — a ReAct agent for tool-using tasks, a Reasoning agent for extraction tasks, etc. All configuration (context providers, error handling) goes on the inner agent, not the outer `ReflexionAgent`.

**When to use:** Tasks that are repeated (or similar), where learning from past mistakes improves future performance.

**When not to use:** One-off tasks (no benefit from reflection), tasks where the first attempt usually succeeds (unnecessary overhead).

> **See also:** [examples/agents/reflexion_agent.py](../../examples/agents/reflexion_agent.py)

## CodeActAgent

Uses Python code as the action modality. The LLM submits code via an internal `execute_code` tool, which the agent executes in a `Sandbox`. Output (stdout, return value, or error traceback) is fed back as an observation. When the LLM responds with plain text (no tool calls), that's the final answer.

**Tool bridge:** Tools can optionally be bridged into the code execution environment — when `tools` are passed to a `CodeActAgent`, they're converted to Python function stubs callable directly in LLM-generated code. The LLM can then write code like `data = fetch_data(query="SELECT COUNT(*) FROM users")` and the tool executes normally behind the scenes.

**Sandbox selection:** Use `DockerSandbox` in production (real containerized execution with resource limits, requires Docker) and `MockSandbox` for testing (in-memory, configurable responses). Executing LLM-generated code without sandboxing is a security risk. See [Safety](safety.md).

**When to use:** Data analysis, computation, tasks naturally expressed as code, complex logic easier to write as code than to decompose into individual tool calls.

**When not to use:** Tasks requiring judgment over computation, tasks where tool calls are clearer, environments where sandboxed code execution isn't available.

> **See also:** [examples/agents/codeact_agent.py](../../examples/agents/codeact_agent.py)

## TreeOfThoughtAgent

Explores multiple reasoning paths simultaneously using tree search. On each step, the agent generates several candidate continuations (branching), evaluates each with a node evaluator, and selects which branches to expand next based on the search strategy.

### Search Strategies

| Strategy | Behavior | Best For |
|---|---|---|
| `SearchStrategy.BFS` | Expand all nodes at the shallowest depth first | Even exploration, finding diverse solutions |
| `SearchStrategy.DFS` | Expand the deepest node first | Finding a complete solution quickly |
| `SearchStrategy.BEST_FIRST` | Expand the highest-scored node (greedy) | When the evaluator is reliable |

The search terminates when a terminal node is found, the node budget (`max_nodes`) is exhausted, or no expandable nodes remain. The best node is selected as output: terminal nodes preferred, then highest-scored non-pruned nodes. Nodes receiving a `REJECT` verdict from the evaluator are pruned.

**When to use:** Complex problems where multiple approaches are viable, creative tasks, problems where the first idea isn't always the best.

**When not to use:** Straightforward tasks (massive overkill — a single ReAct loop is faster and cheaper), tasks requiring tool calls (TreeOfThought doesn't support tools).

> **See also:** [examples/agents/tree_of_thought.py](../../examples/agents/tree_of_thought.py)

## LATSAgent

**L**anguage **A**gent **T**ree **S**earch. The most powerful (and most expensive) agent type. Combines Monte Carlo Tree Search (MCTS) with tool use, evaluation, backpropagation, and optional episodic memory.

Each MCTS iteration:

1. **Select** — traverse the tree from root using UCB1 (Upper Confidence Bound) to find the most promising leaf. UCB1 balances exploitation (high-value nodes) with exploration (under-visited nodes) via the `exploration_constant`.
2. **Expand** — generate `branching_factor` child nodes from the selected leaf. Each child is an LLM response that either calls a tool or produces a final answer.
3. **Evaluate** — score each child node with the `node_evaluator`.
4. **Backpropagate** — update value and visit counts from the child back up to the root.

The cycle repeats until a node is accepted by the evaluator, `max_iterations` is exhausted, or the agent is cancelled.

When `episode_store` is provided, past episodes are retrieved as context before search and results are recorded after (with a reflection if the search failed), enabling cross-run learning on similar tasks.

**When to use:** The hardest problems where simpler agent types fail. Tasks requiring systematic exploration with backtracking.

**When not to use:** Almost everything else. LATS is expensive — each iteration involves selection, expansion, evaluation, and backpropagation. Use simpler agent types first and escalate to LATS only when they're insufficient.

> **See also:** [examples/agents/lats_agent.py](../../examples/agents/lats_agent.py)

## Extending Agent

To create a custom agent type, subclass `Agent` and implement `_execute()`. Your implementation focuses purely on the reasoning strategy — the base class handles everything else:

- `_call_llm()` — LLM calls with automatic context injection, context management, error handling, and event emission
- `_emit_step()` / `_emit_error()` — observability events
- `_evaluate_output()` — output evaluation (if configured)
- `_is_cancelled` — cancellation token check
- `_system_prompt` — the built system prompt (base + contributions)

All base `Agent` parameters (emitter, cancellation token, error handler, context manager, context providers, output evaluator, prompt contributors) are handled by the base class.

> **See also:** Source — `nanitics/core/agent.py`

## Coordination Compatibility

Some multi-agent coordination patterns inject tools into agents at runtime — for example, the Blackboard injects shared memory tools, the MessageBus injects a `publish_message` tool, and PeerNetwork injects `consult_<peer>` tools. These patterns require agents that can **react to tool results mid-execution**: read shared state, adapt reasoning, and take further action based on what they see.

Only `ReActAgent` has this capability. Its reactive loop — interleaved reasoning and action — is what makes mid-execution coordination possible. The other agent types have execution models that can't meaningfully participate:

| Agent | Why it can't do active coordination |
|---|---|
| `ReasoningAgent` | Single LLM call, no tool loop |
| `ReWOOAgent` | Plans all tool calls upfront, cannot adapt after seeing results |
| `CodeActAgent` | Code as action modality, designed for computation not coordination |
| `TreeOfThoughtAgent` | No tool support |
| `LATSAgent` | MCTS tree search, architecturally mismatched with interactive coordination |
| `ReflexionAgent` | Delegates to inner agent (configure the inner agent instead) |

This distinction is not about whether an agent "supports tools" — ReWOO, CodeAct, and LATS all use tools internally. It's about whether the agent's execution loop can **react dynamically** to tool results from coordination infrastructure. Agents declare this capability via the `supports_dynamic_tools` property.

**Bottom line:** If your agent needs to participate in Blackboard, MessageBus, or PeerNetwork coordination, use `ReActAgent`.

## Pitfalls

- **Defaulting to complex agent types.** Start with `ReActAgent`. Only escalate when simpler agents demonstrably fail. `TreeOfThoughtAgent` and `LATSAgent` are 10–100x more expensive per run.
- **Using ReWOO for adaptive tasks.** ReWOO commits to a plan upfront. If tool results should change the strategy, use ReAct instead.
- **Forgetting iteration limits.** `ReActAgent` and `CodeActAgent` default to `max_iterations=10`. For long-running tasks, increase this. Without a limit, a confused agent can loop indefinitely.
- **Skipping the evaluator for tree-search agents.** `TreeOfThoughtAgent` and `LATSAgent` require a `node_evaluator` — the quality of the search depends entirely on how well nodes are scored. Invest in evaluator quality.
- **Using CodeActAgent without a proper sandbox.** `MockSandbox` is for testing only. In production, use `DockerSandbox` with resource limits. Executing LLM-generated code without sandboxing is a security risk.
- **Configuring features on ReflexionAgent instead of the inner agent.** `ReflexionAgent` delegates execution entirely. Context providers, error handling, memory — all go on the inner agent.

### Escalation Path

When an agent type isn't working, escalate deliberately:

1. **ReActAgent** — start here for tool-using tasks
2. If accuracy is too low → add an `OutputEvaluator` for self-revision
3. If the task is repeated and accuracy varies → wrap in `ReflexionAgent` for cross-run learning
4. If multiple solution paths exist and you're stuck in local optima → `TreeOfThoughtAgent`
5. If the problem requires systematic backtracking → `LATSAgent` (last resort)

## Base Agent Parameter Support

Not all agent types support every base `Agent` parameter:

| Agent | `error_handler` | `context_manager` | `context_providers` | `output_evaluator` | `prompt_contributors` |
|---|---|---|---|---|---|
| `ReActAgent` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ReasoningAgent` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ReWOOAgent` | ✗ | ✓ | ✓ | ✓ | ✓ |
| `CodeActAgent` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ReflexionAgent` | ✗ | ✗ | ✗ | ✗ (has `evaluator`) | ✗ |
| `TreeOfThoughtAgent` | ✗ | ✓ | ✓ | via `node_evaluator` | ✓ |
| `LATSAgent` | ✗ | ✓ | ✓ | via `node_evaluator` | ✓ |

**Notes:**
- `ReflexionAgent` delegates all features to the inner agent — configure context, error handling, and evaluation on the inner agent, not the outer.
- `TreeOfThoughtAgent` and `LATSAgent` pass `node_evaluator` through as the base `output_evaluator`.
- `ReWOOAgent` doesn't support `error_handler` because its plan-first architecture has no mid-execution adaptation — tool failures are recorded as step results and passed to the solver phase.
