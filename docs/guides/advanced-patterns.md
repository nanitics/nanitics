# Advanced Patterns

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Specialized primitives that live in `nanitics.specialized`. They are structurally distinct from the core surface but niche — most agentic systems are built without them. Reach for these deliberately, when the shape of your problem matches the pattern.

If you're new to Nanitics, start with the [Start here](README.md#start-here) tier. Come back to this page when one of the patterns below looks like the right tool.

## Reasoning strategies

The default reasoning loop is [`ReActAgent`](agent-types.md#reactagent). The strategies below trade off more LLM calls, more memory, or more code for better outcomes on a narrower set of problems.

- **[`ReWOOAgent`](agent-types.md#rewooagent)** — plan-then-execute. Use when the task decomposes cleanly upfront, tool calls are expensive, and you want to minimise LLM round-trips.
- **[`ReflexionAgent`](agent-types.md#reflexionagent)** — wraps an inner agent with cross-run self-critique. Use when the same task is repeated and quality improves with a learned critique loop.
- **[`TreeOfThoughtAgent`](agent-types.md#treeofthoughtagent)** — branching evaluation-guided search. Use for problems with multiple plausible solution paths that can be scored independently.
- **[`LATSAgent`](agent-types.md#latsagent)** — MCTS-based tree search with backtracking and reward signals. Use for the hardest problems where you can score an outcome and afford many LLM calls.

## Workflow shapes

Beyond the core `Sequential`, `Parallel`, and `DAG` workflows, `nanitics.specialized` adds shapes for control-flow and data-parallel patterns.

- **[`Loop`](orchestration.md#loop)** — repeat a step until a condition holds.
- **[`Conditional`](orchestration.md#conditional)** — branch based on the result of a previous step.
- **[`MapReduce`](orchestration.md#mapreduce)** — fan out over a collection, then aggregate.
- **[`Pipeline`](orchestration.md#pipeline)** — typed step-to-step data passing.

## Multi-agent coordination

The [foundations](multi-agent-foundations.md) (agent-as-tool, broadcast, handoff, blackboard) and core [coordination patterns](multi-agent-coordination.md) (orchestrator, supervisor, judge router) cover most multi-agent systems. The patterns below model agents as independent participants with their own incentives or perspectives.

- **[Bidding](multi-agent-coordination.md#bidding)** — agents bid for work; a strategy allocates. Use when agents have specializations and you want competitive selection rather than a fixed router.
- **[Debate](multi-agent-coordination.md#debate)** — adversarial multi-round refinement. Use when a single agent's first answer is unreliable and adversarial pressure improves it.
- **[Consensus](multi-agent-coordination.md#consensus)** — independent agents converge through deliberation. Use when you want robustness through diversity rather than a single authoritative answer.
- **`MessageBus`** — pub/sub messaging across agents. Use for decoupled coordination where agents react to events instead of being called directly.
- **`PeerNetwork`** — agents discover and call peers without a central orchestrator. Use for decentralized coordination.

## Planning

The [core planning surface](planning.md) covers upfront and adaptive planning with goal tracking. `nanitics.specialized` adds:

- **Hierarchical-decomposition planning** — recursive task breakdown into subtasks until each is executable. Use when the task is too large to plan in one pass and natural sub-goals exist.

## Providers

- **`MistralLLMClient`** — native Mistral client. For most adopters [`LiteLLMClient`](local-llms.md) covers Mistral too; reach for the native client only when you need Mistral-specific features.
