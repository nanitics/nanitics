# Planning

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Agents can operate without explicit plans — a ReAct agent naturally sequences actions through its reasoning loop. But for complex, multi-step tasks, explicit planning provides structure: the agent creates a plan, tracks progress through steps, revises when circumstances change, and can be evaluated on whether it completed what it set out to do.

The planning system provides data models (`Plan`, `PlanStep`, `Goal`), a persistent store, tools that let agents manage plans during execution, context providers that inject plan state into the conversation, evaluators that check plan completion, and prompt contributors that teach agents different planning strategies.

## When to Use Planning

Planning is a capability you add deliberately — it's not always the right choice.

**Use planning when:**

- The task has multiple ordered steps with dependencies between them
- The agent needs to track progress across many iterations
- You want automated evaluation that the agent completed its plan before finishing
- Multiple competing objectives require explicit prioritization
- You need visibility into what the agent intended to do vs. what it actually did

**Skip planning when:**

- The task is simple enough that the agent can sequence actions through natural reasoning (3–5 steps)
- The task structure isn't known in advance and can't be decomposed upfront
- The agent type already enforces structure (e.g., ReWOO has planning built into its architecture)

Planning adds LLM overhead — the agent spends tokens creating, reading, and updating plan state. For simple tasks, this cost outweighs the benefit of structured tracking.

## Choosing a Planning Strategy

Four system prompt contributors teach agents different planning behaviors. The strategy you choose determines how the agent approaches plan creation, execution, and revision.

| Strategy | Workflow | Best For |
|----------|----------|----------|
| **Adaptive** | Create plan → execute → evaluate after each step → revise if needed | Most scenarios. Tasks where the approach may need adjustment based on intermediate results. |
| **Upfront** | Analyze → create complete plan → execute in order → synthesize | Well-defined tasks where the full sequence is knowable in advance. Minimizes LLM reasoning during execution. |
| **Decomposition** | Analyze → decompose into subtasks → execute each independently → assemble | Tasks with multiple distinct concerns that benefit from separation. |
| **Goal Tracking** | Identify goals → track status → prioritize by importance → resolve conflicts | Tasks with multiple competing objectives where priorities matter. |

Each contributor is a `SystemPromptContributor` — add it to your `SystemPromptBuilder` to inject the strategy instructions into the agent's system prompt. The agent still needs planning tools to act on the strategy.

**Choosing between strategies:**

- Start with **Adaptive** — it handles the widest range of tasks and allows mid-course correction.
- Use **Upfront** when you want to minimize token usage during execution and the task is predictable. This is a "soft" version of the ReWOO architecture — the strategy encourages upfront planning but doesn't enforce it structurally.
- Use **Decomposition** when a task naturally splits into independent workstreams (e.g., "research X, analyze Y, write Z"). Consider whether a multi-agent setup with task delegation would be more appropriate than a single agent decomposing internally.
- Use **Goal Tracking** when the agent must balance competing priorities or when success is defined by outcomes rather than step completion.

Strategies can be combined with any evaluator. Pair **Goal Tracking** with `GoalSatisfactionEvaluator` and others with `PlanAdherenceEvaluator`.

## Goal-Based Planning

Plans can include **goals** — desired outcomes with priorities and success criteria. Goals elevate plans from task checklists to objective-driven execution.

Without goals, an agent checks off steps mechanically. With goals, the agent can reason about *why* it's executing steps, re-prioritize when circumstances change, and evaluate whether completing the steps actually achieved the intended outcomes.

Goals nest recursively (subgoals under parent goals), have explicit status tracking (active, achieved, blocked, abandoned), and can be created or updated mid-execution through planning tools. The `GoalTrackingContributor` strategy works best with goal-based plans, but goals can be used with any strategy.

**When to use goals vs. plain steps:**

- Use plain steps when the task is procedural and success means "do these things in order."
- Add goals when the task has qualitative success criteria ("find at least 3 actionable insights") or when the agent should reason about trade-offs between competing objectives.
- Use nested subgoals when a high-level objective decomposes into independently trackable outcomes.

Use `GoalSatisfactionEvaluator` to enforce that all goals are resolved before the agent finishes — it returns `REVISE` if any goals remain active. Use `PlanAdherenceEvaluator` for step-based completion checking.

> **See also:** [examples/planning/planning_goals.py](../../examples/planning/planning_goals.py) — goals, goal tools, goal evaluator

## PlanningCapability

Setting up planning requires wiring together several components: a plan store, planning tools, a context provider (so the agent sees its plan state), and optionally an evaluator. `PlanningCapability` bundles all of this into a single object with automatic plan ID wiring.

When the agent creates a plan via tools, the capability captures the plan ID and automatically links the context provider and evaluator to that plan. This eliminates the most common planning bug: mismatched plan IDs across components.

The capability exposes `.tools`, `.context_provider`, and `.output_evaluator` — pass these to your agent constructor. You can also wire components individually if you need finer control, but `PlanningCapability` is the recommended starting point.

**What it bundles:**

- **Plan store** — persistence for plan state (use `InMemoryPlanStore` for development)
- **Planning tools** — six tools for creating, reading, updating, and revising plans and goals
- **Context provider** — injects current plan state into the agent's context at configurable detail levels (minimal, normal, full). Delivered via the standard `<nanitics:context>` wrapper the SDK applies to every context-provider contribution, so the LLM recognises the plan state as SDK-injected context rather than user speech.
- **Output evaluator** — checks plan/goal completion before accepting the agent's output

> **See also:** [examples/planning/planning.py](../../examples/planning/planning.py) — plans, steps, auto-wiring, revision

## Plan-to-Workflow Bridge

`plan_to_workflow` converts a `TaskPlan` (a tree of `TaskNode` objects with dependency edges) into an executable orchestration workflow. It analyzes the dependency structure and selects the appropriate workflow type automatically:

- Independent tasks → `Parallel`
- Linear chains → `Sequential`
- Mixed dependencies → `DAG`
- Nodes with subtasks → nested sub-workflows

This bridges the gap between planning (what to do) and orchestration (how to execute it). A planning agent can produce a `TaskPlan`, and the system converts it into a concrete execution graph without manual workflow construction.

This is particularly useful in multi-agent systems where one agent plans and another (or a workflow) executes. The planning agent reasons about task structure; `plan_to_workflow` handles the translation to executable form.

> **See also:** [examples/workflows/plan_to_workflow.py](../../examples/workflows/plan_to_workflow.py), [Orchestration](orchestration.md) guide

## Composition with Other Capabilities

Planning interacts with several other SDK capabilities. Understanding these interactions helps you wire things together correctly.

**Planning + Working Memory.** Agents with working memory can write plan-like notes naturally. The parsing utilities (`parse_plan_from_working_memory`, `parse_goals_from_working_memory`) extract structured state from markdown-formatted memory content. Use this lightweight approach when you want plan-like tracking without the full planning tool suite.

**Planning + Evaluation.** Plan-aware evaluators (`PlanAdherenceEvaluator`, `GoalSatisfactionEvaluator`) plug into the standard output evaluation system. They work alongside other evaluators — you can check both plan completion and output quality in the same agent.

**Planning + Multi-Agent.** In supervisor or orchestrator patterns, the coordinator agent typically owns the plan while delegated agents execute individual steps. The coordinator tracks progress and revises the plan based on delegate results. Use `PlanningCapability` on the coordinator, not on the delegates.

**Planning + Context Management.** The `PlanningContextProvider` competes for context window space with other context providers. Choose the detail level carefully — `"full"` detail on a large plan can consume significant tokens. Start with `"normal"` and drop to `"minimal"` if context pressure is an issue.

## Pitfalls

**Over-planning simple tasks.** Not every task needs explicit planning. If an agent can complete a task in 3–5 steps with natural reasoning, planning tools add overhead without benefit. The added tokens for plan creation, context injection, and evaluation can exceed the cost of the actual work.

**Forgetting the context provider.** Planning tools create and update plans, but the agent can only *see* plan state if you include a `PlanningContextProvider` (or use `PlanningCapability` which bundles it). Without it, the agent must call `get_plan` explicitly each iteration, wasting a tool call. This is the most common wiring mistake.

**Mismatching plan ID across components.** When wiring `PlanningContextProvider` and evaluators separately, they must reference the same plan ID. Use `PlanningCapability` to avoid this entirely.

**Not teaching the agent to plan.** Giving an agent planning tools without a strategy contributor means the agent may not know when or how to use them. Always pair planning tools with a strategy contributor.

**Wrong evaluator for the planning style.** `PlanAdherenceEvaluator` checks step completion — it doesn't understand goals. `GoalSatisfactionEvaluator` checks goal resolution — it doesn't care about step status. Match the evaluator to what matters: steps, goals, or both (use two evaluators).

**Too much context detail on large plans.** The `"full"` detail level includes step IDs, dependency lists, and all results. For plans with 10+ steps, this can consume a significant portion of the context window. Start with `"normal"` and only use `"full"` when debugging plan execution issues.

**Revision without bounds.** If using `AdaptivePlanningContributor`, the agent may revise its plan repeatedly. Set `max_revisions` on the evaluator to prevent infinite revision loops. Two to three revisions is a reasonable default.
