# Orchestration

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Orchestration composes steps — agent runs, async functions, or nested workflows — into multi-step workflows. Instead of building complex logic inside a single agent, you decompose work into discrete steps and let a workflow pattern handle sequencing, concurrency, branching, and failure recovery.

The SDK provides 8 workflow patterns, all built on the same `Step`/`StepResult` protocol. Every pattern supports cancellation, checkpoint-based suspension and resumption, and event emission for observability.

## When to Use Orchestration

**Use orchestration when** you have multiple distinct phases of work, each naturally handled by a separate agent or function. Examples: extract → transform → load pipelines, research → draft → review → publish, or fan-out analysis across multiple data sources.

**Don't orchestrate when** a single agent can handle the task. Adding workflow structure to a problem one agent can solve just adds complexity. If you're unsure, start with a single agent and extract steps when the task naturally separates into phases.

**The pre-pattern check.** Many "workflows" are equally well expressed as `await`-chains: a `ReasoningAgent` with an `output_schema` produces typed output, then plain Python `await`s a few async functions that consume the typed output. Reach for a workflow primitive only when its specific value is load-bearing — `Parallel` for true concurrency on independent steps, checkpoint-based suspension and resumption between stages, `DAG` for non-trivial dependency topologies, intermediate-result inspection where the workflow's metadata surface is what your code actually needs. If a 5-line `await` chain expresses the same shape, it is also easier to read and easier to test. See [`examples/agents/dispatch_over_structured_output.py`](../../examples/agents/dispatch_over_structured_output.py) for the canonical shape.

**Common indicators you need orchestration:**

- Different steps require different tools, models, or system prompts
- Steps have clear input/output contracts
- You want to retry or loop on individual steps without re-running everything
- Work can be parallelized for throughput
- Human approval is needed between phases

## Choosing a Pattern

Every execution unit in a workflow implements the `Step` protocol (a `name` property and an async `execute` method). The SDK provides three adapters: `AgentStep` wraps an agent, `FunctionStep` wraps an async function, and `WorkflowStep` wraps a nested workflow. All return `StepResult`, which carries an `output` (passed between steps) and a `metadata` dict (step counts, termination reasons, intermediate results). When the wrapped agent has `output_schema`, `AgentStep` forwards the parsed Pydantic model as the step output; the text response is available in `metadata["text_output"]`. See the source docstrings for details and [examples/workflows/sequential_pipeline.py](../../examples/workflows/sequential_pipeline.py) Section 2b for structured output flow.

| Pattern | Structure | Data Flow | Concurrency | Best For |
|---------|-----------|-----------|-------------|----------|
| **Sequential** | Ordered chain | Output → next input | None | Step-by-step pipelines where each step needs the previous result |
| **Pipeline** | Ordered chain + contracts | Output → next input (validated) | None | Data pipelines with strict inter-stage type guarantees |
| **Parallel** | Fan-out | Same input to all | All steps | Independent analyses of the same input |
| **DAG** | Dependency graph | Routed by dependencies | Where possible | Complex dependency structures with shared intermediate results |
| **Loop** | Single step, repeated | Output → next iteration input | None | Iterative refinement until a quality condition is met |
| **MapReduce** | Fan-out per item | Split → map → reduce | Per item | Applying the same step to many items with aggregation |
| **Conditional** | Router + branches | Input → selected branch | None | Choosing one path based on input characteristics |
| **PlanToWorkflow** | Auto-selected | From plan structure | Depends on plan | Executing a `TaskPlan` from the planning system |

## Workflow Patterns

### Sequential

Executes steps one after another, passing each step's output as the next step's input. Data flows as `input → step[0] → output₀ → step[1] → output₁ → ... → final output`.

The result metadata includes `intermediate_results` (a dict of step name → `StepResult`) and `total_steps_executed`, so you can inspect any stage's output after completion. Each nested `StepResult` preserves its step's metadata, making failure details and agent statistics from earlier stages accessible.

Use Sequential for straightforward pipelines: research → summarize → review, or extract → transform → load.

> **See also:** [examples/workflows/sequential_pipeline.py](../../examples/workflows/sequential_pipeline.py) — FunctionStep chaining, AgentStep data flow, and WorkflowStep composability.

### Pipeline

Like Sequential, but with optional Pydantic-based type validation between stages. Each `Stage` wraps a step and can declare `input_type` and `output_type` as Pydantic models. Before a stage executes, its input is validated; after execution, its output is validated.

If validation fails, a `PipelineContractError` is raised identifying the stage name, stage index, direction (input or output), expected type, and the Pydantic validation error. This catches data contract violations at the boundary rather than letting bad data propagate through the pipeline.

Use Pipeline over Sequential when stages have strict data shape requirements — especially when steps are developed independently and you want to catch integration mismatches early.

> **See also:** [examples/workflows/sequential_pipeline.py](../../examples/workflows/sequential_pipeline.py) — Stage contracts, Pydantic validation, and PipelineContractError diagnostics.

### Parallel

Executes all steps concurrently with the same input. Every step receives the same input and runs independently.

By default, output is a list of each step's output in declaration order. Provide a custom `aggregator` function to combine results differently — e.g., merge into a named dict, pick the best result, or compute a consensus.

Supports `FailurePolicy` — with `ALL_OR_NOTHING` (default), any step failure cancels pending steps and propagates the exception. With `BEST_EFFORT`, failures are tracked in `metadata["failed_steps"]` and successful results are returned. See [Failure Policies](#failure-policies).

> **See also:** [examples/workflows/parallel.py](../../examples/workflows/parallel.py)

### DAG

Executes steps as a directed acyclic graph, running nodes concurrently whenever their dependencies are satisfied. This is the most flexible pattern — it handles diamond dependencies, fan-out/fan-in, and arbitrary dependency topologies.

**Input routing:** Nodes with no dependencies receive the original workflow input. Nodes with one dependency receive that dependency's output directly. Nodes with multiple dependencies receive a dict mapping dependency names to their outputs.

**Output:** Terminal nodes (nodes that no other node depends on) determine the result. One terminal → its output is the result. Multiple terminals → a dict of `{node_name: output}`.

**Validation:** The graph is validated at construction time — missing dependency references and cycles (detected via Kahn's algorithm) raise immediately with descriptive error messages.

Supports `FailurePolicy` and `max_concurrency`. With `BEST_EFFORT`, failed nodes and their transitive dependents are skipped, tracked in `metadata["failed_nodes"]` and `metadata["skipped_nodes"]`.

> **See also:** [examples/workflows/dag.py](../../examples/workflows/dag.py) — diamond execution, input routing, multiple terminals, validation, BEST_EFFORT failure, and concurrency limiting.

### Loop

Repeatedly executes a single step until a condition callback returns `True` or `max_iterations` is reached. The condition receives the step's result and the 1-indexed iteration number. It can be sync or async.

Between iterations, the step's output becomes the next iteration's input, creating a feedback loop. If `max_iterations` is reached without the condition returning `True`, the last result is returned with `metadata["terminated"] = "iteration_limit"`. The actual iteration count is always available in `metadata["iterations"]`.

Use Loop for iterative refinement: an agent drafts, a quality check evaluates, and the loop continues until the output is acceptable. Also useful for convergence tasks where an agent progressively improves a result.

> **See also:** [examples/workflows/loop.py](../../examples/workflows/loop.py) — condition callbacks, iteration limits, AgentStep refinement, async conditions.

### MapReduce

Splits input into items via a `splitter` function, applies a step to each item concurrently, then combines results via a `reducer`. This is the pattern for "do the same thing to many items" — analyze each section of a report, process each file in a batch, evaluate each candidate.

Both `splitter` and `reducer` can be sync or async. The `splitter` receives the workflow input and returns a list of items. The `reducer` receives a list of `StepResult` objects and produces the final output.

`max_concurrency` limits how many items are processed simultaneously. Supports `FailurePolicy` — with `BEST_EFFORT`, failed items are tracked in `metadata["failed_items"]` while successful results are still reduced.

> **See also:** [examples/workflows/map_reduce.py](../../examples/workflows/map_reduce.py) — structural splitting, concurrency control, failure policies, async splitter/reducer.

### Conditional

Routes input to one of multiple named branches based on a router function. The router (sync or async) examines the input and returns a branch name. Only the selected branch executes.

Provide a `default` step for unrecognized branch names — without one, unknown branches raise `ValueError`. The selected branch name is recorded in `metadata["selected_branch"]`. Branches can be any step type, including `WorkflowStep` for nested workflows.

Use Conditional for input-dependent routing: language detection, task classification, or feature-flagged behavior.

> **See also:** [examples/workflows/conditional.py](../../examples/workflows/conditional.py) — sync/async routers, default fallback, missing branch errors, nested workflow branches.

### PlanToWorkflow

Bridges the [planning](planning.md) and orchestration systems. `plan_to_workflow()` converts a `TaskPlan` into an executable workflow. You provide a `step_factory` that converts leaf `TaskNode`s into concrete steps — the function handles everything else.

The dependency structure determines the workflow type automatically:

- **All independent tasks** → `Parallel`
- **Linear dependency chain** → `Sequential`
- **Mixed dependencies** → `DAG`

Nodes with subtasks are recursively converted into nested sub-workflows. The `step_factory` is only called for leaf nodes.

> **See also:** [examples/workflows/plan_to_workflow.py](../../examples/workflows/plan_to_workflow.py) — auto-selecting Parallel/Sequential/DAG from dependency structure, recursive subtask conversion.

## Composability

Workflows can nest inside each other. Since `Workflow` doesn't implement the `Step` protocol directly (its `execute` accepts an extra `resume_from` keyword argument), wrap a workflow with `WorkflowStep` to use it as a step inside another workflow.

This lets you build hierarchical structures. Common compositions:

- **Sequential + Parallel:** A pipeline where one stage fans out to multiple agents, then results are aggregated for the next stage.
- **DAG + Loop:** A dependency graph where one node iteratively refines its output before dependents execute.
- **Conditional + Sequential:** Route to different pipelines based on input characteristics.
- **MapReduce + Sequential:** Process each item through a multi-step pipeline.
- **Sequential + Conditional:** Pre-process input, then route to specialized handlers.

Each nested workflow gets its own span in the trace, so the full hierarchy is observable. There's no depth limit; compose as deeply as the problem requires.

When composing, think about failure boundaries. A nested workflow's failure propagates to the parent as a step failure, which is then handled according to the parent's `FailurePolicy`. This means you can have `BEST_EFFORT` at the outer level and `ALL_OR_NOTHING` within each nested workflow, or vice versa.

## Failure Policies

Two policies control how concurrent patterns handle step failures:

| Policy | Behavior | Use When |
|--------|----------|----------|
| `FailurePolicy.ALL_OR_NOTHING` (default) | First failure cancels pending steps and re-raises | Partial results are meaningless |
| `FailurePolicy.BEST_EFFORT` | Failures recorded in metadata, execution continues | Partial results have value |

`FailurePolicy` applies to `Parallel`, `MapReduce`, and `DAG`. Sequential, Pipeline, and Loop propagate failures immediately — there's no concurrent work to preserve.

With `BEST_EFFORT`, each pattern tracks failures differently: `Parallel` uses `metadata["failed_steps"]`, `MapReduce` uses `metadata["failed_items"]`, and `DAG` uses `metadata["failed_nodes"]` and `metadata["skipped_nodes"]` (transitive dependents of failed nodes are skipped).

## Resumable Workflows

All workflow patterns support suspension and resumption via checkpoints, enabling durable execution. A workflow can pause mid-execution (e.g., waiting for [human approval](human-in-the-loop.md)), persist its state to a `CheckpointStore`, and resume later — potentially in a different process.

**How it works:**

1. Provide a `CheckpointStore` and `run_id` when constructing the workflow.
2. When a step raises `SuspendExecution`, the workflow saves a checkpoint containing completed results and the current position.
3. The workflow emits `ExecutionSuspendedEvent` and re-raises the exception.
4. To resume: load the checkpoint from the store and pass it as `resume_from` to `execute()`. The workflow skips completed steps and picks up where it left off.

Each pattern stores the minimal state needed for resumption — completed results, the suspended step/node/iteration, and any intermediate state specific to the pattern.

**Limitation:** In `Parallel`, `DAG`, and `MapReduce`, only the first concurrent suspension is captured. If multiple branches might need human approval, sequence them instead.

> **See also:** [examples/durability/checkpoint_suspension.py](../../examples/durability/checkpoint_suspension.py) — checkpoint primitives, workflow suspension/resumption, version validation, HITL-integrated suspend/resume.

## Cancellation

Pass a `CancellationToken` to any workflow for cooperative cancellation. Call `token.cancel()` from another coroutine to signal cancellation.

Cancellation behavior differs by pattern:

- **Before execution starts (all patterns):** Raises `WorkflowCancelledError`.
- **Iterative patterns (Sequential, Pipeline, Loop, MapReduce):** Stops after the current step completes. Returns a `StepResult` with `metadata["terminated"] = "cancelled"`. Completed step results are preserved in metadata.
- **Concurrent patterns (Parallel, DAG):** Cancels pending tasks and returns partial results. Already-running steps are allowed to complete.

Cancellation is cooperative — steps are not forcibly terminated mid-execution. The token is checked between steps (iterative patterns) or before starting new tasks (concurrent patterns).

## Events

All workflows emit events through `EventEmitter` for observability. Key events: `WorkflowStartEvent`, `WorkflowStepCompleteEvent` (per step), `WorkflowCompleteEvent`, `WorkflowErrorEvent`, `ExecutionSuspendedEvent`, `ExecutionResumedEvent`, and `CheckpointSavedEvent`.

Each workflow wraps its execution in an emitter span, and individual steps get nested spans, creating a hierarchical trace. This means nested workflows produce a tree of spans that mirrors the workflow composition structure — useful for debugging multi-level orchestrations.

See the [observability guide](observability.md) for how to consume these events.

## Pitfalls

- **Over-orchestrating:** If one agent can do the job, don't split it into a workflow. Orchestration adds coordination overhead.
- **Large step outputs:** Outputs flow between steps as-is. If a step returns a large payload, use a `FunctionStep` to extract just what the next step needs.
- **Missing emitter:** The `emitter` parameter is required. Without it, you get no observability into workflow execution.
- **Checkpoint store without run_id:** Provide both `checkpoint_store` and `run_id` together — checkpoints need a `run_id` to be meaningful.
- **DAG cycles:** Caught at construction time with a clear error message. Restructure your dependencies.
- **Concurrent suspensions:** In `Parallel`, `DAG`, and `MapReduce`, only the first branch suspension is captured. If multiple branches might need human approval, sequence them instead.
