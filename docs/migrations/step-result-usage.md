# Migration — `StepResult.usage` and aggregated workflow usage

## What changed

`StepResult` gained a first-class, typed `usage: Usage | None` field. The
existing string-key contract `StepResult.metadata["usage"]` — a
`model_dump()` of `Usage` written by `AgentStep`, `_BoundAgentStep`, and
`_BoundHandoffStep` — is now deprecated. Both surfaces are populated for the
deprecation window so existing consumers continue to work unchanged. The
typed field is the canonical path going forward.

The `Sequential` and `Pipeline` workflow runners now attach an
**aggregated** `usage` to their returned `StepResult` — the sum of every
sub-step's `usage` (`None` only when every sub-step contributed `None`).
A `WorkflowStep` wrapping a nested workflow folds the nested aggregate
into its parent runner's aggregate by construction, so recursive nesting
needs no special handling.

`SupervisionResult` gained a non-optional `usage: Usage` field. It is the
sum across **every** attempt the `Supervisor` ran during a single
`supervise()` call — accepted attempt plus every retried, reassigned, or
escalated attempt. This is distinct from `result.usage`, which is the
final attempt's usage only.

## Migration path

**Per-step usage** — read off the typed field instead of digging into the
metadata dict.

Before:

```python
step_result = await agent_step.execute("task")
total = step_result.metadata["usage"]["total_tokens"]
```

After:

```python
step_result = await agent_step.execute("task")
total = step_result.usage.total_tokens if step_result.usage else 0
```

**Per-workflow-run usage** — read it directly off the returned
`StepResult`; you no longer need to walk `intermediate_results` and sum
manually.

Before:

```python
result = await sequential.execute("input")
total = sum(
    sub.metadata["usage"]["total_tokens"]
    for sub in result.metadata["intermediate_results"].values()
    if "usage" in sub.metadata
)
```

After:

```python
result = await sequential.execute("input")
total = result.usage.total_tokens if result.usage else 0
```

**Per-supervision usage** — `SupervisionResult.usage` carries the sum
across attempts; `SupervisionResult.result.usage` remains the final
attempt's usage.

```python
sr = await supervisor.supervise(agent, "task")
total_tokens_across_all_attempts = sr.usage.total_tokens
final_attempt_only = sr.result.usage.total_tokens
```

## Behavior differences

- `StepResult.usage is None` is a distinct, typed sentinel meaning "no
  contribution" (e.g. a `FunctionStep` or a workflow whose every step
  was non-LLM). The old `metadata["usage"]` dict never expressed this
  case cleanly — a `FunctionStep` simply omitted the key, leaving
  consumers to guess between "the step ran but produced no usage" and
  "the key is missing because this version doesn't write it."
- `Sequential.result.usage` and `Pipeline.result.usage` are **aggregated**
  values; they are a new surface, not a migration of an existing one. A
  consumer that previously walked `intermediate_results` to compute the
  sum can now read this field instead.
- On a cancellation mid-flight, `Sequential` and `Pipeline` return a
  `StepResult` whose `usage` reflects only the completed sub-steps —
  what was actually spent before the cancellation token fired.
- On resume from a checkpoint, restored sub-step usages are reconstructed
  from checkpoint state and folded into the final aggregate. Pre-bump
  checkpoints (no `"usage"` key in `completed_results`) resume cleanly —
  restored sub-steps simply contribute `None` to the aggregate.
- `SupervisionResult.usage` differs from `SupervisionResult.result.usage`
  whenever there is more than one attempt. The first is the cross-attempt
  sum, the second is the final attempt only.

## Deprecation timeline

- **0.5.0** — `StepResult.usage` and `SupervisionResult.usage` ship as
  typed fields. `StepResult.metadata["usage"]` continues to be written
  by `AgentStep`, `_BoundAgentStep`, `_BoundHandoffStep`, and
  `HandoffStep`; the deprecation is announced.
- **0.6.0+** — the dict mirror remains in place. Consumers should migrate
  off it during this window.
- **1.0.0** — `StepResult.metadata["usage"]` is removed. `StepResult.usage`
  is the only access path.
