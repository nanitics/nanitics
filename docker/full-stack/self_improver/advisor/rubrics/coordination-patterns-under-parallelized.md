---
id: coordination-patterns-under-parallelized
severity: observation
category: coordination-patterns
target_dimension: coordination_patterns
---

# Coordination patterns — Sequential steps that could run in parallel

## When this fires

A `Sequential` or `Pipeline` composition executes independent steps in order
even though the step dependencies form an antichain (no data flows between
them). Symptoms:

- Wall-clock latency matches the sum of per-step latencies when it could
  have matched the max.
- Step outputs feed only the coordinator's final aggregation, never each
  other.

This is `observation`, not `warning`, because correctness is fine; the
concern is latency. Adopters with a strict latency budget benefit most
from this proposal.

## Evidence to cite

- The composition definition (from orchestration start).
- The span boundaries of the sequential steps (start / end timestamps) — the
  advisor can show the observed total vs. the hypothetical parallel total.
- The absence of data flow between steps (each step's inputs are the
  coordinator's input, not a prior step's output).

## Proposal template

- **headline** — "Steps <A>, <B> run sequentially but are data-independent".
- **detail** — Name the steps, show the sequential wall-clock latency from
  the trace, and quote the configuration that pins them to `Sequential`.
- **suggested_action** — Suggest replacing the sequential block with
  `Parallel` (or `MapReduce` where a fan-out / aggregate shape applies),
  noting the expected latency improvement and any error-handling nuance
  introduced by concurrency (e.g., failure policy becomes relevant).
