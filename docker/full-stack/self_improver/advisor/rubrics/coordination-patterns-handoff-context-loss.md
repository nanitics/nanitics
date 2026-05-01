---
id: coordination-patterns-handoff-context-loss
severity: warning
category: coordination-patterns
target_dimension: coordination_patterns
---

# Coordination patterns — Context lost across handoffs

## When this fires

A `HandoffStep` or handoff chain passes control between agents, but the
receiving agent lacks context the upstream agent had accumulated. Symptoms:

- Receiver agent re-derives facts the sender already established
  (re-running the same tool calls, re-extracting the same entities).
- Receiver asks clarifying questions about the task even though the sender's
  reasoning contains the answer.
- `HandoffPayload` uses a minimal strategy (e.g., `RawOutputTransfer`) when
  the observed handoff content clearly benefited from `SummaryTransfer` or
  `TrajectoryTransfer`.

This is `warning`: the pipeline completes, but it wastes tokens re-
establishing state the sender already had.

## Evidence to cite

- The handoff chain definition (from orchestration start).
- A pair of events straddling the handoff — the last `agent.step` before and
  the first `agent.step` after — showing the context loss.
- Downstream tool calls / reasoning re-doing work the sender already did.

## Proposal template

- **headline** — "Handoff from `<sender>` to `<receiver>` drops trajectory
  context the receiver needs".
- **detail** — Quote the receiver's early reasoning that shows the missing
  context, quote the relevant sender output that should have carried
  through, and identify the current transfer strategy.
- **suggested_action** — Recommend a transfer strategy (by name:
  `SummaryTransfer`, `TrajectoryTransfer`, or a custom `CustomTransfer`)
  with a short rationale. When the choice is non-obvious, list trade-offs
  (trajectory token usage vs. summary fidelity).
