---
id: coordination-patterns-no-termination
severity: critical
category: coordination-patterns
target_dimension: coordination_patterns
---

# Coordination patterns — Missing or ineffective termination condition

## When this fires

A multi-agent composition (blackboard, message bus, peer network, debate, or
loop) runs without a working termination condition, or the termination condition
is defined but never satisfied under realistic inputs. Symptoms:

- A `blackboard.round.end` or equivalent event stream that terminates on
  `MaxRoundsTermination` / `MaxMessagesTermination` every run, rather than on
  the intended predicate.
- Participant agents producing near-identical or repetitive contributions in
  late rounds with no new information.
- `BusCompositeTermination` / `BlackboardCompositeTermination` configured
  with conditions that are never reachable given the participants' outputs.

This is `critical` because failure to terminate burns budget and emits
low-quality output (the last few rounds add noise to whatever signal the
composition accumulated).

## Evidence to cite

- The composition configuration (visible in the orchestration start event).
- The sequence of round / message events showing convergence has stalled.
- The terminal event — ideally the intended termination predicate reference.

## Proposal template

- **headline** — "Blackboard / bus / peer network terminates on round cap,
  not on the intended predicate".
- **detail** — Name the composition, quote the configured termination
  condition(s), describe how the trace shows the predicate never fires.
- **suggested_action** — Propose a concrete predicate or composite termination
  grounded in the observed trace — for example, a `NoNewContributions`
  detector over the last N rounds, or a `BlackboardPredicateTermination` that
  checks a specific shared-state invariant. Where appropriate, suggest
  revisiting the orchestration choice (blackboard vs. pipeline) if
  termination turns out to be inherently hard for this workload.
