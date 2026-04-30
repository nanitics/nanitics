---
id: prompts-redundant-instructions
severity: observation
category: prompts
target_dimension: prompts
---

# Prompts — Redundant or accreted instructions

## When this fires

The prompt shows signs of accretion — rules that overlap, contradict, or
restate the same constraint in different words. Typical accretion patterns:

- Multiple "always X" clauses where one would suffice.
- Positive and negative forms of the same rule ("always answer in JSON" and
  "never answer in prose").
- Patches that reference specific past failures ("if the user asks about the
  2023 sales report, respond with ...") rather than generalizable criteria.

This is `observation`, not `warning` or `critical`, because the agent may
still behave correctly; the concern is prompt maintainability and the
documented anti-pattern of prompt length growing with every bug fix.

## Evidence to cite

- The full system prompt text from `agent.start`.
- The rubric body SHOULD NOT reproduce the entire prompt in the proposal's
  evidence; cite index and type, and quote the specific redundant fragments
  in the proposal `detail`.

## Proposal template

- **headline** — "System prompt contains redundant / accreted instructions"
  (tighten as appropriate to the observed redundancy).
- **detail** — Quote the overlapping clauses side-by-side and explain the
  accretion pattern. Note that none of the observed redundancy caused the
  failure under analysis — this is maintainability feedback.
- **suggested_action** — Propose a consolidated rewrite that preserves the
  behavioral intent but removes duplication. Where patches reference past
  cases, suggest promoting them to generalizable criteria and dropping the
  case-specific language — repeated case-specific patches are the #1
  anti-pattern in agent-prompt maintenance.
