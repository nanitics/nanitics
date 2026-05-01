---
id: prompts-missing-exit-criteria
severity: critical
category: prompts
target_dimension: prompts
---

# Prompts — Missing exit criteria

## When this fires

The agent's system prompt does not define an explicit termination condition,
and the trace shows the agent looping beyond a reasonable step count, hitting
the iteration limit, or emitting repeated "let me try again" reasoning turns
without convergence. Symptoms include:

- A trace that terminates on `AgentIterationLimitError` rather than on the
  agent's own final answer.
- Repeated tool calls to the same tool with near-identical arguments late in
  the run.
- Reasoning text that oscillates between "I have enough information" and "I
  should gather more" across consecutive turns.

## Evidence to cite

Prefer the smallest set of events that make the loop visible:

- The `agent.start` event carrying the system prompt text (so the reader can
  verify no exit clause is present).
- Two or three `agent.step` events from the later iterations showing the
  oscillation or redundant tool use.
- The terminal event — whichever of `agent.complete`,
  `AgentIterationLimitError`, or `AgentBudgetExceededError` fired.

Cite event indices precisely; a proposal about "missing exit criteria" that
cannot point to a looping late-run trace slice is weak evidence.

## Proposal template

- **headline** — "System prompt lacks an explicit termination condition"
  (or a closer variant grounded in the specific trace).
- **detail** — Name the agent, summarize the symptom, quote the relevant
  prompt snippet, and describe what the trace showed. Two short paragraphs.
- **suggested_action** — Propose concrete termination language: a "Stop when
  ..." clause tied to the agent's goal (e.g., "Stop when you have produced a
  JSON object with fields `foo` and `bar`, even if you have unanswered
  sub-questions"). Where possible, derive the clause from the agent's actual
  output contract rather than prescribing boilerplate.
