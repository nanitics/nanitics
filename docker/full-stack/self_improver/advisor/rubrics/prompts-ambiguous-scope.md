---
id: prompts-ambiguous-scope
severity: warning
category: prompts
target_dimension: prompts
---

# Prompts — Ambiguous scope boundary

## When this fires

The prompt under-specifies what is in scope vs. out of scope for the agent,
and the trace shows the agent drifting into adjacent work. Symptoms:

- Agent answers correctly but also answers related questions the user never
  asked — e.g., a "classify this email" agent also summarizes the email body.
- Tool calls touch resources or tables outside the agent's stated purpose.
- The agent asks clarifying questions that suggest it does not know where its
  responsibilities end.

This is `warning`, not `critical`, because the agent usually completes the
requested task — scope drift produces noisy outputs but does not by itself
cause wrong answers.

## Evidence to cite

- The system prompt text from `agent.start`.
- One or two trace slices showing the off-scope work (tool calls against
  unrelated resources, reasoning turns on unrelated subproblems).
- The final `agent.complete` output, showing the scope drift surfaced to the
  caller.

## Proposal template

- **headline** — Concise naming of the drift (e.g., "Agent drifts into
  summarization when the task is classification only").
- **detail** — Quote the ambiguous sentence(s) from the prompt, describe the
  observed drift with trace references, and note the output-clarity impact.
- **suggested_action** — Propose a scope stanza for the prompt: what the agent
  is responsible for, and an explicit list of things the agent should NOT do
  (e.g., "Do not summarize the email body. Do not propose follow-up actions.").
  Prefer negative clauses that name the specific drifts observed.
