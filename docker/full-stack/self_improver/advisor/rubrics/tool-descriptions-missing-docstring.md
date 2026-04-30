---
id: tool-descriptions-missing-docstring
severity: critical
category: tool-descriptions
target_dimension: tool_descriptions
---

# Tool descriptions — Missing docstring or empty description

## When this fires

A tool the agent has access to is registered without a description (empty
docstring, empty `description` argument, or a placeholder like "TODO" / "…")
and the trace shows the agent misusing the tool or ignoring it when it was
the right choice. Symptoms:

- Repeated `ToolParameterError` on the tool — the agent guessed at the schema.
- The agent avoids a tool that should have been called for the observed task,
  choosing a worse alternative (or falling back to pure reasoning).
- Tool call arguments that are obviously miscalibrated (e.g., passing a free-
  text query where a structured filter was expected).

This is `critical` because an agent cannot reliably use a tool it cannot
understand. The failure mode is immediate and repeatable.

## Evidence to cite

- The `agent.start` event payload showing the tool schema registered with the
  agent (including the offending empty / placeholder description).
- One or two `tool.call.error` events showing the mis-call pattern.
- An `agent.step` reasoning slice where the agent acknowledges not knowing
  how to use the tool, if present.

## Proposal template

- **headline** — "Tool `<name>` is registered without a usable description".
- **detail** — Name the tool, quote the empty / placeholder description,
  describe the observed misuse (or non-use), and cite the failed call(s).
- **suggested_action** — Propose a concrete docstring that covers: (1) what
  the tool does in one sentence, (2) when the agent should reach for it,
  (3) each parameter's meaning and a minimal example argument. Prefer
  descriptions grounded in the agent's decision model ("use when the user
  asks to X") over purely technical ones ("calls endpoint Y").
