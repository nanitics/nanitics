---
id: tool-descriptions-unclear-parameters
severity: warning
category: tool-descriptions
target_dimension: tool_descriptions
---

# Tool descriptions — Unclear parameter semantics

## When this fires

The tool has a description but individual parameters are under-documented,
and the trace shows the agent picking wrong parameter values or omitting
optional parameters that should have been set. Symptoms:

- Tool calls succeed syntactically but the returned results are less useful
  than they could have been (e.g., `limit=10` when the agent needed 100).
- Default values quietly carry the wrong behavior for the agent's task.
- Enum-typed parameters are called with stringy free-text values that happen
  to validate but miss the intent.

This is `warning`, not `critical`, because the tool calls work — just
suboptimally. The failure is signal loss, not an outright error.

## Evidence to cite

- The tool's registered parameter schema (visible in `agent.start`).
- One or two `tool.call` events showing the sub-optimal arguments.
- The downstream `tool.call.result` events — the thin / wrong-shape result
  that triggered the finding.

## Proposal template

- **headline** — Name the tool and the specific parameter (e.g., "Tool
  `search_knowledge_base` parameter `limit` is under-documented").
- **detail** — Quote the current parameter description (or note its absence),
  describe the observed miscalibration with trace references, and note the
  downstream impact (re-calls, thin results, irrelevant context).
- **suggested_action** — Propose a parameter docstring that explicitly
  describes: typical values, default behavior, interaction with other
  parameters, and an example reflecting a realistic agent call.
