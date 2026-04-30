---
id: tool-descriptions-naming-drift
severity: observation
category: tool-descriptions
target_dimension: tool_descriptions
---

# Tool descriptions — Implementation-leaking names

## When this fires

Tool names reflect their implementation rather than their use — e.g., names
embed provider/vendor identifiers (`openai_retrieval_v2`), internal module
paths (`storage_backend_query`), or legacy identifiers that no longer match
the tool's current role. The trace shows no failure, but the agent spends
reasoning turns justifying why the oddly-named tool is the right choice, or
the docstring has to compensate with extensive preamble.

This is `observation`: the agent works, but adopter-facing ergonomics and
long-term maintainability degrade as the tool surface grows.

## Evidence to cite

- The registered tool name and docstring from `agent.start`.
- A reasoning trace slice (if present) where the agent notes confusion about
  the name or explicitly translates the name to the task it's solving.

## Proposal template

- **headline** — "Tool name `<current>` leaks implementation detail".
- **detail** — Quote the current name and any docstring preamble compensating
  for it. Note whether any observed trace reasoning reflects the confusion.
- **suggested_action** — Propose a task-oriented rename (e.g.,
  `retrieve_customer_records` instead of `openai_retrieval_v2`) with a brief
  rationale. Flag the rename as a breaking change for adopters and suggest
  documenting it in the changelog.
