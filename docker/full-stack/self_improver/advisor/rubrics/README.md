# Advisor Rubric Corpus

This directory is the source of truth for the advisor's classification taxonomy
and proposal authoring guidance. Each rubric file describes one concern the
advisor can surface.

## File format

Every rubric file is UTF-8 Markdown with a required YAML frontmatter block.
Filename must equal `{id}.md` where `id` matches the frontmatter `id`.

```markdown
---
id: prompts-ambiguous-exit-criteria
severity: warning
category: prompts
target_dimension: prompts
---

# Prompts — Ambiguous exit criteria

## When this fires
...

## Evidence to cite
...

## Proposal template
...
```

### Frontmatter schema

| Key                | Values                                                                                                                                      |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `id`               | Globally unique kebab-case identifier across the builtin corpus **and** adopter-authored corpora. Duplicates raise `DuplicateRubricError`. |
| `severity`         | `critical` \| `warning` \| `observation`. Ordering first-class axis in `rank_proposals`.                                                    |
| `category`         | `prompts` \| `tool-descriptions` \| `coordination-patterns` \| `agent-strategy` \| `iteration-budgets` \| `application-logic` \| `configuration` \| `sdk` \| `evaluation` \| `observability` |
| `target_dimension` | `prompts` \| `tool_descriptions` \| `coordination_patterns` \| `agent_strategy` \| `iteration_budgets`                                       |

Unknown frontmatter keys are ignored with no warning — this is deliberate forward-compat.
Missing required keys raise `MalformedRubricError`.

### Body

Free-form Markdown passed verbatim to the specialist agents as LLM context
(not executable logic). A rubric body SHOULD contain, in this order:

1. **When this fires** — trace patterns or signals that indicate the rubric applies.
2. **Evidence to cite** — which event types and fields the specialist should pull
   into the proposal's `evidence` list.
3. **Proposal template** — guidance on what the `headline`, `detail`, and
   `suggested_action` fields should contain.

The body is LLM context; it is not parsed for structure. Bodies may include
additional sections freely.

## Authoring rules

- **Filename matches `id`.** Mismatches raise `RubricFileNameMismatchError`.
- **Frontmatter is required.** Missing or malformed frontmatter raises
  `MalformedRubricError` with the offending file path.
- **`id` uniqueness.** Every shipped `rubric_id` is globally unique. Adopters
  authoring custom rubrics must pick ids that do not collide with any builtin;
  a principled naming convention (e.g., `<org>-<concern>`) prevents collisions
  by construction.
- **No executable logic.** Body text is LLM reasoning context, not a DSL.

## Adopter extension

Adopters authoring custom rubrics place their own `*.md` files in any directory
they control and point the advisor at them:

```python
from pathlib import Path
from self_improver.advisor import load_rubrics

rubrics = load_rubrics(paths=[Path("./my_rubrics")])
```

Custom rubrics are loaded alongside builtins, ranked on equal terms, and carry
`rubric_source=custom` in every proposal they produce — adopter concerns are
not second-class.

To load only adopter rubrics (for focused testing), pass
`include_builtins=False`:

```python
rubrics = load_rubrics(paths=[Path("./my_rubrics")], include_builtins=False)
```

## Current launch corpus

The current corpus covers three target dimensions:

- `prompts` — system prompts, user prompts, and dynamic context injection.
- `tool_descriptions` — tool names, docstrings, and parameter descriptions.
- `coordination_patterns` — multi-agent orchestration, handoffs, and termination.

Rubrics for the remaining `target_dimension` values (`agent_strategy`,
`iteration_budgets`) are intentionally deferred — the enum reserves the slots
so adding their specialists post-launch is additive, not a breaking schema
change.
