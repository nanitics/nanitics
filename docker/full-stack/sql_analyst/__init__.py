"""Self-healing SQL analyst showcase runner.

Bundles a small analyst schema (five tables), deterministic seed data,
a :class:`~nanitics.Supervisor`-driven ``ReActAgent`` that writes SQL
against the schema, and a :class:`GroundTruthEvaluator` that compares
the agent's produced value against hand-computed canonical answers —
no LLM-as-judge.

See ``docker/full-stack/sql_analyst/README.md`` for the full pattern
and the sample-question catalog.
"""
