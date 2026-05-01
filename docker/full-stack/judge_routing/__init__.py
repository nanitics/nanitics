"""Judge-routed request handling showcase runner.

Four specialist agents are routed by a single comparative-judgment LLM
call via :class:`~nanitics.JudgeRouter`; the winning
:class:`~nanitics.ReActAgent` answers using a small in-memory fixture
backing real tool calls (lookup, mutation, search). This is the
counterpart to ``auction-routing``: same specialist roster shape, same
calibrated cost grounding, but the routing decision is centralised so
the judge can compare candidates against each other in one prompt.

See ``docker/full-stack/judge_routing/README.md`` for the pattern,
endpoints, and Observatory trace shape.
"""
