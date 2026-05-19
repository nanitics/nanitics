"""Real-Anthropic smoke validation for ``self_improver.advisor.analyze``.

Feeds the frozen ``smoke_react_agent`` trace envelope into ``analyze()``
against a real Anthropic client. The assertions pin the shape of the
report — trace id, dimensions analyzed, proposal schema, positive usage —
without asserting proposal content (the smoke trace is deliberately bland
and may not fire any rubric).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from self_improver.advisor import MarkdownFormatter, Proposal, analyze

from nanitics.tracing import InMemoryEmitter
from validation.helpers import make_llm_client

FIXTURE = Path(__file__).parent / "fixtures" / "smoke_react_agent.json"


@pytest.mark.quick
async def test_advise_anthropic(traced_emitter: InMemoryEmitter) -> None:
    client = make_llm_client("anthropic")
    report = await analyze(FIXTURE, llm_client=client, emitter=traced_emitter)

    assert report.trace_id == "validation/smoke/smoke.py::test_smoke_react_agent"
    assert report.target_dimensions_analyzed == [
        "prompts",
        "tool_descriptions",
        "coordination_patterns",
    ]
    assert isinstance(report.proposals, list)
    for proposal in report.proposals:
        assert isinstance(proposal, Proposal)
    assert report.usage.input_tokens > 0
    assert report.usage.output_tokens > 0

    markdown = MarkdownFormatter().render(report)
    assert markdown.startswith("# Advisor Report"), markdown[:64]
