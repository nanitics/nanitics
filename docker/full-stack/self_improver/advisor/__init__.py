"""Advisory optimization system — runtime, schema, rubric corpus, and ranking.

Exports the full advisor surface: the public :func:`analyze` entry point and
its :class:`AdvisorReport` data model, the trace adapter protocol and default
Nanitics implementation, the output-formatter protocol with JSON and Markdown
defaults, the proposal data model, the rubric corpus loader, and the ranking
primitive.
"""

from self_improver.advisor.analyze import AdvisorReport, analyze, write_report
from self_improver.advisor.formatters import (
    JSONFormatter,
    MarkdownFormatter,
    OutputFormatter,
)
from self_improver.advisor.proposal import (
    EvidenceReference,
    Proposal,
    ProposalCategory,
    ProposalSeverity,
    RubricSource,
)
from self_improver.advisor.ranking import rank_proposals
from self_improver.advisor.rubric import (
    DuplicateRubricError,
    MalformedRubricError,
    Rubric,
    RubricFileNameMismatchError,
    load_rubrics,
)
from self_improver.advisor.trace_adapter import (
    MalformedTraceError,
    NaniticsTraceAdapter,
    TraceAdapter,
    load_trace,
)

__all__ = [
    "AdvisorReport",
    "DuplicateRubricError",
    "EvidenceReference",
    "JSONFormatter",
    "MalformedRubricError",
    "MalformedTraceError",
    "MarkdownFormatter",
    "NaniticsTraceAdapter",
    "OutputFormatter",
    "Proposal",
    "ProposalCategory",
    "ProposalSeverity",
    "Rubric",
    "RubricFileNameMismatchError",
    "RubricSource",
    "TraceAdapter",
    "analyze",
    "load_rubrics",
    "load_trace",
    "rank_proposals",
    "write_report",
]
