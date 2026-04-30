"""Retrospective self-improver showcase runner.

A deliberately-imperfect :class:`~nanitics.ReActAgent` runs a small
research task against a bundled markdown corpus; the SDK reads that
trace back and hands it to :func:`self_improver.advisor.analyze`, whose
ranked proposals surface prompt-, tool-, and coordination-level
critiques. The critic run is itself wrapped in
``TracedExecutor.execute(...)`` so "trace of trace" shows up in the
Observatory.

See ``docker/full-stack/self_improver/README.md`` for the full pattern
and the two request-body modes (task mode, referenced-trace mode).
"""
