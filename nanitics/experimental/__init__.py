"""Experimental, unstable Nanitics APIs.

Symbols re-exported from ``nanitics.experimental.*`` submodules are
advanced patterns held to a different stability contract than the
committed top-level ``nanitics.__all__`` surface. They may change
without semver-major bumps, and may move or be removed between
minor releases.

If a downstream project depends on something in ``nanitics.experimental.*``,
pin Nanitics to a specific version and read the changelog before
upgrading.

Submodules
----------
- ``nanitics.experimental.strategies`` — advanced agent strategies
  (LATS, Tree of Thought, Reflexion, ReWOO).
- ``nanitics.experimental.coordination`` — advanced multi-agent
  coordination patterns (blackboard, debate, consensus, bidding,
  broadcast, message bus, peer network, judge router, dynamic
  orchestrator).
"""
