# Vision: Nanitics

Nanitics is the open-source Python SDK behind [Propodeum](https://propodeum.com)'s production client engagements. It provides the building blocks — agent types, memory, planning, orchestration, multi-agent coordination, evaluation, human-in-the-loop, observability — that a developer composes into an intelligent application. The SDK has no server, no database, and no web framework; it is a pure library that runs inside whatever Python application you build on top of it.

The SDK is young. It matures through the process of building real applications with real clients, not through speculative design. If you are evaluating Nanitics or adopting it to build your own system, this document is the short version of what the project is for and how it evolves.

## SDK Maturity Model

The SDK is done when it has built multiple reliable applications without needing alterations. Until then:

- Every application is a test of the SDK's design.
- Gaps and friction discovered during application development drive SDK evolution.
- Breaking changes are expected and acceptable pre-1.0 — call sites are updated in the same change.
- The SDK's public API surface should be intentional, not accumulated. `nanitics.__all__` is the authoritative public surface; everything else is internal.

Maturity comes from use, not from feature count.

## Documentation Philosophy

Documentation serves developers adopting the SDK. Two commitments follow from that:

- **Accuracy over comprehensiveness.** Stale docs are worse than no docs — they cause adopters to build on false assumptions. A smaller, accurate documentation set beats a comprehensive stale one.
- **One source of truth per concept.** Docstrings in the source tree carry API details (signatures, fields, constraints). Examples carry runnable usage patterns. Guides under `docs/guides/` carry decision guidance — when to reach for a feature, how to choose between two similar ones. The three layers do not duplicate each other.

When something in the SDK changes, the docstring changes in the same commit, and guides that describe the affected concept are updated alongside it. Drift between code and docs is the failure mode this project treats as a bug.
