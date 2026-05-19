"""Nanitics — Python SDK for single-agent and multi-agent AI systems.

The public surface is **hierarchical**: every name lives in a topic-named
subpackage. The top-level :mod:`nanitics` package exports only
:data:`__version__`. Import from the subpackages directly — there are no
flat re-exports.

Public subpackages
------------------

- :mod:`nanitics.strategies` — agent strategies and the foundational
  ``Agent`` / ``Tool`` / ``SystemPromptBuilder`` primitives (``ReActAgent``,
  ``ReasoningAgent``, ``CodeActAgent``, plus the specialized strategies
  re-exported from :mod:`nanitics.specialized`).
- :mod:`nanitics.memory` — working, shared, semantic, episodic, and
  long-term memory stores.
- :mod:`nanitics.composition` — multi-agent foundations, workflows
  (``Sequential``, ``Parallel``, ``DAG``), ``AgentTool``, ``Broadcast``,
  ``Blackboard``, ``Supervisor``, ``JudgeRouter``, and the durable-run
  / checkpointing stack.
- :mod:`nanitics.tracing` — events, emitters, trace stores, and level
  filtering for the Observatory.
- :mod:`nanitics.errors` — error classes and the error-handling
  capability surface.
- :mod:`nanitics.hitl` — human-in-the-loop primitives: approval,
  revision, and human-input providers.
- :mod:`nanitics.evaluation` — output evaluators, verdicts, and
  contexts.
- :mod:`nanitics.planning` — goal- and plan-based planning primitives.
- :mod:`nanitics.context` — context management: token counting,
  summarization, and truncation.
- :mod:`nanitics.safety` — cancellation, iteration limits, sandboxes.
- :mod:`nanitics.tools` — curated reference :class:`Tool` implementations.
- :mod:`nanitics.infrastructure` — LLM and embedding clients
  (``AnthropicLLMClient``, ``OpenAILLMClient``, ``LiteLLMClient``,
  ``MockLLMClient``, ``VoyageEmbeddingClient``, ``MockEmbeddingClient``).
- :mod:`nanitics.patterns` — named compositions over the core primitives
  (``create_orchestrator``, ``HandoffPayload`` / ``HandoffStep`` /
  ``create_handoff_chain``). Adoption-guidance namespace, not maturity.
- :mod:`nanitics.specialized` — specialized primitives that are
  structurally distinct but niche (``ReWOOAgent``, ``ReflexionAgent``,
  ``TreeOfThoughtAgent``, ``LATSAgent``, ``Loop`` / ``Conditional`` /
  ``MapReduce`` / ``Pipeline`` workflows, ``Bidding`` / ``Debate`` /
  ``Consensus`` coordination, ``MessageBus``, ``PeerNetwork``,
  ``MistralLLMClient``, hierarchical-decomposition planning).
  Adoption-guidance namespace, not maturity.

The union of every public subpackage's ``__all__`` is the authoritative
public surface (see ``docs/deprecation-policy.md``).
"""

from importlib.metadata import version as _metadata_version

__version__ = _metadata_version("nanitics")

__all__ = ["__version__"]
