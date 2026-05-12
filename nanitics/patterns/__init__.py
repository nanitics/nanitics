"""Named compositions over Nanitics primitives.

This namespace holds primitives that are conveniences over the core
surface — orchestrator factories, structured handoff helpers — rather
than structurally distinct primitives. Every symbol here is composition;
you could rebuild it yourself from the primitives in :mod:`nanitics`.

The factories are useful when the pattern is overwhelmingly common and
the boilerplate is real. Reach for them when they fit; reach for the
underlying primitives in :mod:`nanitics` when they do not.

What lives here:

* :class:`HandoffPayload`, :class:`HandoffTransfer`, :class:`HandoffStep`,
  :func:`create_handoff_chain`, :func:`handoff_sender_instructions`,
  :func:`handoff_receiver_instructions` — structured handoff stack over
  the raw context-transfer strategies in :mod:`nanitics`.
* :func:`create_orchestrator`, :func:`orchestrator_prompt_section`,
  :class:`FinalOutputStrategy` — coordinator factory over
  :class:`nanitics.ReActAgent` + :class:`nanitics.AgentTool`.
"""

from nanitics.composition import (
    FinalOutputStrategy,
    HandoffPayload,
    HandoffStep,
    HandoffTransfer,
    create_handoff_chain,
    create_orchestrator,
    handoff_receiver_instructions,
    handoff_sender_instructions,
    orchestrator_prompt_section,
)

__all__ = [
    "FinalOutputStrategy",
    "HandoffPayload",
    "HandoffStep",
    "HandoffTransfer",
    "create_handoff_chain",
    "create_orchestrator",
    "handoff_receiver_instructions",
    "handoff_sender_instructions",
    "orchestrator_prompt_section",
]
