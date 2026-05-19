"""Reusable tool fixtures for validation scripts.

These tools are canned stand-ins for third-party services — their value
is that they raise deterministically, so a validation test can pin the
agent's recovery behavior under a real LLM without depending on a real
external service being flaky at the right moment.
"""

from __future__ import annotations

from nanitics.strategies import (
    Tool,
    tool,
)


def make_failing_tool(
    *,
    name: str,
    description: str,
    message: str = "tool unavailable",
    exc_cls: type[Exception] = RuntimeError,
) -> Tool:
    """Build a tool that raises ``exc_cls(message)`` on every invocation.

    The tool accepts a single ``query`` string parameter (so an LLM is
    free to pass arbitrary text). Pair with a healthy tool to prove that
    an agent pivots away from a failing branch.

    Args:
        name: Tool name exposed to the LLM.
        description: Tool description — write this so the LLM plausibly
            considers invoking the tool for the task at hand.
        message: Exception message. Surfaces via
            ``ActionNode.error_message`` and can be asserted on.
        exc_cls: Exception class to raise. Default ``RuntimeError``.

    Returns:
        A :class:`Tool` suitable for passing to any agent's ``tools=``.
    """

    @tool(name, description)
    async def _always_failing(query: str) -> str:
        del query
        raise exc_cls(message)

    return _always_failing
