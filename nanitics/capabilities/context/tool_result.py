"""Tool-result policies: bounded tool output, symmetric to ``ContextManagement``.

Defines :class:`ToolResultPolicy` and three default implementations:

- :class:`ErrorOnLargeToolResult` — raises when a result exceeds the budget
  (the recommended default; surfaces the failure through the agent's
  error-handling capability as a correction prompt).
- :class:`TruncateToolResult` — head/tail truncates to fit the budget
  (opt-in data loss).
- :class:`SummarizeToolResult` — LLM-summarizes the result; falls back to
  truncate semantics on failure (opt-in data loss + LLM cost).

The policy is invoked at the single seam in
:meth:`~nanitics.strategies.tools.registry.ToolRegistry.dispatch` after the
tool's ``execute()`` returns a freshly-produced :class:`ToolResult`. It
composes orthogonally with :class:`~nanitics.capabilities.context.ContextManager`:
this layer bounds individual tool results; ``ContextManager`` bounds the
total message list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nanitics.capabilities.context.token_counter import TokenCounter
from nanitics.infrastructure.errors import ToolResultTooLargeError
from nanitics.infrastructure.llm.protocol import LLMClient, Message, ToolCall
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import ToolResultPolicyAppliedEvent

if TYPE_CHECKING:
    # ``ToolResult`` lives in ``strategies`` (a higher layer than this capability
    # module). Importing it at runtime inverts the layering and forms a circular
    # import — importing ``strategies.tools.protocol`` pulls in ``strategies``,
    # whose agent strategies import ``ToolResultPolicy`` back from this module
    # while it is still initializing. The class is needed only in annotations
    # (lazy under ``from __future__ import annotations``) and for the two
    # instantiations below, which import it locally. See the agent strategies'
    # top-level ``ToolResultPolicy`` import for the correct (downward) direction.
    from nanitics.strategies.tools.protocol import ToolResult

DEFAULT_TOOL_SUMMARY_PROMPT = (
    "Summarize the following tool result for an agent that needs to act on it.\n"
    "Preserve concrete data values, identifiers, errors, and the structural shape\n"
    'of the result (e.g. "10 rows", "exit code 1"). Drop verbose framing and\n'
    "repeated patterns. Be concise. Do not invent facts."
)


class ToolResultContext(BaseModel):
    """Read-only context passed to :meth:`ToolResultPolicy.apply`.

    Attributes:
        tool_call: The originating :class:`ToolCall` (name, arguments, id).
        token_counter: Counter used for budget arithmetic. Supplied by the
            agent (defaults to ``EstimateTokenCounter`` when the agent does
            not provide one).
        emitter: Optional event emitter. When non-``None``, policy
            implementations emit :class:`ToolResultPolicyAppliedEvent` for
            observability.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tool_call: ToolCall
    token_counter: TokenCounter
    emitter: EventEmitter | None = None


@runtime_checkable
class ToolResultPolicy(Protocol):
    """Manages the size and shape of a :class:`ToolResult` before it enters the message list.

    Symmetric to :class:`~nanitics.capabilities.context.ContextManager` for
    messages. The agent's :class:`~nanitics.strategies.tools.ToolRegistry`
    calls :meth:`apply` after every tool's ``execute()`` returns a freshly
    produced result and before the result is appended to the message list.
    :meth:`reset` is called by the agent at the start of each run.

    Implementations may:

    - Return ``result`` unchanged when within budget.
    - Return a new :class:`ToolResult` with shortened ``content`` and
      updated ``metadata`` (e.g. ``{"truncated": True, "original_tokens": N}``).
    - Raise :class:`~nanitics.infrastructure.errors.ToolResultTooLargeError`
      (or another :class:`~nanitics.infrastructure.errors.ToolError`
      subclass) to surface a hard failure through the registry's existing
      exception path.

    Implementations MUST NOT return ``None``. The registry only invokes
    :meth:`apply` for results with ``executed=True``; wrapper-suppressed
    results are never passed to a policy.
    """

    async def apply(self, result: ToolResult, context: ToolResultContext) -> ToolResult:
        """Return a (possibly transformed) :class:`ToolResult`."""
        ...

    def reset(self) -> None:
        """Reset any per-run state. Called by the agent at the start of each run."""
        ...


def _emit(
    *,
    emitter: EventEmitter | None,
    tool_call: ToolCall,
    policy_class: str,
    action: str,
    original_tokens: int,
    final_tokens: int,
    fell_back: bool = False,
    error: str | None = None,
) -> None:
    if emitter is None:
        return
    emitter.emit(
        ToolResultPolicyAppliedEvent(
            trace_id=emitter.trace_id,
            span_id=emitter.span_id,
            parent_span_id=emitter.parent_span_id,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            policy_class=policy_class,
            action=action,  # type: ignore[arg-type]
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            fell_back=fell_back,
            error=error,
        )
    )


class ErrorOnLargeToolResult:
    """Raises :class:`ToolResultTooLargeError` when a tool result exceeds ``max_tokens``.

    The recommended default. Surfaces the failure through the registry's
    existing :class:`~nanitics.infrastructure.errors.ToolError` exception
    path so the agent's
    :class:`~nanitics.capabilities.errors.ErrorHandler` converts it into a
    correction prompt for the LLM, exactly like any other ``ToolError``.

    Use this first. Reach for :class:`TruncateToolResult` or
    :class:`SummarizeToolResult` only when surfacing the failure is not
    actionable (e.g. you have no way to reformulate the tool call).
    """

    def __init__(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    async def apply(self, result: ToolResult, context: ToolResultContext) -> ToolResult:
        tokens = context.token_counter.count_text(result.content)
        if tokens <= self._max_tokens:
            return result
        _emit(
            emitter=context.emitter,
            tool_call=context.tool_call,
            policy_class=type(self).__name__,
            action="errored",
            original_tokens=tokens,
            final_tokens=0,
            error=f"result exceeded budget ({tokens} > {self._max_tokens} tokens)",
        )
        raise ToolResultTooLargeError(
            f"Tool '{context.tool_call.name}' returned {tokens} tokens; budget is {self._max_tokens}.",
            tool_name=context.tool_call.name,
            result_tokens=tokens,
            max_tokens=self._max_tokens,
        )

    def reset(self) -> None:
        """No-op; this impl has no per-run state."""


def _slice_to_budget(
    content: str,
    *,
    max_tokens: int,
    head_tokens: int | None,
    marker: str,
    counter: TokenCounter,
) -> str:
    """Head/tail slice ``content`` to fit ``max_tokens``.

    When ``head_tokens`` is ``None``, keeps the trailing portion. When
    set, keeps ``head_tokens`` from the front and the remaining budget
    from the tail, joined by ``marker``.

    When the marker alone exceeds ``max_tokens``, falls back to a flat
    tail slice with no marker.
    """
    marker_tokens = counter.count_text(marker)
    if marker_tokens >= max_tokens:
        # Marker won't fit — flat tail slice, no marker.
        chars_per_token = max(1, len(content) // max(1, counter.count_text(content)))
        return content[-(max_tokens * chars_per_token) :]

    # Use the content's own char/token ratio so the slice respects the
    # counter's measurement (the abstraction tests do not assume any
    # particular tokenizer).
    total_tokens = max(1, counter.count_text(content))
    chars_per_token = max(1, len(content) // total_tokens)
    tail_budget = max_tokens - marker_tokens - (head_tokens or 0)
    tail_budget = max(0, tail_budget)
    if head_tokens is None:
        # Tail-only slice: keep the trailing ``max_tokens - marker_tokens``.
        keep_tokens = max_tokens - marker_tokens
        keep_chars = keep_tokens * chars_per_token
        return f"{marker}{content[-keep_chars:]}"
    head_chars = head_tokens * chars_per_token
    tail_chars = tail_budget * chars_per_token
    if tail_chars <= 0:
        return f"{content[:head_chars]}{marker}"
    return f"{content[:head_chars]}{marker}{content[-tail_chars:]}"


class TruncateToolResult:
    """Head/tail truncates ``result.content`` to fit ``max_tokens``.

    Opt-in data-loss behavior. When the content fits, returns it unchanged.
    When it does not, slices to keep ``head_tokens`` from the front (if
    set) and the remaining budget from the tail, joined by ``marker``.
    When ``head_tokens`` is ``None`` (the default), only the trailing
    portion is kept — LLMs most often need the tail of a tool result
    (recent stderr, last log lines).

    The returned :class:`ToolResult` carries ``metadata["truncated"] = True``
    and ``metadata["original_tokens"] = <n>`` in addition to all of the
    original tool's metadata keys.

    Edge case: when the marker alone exceeds ``max_tokens``, the slice
    falls back to a flat tail slice with no marker.
    """

    def __init__(
        self,
        max_tokens: int,
        *,
        head_tokens: int | None = None,
        marker: str = "[…truncated…]",
    ) -> None:
        self._max_tokens = max_tokens
        self._head_tokens = head_tokens
        self._marker = marker

    async def apply(self, result: ToolResult, context: ToolResultContext) -> ToolResult:
        from nanitics.strategies.tools.protocol import ToolResult

        tokens = context.token_counter.count_text(result.content)
        if tokens <= self._max_tokens:
            return result
        sliced = _slice_to_budget(
            result.content,
            max_tokens=self._max_tokens,
            head_tokens=self._head_tokens,
            marker=self._marker,
            counter=context.token_counter,
        )
        new_tokens = context.token_counter.count_text(sliced)
        merged_metadata = {**result.metadata, "truncated": True, "original_tokens": tokens}
        new_result = ToolResult(content=sliced, metadata=merged_metadata, executed=result.executed)
        _emit(
            emitter=context.emitter,
            tool_call=context.tool_call,
            policy_class=type(self).__name__,
            action="truncated",
            original_tokens=tokens,
            final_tokens=new_tokens,
        )
        return new_result

    def reset(self) -> None:
        """No-op; this impl has no per-run state."""


class SummarizeToolResult:
    """LLM-summarizes ``result.content`` when it exceeds ``max_tokens``.

    Opt-in data-loss behavior. Calls ``llm_client.generate(...)`` with
    ``summary_prompt`` as system prompt and the original content as a
    user message. The returned :class:`ToolResult` carries the summary
    as ``content`` and ``metadata["summarized"] = True``,
    ``metadata["original_tokens"] = <n>``.

    Fallback semantics: when the LLM call raises any exception or when
    the resulting summary is still over budget, falls back to
    :class:`TruncateToolResult` semantics. The fallback marks the result
    with ``metadata["summarized"] = False``, ``metadata["truncated"] = True``,
    and ``metadata["fell_back"] = True``, and emits
    :class:`ToolResultPolicyAppliedEvent` with ``action="truncated"`` and
    ``fell_back=True``.
    """

    def __init__(
        self,
        max_tokens: int,
        llm_client: LLMClient,
        *,
        summary_prompt: str = DEFAULT_TOOL_SUMMARY_PROMPT,
    ) -> None:
        self._max_tokens = max_tokens
        self._llm_client = llm_client
        self._summary_prompt = summary_prompt

    async def apply(self, result: ToolResult, context: ToolResultContext) -> ToolResult:
        from nanitics.strategies.tools.protocol import ToolResult

        tokens = context.token_counter.count_text(result.content)
        if tokens <= self._max_tokens:
            return result
        summary: str | None = None
        error: str | None = None
        try:
            response = await self._llm_client.generate(
                system_prompt=self._summary_prompt,
                messages=[Message(role="user", content=result.content)],
            )
            summary = response.content
        except Exception as exc:
            error = str(exc) or type(exc).__name__

        if summary is not None and error is None:
            summary_tokens = context.token_counter.count_text(summary)
            if summary_tokens <= self._max_tokens:
                merged_metadata = {
                    **result.metadata,
                    "summarized": True,
                    "original_tokens": tokens,
                }
                _emit(
                    emitter=context.emitter,
                    tool_call=context.tool_call,
                    policy_class=type(self).__name__,
                    action="summarized",
                    original_tokens=tokens,
                    final_tokens=summary_tokens,
                )
                return ToolResult(
                    content=summary,
                    metadata=merged_metadata,
                    executed=result.executed,
                )
            error = f"summary tokens {summary_tokens} still exceeds budget {self._max_tokens}"

        # Fallback: truncate.
        sliced = _slice_to_budget(
            result.content,
            max_tokens=self._max_tokens,
            head_tokens=None,
            marker="[…truncated…]",
            counter=context.token_counter,
        )
        sliced_tokens = context.token_counter.count_text(sliced)
        merged_metadata = {
            **result.metadata,
            "summarized": False,
            "truncated": True,
            "fell_back": True,
            "original_tokens": tokens,
        }
        _emit(
            emitter=context.emitter,
            tool_call=context.tool_call,
            policy_class=type(self).__name__,
            action="truncated",
            original_tokens=tokens,
            final_tokens=sliced_tokens,
            fell_back=True,
            error=error,
        )
        return ToolResult(content=sliced, metadata=merged_metadata, executed=result.executed)

    def reset(self) -> None:
        """No-op; unlike :class:`~nanitics.capabilities.context.SummarizationPolicy`,
        there is no carried delta state."""
