from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from nanitics.infrastructure.errors import AgentIterationLimitError
from nanitics.infrastructure.llm.protocol import LLMClient, Message, ToolCall, ToolSchema
from nanitics.infrastructure.observability.emitter import EventEmitter
from nanitics.infrastructure.observability.events import (
    CodeExecutionEvent,
    CodeExecutionResultEvent,
    ToolInfo,
    Usage,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.safety.iteration_limits import IterationLimiter
from nanitics.safety.sandbox.protocol import ExecutionResult, Sandbox
from nanitics.strategies.agents.context import ContextManagement, ContextProvider
from nanitics.strategies.agents.errors import ErrorHandling
from nanitics.strategies.agents.evaluation import EvaluationVerdict, OutputEvaluator
from nanitics.strategies.prompts.builder import SystemPromptContributor
from nanitics.strategies.tools import Tool, ToolRegistry

from .base import Agent, AgentInput, AgentResult

# JSON Schema type → Python type annotation mapping
_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

_EXECUTE_CODE_SCHEMA = ToolSchema(
    name="execute_code",
    description=(
        "Execute Python code in a persistent sandbox environment. "
        "Use this tool to run code and observe results. Variables and imports persist between calls."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            }
        },
        "required": ["code"],
    },
)

_CODE_EXECUTION_INSTRUCTIONS = """\
## Code Execution Environment

You have access to a Python execution environment via the `execute_code` tool. \
Use it to run code and observe results.

After each execution, you will see the output (stdout, return value, or error \
traceback) as an observation. Use this feedback to refine your approach.

**Guidelines:**
- Variables and imports persist between executions within the same session.
- Write complete, runnable code — not pseudocode or fragments.
- Use print() to output intermediate results you want to observe.
- The last expression in your code will be captured as a return value.
- If code produces an error, you will see the traceback. Use it to fix and retry.
- When you have the final answer, respond with text only (do not call execute_code)."""


def _python_type(prop: dict[str, Any]) -> str:
    """Convert a JSON Schema property to a Python type annotation string."""
    json_type = prop.get("type", "")
    if isinstance(json_type, list):
        # e.g. ["string", "null"]
        types = [_TYPE_MAP.get(t, "Any") for t in json_type if t != "null"]
        base = types[0] if len(types) == 1 else f"{' | '.join(types)}"
        if "null" in json_type:
            return f"{base} | None"
        return base
    return _TYPE_MAP.get(json_type, "Any")


def _generate_single_stub(schema: ToolSchema) -> str:
    """Generate a Python function stub for a single tool schema.

    The generated function calls ``__call_tool__(name, args)`` which is
    injected into the sandbox namespace by the runner script.
    """
    properties: dict[str, Any] = schema.parameters.get("properties", {})
    required: list[str] = schema.parameters.get("required", [])

    # Build parameter list: required params first, then optional with defaults
    required_params: list[str] = []
    optional_params: list[str] = []

    for param_name, prop in properties.items():
        py_type = _python_type(prop)
        if param_name in required:
            required_params.append(f"{param_name}: {py_type}")
        else:
            default = prop.get("default", None)
            default_repr = repr(default)
            optional_params.append(f"{param_name}: {py_type} = {default_repr}")

    all_params = required_params + optional_params
    params_str = ", ".join(all_params)

    # Build docstring
    doc_lines = [schema.description]
    if properties:
        doc_lines.append("")
        doc_lines.append("Args:")
        for param_name, prop in properties.items():
            desc = prop.get("description", "")
            doc_lines.append(f"    {param_name}: {desc}")

    docstring = "\n    ".join(doc_lines)

    # Build args dict for __call_tool__
    arg_names = list(properties.keys())
    args_dict = ", ".join(f'"{n}": {n}' for n in arg_names)

    lines = [
        f"def {schema.name}({params_str}) -> str:",
        f'    """{docstring}"""',
        f'    return __call_tool__("{schema.name}", {{{args_dict}}})',
        "",
    ]
    return "\n".join(lines)


def generate_tool_stubs(schemas: list[ToolSchema]) -> str:
    """Generate Python code that defines callable tool functions.

    The generated code assumes ``__call_tool__`` is already defined in the
    namespace (injected by the runner script).

    Returns:
        Python source code defining all tool functions.
    """
    parts = [_generate_single_stub(s) for s in schemas]
    return "\n".join(parts)


def generate_tool_documentation(schemas: list[ToolSchema]) -> str:
    """Generate human-readable function documentation for the system prompt.

    Returns:
        Markdown-formatted section describing available functions with
        their signatures and docstrings.
    """
    if not schemas:
        return ""

    lines = ["## Available Functions", "", "You can call these functions directly in your code:", ""]

    for schema in schemas:
        properties: dict[str, Any] = schema.parameters.get("properties", {})
        required: list[str] = schema.parameters.get("required", [])

        # Build signature
        required_params: list[str] = []
        optional_params: list[str] = []
        for param_name, prop in properties.items():
            py_type = _python_type(prop)
            if param_name in required:
                required_params.append(f"{param_name}: {py_type}")
            else:
                default = prop.get("default", None)
                default_repr = repr(default)
                optional_params.append(f"{param_name}: {py_type} = {default_repr}")

        all_params = required_params + optional_params
        params_str = ", ".join(all_params)

        lines.append(f"def {schema.name}({params_str}) -> str:")
        lines.append(f'    """{schema.description}')

        if properties:
            lines.append("")
            lines.append("    Args:")
            for param_name, prop in properties.items():
                desc = prop.get("description", "")
                lines.append(f"        {param_name}: {desc}")

        lines.append('    """')
        lines.append("")

    return "\n".join(lines)


class CodeActAgent(Agent):
    """Agent that uses Python code as the primary action modality.

    Instead of calling tools via structured tool-use, the LLM writes Python
    code in fenced code blocks. Code is executed in a ``Sandbox`` and the
    output (stdout, return value, or error traceback) is fed back as an
    observation. When the LLM responds with plain text (no code blocks),
    that response is the final answer.

    Tools can optionally be bridged into the sandbox as callable Python
    functions, allowing the LLM to call them directly in its code.

    Args:
        name: Identifies the agent in events and traces.
        llm_client: Language model to use.
        emitter: Event emitter for observability.
        system_prompt: Base system prompt text.
        sandbox: Execution environment (``MockSandbox`` for testing,
            ``DockerSandbox`` for production).
        tools: Tools bridged as callable functions in the sandbox.
        max_iterations: Maximum code execution rounds (default: 10).
        max_observation_length: Truncate execution output beyond this
            length (default: 10,000).
        cancellation_token: External cancellation signal.
        error_handler: Error recovery strategy.
        context_manager: Context window management.
        context_providers: Inject context before each LLM call.
        output_evaluator: Quality gate for the final text answer.
        prompt_contributors: Additional system prompt sections.
    """

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        emitter: EventEmitter,
        system_prompt: str,
        sandbox: Sandbox,
        tools: Sequence[Tool] | None = None,
        max_iterations: int = 10,
        max_observation_length: int = 10_000,
        cancellation_token: CancellationToken | None = None,
        error_handler: ErrorHandling | None = None,
        context_manager: ContextManagement | None = None,
        context_providers: list[ContextProvider] | None = None,
        output_evaluator: OutputEvaluator | None = None,
        prompt_contributors: list[SystemPromptContributor] | None = None,
        tool_state: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        if run_id is not None:
            tool_state = dict(tool_state) if tool_state else {}
            tool_state.setdefault("run_id", run_id)
        super().__init__(
            name=name,
            llm_client=llm_client,
            emitter=emitter,
            system_prompt=system_prompt,
            cancellation_token=cancellation_token,
            error_handler=error_handler,
            context_manager=context_manager,
            context_providers=context_providers,
            output_evaluator=output_evaluator,
            prompt_contributors=prompt_contributors,
        )
        self._sandbox = sandbox
        self._max_observation_length = max_observation_length
        self._limiter = IterationLimiter(max_iterations)

        self._tool_registry: ToolRegistry | None = None
        if tools:
            self._tool_registry = ToolRegistry(
                tool_state=tool_state,
                emitter_provider=lambda: self._emitter,
            )
            self._tool_registry.register_all(tools)
            # Wire tool dispatcher into sandbox for tool bridge routing
            if hasattr(self._sandbox, "_tool_dispatcher"):  # pragma: no cover (docker)
                self._sandbox._tool_dispatcher = self._create_tool_dispatcher()

        # Extend system prompt with code execution instructions + tool docs
        extensions = _CODE_EXECUTION_INSTRUCTIONS
        if self._tool_registry:
            tool_docs = generate_tool_documentation(self._tool_registry.list_schemas())
            if tool_docs:
                extensions += "\n\n" + tool_docs
        self._system_prompt += "\n\n" + extensions

    def _agent_type(self) -> str:
        return "codeact"

    def _active_capabilities(self) -> list[str]:
        caps = super()._active_capabilities()
        caps.append("code_execution")
        if self._tool_registry is not None:
            caps.append("tool_use")
        return caps

    def _get_tools_available(self) -> list[str]:
        if self._tool_registry is None:
            return []
        return [s.name for s in self._tool_registry.list_schemas()]

    def _get_tool_schemas(self) -> list[ToolInfo]:
        if self._tool_registry is None:
            return []
        return [
            ToolInfo(
                name=s.name,
                description=s.description,
                requires_approval=s.requires_approval,
            )
            for s in self._tool_registry.list_schemas()
        ]

    async def _execute(self, input: AgentInput) -> AgentResult:
        self._limiter.reset()
        self._error_handler.reset()
        if self._context_manager is not None:
            self._context_manager.reset()

        # Initialize tool function stubs in sandbox namespace
        if self._tool_registry:
            stub_code = generate_tool_stubs(self._tool_registry.list_schemas())
            await self._sandbox.execute(stub_code)

        messages: list[Message] = [Message(role="user", content=input)]
        usages: list[Usage] = []
        step_number = 0
        revision_count = 0
        output: str | None = None
        termination_reason = "complete"

        while True:
            if self._is_cancelled:
                self._emit_safety_cancellation(step_number)
                termination_reason = "cancelled"
                break

            try:
                self._limiter.step()
            except AgentIterationLimitError:
                self._emit_safety_iteration_limit(
                    self._limiter.current_iteration,
                    self._limiter.max_iterations,
                    step_number,
                )
                termination_reason = "iteration_limit"
                break

            step_number += 1

            with self._emitter.span(f"step-{step_number}"):
                response = await self._call_llm(messages, tools=[_EXECUTE_CODE_SCHEMA])
                usages.append(response.usage)

                assistant_content = response.content or ""
                code_calls = [tc for tc in response.tool_calls if tc.name == "execute_code"]

                if not code_calls:
                    # No tool calls — final answer
                    if self._output_evaluator is not None:
                        max_revisions = self._output_evaluator.max_revisions

                        if self._is_truncated(response):
                            if revision_count < max_revisions:
                                messages.append(Message(role="assistant", content=assistant_content))
                                messages.append(Message(role="user", content=self._TRUNCATION_FEEDBACK))
                                self._emit_truncation_events(revision_count, max_revisions)
                                revision_count += 1
                                continue
                            termination_reason = "evaluation_failed"
                            output = assistant_content
                            messages.append(Message(role="assistant", content=assistant_content))
                            self._emit_step(
                                step_number,
                                thought=response.reasoning_text,
                                observation=response.content or None,
                            )
                            break

                        eval_result = await self._evaluate_output(
                            assistant_content,
                            input,
                            messages,
                            revision_count,
                        )
                        if eval_result.verdict == EvaluationVerdict.REVISE and revision_count < max_revisions:
                            messages.append(Message(role="assistant", content=assistant_content))
                            messages.append(Message(role="user", content=eval_result.feedback or ""))
                            self._emit_evaluation_revision(
                                eval_result.feedback or "",
                                revision_count,
                                max_revisions,
                            )
                            revision_count += 1
                            continue
                        if eval_result.verdict == EvaluationVerdict.EVALUATOR_ERROR:
                            termination_reason = "evaluation_skipped"
                        elif eval_result.verdict != EvaluationVerdict.ACCEPT:
                            termination_reason = "evaluation_failed"

                    output = assistant_content
                    messages.append(Message(role="assistant", content=assistant_content))
                    self._emit_step(
                        step_number,
                        thought=response.reasoning_text,
                        observation=response.content or None,
                    )
                    break

                # Tool calls found — execute code sequentially
                observations: list[str] = []
                code_strings: list[str] = []
                tool_result_messages: list[Message] = []
                for tc in code_calls:
                    code = tc.arguments.get("code", "")
                    code_strings.append(code)

                    self._emitter.emit(
                        CodeExecutionEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            agent_name=self._name,
                            code=code,
                            step_number=step_number,
                        )
                    )

                    result = await self._sandbox.execute(code)

                    self._emitter.emit(
                        CodeExecutionResultEvent(
                            trace_id=self._emitter.trace_id,
                            span_id=self._emitter.span_id,
                            parent_span_id=self._emitter.parent_span_id,
                            agent_name=self._name,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            return_value=result.return_value,
                            success=result.success,
                            error=result.error,
                            duration_ms=result.duration_ms,
                            step_number=step_number,
                        )
                    )

                    obs = self._format_observation(result)
                    observations.append(obs)
                    # CodeAct's "tool result" is a sandbox ``ExecutionResult``,
                    # not a ``ToolResult`` from the registry. There is no
                    # ``metadata`` to round-trip onto ``Message.metadata``.
                    # Projecting ``ExecutionResult`` fields into metadata is a
                    # codeact-specific design choice deferred to a future
                    # Phase; ``Message.metadata`` stays ``None`` here by design.
                    tool_result_messages.append(Message(role="tool_result", content=obs, tool_call_id=tc.id))

                observation = "\n\n".join(observations)

                messages.append(
                    Message(
                        role="assistant",
                        content=assistant_content if assistant_content else None,
                        tool_calls=response.tool_calls,
                    )
                )
                messages.extend(tool_result_messages)

                self._emit_step(
                    step_number,
                    thought=response.reasoning_text,
                    action="\n\n".join(code_strings),
                    observation=observation,
                )

        return AgentResult(
            output=output,
            total_steps=step_number,
            termination_reason=termination_reason,
            messages=messages,
            usage=self._aggregate_usage(usages),
        )

    def _format_observation(self, result: ExecutionResult) -> str:
        if result.success:
            parts: list[str] = []
            if result.stdout:
                parts.append(f"[Execution output]\n{self._truncate(result.stdout)}")
            if result.return_value is not None:
                parts.append(f"[Return value]\n{result.return_value}")
            if not parts:
                return "[Execution completed with no output]"
            return "\n\n".join(parts)
        err_parts: list[str] = []
        if result.error:
            err_parts.append(f"[Execution error]\n{result.error}")
        if result.stdout:
            err_parts.append(f"[Partial output]\n{self._truncate(result.stdout)}")
        if not err_parts:
            return "[Execution failed with no error details]"
        return "\n\n".join(err_parts)

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_observation_length:
            return text
        return text[: self._max_observation_length] + "\n... (output truncated)"

    def _create_tool_dispatcher(self) -> Callable[[str, dict[str, Any]], Awaitable[str]]:  # pragma: no cover (docker)
        registry = self._tool_registry
        assert registry is not None

        async def dispatch(name: str, args: dict[str, Any]) -> str:
            tool_call = ToolCall(
                id=f"codeact-{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=args,
            )
            result = await registry.dispatch(tool_call)
            return result.content

        return dispatch
