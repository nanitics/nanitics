from __future__ import annotations

import inspect
import types
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from nanitics.infrastructure.errors import ToolParameterError
from nanitics.infrastructure.llm.protocol import ToolSchema

from .context import ToolContext, _current_tool_context
from .protocol import ToolResult


class FunctionTool:
    """Wraps an async function as a :class:`~nanitics.core.tools.protocol.Tool`.

    Handles parameter schema generation (from a Pydantic model or raw JSON
    Schema), parameter validation, ``ToolContext`` injection, and result
    normalisation (plain strings are wrapped in ``ToolResult``).

    Use the :func:`tool` decorator for the common case.  Use ``FunctionTool``
    directly when you need to create tools programmatically or supply a raw
    JSON Schema.

    Args:
        fn: Async callable implementing the tool logic.  Must return
            ``str`` or ``ToolResult``.
        name: Tool name exposed to the LLM.
        description: Human-readable description the LLM uses to decide
            when to invoke the tool.
        parameters_model: Pydantic model describing the input parameters.
            Mutually exclusive with *parameters_schema*.
        parameters_schema: Raw JSON Schema dict for the parameters.
            Mutually exclusive with *parameters_model*.

    Raises:
        ValueError: If both or neither of *parameters_model* and
            *parameters_schema* are provided.
    """

    def __init__(
        self,
        fn: Callable[..., Awaitable[str | ToolResult]],
        name: str,
        description: str,
        *,
        parameters_model: type[BaseModel] | None = None,
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        if parameters_model is not None and parameters_schema is not None:
            raise ValueError("Provide either parameters_model or parameters_schema, not both")
        if parameters_model is None and parameters_schema is None:
            raise ValueError("Provide either parameters_model or parameters_schema")

        self._fn = fn
        self._name = name
        self._description = description
        self.parameters_model = parameters_model

        # Detect ToolContext parameter
        self._context_param_name: str | None = None
        hints = inspect.get_annotations(fn, eval_str=True)
        for pname, annotation in hints.items():
            if annotation is ToolContext:
                self._context_param_name = pname
                break
            if isinstance(annotation, types.UnionType) and ToolContext in annotation.__args__:
                self._context_param_name = pname
                break

        if parameters_model is not None:
            schema_dict = parameters_model.model_json_schema()
        else:
            assert parameters_schema is not None  # guaranteed by __init__ validation
            schema_dict = parameters_schema

        self._schema = ToolSchema(
            name=name,
            description=description,
            parameters=schema_dict,
        )

    @property
    def schema(self) -> ToolSchema:
        """Return the tool schema describing this tool's interface."""
        return self._schema

    async def execute(self, **params: Any) -> ToolResult:
        """Run the wrapped function with the given parameters.

        If a ``ToolContext`` parameter is declared in the function
        signature, it is injected from the current context variable.
        When a ``parameters_model`` is configured, parameters are
        validated through the Pydantic model before invocation.

        Args:
            **params: Tool parameters as keyword arguments.

        Returns:
            The tool result.  Plain string return values from the
            wrapped function are automatically wrapped in ``ToolResult``.

        Raises:
            ToolParameterError: If Pydantic validation fails.
        """
        if self._context_param_name is not None:
            params[self._context_param_name] = _current_tool_context.get()

        if self.parameters_model is not None:
            try:
                validated = self.parameters_model(**params)
            except ValidationError as e:
                raise ToolParameterError(
                    f"Parameter validation failed for tool '{self._name}': {e}",
                    tool_name=self._name,
                    reason=str(e),
                ) from e
            kwargs = validated.model_dump()
            if self._context_param_name is not None:
                kwargs[self._context_param_name] = params[self._context_param_name]
            result = await self._fn(**kwargs)
        else:
            result = await self._fn(**params)

        if isinstance(result, str):
            return ToolResult(content=result)
        return result


def tool(
    name: str,
    description: str,
    *,
    parameters_model: type[BaseModel] | None = None,
) -> Callable[
    [Callable[..., Awaitable[str | ToolResult]]],
    FunctionTool,
]:
    """Decorator that creates a :class:`FunctionTool` from an async function.

    If *parameters_model* is not provided, a Pydantic model is
    auto-generated from the function's type-annotated signature.  Parameters
    with defaults become optional in the generated schema.

    Args:
        name: Tool name exposed to the LLM.
        description: Description the LLM uses to decide when to invoke
            the tool.
        parameters_model: Optional explicit Pydantic model for parameter
            validation and schema generation.

    Returns:
        A decorator that transforms the function into a ``FunctionTool``.

    Example::

        @tool("get_weather", "Get current weather for a city")
        async def get_weather(city: str) -> str:
            return f"Sunny in {city}"
    """

    def decorator(
        fn: Callable[..., Awaitable[str | ToolResult]],
    ) -> FunctionTool:
        model = parameters_model
        if model is None:
            model = _model_from_function(fn, name)

        return FunctionTool(
            fn=fn,
            name=name,
            description=description,
            parameters_model=model,
        )

    return decorator


def _model_from_function(fn: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    sig = inspect.signature(fn)
    hints = {k: v for k, v in inspect.get_annotations(fn, eval_str=True).items() if k not in ("return", "self", "cls")}

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, Any)
        if annotation is ToolContext:
            continue
        if isinstance(annotation, types.UnionType) and ToolContext in annotation.__args__:
            continue
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (annotation, ...)
        else:
            fields[param_name] = (annotation, param.default)

    return create_model(f"{tool_name}_params", **fields)
