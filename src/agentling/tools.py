"""Tool primitives and schema generation utilities.

This module defines the framework's tool abstraction, the @tool decorator for
wrapping Python functions, lightweight argument validation, and the built-in
final_answer tool used to terminate agent runs explicitly.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import types
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, overload

from .errors import AgentlingError
from .models import ToolSpec

# JSON Schema for a tool's parameters object. This is intentionally separate
# from ToolSpec, which represents the full function tool wrapper.
JsonSchema = dict[str, Any]


class ToolCallError(AgentlingError):
    """Raised when a tool receives invalid model-generated arguments.

    Agent loops can catch this error and return it to the model as an
    observation, allowing the model to recover from invalid tool calls without
    aborting the run.
    """


class Tool:
    """Base class for executable tools.

    Subclass this for stateful tools. For simple stateless functions, prefer the
    @tool decorator.
    """

    name: str
    description: str
    parameters: JsonSchema
    # Optional per-tool time budget in seconds. None means use the agent's
    # tool_timeout default (which is itself None unless the caller sets it).
    timeout: float | None = None
    # Whether this tool is safe to run concurrently with others in the same
    # step. If any tool in a step is not parallel_safe, the step runs in order.
    parallel_safe: bool = True
    # Optional cap on observation length in characters. None means use the
    # agent's max_tool_output_chars default.
    max_output_chars: int | None = None

    async def forward(self, **kwargs: Any) -> Any:
        """Execute the tool implementation.

        Subclasses should override this method. Arguments are validated by
        call() before this method is invoked.
        """

        raise NotImplementedError

    def to_schema(self) -> ToolSpec:
        """Return the provider-facing schema for this tool."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Validate model-generated arguments and execute the tool.

        This is the agent loop entry point. Invalid arguments raise
        ToolCallError so the loop can expose the validation failure back to the
        model.
        """

        if not isinstance(arguments, dict):
            raise ToolCallError(
                f"Tool {self.name!r} expected an arguments object, "
                f"got {type(arguments).__name__}"
            )

        self._validate(arguments)
        return await self.forward(**arguments)

    def _validate(self, arguments: dict[str, Any]) -> None:
        """Validate arguments against the tool's JSON Schema subset."""

        properties: dict[str, Any] = self.parameters.get("properties", {})
        required: list[str] = self.parameters.get("required", [])

        missing = [name for name in required if name not in arguments]
        if missing:
            raise ToolCallError(
                f"Tool {self.name!r} is missing required argument(s): "
                f"{', '.join(missing)}"
            )

        unknown = [name for name in arguments if name not in properties]
        if unknown:
            raise ToolCallError(
                f"Tool {self.name!r} got unexpected argument(s): "
                f"{', '.join(unknown)}"
            )

        for name, value in arguments.items():
            spec = properties[name]

            expected = spec.get("type")
            if expected and not _matches_json_type(value, expected):
                raise ToolCallError(
                    f"Tool {self.name!r} argument {name!r} expects {expected}, "
                    f"got {type(value).__name__}"
                )

            enum = spec.get("enum")
            if enum is not None and value not in enum:
                raise ToolCallError(
                    f"Tool {self.name!r} argument {name!r} must be one of {enum}, "
                    f"got {value!r}"
                )


_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# JSON Schema type name -> Python runtime type(s).
# Used for lightweight validation of model-generated arguments.
_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _matches_json_type(value: Any, json_type: str) -> bool:
    """Return whether a value matches a JSON Schema primitive type."""

    # bool is a subclass of int in Python. Treat it as a distinct JSON boolean
    # so True/False are not accepted for integer or number fields.
    if isinstance(value, bool):
        return json_type == "boolean"

    expected = _PY_TYPES.get(json_type)
    return expected is None or isinstance(value, expected)


def _schema_for_type(python_type: Any) -> JsonSchema:
    """Convert a Python type hint into a JSON Schema property schema.

    This intentionally supports a small, practical subset of Python typing:
    primitive types, list/dict generics, Optional/Union with None, and Literal.
    Unsupported or ambiguous types fall back to string.
    """

    origin = get_origin(python_type)
    args = get_args(python_type)

    # Literal["c", "f"] becomes an enum. A JSON Schema "type" is included only
    # when all literal values share the same primitive type.
    if origin is Literal:
        values = list(args)
        if not values:
            return {"type": "string"}

        value_types = {type(value) for value in values}
        if len(value_types) == 1:
            json_type = _JSON_TYPES.get(next(iter(value_types)), "string")
            return {"type": json_type, "enum": values}

        return {"enum": values}

    # Treat Optional[X] as "the field may be omitted". If the field is supplied,
    # it must still match X. Null is intentionally not advertised as valid.
    if origin is Union or origin is types.UnionType:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _schema_for_type(non_none[0])

        return {"type": "string"}

    # list[str] becomes an array with typed items. Bare list becomes an array
    # without item constraints.
    if origin is list:
        if args:
            return {
                "type": "array",
                "items": _schema_for_type(args[0]),
            }

        return {"type": "array"}

    # dict[...] is represented as an object without per-key constraints.
    if origin is dict:
        return {"type": "object"}

    # For other generic aliases, fall back to the origin's base type.
    if origin is not None:
        return {"type": _JSON_TYPES.get(origin, "string")}

    # Plain types: str, int, float, bool, list, dict, etc.
    return {"type": _JSON_TYPES.get(python_type, "string")}


_ARGS_HEADERS = {
    "args",
    "arguments",
    "parameters",
    "keyword args",
    "keyword arguments",
}

_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z ]*):$")
_ARG_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")


def _parse_docstring(func: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    """Extract a summary and argument descriptions from a docstring.

    This is a pragmatic Google-style parser, not a complete docstring parser.
    It recognizes Args/Arguments/Parameters sections and stops parsing argument
    descriptions when another section header is encountered.
    """

    doc = inspect.getdoc(func) or ""
    summary_lines: list[str] = []
    arg_docs: dict[str, str] = {}

    section: str | None = None
    current_arg: str | None = None

    for raw in doc.splitlines():
        stripped = raw.strip()

        header = _HEADER_RE.match(stripped)
        if header:
            key = header.group(1).strip().lower()
            section = "args" if key in _ARGS_HEADERS else "other"
            current_arg = None
            continue

        if section is None:
            summary_lines.append(stripped)

        elif section == "args":
            match = _ARG_RE.match(stripped)

            if match:
                current_arg = match.group(1)
                arg_docs[current_arg] = match.group(2).strip()

            elif current_arg and stripped:
                arg_docs[current_arg] += " " + stripped

        # Other sections, such as Returns or Raises, are ignored.

    summary = " ".join(line for line in summary_lines if line).strip()
    return summary, arg_docs


def _build_schema(func: Callable[..., Any]) -> JsonSchema:
    """Build a JSON Schema parameters object from a function signature."""

    hints = get_type_hints(func)
    signature = inspect.signature(func)
    _, arg_docs = _parse_docstring(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue

        # Tools are invoked with a JSON object of named arguments. Variadic and
        # positional-only parameters cannot be represented safely, so reject
        # them during tool registration rather than failing during a model run.
        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }:
            raise TypeError(
                f"Tool {func.__name__!r} has unsupported parameter {name!r}. "
                "Tools support only normal or keyword-only parameters "
                "(no *args, **kwargs, or positional-only parameters)."
            )

        schema = _schema_for_type(hints.get(name, str))

        if name in arg_docs:
            schema["description"] = arg_docs[name]

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = param.default

        properties[name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class _FunctionTool(Tool):
    """Tool implementation backed by a plain Python function."""

    def __init__(self, func: Callable[..., Any]) -> None:
        summary, _ = _parse_docstring(func)

        name = func.__name__
        if not _VALID_TOOL_NAME.match(name):
            raise ValueError(
                f"Tool name {name!r} is not a valid tool name: use letters, "
                "digits, underscore, or hyphen, up to 64 characters."
            )

        self._func = func
        self.name = name
        self.description = summary
        self.parameters = _build_schema(func)

    async def forward(self, **kwargs: Any) -> Any:
        # Async functions are awaited directly; a plain sync function runs in a
        # worker thread so a blocking call (CPU or I/O) cannot stall the event
        # loop, the stream, or other tools running in the same step.
        if inspect.iscoroutinefunction(self._func):
            return await self._func(**kwargs)
        return await asyncio.to_thread(self._func, **kwargs)


@overload
def tool(func: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    *,
    timeout: float | None = ...,
    parallel_safe: bool = ...,
    max_output_chars: int | None = ...,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    timeout: float | None = None,
    parallel_safe: bool = True,
    max_output_chars: int | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Create a Tool from a plain Python function.

    The function name becomes the tool name, the docstring summary becomes the
    description, argument descriptions come from a Google-style Args section,
    and type hints become a JSON Schema parameters object. Synchronous and
    asynchronous functions are both supported; sync functions run in a worker
    thread so they cannot block the event loop.

    Use it bare, or with metadata:

        @tool
        def get_weather(city: str) -> str:
            '''Get the current weather for a city.

            Args:
                city: The city to look up.
            '''
            ...

        @tool(timeout=30, max_output_chars=2000, parallel_safe=False)
        def run_query(sql: str) -> str:
            ...
    """

    def make(target: Callable[..., Any]) -> Tool:
        built = _FunctionTool(target)
        built.timeout = timeout
        built.parallel_safe = parallel_safe
        built.max_output_chars = max_output_chars
        return built

    if func is not None:
        return make(func)
    return make


class FinalAnswerTool(Tool):
    """Built-in tool used to return the final answer from an agent run.

    Including final_answer in the tool set makes termination explicit: the model
    signals completion by calling this tool instead of the loop inferring that a
    normal assistant message should end the run.
    """

    name = "final_answer"
    description = "Provide the final answer to the task and end the run."
    parameters: JsonSchema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer to return to the user.",
            }
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    async def forward(self, answer: str) -> str:  # type: ignore[override]
        # The public Tool.forward signature accepts **kwargs because tools are
        # invoked generically. This implementation is intentionally narrower
        # because call() validates arguments against the schema before dispatch.
        return answer
