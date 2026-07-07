from typing import Literal, Optional

import pytest

from agentling.tools import FinalAnswerTool, Tool, ToolCallError, tool


# --------------------------------------------------------------------------- #
# Schema generation
# --------------------------------------------------------------------------- #
def test_basic_types_required_and_descriptions() -> None:
    @tool
    def add(a: int, b: int = 0) -> int:
        """Add two integers.

        Args:
            a: The first number.
            b: The second number.
        """
        return a + b

    fn = add.to_schema()["function"]
    props = fn["parameters"]["properties"]

    assert fn["name"] == "add"
    assert fn["description"] == "Add two integers."
    assert props["a"]["type"] == "integer"
    assert props["a"]["description"] == "The first number."
    assert props["b"]["type"] == "integer"
    assert fn["parameters"]["required"] == ["a"]  # b has a default


def test_literal_becomes_enum() -> None:
    @tool
    def pick(unit: Literal["c", "f"] = "c") -> str:
        return unit

    prop = pick.to_schema()["function"]["parameters"]["properties"]["unit"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["c", "f"]
    assert prop["default"] == "c"  # defaults are now exposed in the schema


def test_optional_is_unwrapped() -> None:
    @tool
    def f(x: Optional[int] = None, y: str | None = None) -> str:
        return "ok"

    schema = f.to_schema()["function"]["parameters"]
    assert schema["properties"]["x"]["type"] == "integer"
    assert schema["properties"]["y"]["type"] == "string"
    assert schema["required"] == []


def test_list_and_dict() -> None:
    @tool
    def f(tags: list[str], meta: dict) -> str:
        return "ok"

    props = f.to_schema()["function"]["parameters"]["properties"]
    assert props["tags"]["type"] == "array"
    assert props["meta"]["type"] == "object"


def test_array_includes_items() -> None:
    @tool
    def f(tags: list[str]) -> str:
        return ",".join(tags)

    prop = f.to_schema()["function"]["parameters"]["properties"]["tags"]
    assert prop == {"type": "array", "items": {"type": "string"}}


def test_default_is_exposed() -> None:
    @tool
    def f(limit: int = 5) -> int:
        return limit

    prop = f.to_schema()["function"]["parameters"]["properties"]["limit"]
    assert prop["default"] == 5


def test_schema_forbids_additional_properties() -> None:
    @tool
    def f(a: int) -> int:
        return a

    assert f.to_schema()["function"]["parameters"]["additionalProperties"] is False


def test_varargs_are_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported parameter"):

        @tool
        def f(*args: str) -> str:
            return "x"


# --------------------------------------------------------------------------- #
# @tool execution (sync + async)
# --------------------------------------------------------------------------- #
async def test_sync_tool_runs() -> None:
    @tool
    def greet(name: str) -> str:
        return f"hi {name}"

    assert await greet.call({"name": "Ada"}) == "hi Ada"


async def test_async_tool_runs() -> None:
    @tool
    async def fetch(url: str) -> str:
        return f"data:{url}"

    assert await fetch.call({"url": "x"}) == "data:x"


# --------------------------------------------------------------------------- #
# Argument validation -> ToolCallError
# --------------------------------------------------------------------------- #
@pytest.fixture
def weather() -> Tool:
    @tool
    def get_weather(city: str, units: Literal["c", "f"] = "c") -> str:
        """Get weather.

        Args:
            city: The city.
        """
        return f"{city}:{units}"

    return get_weather


async def test_valid_call(weather: Tool) -> None:
    assert await weather.call({"city": "Paris"}) == "Paris:c"


async def test_missing_required(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="missing required"):
        await weather.call({})


async def test_unknown_argument(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="unexpected argument"):
        await weather.call({"city": "Paris", "foo": 1})


async def test_wrong_type(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="expects string"):
        await weather.call({"city": 123})


async def test_bad_enum(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="must be one of"):
        await weather.call({"city": "Paris", "units": "kelvin"})


async def test_non_dict_arguments_rejected(weather: Tool) -> None:
    with pytest.raises(ToolCallError, match="arguments object"):
        await weather.call(["Paris"])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# FinalAnswerTool
# --------------------------------------------------------------------------- #
async def test_final_answer_tool() -> None:
    final = FinalAnswerTool()
    props = final.to_schema()["function"]["parameters"]["properties"]

    assert final.name == "final_answer"
    assert "answer" in props
    assert await final.call({"answer": "42"}) == "42"


# --------------------------------------------------------------------------- #
# @tool metadata and name validation
# --------------------------------------------------------------------------- #
def test_bare_tool_decorator_has_default_metadata() -> None:
    @tool
    def t(x: int) -> int:
        """Doc.

        Args:
            x: A number.
        """
        return x

    assert isinstance(t, Tool)
    assert t.timeout is None
    assert t.parallel_safe is True
    assert t.max_output_chars is None


def test_tool_decorator_accepts_metadata() -> None:
    @tool(timeout=1.5, parallel_safe=False, max_output_chars=100)
    def t(x: int) -> int:
        """Doc.

        Args:
            x: A number.
        """
        return x

    assert isinstance(t, Tool)
    assert t.timeout == 1.5
    assert t.parallel_safe is False
    assert t.max_output_chars == 100


def test_invalid_tool_name_is_rejected() -> None:
    def f() -> str:
        """A tool with a bad name."""
        return ""

    f.__name__ = "bad name!"
    with pytest.raises(ValueError, match="valid tool name"):
        tool(f)
