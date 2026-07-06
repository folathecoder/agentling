from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import openai
import pytest

from agentling.models import (
    ChatMessage,
    Delta,
    OpenAIModel,
    ToolCall,
    ToolCallDelta,
    Usage,
    agglomerate_deltas,
)


# --------------------------------------------------------------------------- #
# Helpers — fake the OpenAI client so no network calls happen.
# --------------------------------------------------------------------------- #
def _make_model(create: Any) -> OpenAIModel:
    """Build an OpenAIModel whose client.chat.completions.create is `create`."""

    model = OpenAIModel("test-model", api_key="test", retry_base_delay=0)
    model.client = cast(
        Any,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    return model


def _rate_limit_error() -> openai.RateLimitError:
    """A RateLimitError instance, built without invoking its constructor."""

    return openai.RateLimitError.__new__(openai.RateLimitError)


@pytest.fixture
def model() -> OpenAIModel:
    return OpenAIModel("test-model", api_key="test")


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #
def test_usage_total_tokens() -> None:
    assert Usage(input_tokens=3, output_tokens=4).total_tokens == 7


# --------------------------------------------------------------------------- #
# Outbound conversion: ChatMessage -> OpenAI dicts
# --------------------------------------------------------------------------- #
def test_to_openai_messages_system_and_user(model: OpenAIModel) -> None:
    out = model._to_openai_messages(
        [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
        ]
    )
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_to_openai_messages_assistant_with_tool_calls(model: OpenAIModel) -> None:
    msg = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})],
    )
    out = model._to_openai_messages([msg])
    assert out == [
        {
            "role": "assistant",
            "content": None,  # empty string is coerced to null on the way out
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                }
            ],
        }
    ]


def test_to_openai_messages_tool_result(model: OpenAIModel) -> None:
    out = model._to_openai_messages(
        [ChatMessage(role="tool", content="result", tool_call_id="c1")]
    )
    assert out == [{"role": "tool", "content": "result", "tool_call_id": "c1"}]


def test_tool_message_without_id_raises(model: OpenAIModel) -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        model._to_openai_messages([ChatMessage(role="tool", content="r")])


# --------------------------------------------------------------------------- #
# Inbound conversion helpers
# --------------------------------------------------------------------------- #
def test_from_openai_usage_none(model: OpenAIModel) -> None:
    assert model._from_openai_usage(None) == Usage(0, 0)


def test_from_openai_usage_maps_fields(model: OpenAIModel) -> None:
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    assert model._from_openai_usage(usage) == Usage(7, 3)


def test_from_openai_tool_calls_none(model: OpenAIModel) -> None:
    assert model._from_openai_tool_calls(None) == []


def test_from_openai_tool_calls_maps(model: OpenAIModel) -> None:
    tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="f", arguments='{"a": 1}')
    )
    assert model._from_openai_tool_calls([tc]) == [
        ToolCall(id="c1", name="f", arguments={"a": 1})
    ]


def test_parse_tool_arguments_valid(model: OpenAIModel) -> None:
    assert model._parse_tool_arguments('{"x": 1}') == {"x": 1}


def test_parse_tool_arguments_empty_defaults_to_object(model: OpenAIModel) -> None:
    assert model._parse_tool_arguments("") == {}


def test_parse_tool_arguments_invalid_json_raises(model: OpenAIModel) -> None:
    with pytest.raises(ValueError, match="Invalid tool call arguments"):
        model._parse_tool_arguments("{not json}")


def test_parse_tool_arguments_non_object_raises(model: OpenAIModel) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        model._parse_tool_arguments("[1, 2]")


# --------------------------------------------------------------------------- #
# agglomerate_deltas — reassemble a stream into one ChatMessage
# --------------------------------------------------------------------------- #
def test_agglomerate_concatenates_content() -> None:
    msg = agglomerate_deltas([Delta(content="Hel"), Delta(content="lo"), Delta()])
    assert msg.role == "assistant"
    assert msg.content == "Hello"
    assert msg.tool_calls == []


def test_agglomerate_merges_tool_call_fragments() -> None:
    deltas = [
        Delta(tool_calls=[ToolCallDelta(index=0, id="c1", name="wx", arguments='{"ci')]),
        Delta(tool_calls=[ToolCallDelta(index=0, arguments='ty": "Paris"}')]),
    ]
    msg = agglomerate_deltas(deltas)
    assert msg.tool_calls == [
        ToolCall(id="c1", name="wx", arguments={"city": "Paris"})
    ]


def test_agglomerate_groups_parallel_calls_by_index() -> None:
    deltas = [
        Delta(tool_calls=[ToolCallDelta(index=0, id="a", name="f", arguments="{}")]),
        Delta(tool_calls=[ToolCallDelta(index=1, id="b", name="g", arguments="{}")]),
    ]
    msg = agglomerate_deltas(deltas)
    assert [tc.id for tc in msg.tool_calls] == ["a", "b"]
    assert [tc.name for tc in msg.tool_calls] == ["f", "g"]


def test_agglomerate_captures_usage() -> None:
    msg = agglomerate_deltas([Delta(content="x"), Delta(usage=Usage(5, 2))])
    assert msg.usage == Usage(5, 2)


# --------------------------------------------------------------------------- #
# generate (fake client)
# --------------------------------------------------------------------------- #
async def test_generate_returns_message_with_usage() -> None:
    async def fake_create(**kwargs: Any) -> Any:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", tool_calls=None)
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )

    model = _make_model(fake_create)
    msg = await model.generate([ChatMessage(role="user", content="hi")])

    assert msg.role == "assistant"
    assert msg.content == "hello"
    assert msg.usage == Usage(5, 2)


async def test_generate_parses_tool_calls_and_coerces_content() -> None:
    async def fake_create(**kwargs: Any) -> Any:
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'),
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    model = _make_model(fake_create)
    msg = await model.generate([ChatMessage(role="user", content="weather?")])

    assert msg.content == ""  # None coerced to ""
    assert msg.tool_calls == [
        ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    ]


# --------------------------------------------------------------------------- #
# stream (fake client)
# --------------------------------------------------------------------------- #
async def test_stream_yields_deltas_then_agglomerates() -> None:
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel", tool_calls=None))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="lo", tool_calls=None))],
            usage=None,
        ),
        SimpleNamespace(  # final usage-only chunk
            choices=[],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        ),
    ]

    async def fake_create(**kwargs: Any) -> Any:
        async def gen() -> AsyncIterator[Any]:
            for chunk in chunks:
                yield chunk

        return gen()

    model = _make_model(fake_create)
    deltas = [d async for d in model.stream([ChatMessage(role="user", content="hi")])]
    final = agglomerate_deltas(deltas)

    assert final.content == "Hello"
    assert final.usage == Usage(3, 1)


# --------------------------------------------------------------------------- #
# retry-on-429
# --------------------------------------------------------------------------- #
async def test_retry_then_succeeds() -> None:
    calls = 0
    result = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    async def flaky_create(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _rate_limit_error()
        return result

    model = _make_model(flaky_create)
    msg = await model.generate([ChatMessage(role="user", content="hi")])

    assert msg.content == "ok"
    assert calls == 2  # failed once, retried, then succeeded


async def test_retry_gives_up_after_max_retries() -> None:
    async def always_fails(**kwargs: Any) -> Any:
        raise _rate_limit_error()

    model = _make_model(always_fails)
    model.max_retries = 1  # 2 attempts total

    with pytest.raises(openai.RateLimitError):
        await model.generate([ChatMessage(role="user", content="hi")])
