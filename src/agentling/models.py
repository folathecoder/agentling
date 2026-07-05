from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import openai

Role = Literal["system", "user", "assistant", "tool"]
ToolSpec = dict[str, Any]
OpenAIMessage = dict[str, Any]


@dataclass
class ToolCall:
    """A tool call requested by the model.

    This is provider-neutral. Individual model providers can map their own
    tool-call format into this shape.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallDelta:
    """A partial tool call emitted during streaming.

    Tool calls may arrive in chunks, especially the arguments string.
    The `index` field lets callers merge fragments belonging to the same
    streamed tool call.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class Delta:
    """A single streamed model chunk."""

    content: str | None = None
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    usage: Usage | None = None


@dataclass(frozen=True)
class Usage:
    """Token usage returned by a model provider."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ChatMessage:
    """A provider-neutral chat message.

    This is the main message format used internally by the framework.
    Provider adapters are responsible for converting this into their own
    API-specific formats.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    usage: Usage | None = None


class Model(Protocol):
    """Common interface implemented by all model providers."""

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
    ) -> ChatMessage: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[Delta]: ...


class OpenAIModel:
    """OpenAI Chat Completions adapter.

    Converts the framework's provider-neutral message format into OpenAI's
    chat-completions format, sends the request, then converts the response
    back into a `ChatMessage`.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        context_window: int = 128_000,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ) -> None:
        """Create an OpenAI model adapter.

        Args:
            model: OpenAI model name.
            api_key: Optional API key. If omitted, the OpenAI SDK will use
                its default environment-based configuration.
            base_url: Optional custom base URL for OpenAI-compatible providers.
            context_window: Maximum supported context window for this model.
            max_retries: Number of retries after the initial request.
            retry_base_delay: Initial backoff delay in seconds.
        """

        self.model = model
        self.context_window = context_window
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
    ) -> ChatMessage:
        """Generate a single assistant response."""

        openai_messages = self._to_openai_messages(messages)

        response = await self._create_with_retry(
            model=self.model,
            messages=cast(Any, openai_messages),
            tools=cast(Any, tools) if tools else openai.omit,
        )

        message = response.choices[0].message

        return ChatMessage(
            role="assistant",
            content=message.content or "",
            tool_calls=self._from_openai_tool_calls(message.tool_calls),
            usage=self._from_openai_usage(response.usage),
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[Delta]:
        """Stream response chunks from the model as they are generated."""

        openai_messages = self._to_openai_messages(messages)

        openai_stream = await self._create_with_retry(
            model=self.model,
            messages=cast(Any, openai_messages),
            tools=cast(Any, tools) if tools else openai.omit,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in openai_stream:
            delta = self._from_openai_chunk(chunk)
            if delta is not None:
                yield delta

    async def _create_with_retry(self, **kwargs: Any) -> Any:
        """Call the OpenAI chat-completions API with rate-limit retries.

        Retries cover establishing the request, including opening a stream.
        A 429 raised mid-stream (after chunks have been yielded) is not retried.
        """

        total_attempts = max(1, self.max_retries + 1)

        for attempt in range(total_attempts):
            try:
                return await self.client.chat.completions.create(**kwargs)

            except openai.RateLimitError:
                if attempt == total_attempts - 1:
                    raise

                delay = self.retry_base_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise RuntimeError("Unexpected retry loop exit.")

    def _from_openai_chunk(self, chunk: Any) -> Delta | None:
        """Convert a single OpenAI stream chunk into a framework Delta."""

        usage = self._from_openai_usage(chunk.usage) if chunk.usage else None

        # The final usage-only chunk (from stream_options) carries no choices.
        if not chunk.choices:
            return Delta(usage=usage) if usage else None

        choice_delta = chunk.choices[0].delta

        tool_calls = [
            ToolCallDelta(
                index=tool_call.index,
                id=tool_call.id,
                name=tool_call.function.name if tool_call.function else None,
                arguments=(
                    tool_call.function.arguments if tool_call.function else None
                ),
            )
            for tool_call in (choice_delta.tool_calls or [])
        ]

        return Delta(
            content=choice_delta.content,
            tool_calls=tool_calls,
            usage=usage,
        )

    def _to_openai_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> list[OpenAIMessage]:
        """Convert framework messages into OpenAI chat-completions messages."""

        openai_messages: list[OpenAIMessage] = []

        for message in messages:
            match message.role:
                case "system" | "user":
                    openai_messages.append(
                        {
                            "role": message.role,
                            "content": message.content,
                        }
                    )

                case "assistant":
                    openai_messages.append(self._to_openai_assistant_message(message))

                case "tool":
                    openai_messages.append(self._to_openai_tool_message(message))

                case _:
                    raise ValueError(f"Unsupported message role: {message.role!r}")

        return openai_messages

    def _to_openai_assistant_message(self, message: ChatMessage) -> OpenAIMessage:
        """Convert an assistant message into OpenAI's expected format."""

        openai_message: OpenAIMessage = {
            "role": "assistant",
            "content": message.content or None,
        }

        if message.tool_calls:
            openai_message["tool_calls"] = [
                self._to_openai_tool_call(tool_call) for tool_call in message.tool_calls
            ]

        return openai_message

    def _to_openai_tool_message(self, message: ChatMessage) -> OpenAIMessage:
        """Convert a tool-result message into OpenAI's expected format."""

        if message.tool_call_id is None:
            raise ValueError("Tool messages must include a tool_call_id.")

        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }

    def _to_openai_tool_call(self, tool_call: ToolCall) -> OpenAIMessage:
        """Convert a framework tool call into OpenAI's tool-call format."""

        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                # OpenAI expects tool arguments as a JSON string.
                "arguments": json.dumps(tool_call.arguments),
            },
        }

    def _from_openai_usage(self, usage: Any | None) -> Usage:
        """Convert OpenAI usage metadata into framework usage metadata."""

        if usage is None:
            return Usage(input_tokens=0, output_tokens=0)

        return Usage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    def _from_openai_tool_calls(self, tool_calls: Any | None) -> list[ToolCall]:
        """Convert OpenAI tool calls into framework tool calls."""

        if not tool_calls:
            return []

        return [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=self._parse_tool_arguments(tool_call.function.arguments),
            )
            for tool_call in tool_calls
        ]

    def _parse_tool_arguments(self, arguments: str) -> dict[str, Any]:
        """Parse and validate model-generated tool arguments.

        Tool-call arguments should be a JSON object. If the model emits invalid
        JSON or a non-object value, fail loudly so the caller can decide how to
        handle the error.
        """

        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool call arguments: {arguments!r}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"Tool call arguments must be a JSON object: {arguments!r}"
            )

        return parsed


@dataclass
class _ToolCallFragment:
    """Mutable accumulator for a streamed tool call (internal to agglomeration)."""

    id: str | None = None
    name: str | None = None
    arguments: str = ""


def agglomerate_deltas(deltas: Iterable[Delta]) -> ChatMessage:
    """Reassemble streamed deltas into a single ChatMessage.

    Content fragments are concatenated. Tool-call fragments are merged by their
    `index`: the id and name arrive once, while the arguments string streams in
    pieces and must be concatenated before it can be parsed as JSON.
    """

    content_parts: list[str] = []
    fragments: dict[int, _ToolCallFragment] = {}
    usage: Usage | None = None

    for delta in deltas:
        if delta.content:
            content_parts.append(delta.content)

        for tool_call in delta.tool_calls:
            fragment = fragments.setdefault(tool_call.index, _ToolCallFragment())
            if tool_call.id is not None:
                fragment.id = tool_call.id
            if tool_call.name is not None:
                fragment.name = tool_call.name
            if tool_call.arguments:
                fragment.arguments += tool_call.arguments

        if delta.usage is not None:
            usage = delta.usage

    tool_calls = [
        ToolCall(
            id=fragment.id or "",
            name=fragment.name or "",
            arguments=json.loads(fragment.arguments or "{}"),
        )
        for _index, fragment in sorted(fragments.items())
    ]

    return ChatMessage(
        role="assistant",
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
    )
