"""Memory-persistence demo: dump a session, reload it, and continue.

Runs fully offline with a scripted model, so no API key is needed:

    uv run python examples/cli_memory_chat.py

The first session learns a fact; its typed memory is serialized to JSON; a fresh
session reloads that memory and continues the conversation with reset=False. The
continuation is streamed so the terminal FinalEvent's status is shown too.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentling import (
    Agent,
    ChatMessage,
    Delta,
    FinalEvent,
    Memory,
    ToolCallDelta,
    Usage,
)


class ScriptedModel:
    """A deterministic offline model that replays fixed assistant turns."""

    def __init__(self, turns: Sequence[ChatMessage]) -> None:
        self._turns = list(turns)
        self._index = 0

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> ChatMessage:
        turn = self._turns[self._index]
        self._index += 1
        return turn

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        turn = self._turns[self._index]
        self._index += 1
        if turn.content:
            yield Delta(content=turn.content)
        for index, call in enumerate(turn.tool_calls):
            yield Delta(
                tool_calls=[
                    ToolCallDelta(
                        index=index,
                        id=call.id,
                        name=call.name,
                        arguments=json.dumps(call.arguments),
                    )
                ]
            )
        yield Delta(usage=turn.usage)


def _assistant(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content, usage=Usage(1, 1))


def build_agent() -> Agent:
    """Build an offline agent scripted for a two-turn conversation."""

    model = ScriptedModel(
        [
            _assistant("Nice to meet you, Sam."),
            _assistant("Your name is Sam."),
        ]
    )
    return Agent(model=model)


async def main() -> None:
    agent = build_agent()

    # First session: learn something, then serialize the run to JSON.
    first = agent.start()
    await first.run("Hi, my name is Sam.")
    saved = first.memory.dump_json()
    print(f"Saved {len(first.memory.steps)} steps to JSON.\n")

    # Later, in a fresh session (or another process), restore and continue.
    second = agent.start()
    second.memory = Memory.load_json(saved)

    answer, status = "", "unknown"
    async for event in second.run("What is my name?", reset=False, stream=True):
        if isinstance(event, FinalEvent):
            answer, status = event.answer, event.status
    print(f"[{status}] {answer}")


if __name__ == "__main__":
    asyncio.run(main())
