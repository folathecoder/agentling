"""Failure-recovery demo: watch the agent turn tool errors into observations.

Runs fully offline with a scripted model, so no API key is needed:

    uv run python examples/cli_failure_recovery.py

The scripted model first calls a tool that raises an unexpected exception, then
one that rejects invalid arguments, then retries with valid arguments and
answers. agentling feeds each failure back to the model as an observation
rather than crashing, so the run recovers on its own.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentling import (
    ActionStep,
    Agent,
    ChatMessage,
    Delta,
    ToolCall,
    ToolCallDelta,
    ToolCallError,
    Usage,
    tool,
)


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: The numerator.
        b: The denominator; must be non-zero.
    """
    if b == 0:
        raise ToolCallError("b must be non-zero")
    return a / b


@tool
def fetch_rate() -> float:
    """Fetch a conversion rate from an unreliable upstream service."""
    raise RuntimeError("upstream service unavailable")


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


def _assistant(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(1, 1),
    )


def build_agent() -> Agent:
    """Build an offline agent scripted to fail twice, then recover."""

    model = ScriptedModel(
        [
            _assistant(tool_calls=[ToolCall(id="c1", name="fetch_rate", arguments={})]),
            _assistant(
                tool_calls=[
                    ToolCall(id="c2", name="divide", arguments={"a": 10, "b": 0})
                ]
            ),
            _assistant(
                tool_calls=[
                    ToolCall(id="c3", name="divide", arguments={"a": 10, "b": 2})
                ]
            ),
            _assistant(content="10 divided by 2 is 5."),
        ]
    )
    return Agent(model=model, tools=[divide, fetch_rate])


async def main() -> None:
    session = build_agent().start()
    answer = await session.run("Compute 10 / 2, working around any errors.")

    print(f"Answer: {answer}\n")
    print("How the run got there:")
    for step in session.memory.steps:
        if isinstance(step, ActionStep):
            for result in step.tool_results:
                flag = "error" if result.is_error else "ok"
                print(f"  [{flag}] {result.name}: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
