"""Event-observer demo: watch every event type flow from a single run.

Runs fully offline with a scripted model, so no API key is needed:

    uv run python examples/cli_advanced_observer.py

Iterates the streaming event API directly and prints each TextDelta, tool call,
tool result, step, and the final event. It also wires a context_manager that
trims the prompt and sets a model_timeout budget, two of the production knobs.
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
    StepEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    tool,
)


@tool
def health_check() -> str:
    """Report a one-line service health status."""
    return "all systems nominal"


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


def keep_recent(messages: list[ChatMessage]) -> list[ChatMessage]:
    """A context_manager that keeps the system prompt and the recent tail."""

    if len(messages) <= 6:
        return messages
    return [messages[0], *messages[-5:]]


def build_agent() -> Agent:
    """Build an offline agent that calls one tool, then answers."""

    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="health_check", arguments={})]
            ),
            _assistant(content="All systems nominal."),
        ]
    )
    return Agent(
        model=model,
        tools=[health_check],
        context_manager=keep_recent,
        model_timeout=30.0,
    )


async def observe(agent: Agent) -> dict[str, int]:
    """Stream a run, print each event, and tally how many of each type arrived."""

    counts: dict[str, int] = {}
    session = agent.start()
    async for event in session.run("Check system health.", stream=True):
        counts[type(event).__name__] = counts.get(type(event).__name__, 0) + 1
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
            case ToolCallEvent(tool_call=call):
                print(f"\n-> {call.name}()")
            case ToolResultEvent(result=result):
                print(f"<- {result.content}")
            case StepEvent():
                print("[step recorded]")
            case FinalEvent(answer=answer, status=status):
                print(f"\n= [{status}] {answer}")
    return counts


async def main() -> None:
    counts = await observe(build_agent())
    print("\nEvent counts:", counts)


if __name__ == "__main__":
    asyncio.run(main())
