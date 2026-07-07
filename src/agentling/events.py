"""Streaming events emitted by the agent loop.

Memory is the durable, structured record of a run. Events are runtime
notifications emitted as the run progresses, letting callers render output,
inspect tool execution, and react to completion in real time.

StepEvent connects the live event stream to memory by carrying the step that
was just recorded.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .memory import Step, ToolResult
from .models import ToolCall, Usage


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A chunk of streamed assistant text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """Emitted before the agent executes a tool call."""

    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Emitted after a tool call completes, successfully or with an error."""

    result: ToolResult


@dataclass(frozen=True, slots=True)
class StepEvent:
    """Emitted after a step has been recorded to memory."""

    step: Step


@dataclass(frozen=True, slots=True)
class FinalEvent:
    """Emitted once when the run completes, carrying the answer and total usage."""

    answer: str
    usage: Usage | None = None


# Public union type for the values yielded by the agent's event stream.
Event = TextDelta | ToolCallEvent | ToolResultEvent | StepEvent | FinalEvent


def _truncate(text: str, limit: int = 500) -> str:
    """Shorten long tool output so a live stream stays readable."""

    return text if len(text) <= limit else text[:limit] + "..."


async def print_events(events: AsyncIterator[Event]) -> str:
    """Render an agent's event stream to stdout as it arrives.

    Consumes the iterator from Agent.run(..., stream=True): assistant text
    prints token by token, each tool call and its result get their own line,
    and the final answer is shown at the end. Returns that answer so the caller
    can keep using it once the run has been displayed.
    """

    answer = ""
    mid_line = False  # True while streamed text has left the cursor mid-line.

    async for event in events:
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
                mid_line = True
            case ToolCallEvent(tool_call=call):
                if mid_line:
                    print()
                    mid_line = False
                print(f"-> {call.name}({json.dumps(call.arguments)})")
            case ToolResultEvent(result=result):
                status = "error" if result.is_error else "ok"
                print(f"<- [{status}] {_truncate(result.content)}")
            case FinalEvent(answer=final):
                if mid_line:
                    print()
                    mid_line = False
                answer = final
                print(f"= {final}")
            case StepEvent():
                pass  # Its contents already surfaced through the events above.

    return answer
