"""Streaming events emitted by the agent loop.

Memory is the durable, structured record of a run. Events are runtime
notifications emitted as the run progresses, letting callers render output,
inspect tool execution, and react to completion in real time.

StepEvent connects the live event stream to memory by carrying the step that
was just recorded.
"""

from __future__ import annotations

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
