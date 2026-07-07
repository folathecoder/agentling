"""Typed step records and the agent's memory.

The agent loop records each turn as a typed Step (TaskStep, ActionStep,
FinalStep) rather than as raw messages. Steps carry metadata such as token
usage, timing, and per-tool results and errors, and they know how to render
themselves back into the ChatMessage list the model sees. Memory holds the
steps, renders the full conversation via to_messages(), and round-trips to JSON
for persistence and replay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from .models import ChatMessage, ToolCall, Usage


@dataclass
class ToolResult:
    """The outcome of a single executed tool call."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class TaskStep:
    """A task that started (or, in multi-turn, continued) the run."""

    task: str

    def to_messages(self) -> list[ChatMessage]:
        """Render as a single user message."""

        return [ChatMessage(role="user", content=self.task)]


@dataclass
class ActionStep:
    """One agent loop iteration.

    Stores the assistant message the model produced, any tool results from that
    message, and optional execution metadata (error, usage, timing).
    """

    model_message: ChatMessage
    tool_results: list[ToolResult] = field(default_factory=list)
    error: str | None = None
    usage: Usage | None = None
    duration: float | None = None

    def to_messages(self) -> list[ChatMessage]:
        """Render the assistant turn, then one tool message per result."""

        # model_message is already role="assistant"; reuse it as-is.
        messages: list[ChatMessage] = [self.model_message]

        for result in self.tool_results:
            content = result.content
            # Failed tool calls are returned to the model as observations so the
            # next turn can recover. The tool name is included because a single
            # step may contain several calls.
            if result.is_error:
                content = (
                    f"Error from {result.name!r}: {result.content}. "
                    "Fix the arguments and try again."
                )
            messages.append(
                ChatMessage(
                    role="tool", content=content, tool_call_id=result.tool_call_id
                )
            )

        return messages


@dataclass
class FinalStep:
    """The terminal answer produced by the run."""

    answer: str

    def to_messages(self) -> list[ChatMessage]:
        """Render as the assistant's final message (keeps multi-turn history whole)."""

        return [ChatMessage(role="assistant", content=self.answer)]


# A run's history is a sequence of these three step kinds.
Step = TaskStep | ActionStep | FinalStep


@dataclass
class Memory:
    """Ordered history of an agent run.

    Stores typed steps rather than raw provider messages, renders them into the
    ChatMessage list the model sees, and serializes them for persistence/replay.
    """

    # Discriminator tag written into each serialized step so from_dict() can
    # rebuild the right class. ClassVar keeps it class-level config, not a field.
    _KIND: ClassVar[dict[type, str]] = {
        TaskStep: "task",
        ActionStep: "action",
        FinalStep: "final",
    }

    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        """Append a step to the run's history."""

        self.steps.append(step)

    def reset(self) -> None:
        """Clear all steps (used by run(reset=True) to start fresh)."""

        self.steps.clear()

    def to_messages(self, system_prompt: str) -> list[ChatMessage]:
        """Render the full conversation: system prompt, then every step."""

        # The system prompt is runtime configuration, not run history. Injecting
        # it here means it is always first and never duplicated in stored steps.
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]
        for step in self.steps:
            messages.extend(step.to_messages())
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, tagging each step with its kind."""

        # asdict() recurses the whole nested tree (ChatMessage -> ToolCalls ->
        # Usage, ToolResults) into plain dicts but drops type information, which
        # the load path restores from each step's "type" tag. The version field
        # lets the format evolve without breaking older saved runs.
        return {
            "version": 1,
            "steps": [
                {"type": self._KIND[type(step)], "data": asdict(step)}
                for step in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Rebuild a Memory from the dict produced by to_dict()."""

        return cls(steps=[_step_from_dict(entry) for entry in data.get("steps", [])])

    def dump_json(self) -> str:
        """Serialize the memory to a JSON string."""

        return json.dumps(self.to_dict())

    @classmethod
    def load_json(cls, raw: str) -> Memory:
        """Rebuild a Memory from a JSON string produced by dump_json()."""

        return cls.from_dict(json.loads(raw))


# dataclasses.asdict() converts nested dataclasses to dicts but does not record
# their types, so the load path rebuilds each nested object explicitly.
def _chat_message_from_dict(d: dict[str, Any]) -> ChatMessage:
    """Reconstruct a ChatMessage (and its nested ToolCalls/Usage) from a dict."""

    return ChatMessage(
        role=d["role"],
        content=d["content"],
        tool_calls=[ToolCall(**tc) for tc in d["tool_calls"]],
        tool_call_id=d["tool_call_id"],
        usage=Usage(**d["usage"]) if d["usage"] else None,
    )


def _step_from_dict(entry: dict[str, Any]) -> Step:
    """Reconstruct a Step from a {"type", "data"} entry, dispatching on the tag."""

    kind, data = entry["type"], entry["data"]

    if kind == "task":
        return TaskStep(**data)
    if kind == "final":
        return FinalStep(**data)
    if kind == "action":
        return ActionStep(
            model_message=_chat_message_from_dict(data["model_message"]),
            tool_results=[ToolResult(**tr) for tr in data["tool_results"]],
            error=data["error"],
            usage=Usage(**data["usage"]) if data["usage"] else None,
            duration=data["duration"],
        )

    raise ValueError(f"Unknown step type: {kind!r}")
