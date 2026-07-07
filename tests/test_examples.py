import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import examples.cli_advanced_observer as observer
import examples.cli_failure_recovery as failure_recovery
import examples.cli_math_tutor as math_tutor
import examples.cli_memory_chat as memory_chat
import examples.cli_notes_agent as notes_agent
import examples.cli_repo_assistant as repo_assistant
from agentling import (
    ActionStep,
    ChatMessage,
    Delta,
    Memory,
    TaskStep,
    ToolCall,
    ToolCallDelta,
    ToolCallError,
    ToolResultEvent,
    Usage,
)


class _ScriptedModel:
    """A deterministic fake model for driving the API-backed examples offline."""

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


def _call(call_id: str, name: str, /, **arguments: Any) -> ToolCall:
    # name is positional-only so a tool argument literally called "name"
    # (as in add_note) does not collide with this parameter.
    return ToolCall(id=call_id, name=name, arguments=arguments)


# --------------------------------------------------------------------------- #
# Import / shape
# --------------------------------------------------------------------------- #
def test_offline_examples_expose_build_agent_and_main() -> None:
    for module in (failure_recovery, memory_chat, observer):
        assert callable(module.build_agent)
        assert callable(module.main)


# --------------------------------------------------------------------------- #
# Failure recovery (offline)
# --------------------------------------------------------------------------- #
async def test_failure_recovery_recovers_from_tool_errors() -> None:
    session = failure_recovery.build_agent().start()
    answer = await session.run("compute")

    assert "5" in answer  # the recovered final answer
    action_steps = [s for s in session.memory.steps if isinstance(s, ActionStep)]
    assert action_steps  # at least one step was recorded
    errors = [r for s in action_steps for r in s.tool_results if r.is_error]
    assert errors  # a tool failure surfaced as a recoverable observation


# --------------------------------------------------------------------------- #
# Memory replay (offline)
# --------------------------------------------------------------------------- #
async def test_memory_replay_dumps_reloads_and_continues() -> None:
    agent = memory_chat.build_agent()

    first = agent.start()
    await first.run("Hi, my name is Sam.")
    saved = first.memory.dump_json()
    assert saved  # memory serialized to JSON

    second = agent.start()
    second.memory = Memory.load_json(saved)
    assert second.memory.steps  # restored the prior run's steps

    await second.run("What is my name?", reset=False)
    task_steps = [s for s in second.memory.steps if isinstance(s, TaskStep)]
    assert len(task_steps) > 1  # continuation built on the restored memory


# --------------------------------------------------------------------------- #
# Event observer (offline)
# --------------------------------------------------------------------------- #
async def test_observer_handles_every_event_type() -> None:
    counts = await observer.observe(observer.build_agent())

    for event_type in (
        "TextDelta",
        "ToolCallEvent",
        "ToolResultEvent",
        "StepEvent",
        "FinalEvent",
    ):
        assert counts.get(event_type, 0) >= 1


# --------------------------------------------------------------------------- #
# API-backed examples (driven offline with an injected fake model)
# --------------------------------------------------------------------------- #
def test_api_examples_expose_build_agent_and_main() -> None:
    for module in (math_tutor, repo_assistant, notes_agent):
        assert callable(module.build_agent)
        assert callable(module.main)


async def test_math_tutor_uses_its_tools() -> None:
    model = _ScriptedModel(
        [
            _assistant(tool_calls=[_call("c1", "multiply", a=6, b=7)]),
            _assistant(tool_calls=[_call("c2", "add", a=42, b=3)]),
            _assistant(content="6 times 7 plus 3 is 45."),
        ]
    )

    answer = await math_tutor.build_agent(model=model).run("6*7+3?")
    assert "45" in answer


async def test_repo_assistant_reads_within_root_and_streams(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi there", encoding="utf-8")
    model = _ScriptedModel(
        [
            _assistant(tool_calls=[_call("c1", "read_file", path="hello.txt")]),
            _assistant(content="The file says hi there."),
        ]
    )
    agent = repo_assistant.build_agent(model=model, root=str(tmp_path))

    events = [event async for event in agent.run("read hello.txt", stream=True)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results and results[0].result.content == "hi there"


def test_repo_assistant_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ToolCallError):
        repo_assistant._resolve_within(tmp_path, "../secret.txt")


def test_repo_assistant_sets_output_cap() -> None:
    agent = repo_assistant.build_agent(model=_ScriptedModel([]))
    assert agent.max_tool_output_chars == 2000


async def test_notes_agent_adds_and_searches(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    model = _ScriptedModel(
        [
            _assistant(
                tool_calls=[_call("c1", "add_note", name="todo", text="buy milk")]
            ),
            _assistant(tool_calls=[_call("c2", "search_notes", query="milk")]),
            _assistant(content="Found it in your notes."),
        ]
    )
    session = notes_agent.build_agent(model=model, notes_dir=str(notes_dir)).start()
    await session.run("add and search a note")

    assert (notes_dir / "todo.txt").exists()
    search_step = session.memory.steps[2]
    assert isinstance(search_step, ActionStep)
    assert "todo" in search_step.tool_results[0].content


def test_notes_agent_rejects_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(ToolCallError):
        notes_agent._resolve_within(tmp_path, "../escape.txt")


def test_notes_agent_configures_timeout_and_redaction() -> None:
    agent = notes_agent.build_agent(model=_ScriptedModel([]), notes_dir="unused")
    assert agent.tool_timeout == 10.0
    assert agent.redact_errors is True
