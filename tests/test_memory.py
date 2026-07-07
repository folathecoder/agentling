import json

import pytest

from agentling.errors import MemoryLoadError
from agentling.memory import (
    ActionStep,
    FinalStep,
    Memory,
    TaskStep,
    ToolResult,
)
from agentling.models import ChatMessage, ToolCall, Usage


def _action_step() -> ActionStep:
    """A representative ActionStep: one tool call, one successful result."""
    return ActionStep(
        model_message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"})
            ],
            usage=Usage(10, 5),
        ),
        tool_results=[
            ToolResult(tool_call_id="c1", name="get_weather", content="Sunny")
        ],
        usage=Usage(10, 5),
        duration=1.2,
    )


# --------------------------------------------------------------------------- #
# Per-step rendering
# --------------------------------------------------------------------------- #
def test_task_step_renders_user_message() -> None:
    msgs = TaskStep(task="hi").to_messages()
    assert [m.role for m in msgs] == ["user"]
    assert msgs[0].content == "hi"


def test_action_step_renders_assistant_then_tool() -> None:
    msgs = _action_step().to_messages()
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1].content == "Sunny"
    assert msgs[1].tool_call_id == "c1"


def test_action_step_renders_error_observation() -> None:
    step = ActionStep(
        model_message=ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="f", arguments={})],
        ),
        tool_results=[
            ToolResult(
                tool_call_id="c1",
                name="f",
                content="boom",
                is_error=True,
                error_kind="validation",
            )
        ],
    )
    tool_msg = step.to_messages()[1]
    assert tool_msg.content == "Error from 'f': boom. Fix the arguments and try again."


def test_action_step_execution_error_uses_a_different_hint() -> None:
    step = ActionStep(
        model_message=ChatMessage(role="assistant"),
        tool_results=[
            ToolResult(
                tool_call_id="c1",
                name="f",
                content="network down",
                is_error=True,
                error_kind="execution",
            )
        ],
    )
    tool_msg = step.to_messages()[1]
    assert "different approach" in tool_msg.content
    assert "Fix the arguments" not in tool_msg.content


def test_final_step_renders_assistant_message() -> None:
    msgs = FinalStep(answer="done").to_messages()
    assert [m.role for m in msgs] == ["assistant"]
    assert msgs[0].content == "done"


# --------------------------------------------------------------------------- #
# Memory.to_messages
# --------------------------------------------------------------------------- #
def test_to_messages_full_shape() -> None:
    mem = Memory()
    mem.add(TaskStep(task="weather?"))
    mem.add(_action_step())
    mem.add(FinalStep(answer="Sunny."))

    msgs = mem.to_messages("You are helpful.")
    assert [m.role for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert msgs[0].content == "You are helpful."


def test_to_messages_empty_is_just_the_system_prompt() -> None:
    msgs = Memory().to_messages("sys")
    assert [m.role for m in msgs] == ["system"]


# --------------------------------------------------------------------------- #
# add / reset (multi-turn support)
# --------------------------------------------------------------------------- #
def test_add_and_reset() -> None:
    mem = Memory()
    mem.add(TaskStep(task="a"))
    mem.add(TaskStep(task="b"))
    assert len(mem.steps) == 2

    mem.reset()
    assert mem.steps == []


# --------------------------------------------------------------------------- #
# JSON round-trip (persistence / replay)
# --------------------------------------------------------------------------- #
def test_json_round_trip_preserves_all_steps() -> None:
    mem = Memory()
    mem.add(TaskStep(task="weather?"))
    mem.add(_action_step())
    mem.add(FinalStep(answer="Sunny."))

    restored = Memory.load_json(mem.dump_json())
    assert restored.steps == mem.steps


def test_dump_json_tags_each_step_with_its_kind() -> None:
    mem = Memory()
    mem.add(TaskStep(task="hi"))
    mem.add(_action_step())
    mem.add(FinalStep(answer="bye"))

    data = json.loads(mem.dump_json())
    assert [entry["type"] for entry in data["steps"]] == ["task", "action", "final"]


def test_round_trip_preserves_error_flag() -> None:
    mem = Memory()
    mem.add(
        ActionStep(
            model_message=ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="f", arguments={})],
            ),
            tool_results=[
                ToolResult(tool_call_id="c1", name="f", content="boom", is_error=True)
            ],
        )
    )

    action = Memory.load_json(mem.dump_json()).steps[0]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True


def test_from_dict_rejects_unknown_step_type() -> None:
    with pytest.raises(MemoryLoadError, match="Unknown step type"):
        Memory.from_dict({"version": 1, "steps": [{"type": "bogus", "data": {}}]})


def test_from_dict_rejects_unsupported_version() -> None:
    with pytest.raises(MemoryLoadError, match="version"):
        Memory.from_dict({"version": 999, "steps": []})


def test_from_dict_rejects_non_list_steps() -> None:
    with pytest.raises(MemoryLoadError, match="steps"):
        Memory.from_dict({"version": 1, "steps": "not a list"})
