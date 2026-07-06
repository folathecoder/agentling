from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentling.agent import Agent
from agentling.events import FinalEvent, StepEvent, ToolCallEvent, ToolResultEvent
from agentling.memory import ActionStep, FinalStep, TaskStep
from agentling.models import ChatMessage, Delta, ToolCall, Usage
from agentling.tools import tool


class FakeModel:
    """A scripted model: returns pre-set ChatMessages in order, no network.

    Records the messages it was called with so tests can assert on the loop.
    """

    def __init__(self, responses: Sequence[ChatMessage]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[list[ChatMessage]] = []

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> ChatMessage:
        self.calls.append(list(messages))
        response = self._responses[self._index]
        self._index += 1
        return response

    def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        raise NotImplementedError


def _assistant(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(1, 1),
    )


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


# --------------------------------------------------------------------------- #
# Termination paths
# --------------------------------------------------------------------------- #
async def test_forgiving_termination_returns_content() -> None:
    model = FakeModel([_assistant(content="42")])
    agent = Agent(model=model)

    answer = await agent.run("What is 6 times 7?")

    assert answer == "42"
    assert isinstance(agent.memory.steps[0], TaskStep)
    assert isinstance(agent.memory.steps[-1], FinalStep)


async def test_explicit_final_answer() -> None:
    model = FakeModel(
        [
            _assistant(
                tool_calls=[
                    ToolCall(id="c1", name="final_answer", arguments={"answer": "done"})
                ]
            )
        ]
    )
    agent = Agent(model=model)

    assert await agent.run("finish up") == "done"


# --------------------------------------------------------------------------- #
# Tool execution
# --------------------------------------------------------------------------- #
async def test_tool_call_then_answer() -> None:
    model = FakeModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
            ),
            _assistant(content="The sum is 5."),
        ]
    )
    agent = Agent(model=model, tools=[add])

    answer = await agent.run("add 2 and 3")

    assert answer == "The sum is 5."
    action = agent.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].content == "5"
    assert action.tool_results[0].is_error is False


async def test_multi_step_tool_task() -> None:
    model = FakeModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
            ),
            _assistant(
                tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 5, "b": 4})]
            ),
            _assistant(
                tool_calls=[
                    ToolCall(id="c3", name="final_answer", arguments={"answer": "9"})
                ]
            ),
        ]
    )
    agent = Agent(model=model, tools=[add])

    assert await agent.run("compute a couple of sums") == "9"
    assert len(model.calls) == 3


async def test_tool_error_becomes_observation() -> None:
    @tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    model = FakeModel(
        [
            _assistant(tool_calls=[ToolCall(id="c1", name="boom", arguments={})]),
            _assistant(content="recovered"),
        ]
    )
    agent = Agent(model=model, tools=[boom])

    assert await agent.run("try boom") == "recovered"
    action = agent.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
async def test_streaming_yields_events() -> None:
    model = FakeModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})]
            ),
            _assistant(content="2"),
        ]
    )
    agent = Agent(model=model, tools=[add])

    events = [event async for event in agent.run("add", stream=True)]

    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) for e in events)
    assert any(isinstance(e, StepEvent) for e in events)
    assert isinstance(events[-1], FinalEvent)
    assert events[-1].answer == "2"


# --------------------------------------------------------------------------- #
# max_steps forced answer
# --------------------------------------------------------------------------- #
async def test_max_steps_forces_a_final_answer() -> None:
    # The model never terminates on its own — it always calls a tool.
    looping = [
        _assistant(
            tool_calls=[ToolCall(id=f"c{i}", name="add", arguments={"a": 1, "b": 1})]
        )
        for i in range(2)
    ]
    forced = _assistant(content="forced final")
    model = FakeModel([*looping, forced])
    agent = Agent(model=model, tools=[add], max_steps=2)

    assert await agent.run("loop forever") == "forced final"
    assert len(model.calls) == 3  # 2 loop steps + 1 forced tool-free answer
