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


def _tool_turn(call_id: str, name: str, **arguments: Any) -> ChatMessage:
    """A model turn that requests a single tool call."""
    return _assistant(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)]
    )


# --------------------------------------------------------------------------- #
# Loop detector
# --------------------------------------------------------------------------- #
_REPEAT_MARKER = "already made this exact tool call"


async def test_loop_detector_nudges_on_repeat() -> None:
    model = FakeModel(
        [
            _tool_turn("c1", "add", a=2, b=3),
            _tool_turn("c2", "add", a=2, b=3),  # identical (name, args)
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, tools=[add])
    await agent.run("repeat the same call")

    first, second = agent.memory.steps[1], agent.memory.steps[2]
    assert isinstance(first, ActionStep)
    assert isinstance(second, ActionStep)
    assert _REPEAT_MARKER not in first.tool_results[0].content
    assert _REPEAT_MARKER in second.tool_results[0].content


async def test_loop_detector_ignores_different_args() -> None:
    model = FakeModel(
        [
            _tool_turn("c1", "add", a=2, b=3),
            _tool_turn("c2", "add", a=4, b=5),  # different args -> not a loop
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, tools=[add])
    await agent.run("two different sums")

    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            assert _REPEAT_MARKER not in step.tool_results[0].content


# --------------------------------------------------------------------------- #
# Self-heal: unknown tool and always-raising tool
# --------------------------------------------------------------------------- #
async def test_unknown_tool_recovers() -> None:
    model = FakeModel([_tool_turn("c1", "ghost"), _assistant(content="recovered")])
    agent = Agent(model=model)  # only final_answer is registered
    answer = await agent.run("call a missing tool")

    assert answer == "recovered"
    action = agent.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "Unknown tool" in action.tool_results[0].content


async def test_always_raising_tool_recovers() -> None:
    @tool
    def boom() -> str:
        """Always raises."""
        raise RuntimeError("nope")

    model = FakeModel(
        [
            _tool_turn("c1", "boom"),
            _tool_turn("c2", "final_answer", answer="handled"),
        ]
    )
    agent = Agent(model=model, tools=[boom])
    answer = await agent.run("trigger boom")

    assert answer == "handled"
    action = agent.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "RuntimeError" in action.tool_results[0].content


# --------------------------------------------------------------------------- #
# Interruption + resume
# --------------------------------------------------------------------------- #
async def test_interrupt_stops_run_gracefully() -> None:
    model = FakeModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="unreached")]
    )
    agent = Agent(model=model, tools=[add])

    fired: list[bool] = []

    def interrupt_after_first(step: object) -> None:
        if not fired:
            fired.append(True)
            agent.interrupt()

    agent.step_callbacks.append(interrupt_after_first)

    assert await agent.run("loop") == "Run interrupted."
    assert isinstance(agent.memory.steps[-1], ActionStep)  # paused before a FinalStep
    assert len(model.calls) == 1


async def test_interrupt_then_resume_completes() -> None:
    model = FakeModel(
        [
            _tool_turn("c1", "add", a=1, b=1),
            _assistant(content="finished on resume"),
        ]
    )
    agent = Agent(model=model, tools=[add])

    fired: list[bool] = []

    def interrupt_once(step: object) -> None:
        if not fired:
            fired.append(True)
            agent.interrupt()

    agent.step_callbacks.append(interrupt_once)

    assert await agent.run("start") == "Run interrupted."
    # reset=False keeps the interrupted run's memory; the loop resumes from it.
    assert await agent.run("continue", reset=False) == "finished on resume"
    assert len(model.calls) == 2
    assert sum(isinstance(s, TaskStep) for s in agent.memory.steps) == 2
