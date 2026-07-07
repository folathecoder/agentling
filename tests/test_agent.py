import asyncio
import io
import json
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from agentling.agent import Agent, AgentSession, _truncate_middle
from agentling.errors import ModelError
from agentling.events import (
    FinalEvent,
    StepEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    print_events,
)
from agentling.memory import ActionStep, FinalStep, TaskStep
from agentling.models import ChatMessage, Delta, ToolCall, ToolCallDelta, Usage
from agentling.skills import Skill
from agentling.tools import ToolCallError, tool


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

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        # Replay the next scripted response as a delta stream that
        # agglomerate_deltas rebuilds into the same ChatMessage.
        self.calls.append(list(messages))
        response = self._responses[self._index]
        self._index += 1
        if response.content:
            yield Delta(content=response.content)
        for index, tc in enumerate(response.tool_calls):
            yield Delta(
                tool_calls=[
                    ToolCallDelta(
                        index=index,
                        id=tc.id,
                        name=tc.name,
                        arguments=json.dumps(tc.arguments),
                    )
                ]
            )
        yield Delta(usage=response.usage)


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


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


@tool
async def slow(seconds: float) -> str:
    """Sleep for a while, then return.

    Args:
        seconds: How long to sleep.
    """
    await asyncio.sleep(seconds)
    return "slept"


_REVIEWER_SKILL = Skill(
    name="reviewer",
    description="Review code for bugs and style issues.",
    instructions="Look for off-by-one errors and missing null checks.",
    path=Path("."),
    tools=[],
)

_CALC_SKILL = Skill(
    name="calc",
    description="Do arithmetic with the multiply tool.",
    instructions="Use the multiply tool to compute products.",
    path=Path("."),
    tools=[f"{__name__}:multiply"],
)


# --------------------------------------------------------------------------- #
# Termination paths
# --------------------------------------------------------------------------- #
async def test_forgiving_termination_returns_content() -> None:
    model = FakeModel([_assistant(content="42")])
    agent = Agent(model=model)
    session = agent.start()

    answer = await session.run("What is 6 times 7?")

    assert answer == "42"
    assert isinstance(session.memory.steps[0], TaskStep)
    assert isinstance(session.memory.steps[-1], FinalStep)


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
    session = agent.start()

    answer = await session.run("add 2 and 3")

    assert answer == "The sum is 5."
    action = session.memory.steps[1]
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
    session = agent.start()

    assert await session.run("try boom") == "recovered"
    action = session.memory.steps[1]
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


async def test_streaming_emits_text_deltas() -> None:
    model = FakeModel([_assistant(content="hello world")])
    agent = Agent(model=model)

    texts = [
        event.text
        async for event in agent.run("hi", stream=True)
        if isinstance(event, TextDelta)
    ]
    assert "".join(texts) == "hello world"


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
    return _assistant(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


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
    session = agent.start()
    await session.run("repeat the same call")

    first, second = session.memory.steps[1], session.memory.steps[2]
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
    session = agent.start()
    await session.run("two different sums")

    for step in session.memory.steps:
        if isinstance(step, ActionStep):
            assert _REPEAT_MARKER not in step.tool_results[0].content


# --------------------------------------------------------------------------- #
# Self-heal: unknown tool and always-raising tool
# --------------------------------------------------------------------------- #
async def test_unknown_tool_recovers() -> None:
    model = FakeModel([_tool_turn("c1", "ghost"), _assistant(content="recovered")])
    agent = Agent(model=model)  # only final_answer is registered
    session = agent.start()
    answer = await session.run("call a missing tool")

    assert answer == "recovered"
    action = session.memory.steps[1]
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
    session = agent.start()
    answer = await session.run("trigger boom")

    assert answer == "handled"
    action = session.memory.steps[1]
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
    session = agent.start()

    fired: list[bool] = []

    def interrupt_after_first(step: object) -> None:
        if not fired:
            fired.append(True)
            session.interrupt()

    session.step_callbacks.append(interrupt_after_first)

    assert await session.run("loop") == "Run interrupted."
    assert isinstance(session.memory.steps[-1], ActionStep)  # paused before a FinalStep
    assert len(model.calls) == 1


async def test_interrupt_then_resume_completes() -> None:
    model = FakeModel(
        [
            _tool_turn("c1", "add", a=1, b=1),
            _assistant(content="finished on resume"),
        ]
    )
    agent = Agent(model=model, tools=[add])
    session = agent.start()

    fired: list[bool] = []

    def interrupt_once(step: object) -> None:
        if not fired:
            fired.append(True)
            session.interrupt()

    session.step_callbacks.append(interrupt_once)

    assert await session.run("start") == "Run interrupted."
    # reset=False keeps the interrupted run's memory; the loop resumes from it.
    assert await session.run("continue", reset=False) == "finished on resume"
    assert len(model.calls) == 2
    assert sum(isinstance(s, TaskStep) for s in session.memory.steps) == 2


# --------------------------------------------------------------------------- #
# Skills: progressive disclosure
# --------------------------------------------------------------------------- #
def _load_skill_turn(call_id: str, skill_name: str) -> ChatMessage:
    """A model turn calling load_skill (whose own argument is named 'name')."""
    return _assistant(
        tool_calls=[
            ToolCall(id=call_id, name="load_skill", arguments={"name": skill_name})
        ]
    )


def test_skill_catalog_is_added_to_the_system_prompt() -> None:
    agent = Agent(model=FakeModel([]), skills=[_REVIEWER_SKILL])
    session = agent.start()

    assert "load_skill" in session.tools
    assert "reviewer" in agent.instructions
    assert "Review code for bugs and style issues." in agent.instructions
    # Only the name and description surface up front; the body stays hidden
    # until the skill is loaded.
    assert "off-by-one" not in agent.instructions


def test_no_skills_means_no_load_skill_tool() -> None:
    agent = Agent(model=FakeModel([]))

    assert "load_skill" not in agent.start().tools


async def test_load_skill_reveals_the_body() -> None:
    model = FakeModel(
        [
            _load_skill_turn("c1", "reviewer"),
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, skills=[_REVIEWER_SKILL])
    session = agent.start()

    assert await session.run("review this") == "done"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].content == _REVIEWER_SKILL.instructions
    assert action.tool_results[0].is_error is False


async def test_load_skill_registers_the_skills_tools() -> None:
    model = FakeModel(
        [
            _load_skill_turn("c1", "calc"),
            _tool_turn("c2", "multiply", a=6, b=7),
            _assistant(content="42"),
        ]
    )
    agent = Agent(model=model, skills=[_CALC_SKILL])
    session = agent.start()

    # multiply is hidden until the skill that provides it is loaded.
    assert "multiply" not in session.tools

    assert await session.run("compute six times seven") == "42"

    assert "multiply" in session.tools
    load = session.memory.steps[1]
    assert isinstance(load, ActionStep)
    assert "Tools now available: multiply." in load.tool_results[0].content
    product = session.memory.steps[2]
    assert isinstance(product, ActionStep)
    assert product.tool_results[0].content == "42"


async def test_load_unknown_skill_is_an_error_observation() -> None:
    model = FakeModel(
        [
            _load_skill_turn("c1", "ghost"),
            _assistant(content="recovered"),
        ]
    )
    agent = Agent(model=model, skills=[_REVIEWER_SKILL])
    session = agent.start()

    assert await session.run("load a missing skill") == "recovered"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "Unknown skill" in action.tool_results[0].content


def test_skill_can_be_loaded_from_a_path(tmp_path: Path) -> None:
    folder = tmp_path / "greeter"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: greeter\ndescription: Greet warmly.\n---\nSay hi.\n",
        encoding="utf-8",
    )
    agent = Agent(model=FakeModel([]), skills=[folder])

    assert "greeter" in agent.skills
    assert "greeter" in agent.instructions


# --------------------------------------------------------------------------- #
# Construction guards
# --------------------------------------------------------------------------- #
def test_duplicate_tool_name_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate tool name"):
        Agent(model=FakeModel([]), tools=[add, add])


def test_max_steps_below_one_raises() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Agent(model=FakeModel([]), max_steps=0)


async def test_run_max_steps_below_one_raises() -> None:
    agent = Agent(model=FakeModel([_assistant(content="unused")]))

    with pytest.raises(ValueError, match="at least 1"):
        await agent.run("hi", max_steps=0)


# --------------------------------------------------------------------------- #
# Session isolation
# --------------------------------------------------------------------------- #
async def test_sessions_keep_separate_memory() -> None:
    agent = Agent(model=FakeModel([_assistant(content="one")]))
    s1 = agent.start()
    s2 = agent.start()

    await s1.run("first")

    assert s1.memory is not s2.memory
    assert len(s1.memory.steps) > 0
    assert s2.memory.steps == []  # a sibling session is untouched


async def test_skill_tools_do_not_leak_between_sessions() -> None:
    model = FakeModel(
        [
            _load_skill_turn("c1", "calc"),
            _tool_turn("c2", "multiply", a=2, b=2),
            _assistant(content="4"),
        ]
    )
    agent = Agent(model=model, skills=[_CALC_SKILL])
    loader = agent.start()
    other = agent.start()

    await loader.run("use calc")

    assert "multiply" in loader.tools  # registered by load_skill in this session
    assert "multiply" not in other.tools  # never leaks into a sibling session


async def test_interrupt_affects_only_its_session() -> None:
    agent = Agent(model=FakeModel([_assistant(content="s2 done")]))
    s1 = agent.start()
    s2 = agent.start()

    s1.interrupt()

    # s1 sees its own interrupt and stops before the first step (no model call).
    assert await s1.run("stop") == "Run interrupted."
    # s2 is unaffected and runs to completion.
    assert await s2.run("go") == "s2 done"


# --------------------------------------------------------------------------- #
# Malformed model output (recoverable)
# --------------------------------------------------------------------------- #
class _MalformedThenAnswerModel:
    """Streams one unparseable tool call, then a plain answer on the retry."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> ChatMessage:
        raise NotImplementedError

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        self.calls += 1
        if self.calls == 1:
            yield Delta(
                tool_calls=[
                    ToolCallDelta(index=0, id="c1", name="add", arguments="{bad json")
                ]
            )
        else:
            yield Delta(content="recovered")
        yield Delta(usage=Usage(1, 1))


class _AlwaysMalformedModel:
    """Every turn streams an unparseable tool call."""

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> ChatMessage:
        raise NotImplementedError

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        yield Delta(
            tool_calls=[
                ToolCallDelta(index=0, id="c1", name="add", arguments="{bad json")
            ]
        )
        yield Delta(usage=Usage(1, 1))


async def test_malformed_tool_call_is_recoverable() -> None:
    model = _MalformedThenAnswerModel()
    agent = Agent(model=model)

    assert await agent.run("do something") == "recovered"
    assert model.calls == 2  # first turn malformed, re-prompted, then answered


async def test_persistently_malformed_output_raises_model_error() -> None:
    agent = Agent(model=_AlwaysMalformedModel())

    with pytest.raises(ModelError):
        await agent.run("do something")


# --------------------------------------------------------------------------- #
# Timeouts and cancellation
# --------------------------------------------------------------------------- #
async def test_tool_timeout_becomes_observation() -> None:
    model = FakeModel([_tool_turn("c1", "slow", seconds=5), _assistant(content="done")])
    agent = Agent(model=model, tools=[slow], tool_timeout=0.01)
    session = agent.start()

    assert await session.run("call slow") == "done"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "ToolTimeoutError" in action.tool_results[0].content


async def test_parallel_timeout_keeps_other_results() -> None:
    model = FakeModel(
        [
            _assistant(
                tool_calls=[
                    ToolCall(id="c1", name="slow", arguments={"seconds": 5}),
                    ToolCall(id="c2", name="add", arguments={"a": 2, "b": 3}),
                ]
            ),
            _assistant(content="both handled"),
        ]
    )
    agent = Agent(model=model, tools=[slow, add], tool_timeout=0.01)
    session = agent.start()

    assert await session.run("two tools") == "both handled"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    by_name = {result.name: result for result in action.tool_results}
    assert by_name["slow"].is_error is True
    assert by_name["add"].is_error is False
    assert by_name["add"].content == "5"


class _SlowStreamModel:
    """A model whose stream stalls before producing anything."""

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> ChatMessage:
        raise NotImplementedError

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        await asyncio.sleep(5)
        yield Delta(content="too late")


async def test_model_stream_timeout_raises_model_error() -> None:
    agent = Agent(model=_SlowStreamModel(), model_timeout=0.01)

    with pytest.raises(ModelError):
        await agent.run("hi")


async def test_cancellation_propagates_during_a_tool() -> None:
    model = FakeModel([_tool_turn("c1", "slow", seconds=5), _assistant(content="done")])
    agent = Agent(model=model, tools=[slow])  # no timeout: relies on cancellation

    async def run_it() -> str:
        return await agent.run("call slow")

    task = asyncio.ensure_future(run_it())
    await asyncio.sleep(0.02)  # let the run reach the slow tool

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class _InterruptMidStreamModel:
    """Interrupts its own session partway through streaming."""

    def __init__(self) -> None:
        self.session: AgentSession | None = None

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> ChatMessage:
        raise NotImplementedError

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        yield Delta(content="par")
        assert self.session is not None
        self.session.interrupt()
        yield Delta(content="tial")
        yield Delta(usage=Usage(1, 1))


async def test_interrupt_during_stream_stops_the_run() -> None:
    model = _InterruptMidStreamModel()
    agent = Agent(model=model)
    session = agent.start()
    model.session = session

    assert await session.run("stream") == "Run interrupted."


# --------------------------------------------------------------------------- #
# Tool execution hardening
# --------------------------------------------------------------------------- #
async def test_sync_tools_run_off_the_event_loop_thread() -> None:
    @tool
    def where() -> int:
        """Return the id of the thread the tool ran on."""
        return threading.get_ident()

    model = FakeModel([_tool_turn("c1", "where"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[where])
    session = agent.start()

    await session.run("which thread")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    # The sync tool ran in a worker thread, not the event loop's thread.
    assert action.tool_results[0].content != str(threading.get_ident())


async def test_tool_output_is_truncated() -> None:
    @tool
    def big() -> str:
        """Return a large string."""
        return "x" * 1000

    model = FakeModel([_tool_turn("c1", "big"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[big], max_tool_output_chars=100)
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    content = action.tool_results[0].content
    assert "characters omitted" in content
    assert len(content) < 300


async def test_errors_are_shown_by_default() -> None:
    @tool
    def leak() -> str:
        """Raise with a message."""
        raise RuntimeError("visible detail")

    model = FakeModel([_tool_turn("c1", "leak"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[leak])
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert "visible detail" in action.tool_results[0].content


async def test_error_redaction_hides_the_message() -> None:
    @tool
    def leak() -> str:
        """Raise with a secret."""
        raise RuntimeError("token=SECRET123")

    model = FakeModel([_tool_turn("c1", "leak"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[leak], redact_errors=True)
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    content = action.tool_results[0].content
    assert "SECRET123" not in content  # message suppressed
    assert "RuntimeError" in content  # the type is still surfaced


async def test_tool_call_error_shown_even_when_redacting() -> None:
    @tool
    def bad() -> str:
        """Raise a ToolCallError."""
        raise ToolCallError("use a positive number")

    model = FakeModel([_tool_turn("c1", "bad"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[bad], redact_errors=True)
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert "use a positive number" in action.tool_results[0].content


async def test_non_parallel_safe_tools_run_in_order() -> None:
    order: list[str] = []

    @tool(parallel_safe=False)
    async def first() -> str:
        """First tool."""
        order.append("first-start")
        await asyncio.sleep(0.02)
        order.append("first-end")
        return "1"

    @tool
    async def second() -> str:
        """Second tool."""
        order.append("second")
        return "2"

    model = FakeModel(
        [
            _assistant(
                tool_calls=[
                    ToolCall(id="c1", name="first", arguments={}),
                    ToolCall(id="c2", name="second", arguments={}),
                ]
            ),
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, tools=[first, second])
    session = agent.start()

    await session.run("go")
    # first is not parallel_safe, so the step runs sequentially: it finishes
    # before second starts.
    assert order == ["first-start", "first-end", "second"]


# --------------------------------------------------------------------------- #
# Memory hardening: cumulative usage and the context-window hook
# --------------------------------------------------------------------------- #
async def test_final_event_reports_cumulative_usage() -> None:
    model = FakeModel([_tool_turn("c1", "add", a=1, b=1), _assistant(content="2")])
    agent = Agent(model=model, tools=[add])

    events = [event async for event in agent.run("go", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    # Each FakeModel turn reports Usage(1, 1); two turns sum to Usage(2, 2).
    assert final.usage == Usage(2, 2)


async def test_context_manager_trims_the_prompt() -> None:
    def keep_only_system(messages: list[ChatMessage]) -> list[ChatMessage]:
        return messages[:1]

    model = FakeModel([_assistant(content="ok")])
    agent = Agent(model=model, context_manager=keep_only_system)

    assert await agent.run("hello") == "ok"
    # The model saw only the trimmed message list.
    assert len(model.calls[0]) == 1


# --------------------------------------------------------------------------- #
# Terminal run status and print_events
# --------------------------------------------------------------------------- #
async def test_completed_run_reports_status() -> None:
    model = FakeModel([_assistant(content="done")])
    agent = Agent(model=model)

    events = [event async for event in agent.run("go", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.status == "completed"


async def test_interrupted_run_reports_status() -> None:
    model = FakeModel([_tool_turn("c1", "add", a=1, b=1), _assistant(content="x")])
    agent = Agent(model=model, tools=[add])
    session = agent.start()

    fired: list[bool] = []

    def stop(step: object) -> None:
        if not fired:
            fired.append(True)
            session.interrupt()

    session.step_callbacks.append(stop)

    events = [event async for event in session.run("loop", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.status == "interrupted"


async def test_max_steps_run_reports_status() -> None:
    looping = [_tool_turn(f"c{i}", "add", a=1, b=1) for i in range(2)]
    model = FakeModel([*looping, _assistant(content="forced")])
    agent = Agent(model=model, tools=[add], max_steps=2)

    events = [event async for event in agent.run("loop", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.answer == "forced"
    assert final.status == "max_steps"


async def test_print_events_writes_to_file_and_returns_answer() -> None:
    model = FakeModel([_tool_turn("c1", "add", a=2, b=3), _assistant(content="5")])
    agent = Agent(model=model, tools=[add])

    buffer = io.StringIO()
    answer = await print_events(agent.run("add", stream=True), file=buffer)

    assert answer == "5"
    output = buffer.getvalue()
    assert "add" in output  # the tool call was rendered
    assert "5" in output  # the final answer was rendered


# --------------------------------------------------------------------------- #
# Pre-release edge fixes
# --------------------------------------------------------------------------- #
def test_load_skill_name_is_reserved_when_skills_are_configured() -> None:
    @tool
    def load_skill(name: str) -> str:
        """A tool that collides with the built-in skill loader.

        Args:
            name: Unused.
        """
        return name

    with pytest.raises(ValueError, match="reserved"):
        Agent(model=FakeModel([]), tools=[load_skill], skills=[_REVIEWER_SKILL])


def test_truncate_middle_handles_tiny_limits() -> None:
    assert _truncate_middle("abc", 10) == "abc"  # under the limit, unchanged
    assert _truncate_middle("abcdef", 0) == ""
    assert _truncate_middle("abcdef", 1) == "a"

    windowed = _truncate_middle("x" * 100, 20)
    assert "omitted" in windowed
    assert len(windowed) < 100
