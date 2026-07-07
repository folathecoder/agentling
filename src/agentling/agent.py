"""The agent loop.

Agent ties the framework's primitives (a Model, some Tools, and a Memory of
typed steps) into an async ReAct loop. One async generator, _run_stream, does
the real work and yields Events. run() is a thin dispatcher: it hands back that
stream when stream is True, or drains it and returns the final answer otherwise.
"""

from __future__ import annotations

import asyncio
import time
import json
from dataclasses import replace
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from .events import Event, FinalEvent, StepEvent, ToolCallEvent, ToolResultEvent
from .memory import ActionStep, FinalStep, Memory, Step, TaskStep, ToolResult
from .models import ChatMessage, Model, ToolCall
from .tools import FinalAnswerTool, Tool

DEFAULT_INSTRUCTIONS = (
    "You are a helpful agent. Use the available tools to gather information or "
    "take actions, thinking step by step. When you have the answer, call the "
    "final_answer tool (or simply reply with plain text)."
)

_LOOP_NUDGE = (
    " Note: you already made this exact tool call and got the same result. "
    "Try a different approach."
)


class Agent:
    """An async tool-calling agent.

    Wires a Model, Tools (with final_answer always available), and a Memory into
    a ReAct loop. Call run() to execute a task, optionally streaming Events.
    """

    def __init__(
        self,
        model: Model,
        tools: Sequence[Tool] = (),
        skills: Sequence[Any] = (),
        instructions: str | None = None,
        max_steps: int = 15,
        step_callbacks: Sequence[Callable[[Step], None]] = (),
        parallel_tools: bool = True,
    ) -> None:
        self.model = model
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        self.max_steps = max_steps
        self.step_callbacks = list(step_callbacks)
        self.parallel_tools = parallel_tools
        self.skills = list(skills)  # accepted but not consumed yet

        self.memory = Memory()
        self._interrupt = asyncio.Event()

        all_tools = [*tools, FinalAnswerTool()]
        self.tools: dict[str, Tool] = {tool.name: tool for tool in all_tools}
        self._tool_schemas = [tool.to_schema() for tool in all_tools]

    def run(
        self,
        task: str,
        *,
        stream: bool = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Any:
        """Run the agent on a task.

        With stream=False (the default) this returns a coroutine that resolves
        to the final answer string:

            answer = await agent.run(task)

        With stream=True it returns an async iterator of Events instead:

            async for event in agent.run(task, stream=True):
                ...
        """
        events = self._run_stream(task, reset=reset, max_steps=max_steps)
        if stream:
            return events
        return self._drain(events)

    async def _drain(self, events: AsyncIterator[Event]) -> str:
        """Consume the event stream and return the final answer."""
        answer = ""
        async for event in events:
            if isinstance(event, FinalEvent):
                answer = event.answer
        return answer

    async def _run_stream(
        self, task: str, *, reset: bool = True, max_steps: int | None = None
    ) -> AsyncIterator[Event]:
        """The core loop: drive the model/tool cycle, yielding Events."""
        if reset:
            self.memory.reset()

        self.memory.add(TaskStep(task=task))

        limit = max_steps or self.max_steps

        # Remember the previous step's calls so we can spot an exact repeat.
        previous_signature: tuple[tuple[str, str], ...] | None = None

        for _ in range(limit):
            if self._interrupt.is_set():
                self._interrupt.clear()
                yield FinalEvent(answer="Run interrupted.")
                return

            started = time.monotonic()
            messages = self.memory.to_messages(self.instructions)
            response = await self.model.generate(messages, tools=self._tool_schemas)

            # Forgiving termination: no tool calls means the content is the answer.
            if not response.tool_calls:
                self.memory.add(FinalStep(answer=response.content))
                yield FinalEvent(answer=response.content, usage=response.usage)
                return

            # Fingerprint this step's calls: (name, canonical JSON args) each.
            signature = tuple(
                (tc.name, json.dumps(tc.arguments, sort_keys=True))
                for tc in response.tool_calls
            )
            looping = signature == previous_signature
            previous_signature = signature

            # Announce every call, then run them (concurrently or in order).
            for tool_call in response.tool_calls:
                yield ToolCallEvent(tool_call=tool_call)

            if self.parallel_tools:
                results: list[ToolResult] = await asyncio.gather(
                    *(self._execute_tool(tc) for tc in response.tool_calls)
                )
            else:
                results = [await self._execute_tool(tc) for tc in response.tool_calls]

            # Same calls as the last step: nudge the model to change approach.
            if looping:
                results = [replace(r, content=r.content + _LOOP_NUDGE) for r in results]

            for result in results:
                yield ToolResultEvent(result=result)

            action = ActionStep(
                model_message=response,
                tool_results=results,
                usage=response.usage,
                duration=time.monotonic() - started,
            )
            self.memory.add(action)
            for callback in self.step_callbacks:
                callback(action)
            yield StepEvent(step=action)

            # Explicit termination: the model called final_answer (successfully).
            final = next(
                (r for r in results if r.name == "final_answer" and not r.is_error),
                None,
            )
            if final is not None:
                self.memory.add(FinalStep(answer=final.content))
                yield FinalEvent(answer=final.content, usage=response.usage)
                return

        # Step limit reached: force one tool-free answer.
        messages = self.memory.to_messages(self.instructions)
        messages.append(
            ChatMessage(
                role="user",
                content="Step limit reached. Give your best final answer now.",
            )
        )
        response = await self.model.generate(messages)
        self.memory.add(FinalStep(answer=response.content))
        yield FinalEvent(answer=response.content, usage=response.usage)

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Run one tool call, turning any failure into an error observation."""
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=(
                    f"Unknown tool {tool_call.name!r}. "
                    f"Available: {list(self.tools)}"
                ),
                is_error=True,
            )
        try:
            output = await tool.call(tool_call.arguments)
        except Exception as exc:
            # A failing tool becomes an observation the model can recover from
            # rather than a crash, so catching broadly here is intentional.
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=str(output),
            is_error=False,
        )

    def interrupt(self) -> None:
        """Request a graceful stop before the next step.

        The current run pauses rather than crashing; resume it later with
        run(..., reset=False), which continues from the steps already in memory.
        """
        self._interrupt.set()
