"""The agent loop.

Agent holds immutable configuration (a Model, some Tools, some Skills, and the
run settings) and acts as a factory for sessions. AgentSession owns one
conversation's mutable state (a Memory of typed steps, an interrupt token, and
its own tool set) and runs the ReAct loop.

Splitting the two means a single Agent can be built once and shared safely
across concurrent runs: each run gets its own AgentSession, so memories,
interrupts, and dynamically loaded skill tools never leak between them.

One async generator, AgentSession._run_stream, does the real work and yields
Events. run() is a thin dispatcher: it hands back that stream when stream is
True, or drains it and returns the final answer otherwise.

Skills are disclosed progressively: only their names and descriptions are added
to the system prompt up front. The full instructions, and any tools a skill
declares, arrive when the model calls the built-in load_skill tool.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, overload

from .errors import ModelError, ModelOutputError, ToolTimeoutError
from .events import (
    Event,
    FinalEvent,
    StepEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from .memory import ActionStep, FinalStep, Memory, Step, TaskStep, ToolResult
from .models import ChatMessage, Delta, Model, ToolCall, ToolSpec, agglomerate_deltas
from .skills import Skill
from .tools import FinalAnswerTool, Tool, ToolCallError, tool

DEFAULT_INSTRUCTIONS = (
    "You are a helpful agent. Use the available tools to gather information or "
    "take actions, thinking step by step. When you have the answer, call the "
    "final_answer tool (or simply reply with plain text)."
)

_LOOP_NUDGE = (
    " Note: you already made this exact tool call and got the same result. "
    "Try a different approach."
)

# How many consecutive unparseable model turns to tolerate before giving up.
_MAX_MALFORMED_RETRIES = 2


class Agent:
    """Immutable configuration and a factory for sessions.

    An Agent bundles a Model, the base Tools (final_answer is always added),
    Skills, and the run settings. It holds no per-run state, so one Agent can be
    built once and shared across many concurrent runs. Call start() for a fresh
    session, or run() for a one-shot convenience run.
    """

    def __init__(
        self,
        model: Model,
        tools: Sequence[Tool] = (),
        skills: Sequence[Skill | str | Path] = (),
        instructions: str | None = None,
        max_steps: int = 15,
        step_callbacks: Sequence[Callable[[Step], None]] = (),
        parallel_tools: bool = True,
        tool_timeout: float | None = None,
        model_timeout: float | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.model = model
        self.max_steps = max_steps
        self.parallel_tools = parallel_tools
        self.tool_timeout = tool_timeout
        self.model_timeout = model_timeout
        # Default callbacks applied to every session. A session copies these and
        # may append its own.
        self.step_callbacks = list(step_callbacks)

        # Base tool set: the caller's tools plus the always-available
        # final_answer. A duplicate name here is a programming error, so fail
        # loudly rather than letting one tool shadow another. Each session gets
        # its own copy of this set (see AgentSession).
        base_tools: dict[str, Tool] = {}
        base_schemas: list[ToolSpec] = []
        for base_tool in (*tools, FinalAnswerTool()):
            if base_tool.name in base_tools:
                raise ValueError(f"Duplicate tool name: {base_tool.name!r}")
            base_tools[base_tool.name] = base_tool
            base_schemas.append(base_tool.to_schema())
        self._base_tools = base_tools
        self._base_schemas = base_schemas

        # Load skills up front but reveal only their names and descriptions. The
        # full instructions and any skill tools arrive when the model calls
        # load_skill, which keeps the base context small (progressive disclosure).
        self.skills: dict[str, Skill] = {
            skill.name: skill for skill in (_as_skill(entry) for entry in skills)
        }
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        if self.skills:
            self.instructions += _skill_catalog(self.skills.values())

    def start(self) -> AgentSession:
        """Create a fresh session with its own memory, tools, and interrupt token."""

        return AgentSession(self)

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[False] = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Awaitable[str]: ...

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[True],
        reset: bool = True,
        max_steps: int | None = None,
    ) -> AsyncIterator[Event]: ...

    def run(
        self,
        task: str,
        *,
        stream: bool = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Awaitable[str] | AsyncIterator[Event]:
        """Run a task on a fresh one-shot session.

        Convenience for the common single-run case. Each call gets its own
        session, so concurrent calls on one Agent stay isolated. For a multi-turn
        conversation, or to inspect memory afterwards, use start() and hold on to
        the session.
        """

        session = self.start()
        if stream:
            return session.run(task, stream=True, reset=reset, max_steps=max_steps)
        return session.run(task, stream=False, reset=reset, max_steps=max_steps)


class AgentSession:
    """One conversation's mutable state plus the agent loop.

    A session owns its own Memory, interrupt token, and tool set (a copy of the
    Agent's base tools, so skill tools loaded here never leak into another
    session). Create one with Agent.start().
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.memory = Memory()
        self.step_callbacks = list(agent.step_callbacks)
        self._interrupt = asyncio.Event()

        # Per-session tool view. Copying the agent's base set keeps any skill
        # tools loaded during this run isolated to this session.
        self.tools: dict[str, Tool] = dict(agent._base_tools)
        self._tool_schemas: list[ToolSpec] = list(agent._base_schemas)
        if agent.skills:
            self._register_tool(self._build_load_skill_tool())

    def _register_tool(self, new_tool: Tool) -> None:
        """Add a tool to this session's tool set, skipping names already present.

        Registration is idempotent so a skill can be loaded more than once, or
        declare a tool the session already has, without raising mid-run.
        """

        if new_tool.name in self.tools:
            return
        self.tools[new_tool.name] = new_tool
        self._tool_schemas.append(new_tool.to_schema())

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[False] = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Awaitable[str]: ...

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[True],
        reset: bool = True,
        max_steps: int | None = None,
    ) -> AsyncIterator[Event]: ...

    def run(
        self,
        task: str,
        *,
        stream: bool = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Awaitable[str] | AsyncIterator[Event]:
        """Run the agent on a task within this session.

        With stream=False (the default) this returns an awaitable that resolves
        to the final answer string. With stream=True it returns an async iterator
        of Events. Pass reset=False to continue from this session's existing
        memory (multi-turn).
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

        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        self.memory.add(TaskStep(task=task))

        limit = self.agent.max_steps if max_steps is None else max_steps

        # Remember the previous step's calls so we can spot an exact repeat.
        previous_signature: tuple[tuple[str, str], ...] | None = None

        # Corrective user messages appended to the prompt after the model emits
        # unparseable output, so it can retry. Cleared once a turn parses.
        correction_notes: list[ChatMessage] = []
        malformed_retries = 0

        for _ in range(limit):
            if self._interrupt.is_set():
                self._interrupt.clear()
                yield FinalEvent(answer="Run interrupted.")
                return

            started = time.monotonic()
            messages = self.memory.to_messages(self.agent.instructions)
            messages.extend(correction_notes)

            # Stream the model turn: emit text as it arrives, then rebuild the
            # full ChatMessage from the deltas for the rest of the step to use.
            # model_timeout bounds a hung turn, and the per-chunk interrupt check
            # lets a long stream be stopped without waiting for it to finish.
            deltas: list[Delta] = []
            interrupted = False
            try:
                async with asyncio.timeout(self.agent.model_timeout):
                    async for delta in self.agent.model.stream(
                        messages, tools=self._tool_schemas
                    ):
                        if delta.content:
                            yield TextDelta(text=delta.content)
                        deltas.append(delta)
                        if self._interrupt.is_set():
                            interrupted = True
                            break
            except TimeoutError as exc:
                raise ModelError(
                    f"Model stream exceeded the {self.agent.model_timeout}s timeout."
                ) from exc

            if interrupted:
                self._interrupt.clear()
                yield FinalEvent(answer="Run interrupted.")
                return

            # Malformed tool calls (bad JSON, missing name) are recoverable:
            # re-prompt the model with a correction, up to a small cap, rather
            # than crashing the run.
            try:
                response = agglomerate_deltas(deltas)
            except ModelOutputError as exc:
                malformed_retries += 1
                if malformed_retries > _MAX_MALFORMED_RETRIES:
                    raise ModelError(
                        f"Model produced unparseable tool calls on "
                        f"{malformed_retries} consecutive turns: {exc}"
                    ) from exc
                correction_notes.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"Your previous response could not be parsed: {exc} "
                            "Reply again with valid tool-call arguments as a JSON "
                            "object."
                        ),
                    )
                )
                continue

            # A well-formed turn clears any accumulated corrections.
            correction_notes.clear()
            malformed_retries = 0

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

            if self.agent.parallel_tools:
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
        messages = self.memory.to_messages(self.agent.instructions)
        messages.append(
            ChatMessage(
                role="user",
                content="Step limit reached. Give your best final answer now.",
            )
        )

        deltas = []
        try:
            async with asyncio.timeout(self.agent.model_timeout):
                async for delta in self.agent.model.stream(messages):
                    if delta.content:
                        yield TextDelta(text=delta.content)
                    deltas.append(delta)
        except TimeoutError as exc:
            raise ModelError(
                f"Model stream exceeded the {self.agent.model_timeout}s timeout."
            ) from exc

        try:
            response = agglomerate_deltas(deltas)
        except ModelOutputError as exc:
            raise ModelError(
                f"Model produced unparseable output on the forced final answer: {exc}"
            ) from exc

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
        # A per-tool timeout overrides the agent-wide default.
        timeout = tool.timeout if tool.timeout is not None else self.agent.tool_timeout
        try:
            if timeout is not None:
                output = await asyncio.wait_for(tool.call(tool_call.arguments), timeout)
            else:
                output = await tool.call(tool_call.arguments)
        except TimeoutError:
            # A slow tool is an observation, not a crash. A genuine task
            # cancellation raises CancelledError (a BaseException), which is
            # deliberately not caught here so it propagates and stops the run.
            err = ToolTimeoutError(
                f"tool {tool_call.name!r} exceeded its {timeout}s timeout"
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"{type(err).__name__}: {err}",
                is_error=True,
            )
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

    def _build_load_skill_tool(self) -> Tool:
        """Build the built-in load_skill tool, bound to this session."""

        @tool
        def load_skill(name: str) -> str:
            """Load a skill's full instructions and enable any tools it provides.

            Args:
                name: The name of the skill to load, taken from the catalog in
                    the system prompt.
            """

            skill = self.agent.skills.get(name)
            if skill is None:
                raise ToolCallError(
                    f"Unknown skill {name!r}. Available: {sorted(self.agent.skills)}"
                )

            loaded = skill.load_tools()
            for skill_tool in loaded:
                self._register_tool(skill_tool)

            body = skill.instructions
            if loaded:
                names = ", ".join(skill_tool.name for skill_tool in loaded)
                body += f"\n\nTools now available: {names}."
            return body

        return load_skill

    def interrupt(self) -> None:
        """Request a graceful stop before the next step.

        The current run pauses rather than crashing; resume it later by calling
        run(..., reset=False) on this same session, which continues from the
        steps already in memory.
        """

        self._interrupt.set()


def _as_skill(entry: Skill | str | Path) -> Skill:
    """Coerce a skill entry (a Skill, or a path to a skill folder) into a Skill."""

    return entry if isinstance(entry, Skill) else Skill.from_path(entry)


def _skill_catalog(skills: Iterable[Skill]) -> str:
    """Render the skill catalog appended to the system prompt.

    Only names and descriptions are listed. The full instructions stay out of
    the prompt until the model loads a skill (progressive disclosure).
    """

    lines = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
    return (
        "\n\nYou can load skills: focused instruction sets for particular kinds "
        "of task. When the task matches one, call load_skill(name) to load its "
        "full instructions and any tools it provides before continuing. "
        "Available skills:\n" + lines
    )
