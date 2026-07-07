# agentling

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typed-mypy-blue.svg)](https://mypy-lang.org/)
[![Lint](https://img.shields.io/badge/lint-ruff-orange.svg)](https://docs.astral.sh/ruff/)

A tiny async tool-calling agent framework. The good ideas from larger agent
libraries (a clean ReAct loop, typed memory, streaming, progressive-disclosure
skills) in a codebase small enough to read in one sitting.

agentling is built around one idea: an agent is a loop that turns a model, some
tools, and a memory of what happened into more actions, until it has an answer.
Everything else (streaming, skills, self-healing, persistence) is a thin layer
on top of that loop.

```python
import asyncio

from agentling import Agent, OpenAIModel, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city to look up.
    """
    return f"It is 22C and sunny in {city}."


async def main() -> None:
    agent = Agent(model=OpenAIModel("gpt-4o-mini"), tools=[get_weather])
    print(await agent.run("What's the weather in Paris?"))


asyncio.run(main())
```

## Contents

- [Why agentling](#why-agentling)
- [Install](#install)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [Tools](#tools)
  - [Running: blocking vs streaming](#running-blocking-vs-streaming)
  - [Sessions and concurrency](#sessions-and-concurrency)
  - [Skills](#skills)
  - [Memory and resuming a run](#memory-and-resuming-a-run)
  - [Interruption](#interruption)
  - [Models and other providers](#models-and-other-providers)
- [Architecture](#architecture)
- [Configuration reference](#configuration-reference)
- [Development](#development)
- [License](#license)

## Why agentling

- **Async first.** The loop, tools, and model calls are all `async`. Tool calls
  in a single step run concurrently by default.
- **One code path.** Blocking and streaming share the exact same loop. There is
  a single async generator; blocking mode just drains it.
- **Typed memory.** A run is a list of typed steps, not a bag of raw messages.
  Steps know how to render themselves back into model messages and serialize to
  JSON for persistence and replay.
- **Progressive-disclosure skills.** Drop a `SKILL.md` folder in and the model
  sees only its name and description until it decides to load it. Big skill
  libraries stay cheap.
- **Self-healing.** A tool that raises becomes an observation the model can
  recover from, not a crash.
- **Small and readable.** No metaclasses, no plugin registry, no DSL. Around
  800 lines of source you can actually read.

## Install

```bash
pip install agentling
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add agentling
```

Requires Python 3.11 or newer. The only runtime dependencies are `openai` (the
client used by the built-in provider adapter) and `pyyaml` (for skill
frontmatter).

Set your provider key in the environment:

```bash
export OPENAI_API_KEY="sk-..."
```

## Quickstart

```python
import asyncio

from agentling import Agent, OpenAIModel, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


async def main() -> None:
    agent = Agent(model=OpenAIModel("gpt-4o-mini"), tools=[add])
    answer = await agent.run("What is 19 + 23, and why?")
    print(answer)


asyncio.run(main())
```

## Usage

### Tools

A tool is any Python function wrapped with `@tool`. The function name becomes
the tool name, the docstring summary becomes the description, and the type hints
plus a Google-style `Args:` section become the JSON Schema the model sees. Both
synchronous and asynchronous functions are supported; a synchronous tool runs in
a worker thread so it cannot block the event loop.

> **`tool_timeout` caveat:** a timeout stops the agent from *waiting* on a tool
> and turns it into an observation, but a synchronous tool already running in a
> thread cannot be forcibly cancelled and will finish in the background. Prefer
> async tools, or make blocking tools cooperative, when timeouts matter.

```python
from agentling import tool


@tool
async def search(query: str, limit: int = 5) -> str:
    """Search the docs and return the top matches.

    Args:
        query: What to search for.
        limit: How many results to return.
    """
    ...
```

Supported parameter types map to JSON Schema: `str`, `int`, `float`, `bool`,
`list[...]`, `dict[...]`, `Optional[...]` / `X | None` (treated as "may be
omitted"), and `Literal[...]` (becomes an enum). Parameters without a default
are marked required. `*args`, `**kwargs`, and positional-only parameters are
rejected at registration time, because a tool is always called with a JSON
object of named arguments.

If the model sends arguments that do not match the schema (missing a required
field, wrong type, unknown key), the tool raises a `ToolCallError` which the
loop feeds back to the model as an error observation. The model gets a chance to
fix its call rather than the run blowing up.

Every agent also has a built-in `final_answer` tool. The model can call it to
end the run explicitly, or it can just reply with plain text (see
[forgiving termination](#forgiving-termination)).

### Running: blocking vs streaming

`run()` has two modes that share one implementation.

Blocking mode returns the final answer:

```python
answer: str = await agent.run("Summarize this.")
```

Streaming mode returns an async iterator of typed events:

```python
from agentling import FinalEvent, TextDelta, ToolCallEvent

async for event in agent.run("Summarize this.", stream=True):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[calling {event.tool_call.name}]")
    elif isinstance(event, FinalEvent):
        print(f"\nDone: {event.answer}")
```

There is a ready-made renderer, `print_events`, that consumes the stream, prints
text as it arrives along with each tool call and result, and returns the final
answer. It is the "streaming CLI" in about thirty lines:

```python
import asyncio

from agentling import Agent, OpenAIModel, print_events, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


async def main() -> None:
    agent = Agent(model=OpenAIModel("gpt-4o-mini"), tools=[add])
    answer = await print_events(agent.run("What is 19 + 23?", stream=True))
    print("\nFinal answer:", answer)


asyncio.run(main())
```

### Skills

A skill is a folder containing a `SKILL.md` file: YAML frontmatter followed by a
markdown body of instructions.

```markdown
---
name: code-reviewer
description: Review a code change for bugs, security issues, and style problems.
---

# Code Reviewer

You are reviewing a code change. Work through it methodically and report only
findings you are confident about...
```

Skills use **progressive disclosure**. When you pass skills to an agent, only
their names and descriptions are added to the system prompt as a catalog. The
full instruction body stays out of context until the model calls the built-in
`load_skill(name)` tool, at which point the body is returned as an observation
and any tools the skill declares are registered. This keeps the base prompt
small even with a large library of skills installed.

```python
import asyncio

from agentling import Agent, OpenAIModel, tool


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: Path to the file to read.
    """
    with open(path, encoding="utf-8") as handle:
        return handle.read()


async def main() -> None:
    agent = Agent(
        model=OpenAIModel("gpt-4o-mini"),
        tools=[read_file],
        skills=["examples/skills/code-reviewer"],
    )
    print(await agent.run("Review the code in src/agentling/agent.py"))


asyncio.run(main())
```

A skill can also bring its own tools. List Python entry points in the
frontmatter and they are imported and registered when the skill loads:

```markdown
---
name: linting
description: Lint Python files and report issues.
tools:
  - my_package.lint_tools:run_ruff
---
```

Each entry point is a `"module.path:attribute"` string that must resolve to a
`Tool` (a function decorated with `@tool`). You can pass skills as folder paths
(strings or `Path`) or as pre-built `Skill` objects.

> **Security:** a skill's `tools:` entry point is imported, which runs that
> module's code. Load skills only from sources you trust, exactly as you would a
> Python import. See [SECURITY.md](SECURITY.md) for the full trust model.

### Sessions and concurrency

An `Agent` is immutable configuration (model, tools, skills, settings) and is
safe to build once and share. The per-run state (memory, the interrupt token,
and any skill tools loaded during a run) lives on an `AgentSession`.

`agent.run(task)` is a one-shot convenience: it spins up a fresh session, runs
it, and returns the answer. Because each call gets its own session, concurrent
calls on one shared agent never touch each other's memory or tools. When you
need multi-turn, inspection, or interruption, hold a session with
`agent.start()`.

```python
# one-shot: simplest, and safe under concurrency
answer = await agent.run("A single question")

# hold a session for multi-turn, inspection, or interruption
session = agent.start()
answer = await session.run("First question")
print(session.memory.steps)
```

### Memory and resuming a run

Each session keeps a `Memory` of typed steps. You can serialize it and reload
it later:

```python
session = agent.start()
await session.run("First question")

saved = session.memory.dump_json()
# ... later, in another process ...
from agentling import Memory
restored = agent.start()
restored.memory = Memory.load_json(saved)
```

By default each `run()` starts fresh. Pass `reset=False` to continue from the
session's existing memory, which is how a multi-turn conversation or a resumed
run works:

```python
session = agent.start()
await session.run("First question")
await session.run("A follow-up", reset=False)   # sees the earlier turn
```

### Interruption

Call `session.interrupt()` to request a graceful stop. The current run does not
crash; it finishes at the next step boundary and the session's memory is
preserved, so you can resume it with `run(..., reset=False)`.

```python
session = agent.start()
session.interrupt()            # from a signal handler, another task, a UI button
await session.run(task)        # returns "Run interrupted." at the next boundary
# ... later ...
await session.run(task, reset=False)   # picks up where it left off
```

### Models and other providers

`OpenAIModel` is an adapter for any OpenAI-compatible chat-completions endpoint.
Point it at a different `base_url` to use a compatible provider (a local server,
a gateway, or another vendor's OpenAI-compatible API):

```python
from agentling import OpenAIModel

model = OpenAIModel(
    "llama-3.1-70b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed-locally",
)
```

Transient failures (rate limits, connection or timeout errors, 5xx responses)
are retried with exponential backoff. Permanent errors (a bad request, bad auth)
fail fast without retrying.

Any object implementing the `Model` protocol works, so you can write your own
adapter:

```python
class Model(Protocol):
    async def generate(self, messages, tools=None) -> ChatMessage: ...
    def stream(self, messages, tools=None) -> AsyncIterator[Delta]: ...
```

## Architecture

agentling is six small modules. Each one owns a single concept, and they depend
on each other in one direction only (agent depends on skills, tools, memory,
events, models; nothing depends on agent).

| Module | Responsibility |
| --- | --- |
| [`models.py`](src/agentling/models.py) | Provider-neutral message types (`ChatMessage`, `ToolCall`, `Usage`), streaming types (`Delta`, `ToolCallDelta`), the `Model` protocol, and the `OpenAIModel` adapter. |
| [`tools.py`](src/agentling/tools.py) | The `Tool` abstraction, the `@tool` decorator, JSON Schema generation from function signatures, argument validation, and the built-in `final_answer` tool. |
| [`memory.py`](src/agentling/memory.py) | Typed steps (`TaskStep`, `ActionStep`, `FinalStep`), the `Memory` container, rendering to model messages, and JSON serialization. |
| [`events.py`](src/agentling/events.py) | The streaming event types, the `Event` union, and the `print_events` renderer. |
| [`skills.py`](src/agentling/skills.py) | The `Skill` dataclass, the `SKILL.md` loader (frontmatter plus body), and entry-point tool resolution. |
| [`agent.py`](src/agentling/agent.py) | The `Agent` config/factory, the `AgentSession` that holds one run's state, and the ReAct loop that ties everything together. |

### The agent loop

The whole framework hangs off a single async generator, `AgentSession._run_stream`.
`run()` is a thin dispatcher: in streaming mode it hands back that generator; in
blocking mode it drains the generator and returns the final answer. There is no
second implementation to keep in sync.

```
  run(task, stream=True)                run(task)  (stream=False)
          │                                   │
          ▼                                   ▼
   _run_stream(task)  ◀───────────────  _drain(_run_stream(task))
   (async generator)                    (awaits, returns the answer str)
          │
          │  one iteration == one "step"
          ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. interrupt requested?  -> yield FinalEvent, stop (resumable)  │
  │ 2. Memory.to_messages(instructions)  -> the full prompt         │
  │ 3. Model.stream(messages, tools)  -> Delta stream               │
  │       - each text chunk is yielded as a TextDelta               │
  │       - agglomerate_deltas() rebuilds one ChatMessage           │
  │ 4. no tool calls?  -> that text is the answer; finish           │
  │ 5. for each tool call: yield ToolCallEvent                      │
  │ 6. execute tools (concurrently by default) -> ToolResults       │
  │       - a raised exception becomes an error observation         │
  │       - an exact repeat of last step's calls gets a nudge       │
  │ 7. yield ToolResultEvent per result; record an ActionStep       │
  │ 8. final_answer called?  -> finish with FinalEvent              │
  └───────────────────────────────────────────────────────────────┘
          │
          ▼
  step limit reached -> ask once for a tool-free answer, then finish
```

Each iteration is a step. A step streams one model turn, runs whatever tools the
model asked for, records the outcome, and checks whether the run is done. The
loop ends in one of three ways: the model calls `final_answer`, the model
replies with plain text and no tool calls, or the step limit is hit and the loop
asks for one last tool-free answer.

### Message and model layer

Everything above the provider speaks in framework-neutral types, not vendor
payloads:

- **`ChatMessage`** is the one message type used internally. It has a role,
  content, optional tool calls, an optional tool-call id (for tool results), and
  optional usage.
- **`ToolCall`** is a provider-neutral tool call: an id, a name, and a parsed
  arguments dict.
- **`Usage`** is input and output token counts, with a `total_tokens` property.

`OpenAIModel` is the only place that knows OpenAI's wire format. It converts
`ChatMessage` lists into OpenAI messages on the way out and converts responses
back into `ChatMessage` on the way in, so the rest of the framework never sees a
provider-specific shape. Swapping providers means writing one adapter, not
touching the loop.

**Streaming and reassembly.** `Model.stream` yields `Delta` objects: small
chunks of content or fragments of a tool call. Tool-call arguments in particular
arrive in pieces across many deltas. The module-level `agglomerate_deltas`
function reassembles a delta stream back into a single `ChatMessage`: it
concatenates content, merges tool-call fragments by their index, and captures
the final usage. This is why the loop can stream text to the user and still work
with a complete `ChatMessage` for tool execution: it streams and reassembles at
the same time.

### Memory and typed steps

A run is recorded as a list of typed steps rather than a flat list of messages:

- **`TaskStep`** is the user's task (or a continued turn).
- **`ActionStep`** is one loop iteration: the assistant message the model
  produced, the tool results from it, and metadata (token usage, wall-clock
  duration).
- **`FinalStep`** is the terminal answer.

Each step knows how to render itself into the `ChatMessage` list the model sees,
via `to_messages()`. `Memory.to_messages(system_prompt)` prepends the system
prompt (which is runtime configuration, not history, so it is never stored in a
step) and then renders every step in order. This separation is what makes the
memory serializable: `dump_json` / `load_json` round-trip the whole run, tagging
each step with its kind so it can be rebuilt, which gives you free persistence,
replay, and multi-turn continuation.

### Events and streaming

The loop communicates progress through a small set of frozen event types:

| Event | Meaning |
| --- | --- |
| `TextDelta` | A chunk of streamed assistant text. |
| `ToolCallEvent` | Emitted just before a tool call runs. |
| `ToolResultEvent` | Emitted after a tool call completes (success or error). |
| `StepEvent` | Emitted after a step is recorded to memory; carries the step. |
| `FinalEvent` | Emitted once when the run ends; carries the answer and usage. |

`StepEvent` is the bridge between the live event stream and the durable memory:
it carries the exact `ActionStep` that was just written. `print_events` is a
reference consumer of this stream, but you can write your own to drive a UI, log
to a database, or compute metrics.

### Tools and schema generation

`@tool` wraps a function into a `Tool`. At registration time it reads the
function's signature and docstring and builds a JSON Schema `parameters` object:
type hints become JSON types, the `Args:` section becomes per-parameter
descriptions, and defaults decide what is required. Before a tool runs, the loop
validates the model's arguments against that schema and raises `ToolCallError`
on a mismatch. For stateful tools you can subclass `Tool` directly instead of
using the decorator.

### Skills and progressive disclosure

`Skill.from_path` parses a `SKILL.md` into a name, description, instruction body,
and an optional list of tool entry points. When an agent is built with skills,
it registers a single built-in `load_skill` tool and appends a catalog of
`name: description` lines to the system prompt. It does not put any skill bodies
in the prompt.

When the model calls `load_skill(name)`, the agent returns that skill's full
instruction body as the tool observation and registers any tools the skill
declared. Because the loop reads the live tool set fresh on every step, those
newly registered tools are available on the very next turn. The result is that
the cost of a skill (its instructions, its tools) is only paid once the model
actually decides to use it.

### Cross-cutting behaviors

These are small rules layered onto the loop that make agents robust in practice.

#### Forgiving termination

Not every model reliably calls `final_answer`. So if the model replies with
plain text and no tool calls, agentling treats that text as the answer and ends
the run. Explicit `final_answer` and plain-text replies both work.

#### Self-healing tool errors

`_execute_tool` catches any exception a tool raises and turns it into a
`ToolResult` with `is_error=True`. That error is rendered back to the model as an
observation ("Error from 'search': ... Fix the arguments and try again"), so the
model can correct course. One bad tool call does not kill the run.

#### Loop detection

If a step's tool calls are an exact repeat of the previous step's (same names,
same arguments), the loop appends a short nudge to the observations telling the
model it already made that call and got the same result. This helps the model
break out of a stuck cycle without a hard failure.

#### Graceful interruption and resume

`interrupt()` sets an event that the loop checks at the top of each step. When
set, the loop emits a final "Run interrupted." event and returns without writing
a terminal `FinalStep`. Because memory is left intact and no terminal step is
written, `run(..., reset=False)` can pick the run back up exactly where it
paused.

#### Concurrent tool execution

When a single model turn requests several tool calls, they run concurrently with
`asyncio.gather` by default. Set `parallel_tools=False` to run them in order
instead (useful when tools share state or must not interleave).

## Configuration reference

`Agent(...)`:

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | required | Any object implementing the `Model` protocol. |
| `tools` | `()` | Tools to register (a `final_answer` tool is always added). |
| `skills` | `()` | Skills as folder paths or `Skill` objects. |
| `instructions` | built-in default | The system prompt. A skill catalog is appended when skills are present. |
| `max_steps` | `15` | Maximum loop iterations before a forced answer. Must be at least 1. |
| `step_callbacks` | `()` | Callables invoked with each `ActionStep` as it is recorded. |
| `parallel_tools` | `True` | Run a turn's tool calls concurrently, or in order when `False`. |
| `tool_timeout` | `None` | Per-call time budget (seconds) for tools; a timeout becomes a recoverable observation. |
| `model_timeout` | `None` | Time budget (seconds) for each model turn; exceeding it raises `ModelError`. |
| `max_tool_output_chars` | `None` | Truncate tool observations head and tail beyond this length. |
| `redact_errors` | `False` | Hide unexpected tool-exception messages from the model and log them instead. |
| `context_manager` | `None` | Callable `messages -> messages` applied before each model call, to trim or summarize. |

`OpenAIModel(...)`:

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | required | The model name to request. |
| `api_key` | env | Falls back to the OpenAI SDK's environment configuration. |
| `base_url` | `None` | Point at any OpenAI-compatible endpoint. |
| `context_window` | `128_000` | Advertised context window for this model. |
| `max_retries` | `2` | Retries after the initial request for transient errors. |
| `retry_base_delay` | `0.5` | Initial backoff delay in seconds (doubles each retry). |

`agent.run(task, *, stream=False, reset=True, max_steps=None)`:

- `stream=False` returns an awaitable that resolves to the final answer string.
- `stream=True` returns an async iterator of `Event` objects.
- `reset=False` continues from existing memory instead of starting fresh.
- `max_steps` overrides the agent's limit for this run only.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync                                # install everything, including dev deps
uv run pytest                          # run the test suite
uv run ruff check src tests            # lint
uv run ruff format --check src tests   # formatting (run `ruff format` to fix)
uv run mypy src tests                  # type-check
uv build                               # verify the package builds
```

## Security

Tools and skill-provided tools run as trusted code in your process, and tool
output is fed back to the model. See [SECURITY.md](SECURITY.md) for the trust
model, the `redact_errors` and `max_tool_output_chars` knobs, and how to report
a vulnerability.

## License

[MIT](LICENSE) (c) Folarin Akinloye.

## Acknowledgements

The design borrows the best ideas from the broader agent ecosystem: the clean
ReAct loop and code-first tools popularized by smolagents, and the
progressive-disclosure skill format used by Claude. agentling's contribution is
packaging those ideas into a small, typed, async codebase you can read end to
end.
