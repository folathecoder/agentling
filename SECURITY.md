# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately by emailing me@folarin.dev.
Do not open a public issue for a security report. I will acknowledge receipt
within a few days and keep you posted on the fix.

## Trust model

agentling runs code you give it. Two boundaries are worth calling out:

- **Tools are trusted code.** A `@tool` function runs in your process with your
  privileges. Only register tools you trust, and treat the tool *arguments*
  (which come from the model) as untrusted input inside the tool.
- **Skills with `tools:` execute code.** A `SKILL.md` `tools:` entry point is
  imported with `importlib`, which runs the target module's top-level code.
  Load skills only from sources you trust, exactly as you would a Python import.

## Model and tool output

- Tool return values and error messages are fed back to the model as
  observations. Do not put secrets in tool return values or exception messages.
  Set `Agent(redact_errors=True)` to keep unexpected exception messages out of
  the model's context (the exception type is still shown, and the full detail is
  logged via the `agentling` logger).
- Tool output is injected into the model's context, so treat any external data
  a tool returns as untrusted: it can attempt to steer later steps (prompt
  injection). Bound its size with `max_tool_output_chars`.

## Concurrency

An `Agent` is immutable configuration and is safe to share across concurrent
runs. Per-run state lives on an `AgentSession` (from `agent.start()`, or created
implicitly by `agent.run()`), so concurrent runs never share memory, tools, or
interrupts.
