# agentling roadmap

agentling is a tiny async framework for reliable, observable, tool-using agents:
a clean ReAct loop, typed memory, streaming events, recoverable failures, and
progressive-disclosure skills, in a codebase small enough to read in one
sitting.

This roadmap is **directional, not a schedule**. agentling is alpha (0.x) and
the API may change before 1.0. Priorities can shift with community feedback, and
versions below are indicative — pre-1.0 milestones may split or merge.

## Shipped — v0.1.0

The first release: the core framework plus a production-hardening pass —
`Agent`/`AgentSession`, an OpenAI-compatible model adapter, `@tool`, typed
memory with JSON persistence, streaming events, progressive-disclosure skills,
timeouts/cancellation, malformed-output recovery, and a runnable examples suite.

## Next — Reliability (v0.2)

**Top priority.** Harden and clean up the v0.1 surface before adding features.
No point building on foundations with sharp edges.

- **Session lifecycle.** An idle `interrupt()` no longer silently kills the next
  run, and using one session concurrently raises a clear error instead of
  quietly corrupting memory.
- **Honest errors.** Make the exception hierarchy real — the documented error
  types are actually raised, so `except AgentlingError` behaves as promised.
- **Broader compatibility.** Stop rejecting valid tool calls from
  OpenAI-compatible backends (some omit ids mid-stream); synthesize them
  instead.
- **Sampling controls.** Expose `temperature`, `max_tokens`, `seed`, and friends
  on the model, so deterministic evals and cost caps are possible.
- **Cleaner semantics.** `final_answer` no longer leaks into the event stream,
  context-window trimming applies on every path, and resuming a session no
  longer duplicates the task.
- **Robustness.** Safe, async-capable step callbacks; Python 3.13 and 3.14 in
  CI; stricter typing; and a batch of small correctness fixes.

## Then — Observability and evaluation (v0.3)

Make every run inspectable and testable.

- **Lifecycle tracing.** A dependency-free tracing layer over the whole
  lifecycle — run, step, model call, tool call — capturing inputs, outputs,
  token usage, timing, and errors.
- **OpenTelemetry adapter.** Emit standard GenAI spans so traces flow to
  Langfuse, LangSmith, Arize Phoenix, and any OTLP backend through one
  integration, not a bespoke plugin per vendor.
- **Offline testing and evals.** A public, deterministic testing model (run and
  test your agent with no API key), plus an evaluation harness — datasets and
  evaluators — that works locally or against a hosted experiment backend.

## Later — More models (v0.4)

Meet people where their models already are.

- **OpenAI-compatible tier.** A documented provider matrix and light ergonomics
  for OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, local
  servers (Ollama, vLLM, LM Studio), and Azure OpenAI — most already work by
  pointing at a base URL.
- **Native Anthropic (Claude).** A first-class adapter on the Messages API with
  tool use, streaming, usage, and prompt caching — beyond the lossy
  compatibility endpoint.
- **More.** Native Gemini; optionally a litellm bridge and AWS Bedrock / Vertex.

## Exploring — beyond

Ideas we like but have not committed to a milestone:

- Structured output (JSON mode / response schemas).
- A human-in-the-loop hook to approve, deny, or modify a tool call before it
  runs (guardrails).
- An MCP bridge recipe (wrap an MCP tool as an agentling tool).
- An exhaustive failure-mode test suite and a trust policy for skill-provided
  tools.

## Toward 1.0

Once reliability, observability, and the model surface settle, 1.0 is about
committing to a stable public API and semantic-versioning guarantees.

## Influence the roadmap

This is an open, early project and the priorities above are open to input. Open
an issue to propose something, describe a use case we are missing, or tell us
which item matters most to you. Bug reports and small PRs are especially
welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
