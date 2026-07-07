# Changelog

All notable changes to agentling are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-07-07

The first real release: the core framework plus a production-hardening pass.

### Added

- Async ReAct agent loop with a single streaming code path (`Agent` config and
  factory, `AgentSession` run state); blocking and streaming `run()`.
- Provider-neutral model layer and an OpenAI-compatible adapter with retries.
- `@tool` decorator with JSON Schema generation, argument validation, and
  per-tool metadata (`timeout`, `parallel_safe`, `max_output_chars`).
- Typed memory (`TaskStep` / `ActionStep` / `FinalStep`) with JSON persistence
  and load validation (`MemoryLoadError`).
- Streaming events, a `print_events` renderer, and a terminal run `status`
  (completed / interrupted / max_steps).
- Progressive-disclosure skills (`SKILL.md`) with a built-in `load_skill` tool.
- An exception hierarchy under `AgentlingError`, public API exports, and
  `__all__`.
- Timeouts (`tool_timeout`, `model_timeout`), prompt cancellation, sync tools
  run off the event loop, optional error redaction (`redact_errors`), cumulative
  token usage on the final event, and a `context_manager` hook for long runs.

[0.1.0]: https://github.com/folathecoder/agentling/releases/tag/v0.1.0
