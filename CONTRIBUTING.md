# Contributing to agentling

Thanks for your interest in improving agentling.

## Development setup

agentling uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync   # create the environment and install all dependencies
```

## The checks

These all run in CI and must pass. Run them locally before opening a PR:

```bash
uv run pytest                         # tests
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # formatting (run `ruff format` to fix)
uv run mypy src tests                 # type-check
uv build                              # the package builds
```

## Guidelines

- Keep the framework small and readable; prefer clarity over cleverness.
- Add tests for new behavior, including the failure paths.
- Type everything; the codebase is checked with mypy.
- Match the surrounding style; `ruff format` is the source of truth.
- Avoid new runtime dependencies unless they clearly earn their place.

## Reporting bugs and vulnerabilities

Open a GitHub issue for bugs. For security reports, see [SECURITY.md](SECURITY.md).
