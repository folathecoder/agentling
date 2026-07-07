"""A file-backed notes assistant: add, search, and list notes.

Needs an OpenAI-compatible API key (set OPENAI_API_KEY):

    uv run python examples/cli_notes_agent.py

Demonstrates practical file tools sandboxed to a notes directory, an async
tool (search), a non-parallel-safe mutating tool (add), and the production
knobs tool_timeout and redact_errors. build_agent() takes an optional model so
tests run offline against a temporary directory.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agentling import Agent, Model, OpenAIModel, ToolCallError, tool


def _resolve_within(base: Path, candidate: str) -> Path:
    """Resolve candidate under base, rejecting anything that escapes the root."""

    target = (base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ToolCallError(f"path {candidate!r} escapes the notes directory")
    return target


def build_agent(model: Model | None = None, notes_dir: str = "notes") -> Agent:
    """Build a notes assistant whose tools live under `notes_dir`."""

    base = Path(notes_dir).resolve()

    @tool(parallel_safe=False)
    def add_note(name: str, text: str) -> str:
        """Save a note (overwrites one with the same name).

        Args:
            name: The note's name (no path separators).
            text: The note body.
        """
        base.mkdir(parents=True, exist_ok=True)
        _resolve_within(base, f"{name}.txt").write_text(text, encoding="utf-8")
        return f"saved note {name!r}"

    @tool
    async def search_notes(query: str) -> str:
        """Return the names of notes whose text contains the query.

        Args:
            query: Text to search for.
        """
        base.mkdir(parents=True, exist_ok=True)
        hits = [
            note.stem
            for note in sorted(base.glob("*.txt"))
            if query in note.read_text(encoding="utf-8")
        ]
        return "\n".join(hits) or "no matches"

    @tool
    def list_notes() -> str:
        """List the names of all saved notes."""
        base.mkdir(parents=True, exist_ok=True)
        return "\n".join(sorted(note.stem for note in base.glob("*.txt"))) or "no notes"

    return Agent(
        model=model
        or OpenAIModel(os.environ.get("AGENTLING_EXAMPLE_MODEL", "gpt-4o-mini")),
        tools=[add_note, search_notes, list_notes],
        tool_timeout=10.0,
        redact_errors=True,
    )


async def main() -> None:
    agent = build_agent()
    print(
        await agent.run("Add a note called groceries: milk, eggs. Then list my notes.")
    )


if __name__ == "__main__":
    asyncio.run(main())
