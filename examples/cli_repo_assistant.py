"""A streaming repo assistant with safe, read-only file tools.

Needs an OpenAI-compatible API key (set OPENAI_API_KEY):

    uv run python examples/cli_repo_assistant.py "Summarize the README"

Demonstrates streaming with print_events, file tools sandboxed to a project
root, an output cap (max_tool_output_chars), and loading the bundled
code-reviewer skill. build_agent() takes an optional model so tests run offline.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from agentling import Agent, Model, OpenAIModel, ToolCallError, print_events, tool


def _resolve_within(base: Path, candidate: str) -> Path:
    """Resolve candidate under base, rejecting anything that escapes the root."""

    target = (base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ToolCallError(f"path {candidate!r} escapes the project root")
    return target


def build_agent(model: Model | None = None, root: str = ".") -> Agent:
    """Build a repo assistant whose file tools are sandboxed to `root`."""

    base = Path(root).resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file under the project root.

        Args:
            path: Path relative to the project root.
        """
        return _resolve_within(base, path).read_text(encoding="utf-8")

    @tool
    def list_files(subdir: str = ".") -> str:
        """List the entries of a directory under the project root.

        Args:
            subdir: Directory relative to the project root.
        """
        target = _resolve_within(base, subdir)
        return "\n".join(sorted(entry.name for entry in target.iterdir()))

    return Agent(
        model=model
        or OpenAIModel(os.environ.get("AGENTLING_EXAMPLE_MODEL", "gpt-4o-mini")),
        tools=[read_file, list_files],
        skills=[Path(__file__).parent / "skills" / "code-reviewer"],
        max_tool_output_chars=2000,
    )


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or (
        "List the files in the project root, then summarize the README."
    )
    await print_events(build_agent().run(prompt, stream=True))


if __name__ == "__main__":
    asyncio.run(main())
