"""Markdown skills with progressive disclosure.

A skill is a folder containing a SKILL.md file: YAML frontmatter (name,
description, and optional tool entry points) followed by a markdown body of
instructions. Only the name and description are shown to the model up front, in
a catalog appended to the system prompt. The full body, and any tools the skill
declares, are revealed on demand when the model calls the built-in load_skill
tool. That keeps the base context small until a skill is actually needed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .tools import Tool

SKILL_FILE = "SKILL.md"


@dataclass
class Skill:
    """A loadable unit of instructions plus optional tools.

    name and description are cheap metadata for the catalog. instructions is the
    full markdown body, revealed only when the skill is loaded. tools lists
    Python entry points ("package.module:attribute") that resolve to Tool
    objects and are registered at load time. path is the skill's folder, so the
    instructions can reference files bundled alongside SKILL.md.
    """

    name: str
    description: str
    instructions: str
    path: Path
    tools: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: str | Path) -> Skill:
        """Load a Skill from a folder containing a SKILL.md file."""

        folder = Path(path)
        source = (folder / SKILL_FILE).read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(source)

        try:
            name = frontmatter["name"]
            description = frontmatter["description"]
        except KeyError as exc:
            raise ValueError(
                f"{folder / SKILL_FILE} is missing required frontmatter key {exc}."
            ) from exc

        return cls(
            name=name,
            description=description,
            instructions=body.strip(),
            path=folder,
            tools=list(frontmatter.get("tools") or []),
        )

    def load_tools(self) -> list[Tool]:
        """Import and return the Tool objects named in this skill's frontmatter."""

        return [_resolve_tool(spec) for spec in self.tools]


def _split_frontmatter(source: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into its YAML frontmatter dict and markdown body.

    The frontmatter is the block between the leading '---' fence and the next
    '---' line. A file with no leading fence is treated as all body.
    """

    if not source.startswith("---"):
        return {}, source

    # maxsplit=2 splits on only the first two fences, so a '---' horizontal
    # rule inside the body is left intact. parts[0] is empty because the
    # string starts with the opening fence.
    parts = source.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Unterminated SKILL.md frontmatter (missing closing '---').")

    data = yaml.safe_load(parts[1]) or {}
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping.")

    return data, parts[2]


def _resolve_tool(spec: str) -> Tool:
    """Resolve a 'package.module:attribute' entry point to a Tool instance."""

    module_path, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"Tool entry point {spec!r} must be 'module.path:attribute'.")

    obj = getattr(importlib.import_module(module_path), attr)
    if not isinstance(obj, Tool):
        raise TypeError(
            f"Tool entry point {spec!r} resolved to {type(obj).__name__}, "
            "expected a Tool (did you forget the @tool decorator?)."
        )
    return obj
