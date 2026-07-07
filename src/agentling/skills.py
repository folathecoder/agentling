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

        raw_tools = frontmatter.get("tools", [])
        if raw_tools is None:
            raw_tools = []
        if not isinstance(raw_tools, list) or not all(
            isinstance(item, str) for item in raw_tools
        ):
            raise ValueError(
                f"{folder / SKILL_FILE}: 'tools' must be a list of strings."
            )

        return cls(
            name=name,
            description=description,
            instructions=body.strip(),
            path=folder,
            tools=raw_tools,
        )

    def load_tools(self) -> list[Tool]:
        """Import and return the Tool objects named in this skill's frontmatter."""

        return [_resolve_tool(spec) for spec in self.tools]


def _split_frontmatter(source: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into its YAML frontmatter dict and markdown body.

    Frontmatter is delimited by two fence lines, each exactly '---' on its own.
    A file whose first line is not exactly '---' is treated as all body, so a
    '---' horizontal rule inside the body is never mistaken for a fence.
    """

    lines = source.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, source

    # The body starts after the first closing fence, so a later '---' in the
    # body is left untouched.
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            data = yaml.safe_load("".join(lines[1:index])) or {}
            if not isinstance(data, dict):
                raise ValueError("SKILL.md frontmatter must be a YAML mapping.")
            return data, "".join(lines[index + 1 :])

    raise ValueError("Unterminated SKILL.md frontmatter (missing closing '---').")


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
