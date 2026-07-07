from pathlib import Path

import pytest

from agentling.skills import Skill, _resolve_tool, _split_frontmatter
from agentling.tools import tool


@tool
def sample_tool(x: int) -> int:
    """Double a number.

    Args:
        x: The number to double.
    """
    return x * 2


# A non-Tool module attribute, used to exercise the entry-point type guard.
not_a_tool = 123


def _write_skill(folder: Path, text: str) -> Path:
    """Create a skill folder containing a SKILL.md with the given contents."""

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    return folder


# --------------------------------------------------------------------------- #
# Skill.from_path — happy path
# --------------------------------------------------------------------------- #
def test_from_path_reads_name_description_and_body(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "greeter",
        "---\n"
        "name: greeter\n"
        "description: Greet the user warmly.\n"
        "---\n"
        "# Greeter\n"
        "\n"
        "Say hello.\n",
    )

    skill = Skill.from_path(folder)

    assert skill.name == "greeter"
    assert skill.description == "Greet the user warmly."
    assert skill.instructions == "# Greeter\n\nSay hello."
    assert skill.path == folder
    assert skill.tools == []


def test_from_path_accepts_a_string_path(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "s",
        "---\nname: s\ndescription: d\n---\nbody\n",
    )

    skill = Skill.from_path(str(folder))

    assert skill.name == "s"
    assert skill.path == folder


def test_from_path_parses_a_tools_list(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "reviewer",
        "---\n"
        "name: reviewer\n"
        "description: Review code.\n"
        "tools:\n"
        "  - some.module:thing\n"
        "  - other.module:other\n"
        "---\n"
        "body\n",
    )

    skill = Skill.from_path(folder)

    assert skill.tools == ["some.module:thing", "other.module:other"]


def test_from_path_empty_tools_key_yields_no_tools(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "s",
        "---\nname: s\ndescription: d\ntools:\n---\nbody\n",
    )

    skill = Skill.from_path(folder)

    # `tools:` with no value parses to None, which must normalize to [].
    assert skill.tools == []


def test_from_path_preserves_horizontal_rule_in_body(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "s",
        "---\n"
        "name: s\n"
        "description: d\n"
        "---\n"
        "Section one\n"
        "\n"
        "---\n"
        "\n"
        "Section two\n",
    )

    skill = Skill.from_path(folder)

    # The '---' rule inside the body must survive: only the first two fences
    # delimit the frontmatter.
    assert "---" in skill.instructions
    assert "Section one" in skill.instructions
    assert "Section two" in skill.instructions


# --------------------------------------------------------------------------- #
# Skill.from_path — error paths
# --------------------------------------------------------------------------- #
def test_from_path_missing_name_raises(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "s",
        "---\ndescription: d\n---\nbody\n",
    )

    with pytest.raises(ValueError, match="missing required frontmatter key 'name'"):
        Skill.from_path(folder)


def test_from_path_missing_description_raises(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "s",
        "---\nname: s\n---\nbody\n",
    )

    with pytest.raises(
        ValueError, match="missing required frontmatter key 'description'"
    ):
        Skill.from_path(folder)


def test_from_path_error_names_the_file(tmp_path: Path) -> None:
    folder = _write_skill(tmp_path / "s", "---\ndescription: d\n---\nbody\n")

    with pytest.raises(ValueError, match="SKILL.md"):
        Skill.from_path(folder)


def test_from_path_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Skill.from_path(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# _split_frontmatter
# --------------------------------------------------------------------------- #
def test_split_no_leading_fence_is_all_body() -> None:
    data, body = _split_frontmatter("# Just markdown\n")

    assert data == {}
    assert body == "# Just markdown\n"


def test_split_returns_mapping_and_body() -> None:
    data, body = _split_frontmatter("---\nname: x\ndescription: y\n---\nHello")

    assert data == {"name": "x", "description": "y"}
    assert body == "\nHello"


def test_split_empty_frontmatter_returns_empty_dict() -> None:
    data, body = _split_frontmatter("---\n---\nbody")

    assert data == {}
    assert "body" in body


def test_split_unterminated_frontmatter_raises() -> None:
    with pytest.raises(ValueError, match="Unterminated"):
        _split_frontmatter("---\nname: x\n")


def test_split_non_mapping_frontmatter_raises() -> None:
    with pytest.raises(ValueError, match="YAML mapping"):
        _split_frontmatter("---\njust a scalar\n---\nbody")


# --------------------------------------------------------------------------- #
# _resolve_tool
# --------------------------------------------------------------------------- #
def test_resolve_tool_returns_the_tool_instance() -> None:
    resolved = _resolve_tool(f"{__name__}:sample_tool")

    assert resolved is sample_tool
    assert resolved.name == "sample_tool"


def test_resolve_tool_missing_colon_raises() -> None:
    with pytest.raises(ValueError, match="must be"):
        _resolve_tool("no_colon_here")


def test_resolve_tool_non_tool_target_raises() -> None:
    with pytest.raises(TypeError, match="expected a Tool"):
        _resolve_tool(f"{__name__}:not_a_tool")


def test_resolve_tool_unknown_module_raises() -> None:
    with pytest.raises(ModuleNotFoundError):
        _resolve_tool("agentling._definitely_not_a_module:thing")


def test_resolve_tool_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        _resolve_tool(f"{__name__}:missing_attribute")


# --------------------------------------------------------------------------- #
# Skill.load_tools
# --------------------------------------------------------------------------- #
def test_load_tools_empty_returns_empty_list() -> None:
    skill = Skill(
        name="x", description="y", instructions="", path=Path("."), tools=[]
    )

    assert skill.load_tools() == []


def test_load_tools_resolves_declared_entry_points() -> None:
    skill = Skill(
        name="x",
        description="y",
        instructions="",
        path=Path("."),
        tools=[f"{__name__}:sample_tool"],
    )

    tools = skill.load_tools()

    assert tools == [sample_tool]
    assert tools[0].name == "sample_tool"


def test_from_path_then_load_tools_end_to_end(tmp_path: Path) -> None:
    folder = _write_skill(
        tmp_path / "review",
        "---\n"
        "name: review\n"
        "description: Review code.\n"
        "tools:\n"
        f"  - {__name__}:sample_tool\n"
        "---\n"
        "Review carefully.\n",
    )

    skill = Skill.from_path(folder)
    loaded = skill.load_tools()

    assert skill.instructions == "Review carefully."
    assert [t.name for t in loaded] == ["sample_tool"]


# --------------------------------------------------------------------------- #
# Bundled example
# --------------------------------------------------------------------------- #
def test_bundled_code_reviewer_example_loads() -> None:
    example = (
        Path(__file__).parent.parent
        / "examples"
        / "skills"
        / "code-reviewer"
    )

    skill = Skill.from_path(example)

    assert skill.name == "code-reviewer"
    assert skill.description
    assert skill.instructions.startswith("# Code Reviewer")
