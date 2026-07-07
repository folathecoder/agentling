import agentling


def test_every_name_in_all_resolves() -> None:
    for name in agentling.__all__:
        assert hasattr(agentling, name), f"{name!r} is in __all__ but not importable"


def test_key_symbols_are_the_expected_objects() -> None:
    from agentling import Agent, OpenAIModel, Skill, print_events, tool
    from agentling.agent import Agent as AgentImpl
    from agentling.models import OpenAIModel as OpenAIModelImpl
    from agentling.skills import Skill as SkillImpl

    assert Agent is AgentImpl
    assert OpenAIModel is OpenAIModelImpl
    assert Skill is SkillImpl
    assert callable(tool)
    assert callable(print_events)


def test_tool_call_error_is_under_the_agentling_base() -> None:
    from agentling import AgentlingError, ToolCallError

    assert issubclass(ToolCallError, AgentlingError)


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(agentling.__version__, str)
    assert agentling.__version__
