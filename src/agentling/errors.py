"""Exception hierarchy for agentling.

Everything the framework raises descends from AgentlingError, so a host can
catch all of it with a single except. The subclasses mark the failure domain,
which is what lets different layers treat them differently: some are fatal, some
are surfaced back to the model as recoverable observations, and some are
user-facing configuration errors.
"""

from __future__ import annotations


class AgentlingError(Exception):
    """Base class for every error raised by agentling."""


class AgentRunError(AgentlingError):
    """A run could not start or continue.

    Raised for lifecycle violations such as starting a second run on state that
    is already in use.
    """


class ModelError(AgentlingError):
    """A model provider failed in a way the run cannot recover from."""


class ModelOutputError(AgentlingError):
    """The model produced output the framework could not parse.

    Raised for malformed streamed tool calls (invalid JSON arguments, a
    non-object arguments payload, a missing tool name, or a missing tool-call
    id). The agent loop can turn this into a recoverable observation so the
    model gets a chance to retry.
    """


class ToolExecutionError(AgentlingError):
    """A tool failed unexpectedly while executing.

    This is distinct from ToolCallError, which signals invalid model-supplied
    arguments. ToolExecutionError covers failures inside the tool itself.
    """


class ToolTimeoutError(ToolExecutionError):
    """A tool did not finish within its allotted time."""


class MemoryLoadError(AgentlingError):
    """Serialized memory could not be loaded (unknown version or bad shape)."""


class SkillLoadError(AgentlingError):
    """A skill could not be loaded from its SKILL.md."""
