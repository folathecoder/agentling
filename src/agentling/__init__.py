"""agentling: a tiny async tool-calling agent framework.

This module is the public API. Import from `agentling` directly; the submodules
under it are implementation detail and may be reorganized.
"""

from importlib.metadata import PackageNotFoundError, version

from .agent import Agent
from .errors import (
    AgentlingError,
    AgentRunError,
    MemoryLoadError,
    ModelError,
    ModelOutputError,
    SkillLoadError,
    ToolExecutionError,
    ToolTimeoutError,
)
from .events import (
    Event,
    FinalEvent,
    StepEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    print_events,
)
from .memory import ActionStep, FinalStep, Memory, Step, TaskStep, ToolResult
from .models import (
    ChatMessage,
    Delta,
    Model,
    OpenAIModel,
    ToolCall,
    ToolCallDelta,
    Usage,
    agglomerate_deltas,
)
from .skills import Skill
from .tools import Tool, ToolCallError, tool

try:
    __version__ = version("agentling")
except PackageNotFoundError:  # Running from a source tree without an install.
    __version__ = "0.0.0"

__all__ = [
    # Core
    "Agent",
    "Model",
    "OpenAIModel",
    "Skill",
    "Tool",
    "tool",
    "print_events",
    # Errors
    "AgentlingError",
    "AgentRunError",
    "ModelError",
    "ModelOutputError",
    "ToolCallError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "MemoryLoadError",
    "SkillLoadError",
    # Messages and model types
    "ChatMessage",
    "ToolCall",
    "ToolCallDelta",
    "Delta",
    "Usage",
    "agglomerate_deltas",
    # Memory
    "Memory",
    "Step",
    "TaskStep",
    "ActionStep",
    "FinalStep",
    "ToolResult",
    # Events
    "Event",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "StepEvent",
    "FinalEvent",
    # Version
    "__version__",
]
