"""The smallest useful agentling agent: a math tutor with two tools.

Needs an OpenAI-compatible API key (set OPENAI_API_KEY). The model name can be
overridden with AGENTLING_EXAMPLE_MODEL:

    uv run python examples/cli_math_tutor.py

build_agent() accepts an optional model so tests can inject a fake one and run
without any network access.
"""

from __future__ import annotations

import asyncio
import os

from agentling import Agent, Model, OpenAIModel, tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


def build_agent(model: Model | None = None) -> Agent:
    """Build the tutor agent, defaulting to an OpenAI-compatible model."""

    return Agent(
        model=model
        or OpenAIModel(os.environ.get("AGENTLING_EXAMPLE_MODEL", "gpt-4o-mini")),
        tools=[add, multiply],
        instructions=(
            "You are a patient math tutor. Use the tools to compute, then "
            "explain the result in one sentence."
        ),
    )


async def main() -> None:
    answer = await build_agent().run("What is 6 times 7, then plus 3?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
