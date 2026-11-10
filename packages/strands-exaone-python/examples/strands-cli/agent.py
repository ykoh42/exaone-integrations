"""Run an EXAONE-powered Strands harness from the Strands CLI."""

import os

from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneModel


@tool
def get_example_number() -> int:
    """Return an example number for testing the tool loop."""
    return 7


agent = create_harness(
    model=ExaoneModel(
        base_url=os.environ.get("EXAONE_BASE_URL", "https://api.exaone.ai/v1"),
        preserve_thinking=True,
    ),
    tools=[get_example_number],
    builtin_tools=[],
    builtin_plugins=[],
    session=False,
    memory=False,
    skills=False,
    caching=False,
)
