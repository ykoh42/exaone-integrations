"""Let a Strands agent execute a typed tool with EXAONE."""

import os

from strands import Agent, tool

from strands_exaone import ExaoneModel, ExaoneRetryStrategy


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def main() -> None:
    agent = Agent(
        model=ExaoneModel(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1")),
        retry_strategy=ExaoneRetryStrategy(),
        tools=[multiply],
        system_prompt="Use the multiply tool for multiplication.",
        callback_handler=None,
    )
    result = agent("What is 6 times 7?")
    for block in result.message["content"]:
        if "text" in block:
            print(block["text"])


if __name__ == "__main__":
    main()
