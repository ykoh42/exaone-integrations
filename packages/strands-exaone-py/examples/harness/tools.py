"""Use a custom tool in Strands Harness across two conversation turns."""

import os

from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneModel, ExaoneRetryStrategy


@tool
def get_ticket(ticket_id: str) -> str:
    """Look up a ticket in the example support queue."""
    if ticket_id == "DEMO-42":
        return "DEMO-42: The documentation update is ready for review. Owner: Alex."
    return "Ticket not found in the example queue."


def main() -> None:
    agent = create_harness(
        model=ExaoneModel(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1")),
        retry_strategy=ExaoneRetryStrategy(),
        tools=[get_ticket],
        builtin_tools={"*": False},
        builtin_plugins=[],
        session=False,
        memory=False,
        skills=False,
        caching=False,
        callback_handler=None,
    )
    for prompt in ["Use get_ticket to check DEMO-42.", "Who owns that ticket? Answer in Korean."]:
        result = agent(prompt)
        for block in result.message["content"]:
            if "text" in block:
                print(block["text"])


if __name__ == "__main__":
    main()
