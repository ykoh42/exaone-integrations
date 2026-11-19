"""Use a custom tool in Strands Harness across two conversation turns."""

import argparse

from strands import tool
from strands.middleware import InvokeModelStage
from strands_harness import create_harness

from strands_exaone import ExaoneModel, ExaoneRetryStrategy
from strands_exaone.recovery import recover_tool_calls


@tool
def get_ticket(ticket_id: str) -> str:
    """Look up a ticket in the example support queue."""
    if ticket_id == "DEMO-42":
        return "DEMO-42: The documentation update is ready for review. Owner: Alex."
    return "Ticket not found in the example queue."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-tool-calls", action="store_true", help="Retry an eligible malformed tool call")
    args = parser.parse_args()
    agent = create_harness(
        model=ExaoneModel(),
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
    if args.recover_tool_calls:
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
    for prompt in ["Use get_ticket to check DEMO-42.", "Who owns that ticket? Answer in Korean."]:
        result = agent(prompt)
        for block in result.message["content"]:
            if "text" in block:
                print(block["text"])


if __name__ == "__main__":
    main()
