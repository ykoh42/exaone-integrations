"""Plan and delegate a small task using Deep Agents' virtual filesystem."""

import argparse
import os

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_exaone import ChatEXAONE
from langchain_exaone.recovery import ToolCallRecoveryMiddleware


@tool
def get_release_notes() -> str:
    """Read the synthetic release notes for this example."""
    return "Release 1.0: Added Korean documentation and a typed calculator tool."


def main() -> None:
    """Create and run the release-brief agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-tool-calls", action="store_true", help="Retry an eligible malformed tool call")
    args = parser.parse_args()
    model = ChatEXAONE(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1"))
    agent = create_deep_agent(
        model=model,
        system_prompt="Use the supplied release notes as your only factual source. Write concise reports.",
        tools=[get_release_notes],
        subagents=[
            {
                "name": "release-reviewer",
                "description": "Read and summarize the example release notes.",
                "system_prompt": "Call get_release_notes and summarize the changes accurately.",
                "model": model,
                "tools": [get_release_notes],
            }
        ],
        checkpointer=InMemorySaver(),
        middleware=[ToolCallRecoveryMiddleware()] if args.recover_tool_calls else [],
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Plan a release brief. Delegate reading the release notes to release-reviewer. "
                        "Write a short Korean brief to /release-brief.md, then summarize what you saved."
                    ),
                }
            ]
        },
        {"configurable": {"thread_id": "release-brief"}, "recursion_limit": 40},
    )
    print(result["messages"][-1].text)
    print("Virtual files:", sorted(result.get("files", {})))


if __name__ == "__main__":
    main()
