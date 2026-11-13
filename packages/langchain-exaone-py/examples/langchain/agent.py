"""Let LangChain manage tool execution and conversation checkpoints."""

import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_exaone import ChatEXAONE


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def main() -> None:
    agent = create_agent(
        model=ChatEXAONE(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1")),
        tools=[multiply],
        system_prompt="Use the multiply tool for multiplication.",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "calculator"}, "recursion_limit": 20}
    for prompt in ["What is 6 times 7?", "What numbers did you multiply? Answer in Korean."]:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config)
        print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
