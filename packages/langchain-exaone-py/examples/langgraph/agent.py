"""Build an explicit tool loop with LangGraph and an in-memory checkpointer."""

import os

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_exaone import ChatEXAONE


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def main() -> None:
    model = ChatEXAONE(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1")).bind_tools([multiply])

    def call_model(state: MessagesState) -> dict:
        response = model.invoke([SystemMessage("Use the multiply tool for multiplication."), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([multiply]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    agent = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "calculator"}, "recursion_limit": 20}
    for prompt in ["What is 6 times 7?", "Multiply that result by 2."]:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config)
        print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
