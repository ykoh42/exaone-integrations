"""Stream a response and retain the complete message for the next turn."""

import os

from langchain_core.messages import AIMessageChunk, HumanMessage

from langchain_exaone import ChatEXAONE


def main() -> None:
    """Stream a response and use it in a follow-up turn."""
    model = ChatEXAONE(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1"))
    question = HumanMessage("Explain context windows in two sentences.")
    response = AIMessageChunk(content="")
    for chunk in model.stream([question]):
        print(chunk.text, end="", flush=True)
        response += chunk
    print()
    follow_up = model.invoke([question, response, HumanMessage("Summarize your answer in Korean.")])
    print(follow_up.text)


if __name__ == "__main__":
    main()
