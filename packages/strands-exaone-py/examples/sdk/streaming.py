"""Stream text while the agent retains complete conversation messages."""

import asyncio
import os

from strands import Agent

from strands_exaone import ExaoneModel, ExaoneRetryStrategy


async def main() -> None:
    agent = Agent(
        model=ExaoneModel(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1")),
        retry_strategy=ExaoneRetryStrategy(),
        callback_handler=None,
    )
    for prompt in ["Explain context windows in two sentences.", "Summarize your answer in Korean."]:
        async for event in agent.stream_async(prompt):
            if "data" in event:
                print(event["data"], end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
