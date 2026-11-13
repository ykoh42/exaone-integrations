"""Validate an agent response with a Pydantic schema."""

import os

from pydantic import BaseModel
from strands import Agent

from strands_exaone import ExaoneModel, ExaoneRetryStrategy


class Contact(BaseModel):
    name: str
    email: str


def main() -> None:
    agent = Agent(
        model=ExaoneModel(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1")),
        retry_strategy=ExaoneRetryStrategy(),
        callback_handler=None,
    )
    result = agent("Extract the contact: Alex, alex@example.com.", structured_output_model=Contact)
    print(result.structured_output.model_dump_json())


if __name__ == "__main__":
    main()
