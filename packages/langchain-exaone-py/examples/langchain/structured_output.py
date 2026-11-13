"""Extract a typed contact through LangChain's tool-based output parser."""

import os

from pydantic import BaseModel, Field

from langchain_exaone import ChatEXAONE


class Contact(BaseModel):
    """A contact mentioned in the input."""

    name: str = Field(description="The person's name")
    email: str = Field(description="The person's email address")


def main() -> None:
    model = ChatEXAONE(base_url=os.getenv("EXAONE_BASE_URL", "https://api.exaone.ai/v1"))
    contact = model.with_structured_output(Contact).invoke("Extract the contact: Alex, alex@example.com.")
    print(contact.model_dump_json())


if __name__ == "__main__":
    main()
