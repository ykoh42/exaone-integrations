"""LangChain's standard unit contract checks for ChatEXAONE."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_exaone import ChatEXAONE


class TestChatEXAONEStandard(ChatModelUnitTests):
    """Run the shared LangChain chat-model interface tests."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatEXAONE

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"api_key": "synthetic-test-key"}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"EXAONE_API_KEY": "synthetic-test-key"},
            {},
            {"api_key": "synthetic-test-key"},
        )
