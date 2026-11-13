"""EXAONE model facts and the public Strands override contract."""

import json

import httpx
from pydantic import BaseModel
from strands import Agent
from strands.models import ModelCapabilities, ModelInfo, ModelLimits

from strands_exaone import ExaoneModel


def test_known_model_capabilities_and_context() -> None:
    model = ExaoneModel(api_key="synthetic")
    info = model.model_info
    assert info.limits.context_window == 524_288
    assert info.limits.max_output_tokens is None
    assert info.capabilities.stateful is False
    assert info.capabilities.forced_tool_choice is True
    assert info.capabilities.native_structured_output is True
    assert info.capabilities.prompt_caching is None
    assert model.get_config()["model_info"] is None


def test_unset_context_inherits_known_limit() -> None:
    model = ExaoneModel(api_key="synthetic", context_window_limit=None)
    assert model.model_info.limits.context_window == 524_288
    model.update_config(model_info=ModelInfo(limits=ModelLimits(context_window=None)))
    assert model.model_info.limits.context_window == 524_288


def test_caller_overrides_win_without_replacing_other_facts() -> None:
    model = ExaoneModel(
        api_key="synthetic",
        context_window_limit=4096,
        model_info=ModelInfo(capabilities=ModelCapabilities(native_structured_output=False)),
    )
    assert model.model_info.limits.context_window == 4096
    assert model.model_info.capabilities.native_structured_output is False
    assert model.model_info.capabilities.forced_tool_choice is True
    model.update_config(
        model_info=ModelInfo(
            limits=ModelLimits(context_window=8192, max_output_tokens=1024),
            capabilities=ModelCapabilities(stateful=True, forced_tool_choice=False),
        )
    )
    assert model.model_info.limits.context_window == 8192
    assert model.model_info.limits.max_output_tokens == 1024
    assert model.model_info.capabilities.stateful is True
    assert model.model_info.capabilities.forced_tool_choice is False
    assert model.model_info.capabilities.native_structured_output is True


def test_model_changes_do_not_inherit_exaone_capabilities() -> None:
    model = ExaoneModel(api_key="synthetic")
    model.update_config(model_id="unknown-model")
    assert model.model_info.capabilities.forced_tool_choice is None
    assert model.model_info.capabilities.native_structured_output is None
    model.update_config(model_id="k-exaone_v2.5")
    assert model.model_info.capabilities.native_structured_output is True


async def test_agent_retains_sdk_structured_output_contract() -> None:
    class Contact(BaseModel):
        name: str

    bodies: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": '{"name":"Alex"}'}, "finish_reason": "stop"}]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ExaoneModel(api_key="synthetic", http_client=client, stream=False)
        agent = Agent(model=model, callback_handler=None)
        result = await agent.invoke_async("Extract Alex.", structured_output_model=Contact)
    assert result.structured_output == Contact(name="Alex")
    assert bodies[0]["tools"][0]["function"]["parameters"]["properties"]["name"]["type"] == "string"
