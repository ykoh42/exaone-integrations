"""Unsupported tool reasoning signatures are not silently discarded."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from strands import Agent

from strands_exaone import ExaoneModel


class Answer(BaseModel):
    value: int


def history(signature: str | None) -> list[dict[str, Any]]:
    tool_use: dict[str, Any] = {"name": "lookup", "toolUseId": "call-1", "input": {}}
    if signature is not None:
        tool_use["reasoningSignature"] = signature
    return [
        {"role": "assistant", "content": [{"toolUse": tool_use}]},
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "call-1", "status": "success", "content": [{"text": "7"}]}}],
        },
    ]


@pytest.mark.parametrize("method", ["stream", "structured", "agent"])
async def test_signed_tool_history_is_rejected_before_request(method: str) -> None:
    requests: list[bool] = []

    def handler(_: httpx.Request) -> httpx.Response:
        requests.append(True)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-key", http_client=client)
        with pytest.raises(ValueError, match="tool reasoning signature"):
            if method == "agent":
                agent = Agent(model=model, messages=history("synthetic-signature"), callback_handler=None)
                await agent.invoke_async("Continue")
            else:
                generator = (
                    model.stream(history("synthetic-signature"))
                    if method == "stream"
                    else model.structured_output(Answer, history("synthetic-signature"))
                )
                async for _ in generator:
                    pass
    assert requests == []


@pytest.mark.parametrize("signature", [None, ""])
async def test_unsigned_and_empty_tool_signatures_keep_the_tool_history(signature: str | None) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        chunk = {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
        return httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n").encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-key", http_client=client)
        _ = [event async for event in model.stream(history(signature))]

    messages = requests[0]["messages"]
    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[1] == {"role": "tool", "tool_call_id": "call-1", "content": "7"}
