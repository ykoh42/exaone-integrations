"""Synthetic tool argument literals must survive LangChain streaming and history."""

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_exaone import ChatEXAONE


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("output_version", ["v0", "v1"])
async def test_streamed_tool_argument_literals_survive_next_request(asynchronous: bool, output_version: str) -> None:
    value = 'literal <think> and </tool_call>\\path\nsecond line with "quotes"'
    encoded = json.dumps({"value": value}, ensure_ascii=False, separators=(",", ":"))
    events: list[dict[str, Any]] = []
    for index, part in enumerate(encoded):
        call: dict[str, Any] = {"index": 0, "function": {"arguments": part}}
        if index == 0:
            call.update(id="synthetic-call", type="function")
            call["function"]["name"] = "read_literal"
        events.append({"choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]})
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    first = ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, content=first)
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})

    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = ChatEXAONE(
        api_key="fixture", http_client=sync_client, http_async_client=async_client, output_version=output_version
    )
    try:
        chunks = (
            [chunk async for chunk in model.astream("Use the tool.")]
            if asynchronous
            else list(model.stream("Use the tool."))
        )
        result = chunks[0]
        for chunk in chunks[1:]:
            result += chunk
        assert result.tool_calls == [
            {"name": "read_literal", "args": {"value": value}, "id": "synthetic-call", "type": "tool_call"}
        ]
        history = [
            HumanMessage(content="Use the tool."),
            result,
            ToolMessage(content="ok", tool_call_id="synthetic-call"),
        ]
        if asynchronous:
            await model.ainvoke(history)
        else:
            model.invoke(history)
    finally:
        sync_client.close()
        await async_client.aclose()

    assert len(requests) == 2
    assistant = next(message for message in requests[1]["messages"] if message.get("tool_calls"))
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"value": value}
