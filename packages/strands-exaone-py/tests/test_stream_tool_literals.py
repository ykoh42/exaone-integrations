"""Synthetic tool arguments must survive streaming and the Strands agent loop."""

import json
from typing import Any

import httpx
from strands import Agent, tool

from strands_exaone import ExaoneModel


async def test_streamed_tool_argument_literals_reach_tool_and_history() -> None:
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
    final = b'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    requests: list[dict[str, Any]] = []
    received: list[str] = []

    @tool
    def read_literal(value: str) -> str:
        """Return a synthetic result after receiving text."""
        received.append(value)
        return "synthetic result"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=first if len(requests) == 1 else final)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client)
        agent = Agent(model=model, tools=[read_literal], callback_handler=None)
        await agent.invoke_async("Use the tool.")

    assert received == [value]
    assert len(requests) == 2
    assistant = next(message for message in requests[1]["messages"] if message.get("tool_calls"))
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"value": value}
