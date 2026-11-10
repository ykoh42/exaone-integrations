"""Integration tests using the Strands harness and a mock EXAONE transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
from harness_support import chunk, events, make_harness, response, tool_delta
from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneModel


async def test_harness_receives_synthetic_stream_usage() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response(
            events(
                chunk({"content": "ok"}),
                chunk(finish="stop"),
                {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 4, "total_tokens": 127}},
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client))
        result = await agent.invoke_async("Hello.")
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert result.context_size == 123
    assert result.metrics.accumulated_usage == {"inputTokens": 123, "outputTokens": 4, "totalTokens": 127}


async def test_harness_structured_output_after_tool_and_reasoning() -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    requests: list[dict[str, Any]] = []
    calls: list[bool] = []

    @tool
    def get_number() -> str:
        """Return the synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(
                events(
                    chunk({"reasoning_content": "use number"}),
                    chunk(tool_delta("number_1", "get_number")),
                    chunk(finish="tool_calls"),
                )
            )
        if len(requests) == 2:
            call = {
                "tool_calls": [
                    {"index": 0, "id": "output_1", "function": {"name": "Answer", "arguments": '{"value":7}'}}
                ]
            }
            return response(
                events(chunk({"reasoning_content": "format result"}), chunk(call), chunk(finish="tool_calls"))
            )
        return response(events(chunk({"content": "7"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_number])
        result = await agent.invoke_async("Get the number in structured output.", structured_output_model=Answer)
        assert result.structured_output == Answer(value=7)
        await agent.invoke_async("What was the number?")
    assert calls == [True]
    assert len(requests) == 3
    assert any(message.get("reasoning_content") == "use number" for message in requests[1]["messages"])
    assert any(message.get("tool_call_id") == "number_1" for message in requests[1]["messages"])
    assert any(message.get("reasoning_content") == "format result" for message in requests[2]["messages"])


async def test_harness_agentic_context_tool_and_synthetic_usage() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "pin_1",
                        "function": {"name": "pin_context", "arguments": '{"select":"last_turn"}'},
                    }
                ]
            }
            return response(
                events(
                    chunk({"reasoning_content": "pin this turn"}),
                    chunk(call),
                    chunk(finish="tool_calls"),
                    {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 4, "total_tokens": 127}},
                )
            )
        return response(events(chunk({"content": "Pinned."}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            context_manager="agentic",
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Pin this synthetic turn.")
    assert len(requests) == 2
    names = {spec["function"]["name"] for spec in requests[0]["tools"]}
    assert {"pin_context", "summarize_context", "truncate_context"} <= names
    assert any(message.get("reasoning_content") == "pin this turn" for message in requests[1]["messages"])
    assert any(
        message.get("tool_call_id") == "pin_1" and "Pinned" in message["content"] for message in requests[1]["messages"]
    )


async def test_harness_structured_output_retries_invalid_tool_value() -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        arguments = '{"value":"not-an-integer"}' if len(requests) == 1 else '{"value":7}'
        call = {
            "tool_calls": [
                {"index": 0, "id": f"output_{len(requests)}", "function": {"name": "Answer", "arguments": arguments}}
            ]
        }
        return response(
            events(chunk({"reasoning_content": f"attempt {len(requests)}"}), chunk(call), chunk(finish="tool_calls"))
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client))
        result = await agent.invoke_async(
            "Return a number.", structured_output_model=Answer, limits={"structured_output_attempts": 2}
        )
    assert result.structured_output == Answer(value=7)
    assert len(requests) == 2
    assert any(message.get("reasoning_content") == "attempt 1" for message in requests[1]["messages"])
    assert any(
        message.get("tool_call_id") == "output_1" and "value" in message["content"]
        for message in requests[1]["messages"]
    )


async def test_harness_continues_after_synthetic_extreme_usage() -> None:
    requests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    huge = 10**400

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response(
            events(
                chunk({"content": "ok"}),
                chunk(finish="stop"),
                {"choices": [], "usage": {"prompt_tokens": huge, "completion_tokens": 1, "total_tokens": huge}},
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client, metadata_callback=records.append))
        first = await agent.invoke_async("First turn.")
        second = await agent.invoke_async("Second turn.")
    assert len(requests) == 2
    assert first.metrics.accumulated_usage["inputTokens"] == 0
    assert second.metrics.accumulated_usage["inputTokens"] == 0
    assert all(record["usage"]["prompt_tokens"] == huge for record in records)
