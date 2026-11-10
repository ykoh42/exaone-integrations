"""Contract tests against a mock EXAONE HTTP transport."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from strands import Agent, tool

from strands_exaone import ExaoneError, ExaoneModel, ExaoneThrottledError


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def sse(*events: dict[str, Any], done: bool = True) -> bytes:
    data = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
    if done:
        data += "data: [DONE]\n\n"
    return data.encode()


def chunk(delta: dict[str, Any] | None = None, finish: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **kwargs}


def model_for(handler: Any, **kwargs: Any) -> tuple[ExaoneModel, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ExaoneModel(api_key="test-secret", http_client=client, **kwargs), client


async def collect(model: ExaoneModel, **kwargs: Any) -> list[dict[str, Any]]:
    return [event async for event in model.stream([{"role": "user", "content": [{"text": "안녕"}]}], **kwargs)]


async def test_stream_sse_utf8_and_usage_callback() -> None:
    captured: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    stream_bytes = b": comment\r\n\r\n" + sse(
        chunk({"role": "assistant", "reasoning_content": "생각"}),
        chunk({"content": "\n\n안녕"}),
        chunk(finish="stop"),
        {
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "reasoning_tokens": 2},
        },
    )
    chunks = Chunks([stream_bytes[index : index + 1] for index in range(len(stream_bytes))])

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream", "x-request-id": "req-1"}, stream=chunks
        )

    model, client = model_for(handler, metadata_callback=observed.append, preserve_thinking=False, include_usage=False)
    try:
        events = await collect(model)
        assert captured[0]["chat_template_kwargs"] == {"preserve_thinking": False}
        assert captured[0]["stream_options"] == {"include_usage": False}
        assert events[0] == {"messageStart": {"role": "assistant"}}
        assert {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "생각"}}}} in events
        assert {"contentBlockDelta": {"delta": {"text": "\n\n안녕"}}} in events
        assert events[-1]["metadata"]["usage"] == {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7}
        assert observed[0]["usage"]["reasoning_tokens"] == 2
        assert observed[0]["request_id"] == "req-1"
        assert chunks.closed
    finally:
        await client.aclose()


async def test_multiline_sse_event_and_missing_done() -> None:
    payload = b'data: {"choices":\ndata: []}\n\ndata: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    model, client = model_for(lambda request: httpx.Response(200, content=payload))
    try:
        with pytest.raises(ExaoneError, match="without \\[DONE\\]"):
            await collect(model)
    finally:
        await client.aclose()


async def test_interleaved_tools_are_assembled_and_replayed() -> None:
    bodies: list[dict[str, Any]] = []
    events = sse(
        chunk({"reasoning_content": "reason"}),
        chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "alpha", "arguments": '{"x":'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "beta", "arguments": '{"x":'}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "alpha", "arguments": "1}"}}]}),
        chunk({"tool_calls": [{"index": 1, "function": {"arguments": "2}"}}]}),
        chunk(finish="tool_calls"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=events)

    model, client = model_for(handler, preserve_thinking=True)
    try:
        result = await collect(model)
        starts = [
            e["contentBlockStart"]["start"]["toolUse"]
            for e in result
            if e.get("contentBlockStart", {}).get("start", {}).get("toolUse")
        ]
        assert starts == [{"name": "alpha", "toolUseId": "a"}, {"name": "beta", "toolUseId": "b"}]
        assert result[-1] == {"messageStop": {"stopReason": "tool_use"}}
        history = [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "reason"}}},
                    {"toolUse": {"name": "alpha", "toolUseId": "a", "input": {"x": 1}}},
                    {"toolUse": {"name": "beta", "toolUseId": "b", "input": {"x": 2}}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "a", "status": "success", "content": [{"text": "11"}]}},
                    {"toolResult": {"toolUseId": "b", "status": "success", "content": [{"json": {"value": 22}}]}},
                ],
            },
        ]
        body = model._request(history, None, None, None, None, stream=True)
        assert body["messages"][0]["reasoning_content"] == "reason"
        assert [m["tool_call_id"] for m in body["messages"][1:]] == ["a", "b"]
        assert json.loads(body["messages"][0]["tool_calls"][1]["function"]["arguments"]) == {"x": 2}
    finally:
        await client.aclose()


async def test_agent_tool_round_trip() -> None:
    bodies: list[dict[str, Any]] = []

    @tool
    def get_test_number() -> str:
        """Return the test number."""
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            payload = sse(
                chunk({"reasoning_content": "thinking"}),
                chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "id": "call-1", "function": {"name": "get_test_number", "arguments": "{}"}}
                        ]
                    }
                ),
                chunk(finish="tool_calls"),
            )
        else:
            payload = sse(chunk({"content": "The number is 7."}), chunk(finish="stop"))
        return httpx.Response(200, content=payload)

    model, client = model_for(handler, preserve_thinking=True)
    try:
        agent = Agent(model=model, tools=[get_test_number], callback_handler=None)
        result = await agent.invoke_async("Use the tool")
        assert "7" in str(result)
        assert len(bodies) == 2
        assistant = next(m for m in bodies[1]["messages"] if m["role"] == "assistant")
        assert assistant["reasoning_content"] == "thinking"
        assert any(m.get("tool_call_id") == "call-1" and "7" in m["content"] for m in bodies[1]["messages"])
    finally:
        await client.aclose()


async def test_non_stream_and_structured_output() -> None:
    bodies: list[dict[str, Any]] = []

    class Item(BaseModel):
        count: int

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        response = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"content": '{"count":3}', "reasoning_content": "reason"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6, "reasoning_tokens": 1},
        }
        return httpx.Response(200, json=response)

    model, client = model_for(handler, stream=False, max_tokens=0, response_format={"type": "json_object"})
    try:
        events = await collect(model)
        assert bodies[0]["stream"] is False
        assert bodies[0]["max_tokens"] == 0
        assert bodies[0]["response_format"] == {"type": "json_object"}
        assert any(e.get("contentBlockDelta", {}).get("delta", {}).get("text") == '{"count":3}' for e in events)
        output = [
            item async for item in model.structured_output(Item, [{"role": "user", "content": [{"text": "count"}]}])
        ]
        assert output == [{"output": Item(count=3)}]
        assert bodies[1]["response_format"]["type"] == "json_schema"
        assert bodies[1]["stream"] is False
    finally:
        await client.aclose()


async def test_http_error_shapes_redact_key_and_preserve_retry_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "bad test-secret", "type": "rate_limit_error", "code": "rate_limit_exceeded"}},
            headers={"retry-after-ms": "631", "x-request-id": "request-1"},
        )

    model, client = model_for(handler)
    try:
        with pytest.raises(ExaoneThrottledError) as info:
            await collect(model)
        assert info.value.status_code == 429
        assert info.value.code == "rate_limit_exceeded"
        assert info.value.headers["retry-after-ms"] == "631"
        assert info.value.request_id == "request-1"
        assert "test-secret" not in str(info.value.body)
        assert "test-secret" not in str(info.value)
    finally:
        await client.aclose()


async def test_unknown_422_preserves_detail_body() -> None:
    model, client = model_for(lambda request: httpx.Response(422, json={"detail": [{"msg": "invalid"}]}))
    try:
        with pytest.raises(ExaoneError) as info:
            await collect(model)
        assert info.value.status_code == 422
        assert info.value.body == {"detail": [{"msg": "invalid"}]}
    finally:
        await client.aclose()


async def test_incomplete_and_malformed_tool_arguments_never_emit_tool_block() -> None:
    raw = sse(
        chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "one", "arguments": "{"}}]}),
        chunk(finish="tool_calls"),
    )
    model, client = model_for(lambda request: httpx.Response(200, content=raw))
    events: list[dict[str, Any]] = []
    try:
        with pytest.raises(ExaoneError, match="Invalid tool arguments"):
            async for event in model.stream([{"role": "user", "content": [{"text": "hello"}]}]):
                events.append(event)
        assert not any(e.get("contentBlockStart", {}).get("start", {}).get("toolUse") for e in events)
    finally:
        await client.aclose()


async def test_duplicate_tool_ids_never_emit_tool_blocks() -> None:
    calls = [
        {"index": 0, "id": "same", "function": {"name": "first", "arguments": "{}"}},
        {"index": 1, "id": "same", "function": {"name": "second", "arguments": "{}"}},
    ]
    raw = sse(chunk({"tool_calls": calls}), chunk(finish="tool_calls"))
    model, client = model_for(lambda request: httpx.Response(200, content=raw))
    events: list[dict[str, Any]] = []
    try:
        with pytest.raises(ExaoneError, match="Duplicate tool call IDs"):
            async for event in model.stream([{"role": "user", "content": [{"text": "hello"}]}]):
                events.append(event)
        assert not any(e.get("contentBlockStart", {}).get("start", {}).get("toolUse") for e in events)
    finally:
        await client.aclose()


async def test_concurrent_metadata_is_request_scoped() -> None:
    observed: list[dict[str, Any]] = []

    async def callback(meta: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        observed.append(meta)

    def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["messages"][0]["content"]
        payload = sse(
            chunk({"content": text}),
            chunk(finish="stop"),
            {
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "reasoning_tokens": 0},
            },
        )
        return httpx.Response(200, content=payload, headers={"x-request-id": text})

    model, client = model_for(handler, metadata_callback=callback)
    try:

        async def invoke(text: str) -> None:
            _ = [e async for e in model.stream([{"role": "user", "content": [{"text": text}]}])]

        await asyncio.gather(invoke("one"), invoke("two"))
        assert {item["request_id"] for item in observed} == {"one", "two"}
        assert all(item["usage"]["reasoning_tokens"] == 0 for item in observed)
    finally:
        await client.aclose()


async def test_callback_failure_propagates() -> None:
    def callback(_: Any) -> None:
        raise RuntimeError("callback failed")

    model, client = model_for(
        lambda request: httpx.Response(200, content=sse(chunk({"content": "ok"}), chunk(finish="stop"))),
        metadata_callback=callback,
    )
    try:
        with pytest.raises(RuntimeError, match="callback failed"):
            await collect(model)
    finally:
        await client.aclose()


def test_configuration_and_unsupported_media() -> None:
    model = ExaoneModel(api_key="key", params={"n": 1}, top_p=0, top_k=-1)
    assert "api_key" not in model.get_config()
    with pytest.raises(ValueError, match="exactly one"):
        model.update_config(params={"n": 2})
    with pytest.raises(ValueError, match="exactly one"):
        model.update_config(params={"n": True})
    assert model.get_config()["params"] == {"n": 1}
    with pytest.raises(ValueError, match="Unsupported"):
        model._request([{"role": "user", "content": [{"image": {}}]}], None, None, None, None, stream=True)


async def test_cancellation_while_waiting_for_stream_closes_response() -> None:
    entered = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self) -> None:
            self.closed = True

    response_stream = SlowStream()
    model, client = model_for(lambda request: httpx.Response(200, stream=response_stream))
    signal = threading.Event()
    try:
        task = asyncio.create_task(collect(model, cancel_signal=signal))
        await asyncio.wait_for(entered.wait(), 2)
        signal.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert response_stream.closed
        assert not client.is_closed
    finally:
        await client.aclose()
