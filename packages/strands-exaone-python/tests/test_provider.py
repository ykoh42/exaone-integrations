"""Contract tests against a mock EXAONE HTTP transport."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from itertools import zip_longest
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


@pytest.mark.parametrize(
    ("config", "expected_temperature", "expected_top_p", "expected_preserve"),
    [
        ({}, 1.0, 0.95, True),
        ({"temperature": 0.0, "top_p": 0.0, "preserve_thinking": False}, 0.0, 0.0, False),
        (
            {"params": {"temperature": 0.0, "top_p": 0.0, "chat_template_kwargs": {"preserve_thinking": False}}},
            0.0,
            0.0,
            False,
        ),
        (
            {
                "temperature": 0.4,
                "top_p": 0.7,
                "preserve_thinking": True,
                "params": {"temperature": 0.0, "top_p": 0.0, "chat_template_kwargs": {"preserve_thinking": False}},
            },
            0.4,
            0.7,
            True,
        ),
    ],
)
async def test_agentic_request_defaults_and_explicit_precedence(
    config: dict[str, Any], expected_temperature: float, expected_top_p: float, expected_preserve: bool
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=sse(chunk({"content": "ok"}), chunk(finish="stop")))

    model, client = model_for(handler, **config)
    try:
        await collect(model)
        assert requests[0]["temperature"] == expected_temperature
        assert requests[0]["top_p"] == expected_top_p
        assert requests[0]["chat_template_kwargs"]["preserve_thinking"] is expected_preserve
    finally:
        await client.aclose()


async def test_agentic_defaults_follow_updates_and_keep_other_params() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=sse(chunk({"content": "ok"}), chunk(finish="stop")))

    params = {
        "temperature": 0.0,
        "top_p": 0.0,
        "seed": 42,
        "chat_template_kwargs": {"preserve_thinking": False, "custom": "value"},
    }
    model, client = model_for(handler, params=params, temperature=0.4, top_p=0.7, preserve_thinking=True)
    try:
        await collect(model)
        model.update_config(temperature=None, top_p=None, preserve_thinking=None)
        await collect(model)
        model.update_config(params={"chat_template_kwargs": {"custom": "value"}})
        await collect(model)
        assert [request["temperature"] for request in requests] == [0.4, 0.0, 1.0]
        assert [request["top_p"] for request in requests] == [0.7, 0.0, 0.95]
        assert [request["chat_template_kwargs"]["preserve_thinking"] for request in requests] == [True, False, True]
        assert all(request["chat_template_kwargs"]["custom"] == "value" for request in requests)
        assert requests[0]["seed"] == requests[1]["seed"] == 42
    finally:
        await client.aclose()


async def test_structured_output_uses_agentic_request_defaults() -> None:
    class Item(BaseModel):
        count: int

    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"count":3}'}}]})

    model, client = model_for(handler)
    try:
        output = [
            event async for event in model.structured_output(Item, [{"role": "user", "content": [{"text": "count"}]}])
        ]
        assert output == [{"output": Item(count=3)}]
        assert requests[0]["temperature"] == 1.0
        assert requests[0]["top_p"] == 0.95
        assert requests[0]["chat_template_kwargs"]["preserve_thinking"] is True
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
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
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


@pytest.mark.parametrize("stream", [True, False])
async def test_tool_round_trip_preserves_text_whitespace(stream: bool) -> None:
    requests: list[dict[str, Any]] = []
    original_text = "\n\nUseful text  \n"

    @tool
    def get_test_number() -> str:
        """Return the test number."""
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {"id": "call-1", "type": "function", "function": {"name": "get_test_number", "arguments": "{}"}}
            if stream:
                payload = sse(
                    chunk({"content": "\n"}),
                    chunk({"content": "\nUseful text  \n"}),
                    chunk({"tool_calls": [{"index": 0, **call}]}),
                    chunk(finish="tool_calls"),
                )
                return httpx.Response(200, content=payload)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": original_text, "tool_calls": [call]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        if stream:
            return httpx.Response(200, content=sse(chunk({"content": "done"}), chunk(finish="stop")))
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "done"}, "finish_reason": "stop"}]}
        )

    model, client = model_for(handler, stream=stream)
    try:
        agent = Agent(model=model, tools=[get_test_number], callback_handler=None)
        await agent.invoke_async("Use the tool.")
        assert len(requests) == 2
        assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
        assert assistant["content"] == original_text
        assert any(message.get("tool_call_id") == "call-1" for message in requests[1]["messages"])
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

    model, client = model_for(
        handler,
        stream=False,
        max_tokens=0,
        response_format={"type": "json_object"},
        params={"stream_options": {"include_usage": False}},
    )
    try:
        events = await collect(model)
        assert bodies[0]["stream"] is False
        assert "stream_options" not in bodies[0]
        assert bodies[0]["max_tokens"] == 0
        assert bodies[0]["response_format"] == {"type": "json_object"}
        assert any(e.get("contentBlockDelta", {}).get("delta", {}).get("text") == '{"count":3}' for e in events)
        output = [
            item async for item in model.structured_output(Item, [{"role": "user", "content": [{"text": "count"}]}])
        ]
        assert output == [{"output": Item(count=3)}]
        assert bodies[1]["response_format"]["type"] == "json_schema"
        assert bodies[1]["stream"] is False
        assert "stream_options" not in bodies[1]
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


@pytest.mark.parametrize(
    ("payload", "error_message"),
    [
        (b'{"sensitive_output":', "Malformed JSON in EXAONE response"),
        (b'["sensitive_output"]', "Invalid EXAONE response payload"),
        (b"null", "Invalid EXAONE response payload"),
        (b'{"choices":{"sensitive_output":{}}}', "Expected exactly one structured output choice"),
        (b'{"choices":["sensitive_output"]}', "Invalid structured output choice"),
        (
            b'{"choices":[{"finish_reason":"stop","message":"sensitive_output"}]}',
            "Structured output did not include a message",
        ),
        (
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"count\\":\\"sensitive_output\\"}"}}]}',
            "Structured output did not match the requested schema",
        ),
    ],
)
async def test_invalid_structured_response_is_sanitized(payload: bytes, error_message: str) -> None:
    class Item(BaseModel):
        count: int

    model, client = model_for(lambda request: httpx.Response(200, content=payload))
    try:
        with pytest.raises(ExaoneError, match=error_message) as error:
            _ = [
                event
                async for event in model.structured_output(Item, [{"role": "user", "content": [{"text": "test"}]}])
            ]
        assert "sensitive_output" not in str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__context__ is None
    finally:
        await client.aclose()


@pytest.mark.parametrize("mode", ["stream", "nonstream", "structured"])
async def test_http_success_with_error_envelope_preserves_safe_details(mode: str) -> None:
    class Item(BaseModel):
        count: int

    payload = {"error": {"message": "invalid test-secret", "code": "synthetic_code", "param": "input"}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "request-1"},
            content=sse(payload) if mode == "stream" else json.dumps(payload).encode(),
        )

    metadata: list[dict[str, Any]] = []
    model, client = model_for(handler, stream=mode == "stream", metadata_callback=metadata.append)
    try:
        with pytest.raises(ExaoneError) as error:
            if mode == "structured":
                _ = [
                    event
                    async for event in model.structured_output(Item, [{"role": "user", "content": [{"text": "test"}]}])
                ]
            else:
                await collect(model)
        assert error.value.status_code == 200
        assert error.value.code == "synthetic_code"
        assert error.value.param == "input"
        assert error.value.request_id == "request-1"
        assert error.value.body == {
            "error": {"message": "invalid [REDACTED]", "code": "synthetic_code", "param": "input"}
        }
        assert "test-secret" not in str(error.value)
        assert metadata == []
    finally:
        await client.aclose()


@pytest.mark.parametrize("mode", ["stream", "nonstream", "structured"])
async def test_null_error_field_is_accepted_in_successful_responses(mode: str) -> None:
    class Item(BaseModel):
        count: int

    normal = {"error": None, "choices": [{"finish_reason": "stop", "message": {"content": '{"count":3}'}}]}
    stream = sse(chunk({"content": "ok"}, error=None), chunk(finish="stop", error=None))
    model, client = model_for(
        lambda request: httpx.Response(200, content=stream if mode == "stream" else json.dumps(normal).encode()),
        stream=mode == "stream",
    )
    try:
        if mode == "structured":
            result = [
                event
                async for event in model.structured_output(Item, [{"role": "user", "content": [{"text": "test"}]}])
            ]
            assert result == [{"output": Item(count=3)}]
        else:
            result = await collect(model)
            assert any(event.get("messageStop", {}).get("stopReason") == "end_turn" for event in result)
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


@pytest.mark.parametrize(
    ("field", "fragment"),
    [("id", "a"), ("id", "call_123"), ("name", "a"), ("name", "example_tool")],
)
async def test_repeated_tool_identifier_fragments_are_assembled(field: str, fragment: str) -> None:
    first_call: dict[str, Any] = {
        "index": 0,
        "id": "call_123",
        "function": {"name": "example_tool", "arguments": "{}"},
    }
    second_call: dict[str, Any] = {"index": 0}
    if field == "id":
        first_call["id"] = fragment
        second_call["id"] = fragment
    else:
        first_call["function"]["name"] = fragment
        second_call["function"] = {"name": fragment}
    raw = sse(chunk({"tool_calls": [first_call]}), chunk({"tool_calls": [second_call]}), chunk(finish="tool_calls"))
    model, client = model_for(lambda request: httpx.Response(200, content=raw))
    try:
        result = await collect(model)
        starts = [
            event["contentBlockStart"]["start"]["toolUse"]
            for event in result
            if "toolUse" in event.get("contentBlockStart", {}).get("start", {})
        ]
        expected_name = fragment + fragment if field == "name" else "example_tool"
        expected_id = fragment + fragment if field == "id" else "call_123"
        assert starts == [{"name": expected_name, "toolUseId": expected_id}]
    finally:
        await client.aclose()


async def test_split_tool_name_executes_the_complete_name_only() -> None:
    calls: list[str] = []
    requests: list[dict[str, Any]] = []

    @tool
    def a() -> str:
        """Return a value from the first tool."""
        calls.append("a")
        return "first"

    @tool
    def aa() -> str:
        """Return a value from the second tool."""
        calls.append("aa")
        return "second"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            payload = sse(
                chunk({"tool_calls": [{"index": 0, "id": "call_", "function": {"name": "a", "arguments": "{"}}]}),
                chunk({"tool_calls": [{"index": 0, "id": "1", "function": {"name": "a", "arguments": "}"}}]}),
                chunk(finish="tool_calls"),
            )
        else:
            payload = sse(chunk({"content": "done"}), chunk(finish="stop"))
        return httpx.Response(200, content=payload)

    model, client = model_for(handler)
    try:
        agent = Agent(model=model, tools=[a, aa], callback_handler=None)
        await agent.invoke_async("Use the second tool.")
        assert calls == ["aa"]
        assert any(message.get("tool_call_id") == "call_1" for message in requests[1]["messages"])
    finally:
        await client.aclose()


async def test_distinct_tool_identifier_fragments_are_assembled() -> None:
    raw = sse(
        chunk({"tool_calls": [{"index": 0, "id": "call_", "function": {"name": "get_", "arguments": "{"}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "123", "function": {"name": "value", "arguments": "}"}}]}),
        chunk(finish="tool_calls"),
    )
    model, client = model_for(lambda request: httpx.Response(200, content=raw))
    try:
        result = await collect(model)
        starts = [
            event["contentBlockStart"]["start"]["toolUse"]
            for event in result
            if "toolUse" in event.get("contentBlockStart", {}).get("start", {})
        ]
        assert starts == [{"name": "get_value", "toolUseId": "call_123"}]
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("id_parts", "name_parts"),
    [
        (["c", "a", "l", "l", "_", "1"], ["a"]),
        (["call_", "1"], ["a", "a"]),
        (["id", "id"], ["get_", "weather"]),
        (["call_ab", "12"], ["lookup", "_", "42"]),
        (["x", "x", "x"], ["r", "e", "a", "d", "_", "data"]),
    ],
)
async def test_varied_valid_tool_delta_boundaries_are_preserved(id_parts: list[str], name_parts: list[str]) -> None:
    argument_parts = ['{"query":', '"value"', "}"]
    chunks = [chunk({"tool_calls": [{"index": 0, "id": None, "function": None}]})]
    for ident, name, arguments in zip_longest(id_parts, name_parts, argument_parts):
        fragment: dict[str, Any] = {"index": 0}
        if ident is not None:
            fragment["id"] = ident
        function = {key: value for key, value in (("name", name), ("arguments", arguments)) if value is not None}
        if function:
            fragment["function"] = function
        chunks.append(chunk({"tool_calls": [fragment]}))
    chunks.append(chunk(finish="tool_calls"))
    model, client = model_for(lambda request: httpx.Response(200, content=sse(*chunks)))
    try:
        events = await collect(model)
        starts = [
            event["contentBlockStart"]["start"]["toolUse"]
            for event in events
            if "toolUse" in event.get("contentBlockStart", {}).get("start", {})
        ]
        assert starts == [{"name": "".join(name_parts), "toolUseId": "".join(id_parts)}]
        inputs = [
            event["contentBlockDelta"]["delta"]["toolUse"]["input"]
            for event in events
            if "toolUse" in event.get("contentBlockDelta", {}).get("delta", {})
        ]
        assert json.loads("".join(inputs)) == {"query": "value"}
    finally:
        await client.aclose()


@pytest.mark.parametrize("index", [False, True])
async def test_boolean_tool_call_indexes_are_rejected(index: bool) -> None:
    raw = sse(
        chunk({"tool_calls": [{"index": index, "id": "call_1", "function": {"name": "tool", "arguments": "{}"}}]}),
        chunk(finish="tool_calls"),
    )
    model, client = model_for(lambda request: httpx.Response(200, content=raw))
    observed: list[dict[str, Any]] = []
    try:
        with pytest.raises(ExaoneError, match="valid index"):
            async for event in model.stream([{"role": "user", "content": [{"text": "hello"}]}]):
                observed.append(event)
        assert not any(event.get("contentBlockStart", {}).get("start", {}).get("toolUse") for event in observed)
    finally:
        await client.aclose()


@pytest.mark.parametrize("index", [False, True])
async def test_boolean_response_choice_indexes_are_rejected(index: bool) -> None:
    raw = sse({"choices": [{"index": index, "delta": {"content": "hello"}, "finish_reason": "stop"}]})
    model, client = model_for(lambda request: httpx.Response(200, content=raw))
    try:
        with pytest.raises(ExaoneError, match="choice index"):
            await collect(model)
    finally:
        await client.aclose()


async def test_optional_tool_delta_fields_accept_null_missing_and_empty_strings() -> None:
    raw = sse(
        chunk({"tool_calls": [{"index": 0, "id": None, "function": None}]}),
        chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "tool", "arguments": None}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "", "function": {"name": "", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"name": None, "arguments": "{}"}}]}),
        chunk(finish="tool_calls"),
    )
    model, client = model_for(lambda request: httpx.Response(200, content=raw))
    try:
        events = await collect(model)
        starts = [
            event["contentBlockStart"]["start"]["toolUse"]
            for event in events
            if "toolUse" in event.get("contentBlockStart", {}).get("start", {})
        ]
        assert starts == [{"name": "tool", "toolUseId": "call_1"}]
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "tool_calls",
    [
        False,
        {},
        "invalid",
        [None],
        [[]],
        [{"index": 0, "id": False}],
        [{"index": 0, "id": 0}],
        [{"index": 0, "id": []}],
        [{"index": 0, "function": False}],
        [{"index": 0, "function": 0}],
        [{"index": 0, "function": []}],
        [{"index": 0, "function": "invalid"}],
        [{"index": 0, "function": {"name": False}}],
        [{"index": 0, "function": {"name": []}}],
        [{"index": 0, "function": {"arguments": False}}],
        [{"index": 0, "function": {"arguments": 0}}],
        [{"index": 0, "function": {"arguments": {}}}],
    ],
)
async def test_invalid_tool_delta_types_fail_before_tool_events(tool_calls: Any) -> None:
    raw = sse(chunk({"tool_calls": tool_calls}), chunk(finish="tool_calls"))
    metadata: list[dict[str, Any]] = []
    model, client = model_for(lambda request: httpx.Response(200, content=raw), metadata_callback=metadata.append)
    observed: list[dict[str, Any]] = []
    try:
        with pytest.raises(ExaoneError, match="Invalid EXAONE tool-call delta"):
            async for event in model.stream([{"role": "user", "content": [{"text": "hello"}]}]):
                observed.append(event)
        assert not any(event.get("contentBlockStart", {}).get("start", {}).get("toolUse") for event in observed)
        assert metadata == []
    finally:
        await client.aclose()


@pytest.mark.parametrize("tool_calls", [False, {}, "invalid", [None], [[]]])
async def test_invalid_nonstream_tool_types_fail_before_tool_events(tool_calls: Any) -> None:
    response = {"choices": [{"index": 0, "message": {"tool_calls": tool_calls}, "finish_reason": "tool_calls"}]}
    model, client = model_for(
        lambda request: httpx.Response(200, json=response), stream=False, metadata_callback=lambda metadata: None
    )
    observed: list[dict[str, Any]] = []
    try:
        with pytest.raises(ExaoneError, match="Invalid EXAONE tool-call delta"):
            async for event in model.stream([{"role": "user", "content": [{"text": "hello"}]}]):
                observed.append(event)
        assert not any(event.get("contentBlockStart", {}).get("start", {}).get("toolUse") for event in observed)
    finally:
        await client.aclose()


async def test_nonstream_null_tool_calls_are_accepted() -> None:
    response = {"choices": [{"index": 0, "message": {"content": "ok", "tool_calls": None}, "finish_reason": "stop"}]}
    model, client = model_for(lambda request: httpx.Response(200, json=response), stream=False)
    try:
        events = await collect(model)
        assert any(event.get("messageStop", {}).get("stopReason") == "end_turn" for event in events)
    finally:
        await client.aclose()


@pytest.mark.parametrize("case", ["sse", "tool_arguments"])
async def test_malformed_stream_errors_do_not_retain_raw_content(case: str) -> None:
    if case == "sse":
        payload = b'data: {"sensitive_output":\n\n'
        message = "Malformed JSON in EXAONE event stream"
    else:
        payload = sse(
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "a", "function": {"name": "tool", "arguments": '{"sensitive_output":'}}
                    ]
                }
            ),
            chunk(finish="tool_calls"),
        )
        message = "Invalid tool arguments in EXAONE response"
    model, client = model_for(lambda request: httpx.Response(200, content=payload))
    try:
        with pytest.raises(ExaoneError, match=message) as error:
            await collect(model)
        assert "sensitive_output" not in str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__context__ is None
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


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, {"include_usage": True}),
        ({"include_usage": False}, {"include_usage": False}),
        (
            {"params": {"stream_options": {"include_usage": False, "custom": "retained"}}},
            {"include_usage": False, "custom": "retained"},
        ),
        (
            {"include_usage": True, "params": {"stream_options": {"include_usage": False, "custom": "retained"}}},
            {"include_usage": True, "custom": "retained"},
        ),
    ],
)
async def test_stream_usage_default_and_overrides(config: dict[str, Any], expected: dict[str, Any]) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(chunk(finish="stop")))

    model, client = model_for(handler, **config)
    try:
        await collect(model)
        assert requests[-1]["stream_options"] == expected
        model.update_config(include_usage=False)
        await collect(model)
        assert requests[-1]["stream_options"]["include_usage"] is False
        model.update_config(include_usage=None)
        await collect(model)
        assert requests[-1]["stream_options"]["include_usage"] == config.get("params", {}).get(
            "stream_options", {}
        ).get("include_usage", True)
    finally:
        await client.aclose()


async def test_nonstream_omits_stream_options() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    model, client = model_for(
        handler, stream=False, include_usage=True, params={"stream_options": {"include_usage": False}}
    )
    try:
        await collect(model)
        assert "stream_options" not in requests[-1]
    finally:
        await client.aclose()


def test_stream_options_must_be_a_mapping_or_none() -> None:
    with pytest.raises(ValueError, match="stream_options must be a dictionary or None"):
        ExaoneModel(api_key="test-only", params={"stream_options": []})
    model = ExaoneModel(api_key="test-only", params={"stream_options": None})
    with pytest.raises(ValueError, match="stream_options must be a dictionary or None"):
        model.update_config(params={"stream_options": "invalid"})
    assert model.get_config()["params"] == {"stream_options": None}


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        (None, [{"role": "system", "content": "legacy"}]),
        ([], []),
        (
            [{"text": "first"}, {"text": "second"}],
            [{"role": "system", "content": "first"}, {"role": "system", "content": "second"}],
        ),
    ],
)
async def test_system_block_argument_is_authoritative(blocks: Any, expected: list[dict[str, str]]) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(chunk(finish="stop")))

    model, client = model_for(handler)
    try:
        await collect(model, system_prompt="legacy", system_prompt_content=blocks)
    finally:
        await client.aclose()
    assert [message for message in requests[0]["messages"] if message["role"] == "system"] == expected
    assert "tool_choice" not in requests[0]


async def test_unsupported_tool_choice_fails_before_http_request() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        raise AssertionError("No request should be sent")

    model, client = model_for(handler)
    try:
        with pytest.raises(ValueError, match="Unsupported tool choice"):
            await collect(model, tool_choice={"unknown": {}})
    finally:
        await client.aclose()
    assert request_count == 0


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
@pytest.mark.parametrize("invalid", [True, -1, 2**53, 10**400])
async def test_invalid_usage_is_omitted_from_sdk_metrics_but_preserved_in_raw_metadata(
    field: str, invalid: Any
) -> None:
    usage: dict[str, Any] = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    usage[field] = invalid
    records: list[dict[str, Any]] = []
    payload = sse(chunk({"content": "ok"}), chunk(finish="stop"), {"choices": [], "usage": usage})
    model, client = model_for(
        lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload),
        metadata_callback=records.append,
    )
    try:
        received = await collect(model)
    finally:
        await client.aclose()
    assert any(event.get("contentBlockDelta", {}).get("delta", {}).get("text") == "ok" for event in received)
    assert all("metadata" not in event for event in received)
    assert records[0]["usage"] == usage


@pytest.mark.parametrize("value", [0, 2**53 - 1])
async def test_valid_usage_is_forwarded_to_sdk_metrics(value: int) -> None:
    usage = {"prompt_tokens": value, "completion_tokens": value, "total_tokens": value}
    payload = sse(chunk(finish="stop"), {"choices": [], "usage": usage})
    model, client = model_for(
        lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)
    )
    try:
        received = await collect(model)
    finally:
        await client.aclose()
    assert received[-1]["metadata"]["usage"] == {"inputTokens": value, "outputTokens": value, "totalTokens": value}
