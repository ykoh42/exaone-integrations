"""Metadata and opt-in retries at the public model transport boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from strands_exaone import ExaoneError, ExaoneModel, ExaoneThrottledError

MESSAGE = [{"role": "user", "content": [{"text": "test"}]}]


def sse(*payloads: dict[str, Any]) -> bytes:
    return ("".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads) + "data: [DONE]\n\n").encode()


def success() -> bytes:
    return sse({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]})


async def events(model: ExaoneModel, **kwargs: Any) -> list[dict[str, Any]]:
    return [event async for event in model.stream(MESSAGE, **kwargs)]


async def test_retry_429_honors_longer_header_and_success() -> None:
    times: list[float] = []
    closed = []

    def handler(request: httpx.Request) -> httpx.Response:
        times.append(asyncio.get_running_loop().time())
        if len(times) == 1:
            response = httpx.Response(429, headers={"retry-after": "0.01", "retry-after-ms": "40"}, json={})
            closed.append(response)
            return response
        return httpx.Response(200, content=success())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=1, max_retry_delay=1)
        result = await events(model)
    assert len(times) == 2
    assert times[1] - times[0] >= 0.03
    assert closed[0].is_closed
    assert any("messageStop" in event for event in result)


@pytest.mark.parametrize("status", [400, 401, 404, 422])
async def test_non_retryable_http_errors_are_not_repeated(status: int) -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(status, json={"error": {"message": "bad request"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=3, base_retry_delay=0)
        with pytest.raises(ExaoneError):
            await events(model)
    assert count == 1


async def test_retry_limit_and_server_delay_ceiling_preserve_error() -> None:
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(429, headers={"retry-after": "10"}, json={"error": {"message": "busy"}})
        responses.append(response)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=2, max_retry_delay=1)
        with pytest.raises(ExaoneThrottledError) as exc:
            await events(model)
    assert len(responses) == 1
    assert exc.value.headers["retry-after"] == "10"
    assert responses[0].is_closed


async def test_http_date_retry_after_over_ceiling_preserves_error() -> None:
    count = 0
    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(503, headers={"retry-after": date}, json={"error": {"message": "busy"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=1, max_retry_delay=1)
        with pytest.raises(ExaoneError) as exc:
            await events(model)
    assert count == 1
    assert exc.value.headers["retry-after"] == date


@pytest.mark.parametrize("header", ["retry-after", "retry-after-ms"])
async def test_overflowing_retry_after_does_not_retry(header: str) -> None:
    count = 0
    delay = "9" * 400

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(429, headers={header: delay}, json={"error": {"message": "busy"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=1, max_retry_delay=30)
        with pytest.raises(ExaoneThrottledError) as exc:
            await events(model)
    assert count == 1
    assert exc.value.headers[header] == delay


async def test_connection_failure_and_exhausted_server_errors() -> None:
    count = 0

    def connect_then_fail(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(connect_then_fail)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=2, base_retry_delay=0)
        with pytest.raises(ExaoneError) as exc:
            await events(model)
    assert count == 3
    assert exc.value.status_code == 503


async def test_no_retry_after_stream_has_started() -> None:
    count = 0

    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("connection interrupted")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, stream=Interrupted())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=3, base_retry_delay=0)
        observed = []
        with pytest.raises(httpx.ReadError):
            async for event in model.stream(MESSAGE):
                observed.append(event)
    assert count == 1
    assert any(event.get("contentBlockDelta", {}).get("delta", {}).get("text") == "partial" for event in observed)


async def test_cancel_during_retry_wait() -> None:
    count = 0
    issued = asyncio.Event()
    signal = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        issued.set()
        return httpx.Response(429, headers={"retry-after": "5"}, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=3, max_retry_delay=10)
        task = asyncio.create_task(events(model, cancel_signal=signal))
        await issued.wait()
        await asyncio.sleep(0.01)
        signal.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    assert count == 1


async def test_stream_metadata_merges_logprobs_and_redacts_key() -> None:
    records = []
    chunks = [
        {
            "id": "id-test-secret",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "model",
            "metadata": {"test-secret": "test-secret"},
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "private reasoning"},
                    "finish_reason": None,
                    "logprobs": {"content": [{"token": "a"}]},
                }
            ],
        },
        {
            "id": "id-test-secret",
            "system_fingerprint": "fp",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "private answer"},
                    "logprobs": {"content": [{"token": "test-secret"}], "refusal": None},
                    "matched_stop": None,
                    "finish_reason": "stop",
                }
            ],
        },
        {"choices": [], "service_tier": None, "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=sse(*chunks)))
    ) as client:
        model = ExaoneModel(api_key="test-secret", http_client=client, metadata_callback=records.append)
        await events(model)
    response = records[0]["response"]
    assert response["id"] == "id-[REDACTED]"
    assert response["metadata"] == {"[REDACTED]": "[REDACTED]"}
    assert response["system_fingerprint"] == "fp"
    assert response["service_tier"] is None
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["matched_stop"] is None
    assert response["choices"][0]["logprobs"] == {"content": [{"token": "a"}, {"token": "[REDACTED]"}], "refusal": None}
    assert "private answer" not in str(response) and "private reasoning" not in str(response)


async def test_nonstream_and_structured_output_metadata() -> None:
    records = []

    class Answer(BaseModel):
        count: int

    payload = {
        "id": "one",
        "metadata": {"weight_version": "v1"},
        "moderation": None,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "matched_stop": "end",
                "logprobs": None,
                "message": {"content": '{"count":3}', "reasoning_content": "secret"},
            }
        ],
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        model = ExaoneModel(api_key="key", http_client=client, metadata_callback=records.append, stream=False)
        await events(model)
        result = [event async for event in model.structured_output(Answer, MESSAGE)]
    assert result == [{"output": Answer(count=3)}]
    assert len(records) == 2
    for record in records:
        assert record["response"]["metadata"] == {"weight_version": "v1"}
        assert record["response"]["choices"] == [
            {"index": 0, "finish_reason": "stop", "matched_stop": "end", "logprobs": None}
        ]
        assert "secret" not in str(record["response"])


async def test_message_metadata_keeps_stream_fragments_and_nonstream_fields() -> None:
    records = []
    stream_chunks = [
        {"choices": [{"index": 0, "delta": {"refusal": "can", "annotations": [{"id": 1}], "audio": {"data": "a"}}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "refusal": "not",
                        "annotations": [{"id": 2}],
                        "audio": {"data": "b"},
                        "function_call": {"arguments": "{"},
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    ]
    message = {
        "content": "answer",
        "reasoning_content": "reasoning",
        "tool_calls": [],
        "refusal": None,
        "annotations": [{"url": "https://example.test"}],
        "audio": {"id": "audio-1"},
        "function_call": {"name": "test"},
    }
    payload = {"choices": [{"index": 0, "finish_reason": "stop", "message": message}]}

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["stream"]:
            return httpx.Response(200, content=sse(*stream_chunks))
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, metadata_callback=records.append)
        await events(model)
        model.update_config(stream=False)
        await events(model)
    first = records[0]["response"]["choices"][0]
    assert first["message"] == {"refusal": "cannot", "annotations": [{"id": 1}, {"id": 2}]}
    assert first["message_deltas"] == [
        {"audio": {"data": "a"}},
        {"audio": {"data": "b"}, "function_call": {"arguments": "{"}},
    ]
    second = records[1]["response"]["choices"][0]["message"]
    assert second == {key: message[key] for key in ("refusal", "annotations", "audio", "function_call")}


async def test_structured_output_retries_before_response() -> None:
    count = 0

    class Answer(BaseModel):
        count: int

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"count":3}'}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=1)
        result = [event async for event in model.structured_output(Answer, MESSAGE)]
    assert result == [{"output": Answer(count=3)}]
    assert count == 2


@pytest.mark.parametrize("stream", [True, False])
async def test_unrecognized_finish_reason_is_exposed_as_error(stream: bool) -> None:
    chunk = {"choices": [{"index": 0, "delta": {"content": "{"}, "finish_reason": "abort"}]}
    response = {"choices": [{"index": 0, "message": {"content": "{"}, "finish_reason": "abort"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(chunk)) if stream else httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-secret", http_client=client, stream=stream)
        with pytest.raises(ExaoneError, match="abort") as exc:
            await events(model)
    assert exc.value.finish_reason == "abort"


async def test_structured_abort_preserves_reason_and_masks_key() -> None:
    class Answer(BaseModel):
        count: int

    finish_reason = "abort test-secret"
    response = {"choices": [{"message": {"content": "{"}, "finish_reason": finish_reason}]}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    ) as client:
        model = ExaoneModel(api_key="test-secret", http_client=client)
        with pytest.raises(ExaoneError, match="abort") as exc:
            _ = [event async for event in model.structured_output(Answer, MESSAGE)]
    assert exc.value.finish_reason == "abort [REDACTED]"
    assert "test-secret" not in str(exc.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"max_retries": True},
        {"max_retries": 1.5},
        {"base_retry_delay": -1},
        {"base_retry_delay": None},
        {"max_retry_delay": float("inf")},
    ],
)
def test_invalid_retry_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExaoneModel(api_key="key", **kwargs)


def test_missing_key_error_names_configuration_options() -> None:
    with pytest.raises(ValueError, match=r"EXAONE_API_KEY.*api_key"):
        ExaoneModel(api_key="")
