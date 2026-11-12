"""Metadata and opt-in retries at the public model transport boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from strands_exaone import ExaoneError, ExaoneModel, ExaoneThrottledError
from strands_exaone import model as model_module
from strands_exaone import retry as retry_module

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


@pytest.mark.parametrize(
    ("retry_after", "retry_after_ms", "delay"),
    [("10", "9598", 10.0), ("16", "15659", 16.0)],
)
async def test_captured_exaone_retry_hints_honor_the_longer_value(
    retry_after: str, retry_after_ms: str, delay: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(model_module.asyncio, "sleep", record_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={
                    "retry-after": retry_after,
                    "retry-after-ms": retry_after_ms,
                    "x-second-rate-limit": "1",
                    "x-second-rate-limit-remaining": "1",
                    "x-minute-rate-limit": "60",
                    "x-minute-rate-limit-remaining": "60",
                    "x-day-rate-limit": "86400",
                    "x-day-rate-limit-remaining": "86086",
                },
                json={},
            )
        return httpx.Response(200, content=success())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=1)
        await events(model)
    assert attempts == 2
    assert delays == [delay]


async def test_captured_429_without_retry_hints_uses_bounded_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    delays: list[float] = []
    random_bounds: list[tuple[float, float]] = []

    def midpoint(low: float, high: float) -> float:
        random_bounds.append((low, high))
        return (low + high) / 2

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(retry_module.random, "uniform", midpoint)
    monkeypatch.setattr(model_module.asyncio, "sleep", record_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={
                    "x-second-rate-limit": "1",
                    "x-second-rate-limit-remaining": "1",
                    "x-minute-rate-limit": "60",
                    "x-minute-rate-limit-remaining": "60",
                    "x-day-rate-limit": "86400",
                    "x-day-rate-limit-remaining": "86086",
                },
                json={},
            )
        return httpx.Response(200, content=success())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="key", http_client=client, max_retries=1, base_retry_delay=1, max_retry_delay=30)
        await events(model)
    assert attempts == 2
    assert random_bounds == [(1, 1)]
    assert delays == [1]


def test_retry_without_hints_stops_when_cap_is_below_minimum() -> None:
    assert retry_module.retry_delay({}, attempt=0, base=1, maximum=0.5) is None


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


@pytest.mark.parametrize(
    ("status", "body", "error_type", "code", "param", "message"),
    [
        (
            400,
            {"error": {"type": "invalid_request_error", "code": None, "param": "top_k", "message": "invalid value"}},
            "invalid_request_error",
            None,
            "top_k",
            "invalid value",
        ),
        (
            400,
            {"result_code": "INVALID_INPUT", "description": "invalid request"},
            None,
            "INVALID_INPUT",
            None,
            "invalid request",
        ),
        (
            422,
            {"result_code": "INVALID_INPUT", "description": "invalid request"},
            None,
            "INVALID_INPUT",
            None,
            "invalid request",
        ),
    ],
)
async def test_http_errors_expose_normalized_diagnostics(
    status: int, body: dict[str, Any], error_type: str | None, code: Any, param: Any, message: str
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, json=body, headers={"x-request-id": "request-1"})
        )
    ) as client:
        model = ExaoneModel(api_key="key", http_client=client)
        with pytest.raises(ExaoneError) as caught:
            await events(model)
    error = caught.value
    assert str(error) == message
    assert error.status_code == status
    assert error.error_type == error_type
    assert error.code == code
    assert error.param == param
    assert error.request_id == "request-1"
    assert error.body == body


async def test_success_status_error_envelope_uses_generic_message_without_http_status() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=sse({"error": {"code": "bad_response"}})))
    ) as client:
        model = ExaoneModel(api_key="key", http_client=client)
        with pytest.raises(ExaoneError, match="^EXAONE returned an error in the response body$"):
            await events(model)


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


async def test_latency_metric_excludes_metadata_callback_time(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]

    async def on_metadata(_metadata: dict[str, Any]) -> None:
        clock[0] += 12.0

    monkeypatch.setattr(model_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=sse(
                    {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
                ),
            )
        )
    ) as client:
        model = ExaoneModel(api_key="key", http_client=client, metadata_callback=on_metadata)
        result = await events(model)

    metrics = next(
        event["metadata"]["metrics"] for event in result if "metadata" in event and "metrics" in event["metadata"]
    )
    assert metrics["latencyMs"] == 0


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


@pytest.mark.parametrize("failure", ["connect", "read"])
async def test_transport_errors_do_not_expose_credentials(failure: str) -> None:
    import traceback

    secret = "synthetic-secret-key"
    count = 0

    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            try:
                raise ValueError(f"nested {secret}")
            except ValueError as cause:
                raise httpx.ReadError(f"interrupted {secret}") from cause

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if failure == "connect":
            try:
                raise ValueError(f"nested {secret}")
            except ValueError as cause:
                raise httpx.ConnectError(f"connection {secret}", request=request) from cause
        return httpx.Response(200, stream=Interrupted())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key=secret, http_client=client, max_retries=1, base_retry_delay=0)
        error_type = httpx.ConnectError if failure == "connect" else httpx.ReadError
        with pytest.raises(error_type) as caught:
            await events(model)
    assert count == (2 if failure == "connect" else 1)
    error = caught.value
    assert secret not in str(error)
    assert secret not in "".join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert error.__context__ is None
    with pytest.raises(RuntimeError, match="request property has not been set"):
        _ = error.request


async def test_structured_output_transport_error_does_not_retain_original() -> None:
    import traceback

    from pydantic import BaseModel

    class Output(BaseModel):
        value: int

    secret = "synthetic-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            raise ValueError(f"nested {secret}")
        except ValueError as cause:
            raise httpx.ConnectError(f"connection {secret}", request=request) from cause

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key=secret, http_client=client)
        with pytest.raises(httpx.ConnectError) as caught:
            _ = [event async for event in model.structured_output(Output, MESSAGE)]
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    with pytest.raises(RuntimeError, match="request property has not been set"):
        _ = caught.value.request


async def test_closing_model_stream_closes_http_response() -> None:
    class OpenStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

        async def aclose(self) -> None:
            self.closed = True

    stream = OpenStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        generator = model.stream(MESSAGE)
        await anext(generator)
        await generator.aclose()
        assert stream.closed


@pytest.mark.parametrize("failure", ["connect", "read"])
async def test_custom_transport_error_uses_safe_standard_base_type(failure: str) -> None:
    import traceback

    secret = "synthetic-secret-key"
    count = 0

    class CustomConnectError(httpx.ConnectError):
        def __init__(self, message: str, required: str, *, request: httpx.Request) -> None:
            super().__init__(message, request=request)
            self.required = required

        def __str__(self) -> str:
            raise RuntimeError(f"custom formatter {secret}")

    class CustomReadError(httpx.ReadError):
        def __init__(self, message: str, required: str) -> None:
            super().__init__(message)
            self.required = required

    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            try:
                raise ValueError(f"nested {secret}")
            except ValueError as cause:
                raise CustomReadError(f"interrupted {secret}", secret) from cause

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if failure == "connect":
            try:
                raise ValueError(f"nested {secret}")
            except ValueError as cause:
                raise CustomConnectError(f"connection {secret}", secret, request=request) from cause
        return httpx.Response(200, stream=Interrupted())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key=secret, http_client=client, max_retries=1, base_retry_delay=0)
        with pytest.raises(httpx.RequestError) as caught:
            await events(model)
    error = caught.value
    assert type(error) is (httpx.ConnectError if failure == "connect" else httpx.ReadError)
    assert count == (2 if failure == "connect" else 1)
    assert secret not in str(error)
    assert secret not in "".join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert error.__context__ is None
    with pytest.raises(RuntimeError, match="request property has not been set"):
        _ = error.request


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("invalid_url", httpx.InvalidURL),
        ("cookie", httpx.CookieConflict),
        ("stream", httpx.StreamError),
        ("consumed", httpx.StreamConsumed),
        ("status", httpx.HTTPError),
    ],
)
async def test_other_httpx_errors_are_redacted_without_original_chain(
    kind: str, expected_type: type[Exception]
) -> None:
    import traceback

    secret = "synthetic-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        if kind == "invalid_url":
            error: Exception = httpx.InvalidURL(f"bad url {secret}")
        elif kind == "cookie":
            error = httpx.CookieConflict(f"bad cookie {secret}")
        elif kind == "stream":
            error = httpx.StreamError(f"bad stream {secret}")
        elif kind == "consumed":
            error = httpx.StreamConsumed()
            error.args = (f"bad stream {secret}",)
        else:
            error = httpx.HTTPStatusError(f"bad status {secret}", request=request, response=httpx.Response(400))
        try:
            raise ValueError(f"nested {secret}")
        except ValueError as cause:
            raise error from cause

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key=secret, http_client=client)
        with pytest.raises(expected_type) as caught:
            await events(model)
    assert type(caught.value) is expected_type
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


async def test_http_status_response_hook_retains_only_safe_status() -> None:
    import traceback

    secret = "synthetic-secret-key"
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(422, text=f"private response {secret}")

    async def raise_for_status(response: httpx.Response) -> None:
        try:
            raise ValueError(f"nested {secret}")
        except ValueError as cause:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise error from cause

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), event_hooks={"response": [raise_for_status]}
    ) as client:
        model = ExaoneModel(api_key=secret, base_url=f"https://{secret}.example/v1", http_client=client)
        with pytest.raises(httpx.HTTPError) as caught:
            await events(model)
    error = caught.value
    assert type(error) is httpx.HTTPError
    assert getattr(error, "status_code", None) == 422
    assert request_count == 1
    assert secret not in "".join(traceback.format_exception(error))
    assert secret not in repr(vars(error))
    with pytest.raises(RuntimeError, match="request property has not been set"):
        _ = error.request
    assert not hasattr(error, "response")
    assert error.__cause__ is None
    assert error.__context__ is None


async def test_structured_output_uses_request_config_snapshot_during_retry() -> None:
    import asyncio

    from pydantic import BaseModel

    class Output(BaseModel):
        value: int

    entered = asyncio.Event()
    release = asyncio.Event()
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            entered.set()
            await release.wait()
            raise httpx.ConnectError("synthetic failure", request=request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"value":7}'}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(
            api_key="test-only-key", http_client=client, max_retries=1, base_retry_delay=0, temperature=0.4
        )

        async def collect_structured() -> list[dict[str, Any]]:
            return [event async for event in model.structured_output(Output, MESSAGE)]

        task = asyncio.create_task(collect_structured())
        await asyncio.wait_for(entered.wait(), timeout=2)
        model.update_config(max_retries=0, temperature=0.7)
        release.set()
        output = await task
        later = await collect_structured()
    assert output == [{"output": Output(value=7)}]
    assert later == [{"output": Output(value=7)}]
    assert [request["temperature"] for request in requests] == [0.4, 0.4, 0.7]
