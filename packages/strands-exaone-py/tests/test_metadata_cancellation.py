"""Cancellation at the metadata callback boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from strands import tool
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands_harness import create_harness

from strands_exaone import ExaoneError, ExaoneModel

MESSAGE = [{"role": "user", "content": [{"text": "test"}]}]


class Answer(BaseModel):
    value: int


class ResponseBody(httpx.AsyncByteStream):
    def __init__(self, payload: bytes, on_end: threading.Event | None = None) -> None:
        self.payload = payload
        self.on_end = on_end
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.payload
        if self.on_end is not None:
            self.on_end.set()

    async def aclose(self) -> None:
        self.closed = True


def payload(method: str) -> bytes:
    if method == "structured":
        return json.dumps(
            {"choices": [{"index": 0, "message": {"content": '{"value": 7}'}, "finish_reason": "stop"}]}
        ).encode()
    choices = [
        {"choices": [{"index": 0, "delta": {"content": "test answer"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    ]
    return ("".join(f"data: {json.dumps(event)}\n\n" for event in choices) + "data: [DONE]\n\n").encode()


@pytest.mark.parametrize("method", ["stream", "structured"])
@pytest.mark.parametrize("cancel", ["signal", "task"])
async def test_cancellation_during_async_metadata_callback(method: str, cancel: str) -> None:
    entered = asyncio.Event()
    finalized = asyncio.Event()
    signal = threading.Event()
    observed: list[dict[str, Any]] = []

    async def callback(_: dict[str, Any]) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    body = ResponseBody(payload(method))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        model = ExaoneModel(api_key="test-key", http_client=client, metadata_callback=callback)

        async def collect() -> None:
            generator = (
                model.stream(MESSAGE, cancel_signal=signal)
                if method == "stream"
                else model.structured_output(Answer, MESSAGE, cancel_signal=signal)
            )
            async for event in generator:
                observed.append(event)

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            if cancel == "signal":
                signal.set()
            else:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    assert finalized.is_set()
    assert body.closed
    assert not any("output" in event or "metadata" in event for event in observed)
    if method == "stream":
        assert any("messageStop" in event for event in observed)


@pytest.mark.parametrize("method", ["stream", "structured"])
async def test_cancelled_before_metadata_does_not_start_callback(method: str) -> None:
    signal = threading.Event()
    called: list[bool] = []
    body = ResponseBody(payload(method), on_end=signal if method == "structured" else None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        model = ExaoneModel(api_key="test-key", http_client=client, metadata_callback=lambda _: called.append(True))
        with pytest.raises(asyncio.CancelledError):
            if method == "structured":
                async for _ in model.structured_output(Answer, MESSAGE, cancel_signal=signal):
                    pass
            else:
                async for event in model.stream(MESSAGE, cancel_signal=signal):
                    if "messageStop" in event:
                        signal.set()
    assert called == []
    assert body.closed


@pytest.mark.parametrize("method", ["stream", "structured"])
async def test_cancelled_before_completion_without_metadata_callback(method: str) -> None:
    signal = threading.Event()
    observed: list[dict[str, Any]] = []
    body = ResponseBody(payload(method), on_end=signal if method == "structured" else None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        model = ExaoneModel(api_key="test-key", http_client=client)
        with pytest.raises(asyncio.CancelledError):
            generator = (
                model.stream(MESSAGE, cancel_signal=signal)
                if method == "stream"
                else model.structured_output(Answer, MESSAGE, cancel_signal=signal)
            )
            async for event in generator:
                observed.append(event)
                if method == "stream" and "messageStop" in event:
                    signal.set()
    assert body.closed
    assert not any("output" in event or "metadata" in event for event in observed)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("with_callback", [False, True])
async def test_cancel_after_tool_start_stops_buffered_events(stream: bool, with_callback: bool) -> None:
    signal = threading.Event()
    callbacks: list[bool] = []
    observed: list[dict[str, Any]] = []
    call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    if stream:
        chunk = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, **call}]}}]}
        finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        raw = (f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n").encode()
    else:
        raw = json.dumps(
            {
                "choices": [
                    {"index": 0, "message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}
                ]
            }
        ).encode()
    body = ResponseBody(raw)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        model = ExaoneModel(
            api_key="test-key",
            http_client=client,
            stream=stream,
            metadata_callback=(lambda _: callbacks.append(True)) if with_callback else None,
        )
        with pytest.raises(asyncio.CancelledError):
            async for event in model.stream(MESSAGE, cancel_signal=signal):
                observed.append(event)
                if "toolUse" in event.get("contentBlockStart", {}).get("start", {}):
                    signal.set()
    assert body.closed
    assert signal.is_set()
    assert callbacks == []
    assert not any("contentBlockDelta" in event or "messageStop" in event or "metadata" in event for event in observed)


@pytest.mark.parametrize("with_callback", [False, True])
async def test_harness_cancel_after_tool_start_does_not_execute_tool(with_callback: bool) -> None:
    signal = threading.Event()
    calls: list[bool] = []
    callbacks: list[bool] = []
    requests: list[bool] = []

    @tool
    def lookup() -> str:
        """Return a test value."""
        calls.append(True)
        return "7"

    call = {"index": 0, "id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    chunk = {"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    body = ResponseBody((f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n").encode())

    def handler(_: httpx.Request) -> httpx.Response:
        requests.append(True)
        return httpx.Response(200, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(
                api_key="test-key",
                http_client=client,
                metadata_callback=(lambda _: callbacks.append(True)) if with_callback else None,
            ),
            tools=[lookup],
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            context_manager=False,
            callback_handler=None,
            retry_strategy=None,
        )
        try:
            async for event in agent.stream_async("test", cancel_signal=signal):
                start = event.get("event", {}).get("contentBlockStart", {}).get("start", {})
                if "toolUse" in start:
                    signal.set()
        except asyncio.CancelledError:
            pass
    assert signal.is_set()
    assert calls == []
    # The SDK forwards the caller's signal asynchronously; callback timing can vary.
    assert len(callbacks) <= int(with_callback)
    assert requests == [True]
    assert body.closed


@pytest.mark.parametrize("method", ["stream", "structured"])
async def test_cancel_signal_during_sync_metadata_callback(method: str) -> None:
    signal = threading.Event()
    entered = threading.Event()
    observed: list[dict[str, Any]] = []

    def callback(_: dict[str, Any]) -> None:
        entered.set()
        assert signal.wait(timeout=2), "Cancellation was not signalled"

    def cancel() -> None:
        entered.wait(timeout=2)
        signal.set()

    thread = threading.Thread(target=cancel)
    body = ResponseBody(payload(method))
    try:
        thread.start()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
        ) as client:
            model = ExaoneModel(api_key="test-key", http_client=client, metadata_callback=callback)
            with pytest.raises(asyncio.CancelledError):
                generator = (
                    model.stream(MESSAGE, cancel_signal=signal)
                    if method == "stream"
                    else model.structured_output(Answer, MESSAGE, cancel_signal=signal)
                )
                async for event in generator:
                    observed.append(event)
    finally:
        thread.join(timeout=2)
    assert entered.is_set() and signal.is_set()
    assert body.closed
    assert not any("output" in event or "metadata" in event for event in observed)


@pytest.mark.parametrize("method", ["stream", "structured"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_metadata_callback_exception_propagates(method: str, asynchronous: bool) -> None:
    def callback(_: dict[str, Any]) -> None:
        raise ValueError("callback failed")

    async def async_callback(_: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        raise ValueError("callback failed")

    body = ResponseBody(payload(method))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        model = ExaoneModel(
            api_key="test-key", http_client=client, metadata_callback=async_callback if asynchronous else callback
        )
        with pytest.raises(ValueError, match="callback failed"):
            generator = (
                model.stream(MESSAGE, cancel_signal=threading.Event())
                if method == "stream"
                else model.structured_output(Answer, MESSAGE, cancel_signal=threading.Event())
            )
            async for _ in generator:
                pass
    assert body.closed


@pytest.mark.parametrize("method", ["stream", "structured"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error_type", [ModelThrottledException, ContextWindowOverflowException])
async def test_metadata_callback_sdk_control_errors_are_not_model_errors(
    method: str, asynchronous: bool, error_type: type[Exception]
) -> None:
    original = error_type("synthetic callback failure")

    def callback(_: dict[str, Any]) -> None:
        raise original

    async def async_callback(_: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        raise original

    body = ResponseBody(payload(method))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        model = ExaoneModel(
            api_key="test-key", http_client=client, metadata_callback=async_callback if asynchronous else callback
        )
        with pytest.raises(ExaoneError) as caught:
            generator = (
                model.stream(MESSAGE, cancel_signal=threading.Event())
                if method == "stream"
                else model.structured_output(Answer, MESSAGE, cancel_signal=threading.Event())
            )
            async for _ in generator:
                pass
    assert caught.value.__cause__ is original
    assert body.closed


@pytest.mark.parametrize("method", ["stream", "structured"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cancel_signal", [False, True])
@pytest.mark.parametrize(
    "error_type",
    [ValueError, httpx.ReadError, ModelThrottledException, ContextWindowOverflowException, asyncio.CancelledError],
)
async def test_callback_error_with_and_without_cancellation(
    method: str, asynchronous: bool, cancel_signal: bool, error_type: type[BaseException]
) -> None:
    signal = threading.Event()
    original = error_type("synthetic callback failure")

    def callback(_: dict[str, Any]) -> None:
        if cancel_signal:
            signal.set()
        raise original

    async def async_callback(metadata: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        callback(metadata)

    body = ResponseBody(payload(method))
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    previous_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unhandled.append(context))
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
        ) as client:
            model = ExaoneModel(
                api_key="test-key", http_client=client, metadata_callback=async_callback if asynchronous else callback
            )
            if cancel_signal or error_type is asyncio.CancelledError:
                expected: type[BaseException] = asyncio.CancelledError
            elif error_type in (ModelThrottledException, ContextWindowOverflowException):
                expected = ExaoneError
            else:
                expected = error_type
            with pytest.raises(expected) as caught:
                generator = (
                    model.stream(MESSAGE, cancel_signal=signal)
                    if method == "stream"
                    else model.structured_output(Answer, MESSAGE, cancel_signal=signal)
                )
                async for _ in generator:
                    pass
    finally:
        loop.set_exception_handler(previous_exception_handler)
    assert body.closed
    assert unhandled == []
    if not cancel_signal and error_type in (ModelThrottledException, ContextWindowOverflowException):
        assert caught.value.__cause__ is original
    if error_type is asyncio.CancelledError and not asynchronous:
        assert caught.value is original


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error_type", [ModelThrottledException, ContextWindowOverflowException])
async def test_callback_sdk_control_errors_do_not_retry_or_compact(
    asynchronous: bool, error_type: type[Exception]
) -> None:
    original = error_type("synthetic callback failure")
    requests: list[dict[str, Any]] = []
    bodies: list[ResponseBody] = []
    callbacks: list[bool] = []
    visible: list[str] = []
    history = [
        {"role": "user", "content": [{"text": f"question {index}"}]}
        if index % 2 == 0
        else {"role": "assistant", "content": [{"text": f"answer {index}"}]}
        for index in range(8)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = ResponseBody(payload("stream"))
        bodies.append(body)
        return httpx.Response(200, stream=body)

    def callback(_: dict[str, Any]) -> None:
        callbacks.append(True)
        if len(callbacks) == 1:
            raise original

    async def async_callback(metadata: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        callback(metadata)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(
                api_key="test-key", http_client=client, metadata_callback=async_callback if asynchronous else callback
            ),
            messages=deepcopy(history),
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
        )
        agent._retry_strategy._calculate_delay = lambda _: 0
        with pytest.raises(ExaoneError) as caught:
            async for event in agent.stream_async("latest turn"):
                if "data" in event:
                    visible.append(event["data"])
    assert caught.value.__cause__ is original
    assert len(requests) == len(callbacks) == 1
    assert visible == ["test answer"]
    assert all(body.closed for body in bodies)
    after = [{"role": message["role"], "content": message["content"]} for message in agent.messages]
    assert after == history + [{"role": "user", "content": [{"text": "latest turn"}]}]


@pytest.mark.parametrize("method", ["stream", "structured"])
async def test_sync_metadata_callback_completes_without_cancellation(method: str) -> None:
    received: list[dict[str, Any]] = []
    body = ResponseBody(payload(method))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        model = ExaoneModel(api_key="test-key", http_client=client, metadata_callback=received.append)
        generator = (
            model.stream(MESSAGE, cancel_signal=threading.Event())
            if method == "stream"
            else model.structured_output(Answer, MESSAGE, cancel_signal=threading.Event())
        )
        observed = [event async for event in generator]
    assert len(received) == 1
    assert body.closed
    if method == "stream":
        assert any("messageStop" in event for event in observed)
    else:
        assert observed == [{"output": Answer(value=7)}]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_harness_does_not_execute_tool_when_metadata_is_cancelled(asynchronous: bool) -> None:
    signal = threading.Event()
    entered = asyncio.Event()
    sync_entered = threading.Event()
    finalized = asyncio.Event()
    calls: list[bool] = []
    observed: list[dict[str, Any]] = []

    async def callback(_: dict[str, Any]) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    def sync_callback(_: dict[str, Any]) -> None:
        sync_entered.set()
        signal.wait(timeout=2)
        finalized.set()

    def cancel_sync_callback() -> None:
        if sync_entered.wait(timeout=2):
            signal.set()

    @tool
    def lookup() -> str:
        """Return a test value."""
        calls.append(True)
        return "7"

    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    body = ResponseBody((f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(finish)}\n\ndata: [DONE]\n\n").encode())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))) as client:
        agent = create_harness(
            model=ExaoneModel(
                api_key="test-key", http_client=client, metadata_callback=callback if asynchronous else sync_callback
            ),
            tools=[lookup],
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            context_manager=False,
            callback_handler=None,
            retry_strategy=None,
        )

        async def collect() -> None:
            async for event in agent.stream_async("test", cancel_signal=signal):
                observed.append(event)

        task = asyncio.create_task(collect())
        thread = threading.Thread(target=cancel_sync_callback) if not asynchronous else None
        if thread is not None:
            thread.start()
        try:
            if asynchronous:
                await asyncio.wait_for(entered.wait(), timeout=2)
                signal.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=1)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if thread is not None:
                thread.join(timeout=2)
    assert finalized.is_set()
    assert body.closed
    assert calls == []
    if not asynchronous:
        assert any(event.get("result") and event["result"].stop_reason == "cancelled" for event in observed)
