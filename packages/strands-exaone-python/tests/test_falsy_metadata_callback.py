"""Metadata callbacks are optional callables, including falsey callable objects."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from strands_harness import create_harness

from strands_exaone import ExaoneError, ExaoneModel

MESSAGE = [{"role": "user", "content": [{"text": "test"}]}]


class Answer(BaseModel):
    value: int


class Collector:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def __bool__(self) -> bool:
        return False

    def __call__(self, metadata: dict[str, Any]) -> None:
        self.records.append(metadata)


class AsyncCollector(Collector):
    async def __call__(self, metadata: dict[str, Any]) -> None:  # type: ignore[override]
        self.records.append(metadata)


def response(stream: bool, success: bool) -> httpx.Response:
    finish = "stop" if success else "synthetic_invalid_finish"
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "reasoning_tokens": 1}
    if stream:
        chunks = [
            {"id": "synthetic-id", "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": finish}]},
            {"choices": [], "usage": usage},
        ]
        return httpx.Response(
            200, content=("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
        )
    return httpx.Response(
        200,
        json={
            "id": "synthetic-id",
            "choices": [{"index": 0, "message": {"content": '{"value":7}'}, "finish_reason": finish}],
            "usage": usage,
        },
    )


@pytest.mark.parametrize("method", ["stream", "structured"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("success", [False, True])
async def test_falsey_callback_receives_success_metadata(
    method: str, asynchronous: bool, override: bool, success: bool
) -> None:
    collector = AsyncCollector() if asynchronous else Collector()
    unused: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response(method == "stream", success))
    ) as client:
        model = ExaoneModel(
            api_key="test-key", http_client=client, metadata_callback=unused.append if override else collector
        )
        kwargs = {"metadata_callback": collector} if override else {}
        generator = (
            model.stream(MESSAGE, **kwargs)
            if method == "stream"
            else model.structured_output(Answer, MESSAGE, **kwargs)
        )
        if success:
            _ = [event async for event in generator]
        else:
            with pytest.raises(ExaoneError):
                _ = [event async for event in generator]

    assert unused == []
    assert len(collector.records) == int(success)
    if success:
        assert collector.records[0]["response"]["id"] == "synthetic-id"
        assert collector.records[0]["usage"]["reasoning_tokens"] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("success", [False, True])
async def test_harness_falsey_callback_receives_success_metadata(asynchronous: bool, success: bool) -> None:
    collector = AsyncCollector() if asynchronous else Collector()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response(True, success))) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-key", http_client=client, metadata_callback=collector),
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
        if success:
            await agent.invoke_async("test")
        else:
            with pytest.raises(ExaoneError):
                await agent.invoke_async("test")
    assert len(collector.records) == int(success)
    if success:
        assert collector.records[0]["response"]["id"] == "synthetic-id"


@pytest.mark.parametrize("method", ["stream", "structured"])
async def test_none_override_disables_metadata_callback(method: str) -> None:
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response(method == "stream", True))) as client:
        model = ExaoneModel(api_key="test-key", http_client=client, metadata_callback=records.append)
        generator = (
            model.stream(MESSAGE, metadata_callback=None)
            if method == "stream"
            else model.structured_output(Answer, MESSAGE, metadata_callback=None)
        )
        _ = [event async for event in generator]
    assert records == []
