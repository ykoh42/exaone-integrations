"""Synthetic HTTP fixtures for Strands Harness integration tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from strands import Agent
from strands_harness import create_harness

from strands_exaone import ExaoneModel


class SplitBytes(httpx.AsyncByteStream):
    """Split the event stream independently of JSON, SSE, and UTF-8 boundaries."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for offset in range(0, len(self._data), 7):
            yield self._data[offset : offset + 7]


def chunk(delta: dict[str, Any] | None = None, finish: str | None = None) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def events(*chunks: dict[str, Any], done: bool = True) -> bytes:
    payload = "".join(f"data: {json.dumps(item, ensure_ascii=False)}\n\n" for item in chunks)
    return (payload + ("data: [DONE]\n\n" if done else "")).encode()


def tool_delta(ident: str, name: str) -> dict[str, Any]:
    return {"tool_calls": [{"index": 0, "id": ident, "function": {"name": name, "arguments": "{}"}}]}


def response(data: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SplitBytes(data))


def make_harness(model: ExaoneModel, tools: list[Any] | None = None) -> Agent:
    # Leave context_manager at its actual harness default ("auto").
    return create_harness(
        model=model,
        tools=tools,
        builtin_tools=[],
        builtin_plugins=[],
        session=False,
        memory=False,
        skills=False,
        caching=False,
        callback_handler=None,
        retry_strategy=None,
    )
