"""Integration tests using the Strands harness and a mock EXAONE transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
from harness_support import chunk, events, response
from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneModel


async def test_harness_background_tool_round_trip_with_reasoning() -> None:
    requests: list[dict[str, Any]] = []
    calls: list[str] = []

    @tool
    def get_number() -> str:
        """Return the synthetic number."""
        calls.append("called")
        return "BACKGROUND_NUMBER_7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "background_1",
                        "function": {"name": "get_number", "arguments": '{"_background_execution":true}'},
                    }
                ]
            }
            return response(
                events(chunk({"reasoning_content": "run in background"}), chunk(call), chunk(finish="tool_calls"))
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[get_number],
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Run the number task in the background.")
    assert calls == ["called"]
    assert any(
        message.get("reasoning_content") == "run in background"
        for request in requests[1:]
        for message in request["messages"]
    )
    assert any(
        "BACKGROUND_NUMBER_7" in str(message.get("content", ""))
        for request in requests[1:]
        for message in request["messages"]
        if message["role"] == "tool"
    )


async def test_harness_background_tool_failure_is_delivered() -> None:
    requests: list[dict[str, Any]] = []
    calls: list[bool] = []

    @tool
    def fail_task() -> str:
        """Raise a synthetic failure."""
        calls.append(True)
        raise ValueError("SYNTHETIC_BACKGROUND_FAILURE")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "background_1",
                        "function": {"name": "fail_task", "arguments": '{"_background_execution":true}'},
                    }
                ]
            }
            return response(events(chunk({"reasoning_content": "run task"}), chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "handled"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[fail_task],
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Run the failing task.")
    assert calls == [True]
    assert any(
        message.get("reasoning_content") == "run task" for request in requests[1:] for message in request["messages"]
    )
    assert any(
        "SYNTHETIC_BACKGROUND_FAILURE" in str(message.get("content", ""))
        for request in requests[1:]
        for message in request["messages"]
        if message["role"] == "tool"
    )


async def test_harness_cancels_running_background_tool() -> None:
    import asyncio
    import re
    import threading

    started = threading.Event()
    finished = threading.Event()
    requests: list[dict[str, Any]] = []

    @tool
    async def wait_task() -> str:
        """Wait until the synthetic task is cancelled."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
        return "unexpected completion"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "background_1",
                        "function": {"name": "wait_task", "arguments": '{"_background_execution":true}'},
                    }
                ]
            }
            return response(
                events(chunk({"reasoning_content": "start background"}), chunk(call), chunk(finish="tool_calls"))
            )
        if len(requests) == 2:
            assert started.wait(2)
            result = next(
                message for message in requests[-1]["messages"] if message.get("tool_call_id") == "background_1"
            )
            match = re.search(r"Task ID: ([^\s]+)", result["content"])
            assert match is not None
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "cancel_1",
                        "function": {
                            "name": "strands_manage_background_task",
                            "arguments": json.dumps({"mode": "cancel", "task_id": match.group(1)}),
                        },
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "cancelled"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[wait_task],
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Start and cancel the task.")
    assert finished.wait(2)
    assert any(
        message.get("tool_call_id") == "cancel_1" and "cancel" in message["content"].lower()
        for request in requests[2:]
        for message in request["messages"]
    )
    assert any(
        message.get("reasoning_content") == "start background"
        for request in requests[1:]
        for message in request["messages"]
    )
