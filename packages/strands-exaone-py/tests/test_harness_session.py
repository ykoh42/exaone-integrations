"""Integration tests using the Strands harness and a mock EXAONE transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from harness_support import chunk, events, response, tool_delta
from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneModel


async def test_harness_session_resume_preserves_synthetic_tool_history(tmp_path: Any) -> None:
    calls: list[str] = []
    requests: list[dict[str, Any]] = []

    @tool
    def get_number() -> str:
        """Return a fixed test number."""
        calls.append("called")
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(
                events(
                    chunk({"reasoning_content": "before tool"}),
                    chunk(tool_delta("call_1", "get_number")),
                    chunk(finish="tool_calls"),
                )
            )
        if len(requests) == 2:
            return response(
                events(chunk({"reasoning_content": "after tool"}), chunk({"content": "7"}), chunk(finish="stop"))
            )
        if len(requests) == 3:
            return response(events(chunk({"content": "still 7"}), chunk(finish="stop")))
        raise AssertionError("Unexpected extra EXAONE request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        options: dict[str, Any] = {
            "model": model,
            "tools": [get_number],
            "builtin_tools": [],
            "builtin_plugins": [],
            "session": {"id": "synthetic-resume", "dir": str(tmp_path)},
            "memory": False,
            "skills": False,
            "caching": False,
            "callback_handler": None,
            "retry_strategy": None,
        }
        first = create_harness(**options)
        await first.invoke_async("Get a number.")
        resumed = create_harness(**options)
        await resumed.invoke_async("What was the number?")

    assert calls == ["called"]
    assert len(requests) == 3
    messages = requests[-1]["messages"]
    assert [message["reasoning_content"] for message in messages if message.get("reasoning_content")] == [
        "before tool",
        "after tool",
    ]
    assert [message["tool_call_id"] for message in messages if message["role"] == "tool"] == ["call_1"]
    assert messages[-1]["content"] == "What was the number?"


async def test_harness_skill_discovery_and_activation(tmp_path: Any) -> None:
    skill_dir = tmp_path / "example-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: example-skill\ndescription: Use to format a synthetic answer.\n---\n\n"
        "Return the marker SKILL_SYNTHETIC.\n"
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "skill_1",
                        "function": {"name": "skills", "arguments": json.dumps({"skill_name": "example-skill"})},
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "SKILL_SYNTHETIC"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=str(skill_dir),
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Use the formatting skill.")
    assert len(requests) == 2
    assert "example-skill" in str(requests[0]["messages"])
    result = next(message for message in requests[-1]["messages"] if message.get("tool_call_id") == "skill_1")
    assert "SKILL_SYNTHETIC" in result["content"]


async def test_harness_file_memory_search_and_injection(tmp_path: Any) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "memory_1",
                        "function": {
                            "name": "search_memory",
                            "arguments": json.dumps({"query": "favorite synthetic color"}),
                        },
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        if len(requests) == 2:
            return response(events(chunk({"content": "blue"}), chunk(finish="stop")))
        return response(events(chunk({"content": "[]"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory={"dir": str(tmp_path)},
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        assert agent.memory_manager is not None
        await agent.memory_manager.add("The favorite synthetic color is blue.")
        await agent.invoke_async("What is the favorite synthetic color?")
        await agent.memory_manager.flush()
    assert len(requests) == 3  # The third request is the harness's mock memory extraction on flush.
    assert "blue" in str(requests[0]["messages"])
    result = next(message for message in requests[1]["messages"] if message.get("tool_call_id") == "memory_1")
    assert "favorite synthetic color" in result["content"]
    assert "blue" in result["content"]


async def test_harness_session_resume_after_partial_stream_failure(tmp_path: Any) -> None:
    from strands.types.exceptions import EventLoopException

    calls: list[bool] = []
    requests: list[dict[str, Any]] = []

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield events(chunk({"reasoning_content": "incomplete"}), done=False)
            raise httpx.ReadError("synthetic interruption")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(
                events(
                    chunk({"reasoning_content": "before tool"}),
                    chunk(tool_delta("call_1", "get_number")),
                    chunk(finish="tool_calls"),
                )
            )
        if len(requests) == 2:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Interrupted())
        return response(events(chunk({"content": "7"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        options: dict[str, Any] = {
            "model": ExaoneModel(api_key="test-only-key", http_client=client),
            "tools": [get_number],
            "builtin_tools": [],
            "builtin_plugins": [],
            "session": {"id": "interrupted", "dir": str(tmp_path)},
            "memory": False,
            "skills": False,
            "caching": False,
            "callback_handler": None,
            "retry_strategy": None,
        }
        with pytest.raises(EventLoopException):
            await create_harness(**options).invoke_async("Get the number.")
        resumed = create_harness(**options)
        await resumed.invoke_async("Continue the previous task.")
    assert calls == [True]
    assert len(requests) == 3
    history = requests[-1]["messages"]
    assert any(message.get("reasoning_content") == "before tool" for message in history)
    assert any(message.get("tool_call_id") == "call_1" for message in history)
    assert all(message.get("reasoning_content") != "incomplete" for message in history)
