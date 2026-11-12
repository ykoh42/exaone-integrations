"""Preserve Strands tool error status in EXAONE tool-message text."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneModel
from strands_exaone.formatting import format_messages


@pytest.mark.parametrize(
    ("status", "content", "expected"),
    [
        ("error", [{"text": "denied"}], "[ERROR] denied"),
        ("error", [{"json": {"reason": "denied"}}], '[ERROR] {"reason":"denied"}'),
        ("error", [], "[ERROR] "),
        ("success", [{"text": "done"}], "done"),
        ("success", [{"json": {"value": 7}}], '{"value":7}'),
        ("success", [], ""),
        (None, [{"text": "legacy"}], "legacy"),
    ],
)
def test_tool_result_status_wire_text(status: str | None, content: list[dict[str, Any]], expected: str) -> None:
    result: dict[str, Any] = {"toolUseId": "call-1", "content": content}
    if status is not None:
        result["status"] = status
    messages = format_messages([{"role": "user", "content": [{"toolResult": result}]}], None, None)
    assert messages == [{"role": "tool", "tool_call_id": "call-1", "content": expected}]


@pytest.mark.parametrize("stream", [False, True])
async def test_harness_preserves_mixed_tool_result_statuses_and_ids(stream: bool) -> None:
    requests: list[dict[str, Any]] = []
    calls: list[str] = []

    @tool
    def failed_text() -> dict[str, Any]:
        """Return a synthetic text failure."""
        calls.append("failed_text")
        return {"status": "error", "content": [{"text": "denied"}]}

    @tool
    def failed_json() -> dict[str, Any]:
        """Return a synthetic JSON failure."""
        calls.append("failed_json")
        return {"status": "error", "content": [{"json": {"reason": "denied"}}]}

    @tool
    def failed_empty() -> dict[str, Any]:
        """Return a synthetic empty failure."""
        calls.append("failed_empty")
        return {"status": "error", "content": []}

    @tool
    def success_text() -> dict[str, Any]:
        """Return a synthetic successful text result."""
        calls.append("success_text")
        return {"status": "success", "content": [{"text": "done"}]}

    @tool
    def success_json() -> dict[str, Any]:
        """Return a synthetic successful JSON result."""
        calls.append("success_json")
        return {"status": "success", "content": [{"json": {"value": 7}}]}

    @tool
    def success_empty() -> dict[str, Any]:
        """Return a synthetic empty successful result."""
        calls.append("success_empty")
        return {"status": "success", "content": []}

    names = ("failed_text", "failed_json", "failed_empty", "success_text", "success_json", "success_empty")
    tool_calls = [
        {"index": index, "id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": "{}"}}
        for index, name in enumerate(names)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            if stream:
                chunk = {"choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": "tool_calls"}]}
                return httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n").encode())
            calls_without_indices = [
                {key: value for key, value in call.items() if key != "index"} for call in tool_calls
            ]
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"index": 0, "message": {"tool_calls": calls_without_indices}, "finish_reason": "tool_calls"}
                    ]
                },
            )
        if stream:
            chunk = {"choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}]}
            return httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n").encode())
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "done"}, "finish_reason": "stop"}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-key", http_client=client, stream=stream),
            tools=[failed_text, failed_json, failed_empty, success_text, success_json, success_empty],
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
        await agent.invoke_async("Run the synthetic tools")

    assert sorted(calls) == sorted(names)
    assert len(requests) == 2
    results = {
        message["tool_call_id"]: message["content"] for message in requests[1]["messages"] if message["role"] == "tool"
    }
    assert results == {
        "call-0": "[ERROR] denied",
        "call-1": '[ERROR] {"reason":"denied"}',
        "call-2": "[ERROR] ",
        "call-3": "done",
        "call-4": '{"value":7}',
        "call-5": "",
    }
