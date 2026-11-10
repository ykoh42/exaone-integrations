"""Integration tests using the Strands harness and a mock EXAONE transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from strands import Agent, tool
from strands_harness import create_harness

from strands_exaone import ExaoneError, ExaoneModel


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


async def test_harness_preserves_interleaved_thinking_and_whitespace_across_user_turns() -> None:
    calls: list[str] = []
    requests: list[dict[str, Any]] = []
    original_text = "\n\nMeaningful text  \n"

    @tool
    def first_number() -> str:
        """Return the first fixed test number."""
        calls.append("first")
        return "3"

    @tool
    def second_number() -> str:
        """Return the second fixed test number."""
        calls.append("second")
        return "4"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        match len(requests):
            case 1:
                return response(
                    events(
                        chunk({"reasoning_content": "reason "}),
                        chunk({"reasoning_content": "one"}),
                        chunk({"content": "\n"}),
                        chunk({"content": "\nMeaningful text  \n"}),
                        chunk(tool_delta("call_1", "first_number")),
                        chunk(finish="tool_calls"),
                    )
                )
            case 2:
                return response(
                    events(
                        chunk({"reasoning_content": "reason two"}),
                        chunk(tool_delta("call_2", "second_number")),
                        chunk(finish="tool_calls"),
                    )
                )
            case 3:
                return response(
                    events(
                        chunk({"reasoning_content": "reason final"}),
                        chunk({"content": "The sum is 7."}),
                        chunk(finish="stop"),
                    )
                )
            case 4:
                return response(events(chunk({"content": "The sum was 7."}), chunk(finish="stop")))
            case _:
                raise AssertionError("Unexpected extra EXAONE request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = make_harness(model, [first_number, second_number])
        assert agent.context_manager is not None
        await agent.invoke_async("Use both number tools.")
        await agent.invoke_async("What was the sum?")

    assert calls == ["first", "second"]
    assert len(requests) == 4
    assert all(request["temperature"] == 1.0 and request["top_p"] == 0.95 for request in requests)
    assert all(request["chat_template_kwargs"]["preserve_thinking"] is True for request in requests)
    first_assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
    assert first_assistant["reasoning_content"] == "reason one"
    assert first_assistant["content"] == original_text
    assert any(message.get("tool_call_id") == "call_1" for message in requests[1]["messages"])
    later_messages = requests[3]["messages"]
    assistant_reasons = [
        message.get("reasoning_content") for message in later_messages if message["role"] == "assistant"
    ]
    assert assistant_reasons == ["reason one", "reason two", "reason final"]
    assert [message["tool_call_id"] for message in later_messages if message["role"] == "tool"] == ["call_1", "call_2"]
    assert later_messages[-1]["role"] == "user"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, (1.0, 0.95, True)),
        ({"temperature": 0.0, "top_p": 0.0, "preserve_thinking": False}, (0.0, 0.0, False)),
        (
            {"params": {"temperature": 0.0, "top_p": 0.2, "chat_template_kwargs": {"preserve_thinking": False}}},
            (0.0, 0.2, False),
        ),
    ],
)
async def test_harness_respects_provider_defaults_and_user_overrides(
    config: dict[str, Any], expected: tuple[float, float, bool]
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response(events(chunk({"content": "ok"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client, **config)
        agent = make_harness(model)
        assert agent.context_manager is not None
        await agent.invoke_async("Say hello.")
    assert len(requests) == 1
    assert (
        requests[0]["temperature"],
        requests[0]["top_p"],
        requests[0]["chat_template_kwargs"]["preserve_thinking"],
    ) == expected


@pytest.mark.parametrize("failure", ["truncated", "invalid_arguments", "stream_error"])
async def test_harness_does_not_execute_incomplete_tool_calls(failure: str) -> None:
    calls: list[str] = []
    request_count = 0

    @tool
    def unsafe_if_executed() -> str:
        """Return a fixed value."""
        calls.append("called")
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        arguments = "{" if failure == "invalid_arguments" else "{}"
        call = {
            "tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "unsafe_if_executed", "arguments": arguments}}
            ]
        }
        if failure == "stream_error":
            return response(events(chunk(call), {"error": {"message": "synthetic failure"}}))
        return response(events(chunk(call), chunk(finish="tool_calls"), done=failure != "truncated"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = make_harness(model, [unsafe_if_executed])
        assert agent.context_manager is not None
        with pytest.raises(ExaoneError):
            await agent.invoke_async("Use the tool.")
    assert calls == []
    assert request_count == 1


async def test_harness_builtin_tools_with_synthetic_responses(tmp_path: Any) -> None:
    """Exercise local built-ins through the real harness with mock model output."""
    import shlex

    path = tmp_path / "sample.txt"
    requests: list[dict[str, Any]] = []
    steps = [
        ("write", {"path": str(path), "content": "alpha\nneedle original\n"}),
        ("read", {"path": str(path)}),
        ("edit", {"path": str(path), "old_str": "needle original", "new_str": "needle revised"}),
        ("shell", {"command": f"grep -n needle {shlex.quote(str(path))}"}),
        ("todo_write", {"todos": [{"content": "Check file", "activeForm": "Checking file", "status": "in_progress"}]}),
        ("programmatic_tool_caller", {"code": f"result = await read(path={str(path)!r})\nprint(result)"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        index = len(requests) - 1
        if index == len(steps):
            return response(events(chunk({"content": "done"}), chunk(finish="stop")))
        name, arguments = steps[index]
        call = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{index}",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ]
        }
        return response(events(chunk({"reasoning_content": f"step {index}"}), chunk(call), chunk(finish="tool_calls")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = create_harness(
            model=model,
            builtin_tools=["read", "write", "edit", "shell", "programmatic_tool_caller"],
            builtin_plugins=["todos"],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Update and inspect the local file.")

    assert path.read_text() == "alpha\nneedle revised\n"
    assert len(requests) == len(steps) + 1
    messages = requests[-1]["messages"]
    results = {message["tool_call_id"]: message["content"] for message in messages if message["role"] == "tool"}
    assert set(results) == {f"call_{index}" for index in range(len(steps))}
    assert "needle original" in str(results["call_1"])
    assert "needle revised" in str(results["call_3"])
    assert "needle revised" in str(results["call_5"])
    assert [message["reasoning_content"] for message in messages if message.get("reasoning_content")] == [
        f"step {index}" for index in range(len(steps))
    ]


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


async def test_harness_parallel_fragmented_tool_calls() -> None:
    calls: list[str] = []
    requests: list[dict[str, Any]] = []

    @tool
    def first_number() -> str:
        """Return one."""
        calls.append("first")
        return "1"

    @tool
    def second_number() -> str:
        """Return two."""
        calls.append("second")
        return "2"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(
                events(
                    chunk({"reasoning_content": "both tools"}),
                    chunk(
                        {
                            "tool_calls": [
                                {"index": 0, "id": "call_", "function": {"name": "first_", "arguments": "{"}},
                                {"index": 1, "id": "call_", "function": {"name": "second_", "arguments": "{"}},
                            ]
                        }
                    ),
                    chunk(
                        {
                            "tool_calls": [
                                {"index": 1, "id": "2", "function": {"name": "number", "arguments": "}"}},
                                {"index": 0, "id": "1", "function": {"name": "number", "arguments": "}"}},
                            ]
                        }
                    ),
                    chunk(finish="tool_calls"),
                )
            )
        if len(requests) == 2:
            return response(events(chunk({"content": "3"}), chunk(finish="stop")))
        raise AssertionError("Unexpected extra EXAONE request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        await make_harness(model, [first_number, second_number]).invoke_async("Get both numbers.")

    assert set(calls) == {"first", "second"}
    assert len(calls) == 2
    messages = requests[-1]["messages"]
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "both tools"
    assert {call["id"]: call["function"]["name"] for call in assistant["tool_calls"]} == {
        "call_1": "first_number",
        "call_2": "second_number",
    }
    assert {message["tool_call_id"] for message in messages if message["role"] == "tool"} == {"call_1", "call_2"}


@pytest.mark.parametrize("failure", ["cancel", "read_error"])
async def test_harness_does_not_retry_or_run_tools_after_partial_response(failure: str) -> None:
    import asyncio

    calls: list[str] = []
    request_count = 0
    started = asyncio.Event()
    closed = False

    @tool
    def unsafe_if_executed() -> str:
        """Return a fixed value."""
        calls.append("called")
        return "7"

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield events(
                chunk({"reasoning_content": "partial"}), chunk(tool_delta("call_1", "unsafe_if_executed")), done=False
            )
            started.set()
            if failure == "read_error":
                raise httpx.ReadError("synthetic stream interruption")
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=InterruptedStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client, max_retries=2, base_retry_delay=0)
        task = asyncio.create_task(make_harness(model, [unsafe_if_executed]).invoke_async("Use the tool."))
        await asyncio.wait_for(started.wait(), timeout=2)
        if failure == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(httpx.ReadError):
                await task
    assert calls == []
    assert request_count == 1
    assert closed


async def test_harness_read_image_as_text_with_media_disabled(tmp_path: Any) -> None:
    """The text-only model can receive the harness's description of a file."""
    import base64

    path = tmp_path / "pixel.png"
    path.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6n8AAAAASUVORK5CYII=")
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "read_1",
                        "function": {"name": "read", "arguments": json.dumps({"path": str(path)})},
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "I cannot inspect the image."}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools={"*": False, "read": {"media": False}},
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Read the image.")
    assert len(requests) == 2
    result = next(message for message in requests[-1]["messages"] if message.get("tool_call_id") == "read_1")
    assert isinstance(result["content"], str)
    assert result["content"]


async def test_harness_receives_synthetic_stream_usage() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response(
            events(
                chunk({"content": "ok"}),
                chunk(finish="stop"),
                {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 4, "total_tokens": 127}},
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client))
        result = await agent.invoke_async("Hello.")
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert result.context_size == 123
    assert result.metrics.accumulated_usage == {"inputTokens": 123, "outputTokens": 4, "totalTokens": 127}


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


@pytest.mark.parametrize("approved", [True, False])
async def test_harness_tool_intervention_preserves_reasoning(tmp_path: Any, approved: bool) -> None:
    import shlex

    from strands.vended_interventions.hitl import HumanInTheLoop

    marker = tmp_path / "marker"
    decisions: list[bool] = []
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": f"touch {shlex.quote(str(marker))}"}),
                        },
                    }
                ]
            }
            return response(events(chunk({"reasoning_content": "synthetic"}), chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    def ask(prompt: str, **kwargs: Any) -> str:
        decisions.append(approved)
        return "yes" if approved else "no"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["shell"],
            builtin_plugins=[],
            interventions=HumanInTheLoop(ask=ask),
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Run the local test.")
    assert decisions == [approved]
    assert marker.exists() is approved
    assert len(requests) == 2
    messages = requests[-1]["messages"]
    assert any(message.get("reasoning_content") == "synthetic" for message in messages)
    assert any(message.get("tool_call_id") == "call_1" for message in messages)


async def test_harness_subagent_uses_same_mock_provider() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "delegate_1",
                        "function": {
                            "name": "subagent",
                            "arguments": json.dumps({"task": "Return CHILD_SYNTHETIC.", "tools": []}),
                        },
                    }
                ]
            }
            return response(
                events(chunk({"reasoning_content": "delegate task"}), chunk(call), chunk(finish="tool_calls"))
            )
        if any(
            "CHILD_SYNTHETIC" in str(message.get("content", ""))
            for message in requests[-1]["messages"]
            if message["role"] == "user"
        ):
            return response(
                events(
                    chunk({"reasoning_content": "CHILD_REASONING"}),
                    chunk({"content": "CHILD_SYNTHETIC"}),
                    chunk(finish="stop"),
                )
            )
        return response(events(chunk({"content": "Parent finished."}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["subagent"],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Delegate the test task.")
    assert len(requests) >= 3
    assert any("CHILD_SYNTHETIC" in str(request["messages"]) for request in requests[1:])
    assert any(
        "CHILD_SYNTHETIC" in str(message.get("content", ""))
        for request in requests[1:]
        for message in request["messages"]
        if message.get("role") == "tool"
    )
    assert all(
        "CHILD_REASONING" not in str(message.get("content", ""))
        for request in requests[1:]
        for message in request["messages"]
        if message.get("role") == "tool"
    )
    assert any(
        message.get("reasoning_content") == "delegate task"
        for request in requests[1:]
        for message in request["messages"]
    )


async def test_harness_web_fetch_with_synthetic_page_and_summarizer(monkeypatch: Any) -> None:
    import importlib

    fetch_module = importlib.import_module("strands_harness.tools.web_fetch")
    requests: list[dict[str, Any]] = []

    async def mock_fetch(url: str) -> tuple[str, str]:
        assert url == "https://example.com/synthetic"
        return url, "The SYNTHETIC_PAGE value is blue."

    monkeypatch.setattr(fetch_module, "_fetch_direct", mock_fetch)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "fetch_1",
                        "function": {
                            "name": "web_fetch",
                            "arguments": json.dumps(
                                {"url": "https://example.com/synthetic", "prompt": "What is the value?"}
                            ),
                        },
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        if len(requests) == 2:
            return response(
                events(
                    chunk({"reasoning_content": "SUMMARIZER_REASONING"}),
                    chunk({"content": "The value is blue."}),
                    chunk(finish="stop"),
                )
            )
        return response(events(chunk({"content": "blue"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools={"*": False, "web_fetch": {"transport": "direct"}},
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Check the synthetic page.")
    assert len(requests) == 3
    assert "SYNTHETIC_PAGE" in str(requests[1]["messages"])
    result = next(message for message in requests[2]["messages"] if message.get("tool_call_id") == "fetch_1")
    assert "The value is blue." in result["content"]
    assert "SYNTHETIC_PAGE" not in result["content"]
    assert "SUMMARIZER_REASONING" not in result["content"]


async def test_harness_default_environment_and_tools_with_mock_model(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Follow the synthetic ENVIRONMENT_MARKER instruction.\n")
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = "ok" if len(requests) == 1 else "[]"
        return response(events(chunk({"content": content}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(model=ExaoneModel(api_key="test-only-key", http_client=client), callback_handler=None)
        await agent.invoke_async("Say hello.")
        assert agent.memory_manager is not None
        await agent.memory_manager.flush()
    assert "ENVIRONMENT_MARKER" in str(requests[0]["messages"])
    names = {spec["function"]["name"] for spec in requests[0]["tools"]}
    assert {
        "shell",
        "read",
        "write",
        "edit",
        "web_fetch",
        "programmatic_tool_caller",
        "subagent",
        "todo_write",
    } <= names
    assert "web_search" not in names


async def test_harness_local_mcp_tool_round_trip(tmp_path: Any) -> None:
    import sys

    server = tmp_path / "mock_mcp_server.py"
    server.write_text(
        "from mcp.server.mcpserver import MCPServer\n"
        "server = MCPServer('test')\n"
        "@server.tool()\n"
        "def add_numbers(a: int, b: int) -> str:\n"
        "    return str(a + b)\n"
        "if __name__ == '__main__':\n"
        "    server.run('stdio')\n"
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            names = {spec["function"]["name"] for spec in requests[0]["tools"]}
            assert "local_add_numbers" in names
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "local_add_numbers", "arguments": '{"a":3,"b":4}'},
                    }
                ]
            }
            return response(events(chunk({"reasoning_content": "synthetic"}), chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            mcp_servers={"local": {"command": sys.executable, "args": [str(server)]}},
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Add two test values.")
    assert len(requests) == 2
    messages = requests[1]["messages"]
    assert any(message.get("reasoning_content") == "synthetic" for message in messages)
    assert any(message.get("tool_call_id") == "call_1" and "7" in message["content"] for message in messages)


async def test_harness_nonstream_tool_and_reasoning_round_trip() -> None:
    requests: list[dict[str, Any]] = []
    calls: list[bool] = []

    @tool
    def get_number() -> str:
        """Return the synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            message = {
                "reasoning_content": "before tool",
                "content": "\n\nChecking.  \n",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_number", "arguments": "{}"}}
                ],
            }
            return httpx.Response(
                200, json={"choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]}
            )
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "7"}, "finish_reason": "stop"}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client, stream=False), [get_number])
        await agent.invoke_async("Get a number.")
    assert calls == [True]
    assert len(requests) == 2
    assert all(request["stream"] is False and "stream_options" not in request for request in requests)
    assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "before tool"
    assert assistant["content"] == "\n\nChecking.  \n"
    assert any(message.get("tool_call_id") == "call_1" for message in requests[1]["messages"])


async def test_harness_structured_output_after_tool_and_reasoning() -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    requests: list[dict[str, Any]] = []
    calls: list[bool] = []

    @tool
    def get_number() -> str:
        """Return the synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(
                events(
                    chunk({"reasoning_content": "use number"}),
                    chunk(tool_delta("number_1", "get_number")),
                    chunk(finish="tool_calls"),
                )
            )
        if len(requests) == 2:
            call = {
                "tool_calls": [
                    {"index": 0, "id": "output_1", "function": {"name": "Answer", "arguments": '{"value":7}'}}
                ]
            }
            return response(
                events(chunk({"reasoning_content": "format result"}), chunk(call), chunk(finish="tool_calls"))
            )
        return response(events(chunk({"content": "7"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_number])
        result = await agent.invoke_async("Get the number in structured output.", structured_output_model=Answer)
        assert result.structured_output == Answer(value=7)
        await agent.invoke_async("What was the number?")
    assert calls == [True]
    assert len(requests) == 3
    assert any(message.get("reasoning_content") == "use number" for message in requests[1]["messages"])
    assert any(message.get("tool_call_id") == "number_1" for message in requests[1]["messages"])
    assert any(message.get("reasoning_content") == "format result" for message in requests[2]["messages"])


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


async def test_harness_logs_redact_partial_transport_error_after_tool(caplog: Any) -> None:
    import traceback

    from strands.types.exceptions import EventLoopException

    secret = "synthetic-secret-key"
    calls: list[bool] = []
    count = 0

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield events(chunk({"content": "partial"}), done=False)
            try:
                raise ValueError(f"nested {secret}")
            except ValueError as cause:
                raise httpx.ReadError(f"interrupted {secret}") from cause

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            return response(events(chunk(tool_delta("call_1", "get_number")), chunk(finish="tool_calls")))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Interrupted())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key=secret, http_client=client, max_retries=2), [get_number])
        with pytest.raises(EventLoopException) as caught:
            await agent.invoke_async("Use the tool.")
    assert count == 2
    assert calls == [True]
    assert secret not in caplog.text
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert isinstance(caught.value.__cause__, httpx.ReadError)
    assert caught.value.__cause__.__cause__ is None
    assert caught.value.__cause__.__context__ is None


async def test_harness_agentic_context_tool_and_synthetic_usage() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "pin_1",
                        "function": {"name": "pin_context", "arguments": '{"select":"last_turn"}'},
                    }
                ]
            }
            return response(
                events(
                    chunk({"reasoning_content": "pin this turn"}),
                    chunk(call),
                    chunk(finish="tool_calls"),
                    {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 4, "total_tokens": 127}},
                )
            )
        return response(events(chunk({"content": "Pinned."}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            context_manager="agentic",
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Pin this synthetic turn.")
    assert len(requests) == 2
    names = {spec["function"]["name"] for spec in requests[0]["tools"]}
    assert {"pin_context", "summarize_context", "truncate_context"} <= names
    assert any(message.get("reasoning_content") == "pin this turn" for message in requests[1]["messages"])
    assert any(
        message.get("tool_call_id") == "pin_1" and "Pinned" in message["content"] for message in requests[1]["messages"]
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


async def test_harness_structured_output_retries_invalid_tool_value() -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        arguments = '{"value":"not-an-integer"}' if len(requests) == 1 else '{"value":7}'
        call = {
            "tool_calls": [
                {"index": 0, "id": f"output_{len(requests)}", "function": {"name": "Answer", "arguments": arguments}}
            ]
        }
        return response(
            events(chunk({"reasoning_content": f"attempt {len(requests)}"}), chunk(call), chunk(finish="tool_calls"))
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client))
        result = await agent.invoke_async(
            "Return a number.", structured_output_model=Answer, limits={"structured_output_attempts": 2}
        )
    assert result.structured_output == Answer(value=7)
    assert len(requests) == 2
    assert any(message.get("reasoning_content") == "attempt 1" for message in requests[1]["messages"])
    assert any(
        message.get("tool_call_id") == "output_1" and "value" in message["content"]
        for message in requests[1]["messages"]
    )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ({"auto": {}}, "auto"),
        ({"any": {}}, "required"),
        ({"tool": {"name": "get_number"}}, {"type": "function", "function": {"name": "get_number"}}),
    ],
)
async def test_harness_tool_choice_and_system_text_blocks(choice: dict[str, Any], expected: Any) -> None:
    from strands.middleware import InvokeModelStage

    requests: list[dict[str, Any]] = []

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response(events(chunk({"content": "synthetic"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[get_number],
            system_prompt=[{"text": "First.\n"}, {"text": "Second."}],
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        agent.add_middleware(InvokeModelStage.Input, lambda context: context.replace(tool_choice=choice))
        await agent.invoke_async("Use a number.")
    assert len(requests) == 1
    assert requests[0]["tool_choice"] == expected
    assert any(spec["function"]["name"] == "get_number" for spec in requests[0]["tools"])
    assert [message for message in requests[0]["messages"] if message["role"] == "system"] == [
        {"role": "system", "content": "First.\n"},
        {"role": "system", "content": "Second."},
    ]


@pytest.mark.parametrize(
    "block",
    [{"cachePoint": {"type": "default"}}, {"image": {"format": "png", "source": {"bytes": b"synthetic"}}}],
)
async def test_harness_rejects_nontext_system_blocks_before_request(block: dict[str, Any]) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        raise AssertionError("No request should be sent")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            system_prompt=[{"text": "Synthetic instruction."}, block],
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        with pytest.raises(ValueError, match="text-only system prompt blocks"):
            await agent.invoke_async("Hello.")
    assert request_count == 0


async def test_harness_continues_after_synthetic_extreme_usage() -> None:
    requests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    huge = 10**400

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response(
            events(
                chunk({"content": "ok"}),
                chunk(finish="stop"),
                {"choices": [], "usage": {"prompt_tokens": huge, "completion_tokens": 1, "total_tokens": huge}},
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client, metadata_callback=records.append))
        first = await agent.invoke_async("First turn.")
        second = await agent.invoke_async("Second turn.")
    assert len(requests) == 2
    assert first.metrics.accumulated_usage["inputTokens"] == 0
    assert second.metrics.accumulated_usage["inputTokens"] == 0
    assert all(record["usage"]["prompt_tokens"] == huge for record in records)
