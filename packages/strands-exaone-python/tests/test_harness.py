"""Integration tests using the Strands harness and a mock EXAONE transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from harness_support import chunk, events, make_harness, response, tool_delta
from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneError, ExaoneModel


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
