"""Integration tests using the Strands harness and a mock EXAONE transport."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from harness_support import chunk, events, make_harness, response, tool_delta
from strands import Agent, tool
from strands_harness import create_harness

from strands_exaone import ExaoneError, ExaoneModel, ExaoneRetryStrategy, ExaoneThrottledError


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


async def test_exaone_retry_strategy_honors_retry_after_in_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    requests = 0

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "1", "retry-after-ms": "631"},
                json={"error": {"type": "rate_limit_error", "message": "busy"}},
            )
        return response(events(chunk({"content": "ok"}), chunk(finish="stop")))

    monkeypatch.setattr("strands_exaone.retry_strategy.asyncio.sleep", record_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = make_harness(
            model,
            retry_strategy=ExaoneRetryStrategy(max_attempts=2, base_retry_delay=0.01, max_retry_delay=1),
        )
        result = await agent.invoke_async("Say ok.")

    assert result.message["content"]
    assert requests == 2
    assert delays == [1.0]


async def test_exaone_retry_strategy_surfaces_retry_after_above_cap() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            429,
            headers={"retry-after-ms": "100"},
            json={"error": {"type": "rate_limit_error", "message": "busy"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = make_harness(
            model,
            retry_strategy=ExaoneRetryStrategy(max_attempts=2, base_retry_delay=0.01, max_retry_delay=0.05),
        )
        with pytest.raises(ExaoneThrottledError):
            await agent.invoke_async("Say ok.")

    assert requests == 1


async def test_exaone_retry_strategy_bounds_total_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0

    async def skip_sleep(_: float) -> None:
        return None

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            429,
            headers={"retry-after-ms": "0"},
            json={"error": {"type": "rate_limit_error", "message": "busy"}},
        )

    monkeypatch.setattr("strands_exaone.retry_strategy.asyncio.sleep", skip_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = make_harness(
            model,
            retry_strategy=ExaoneRetryStrategy(max_attempts=3, base_retry_delay=0.01, max_retry_delay=1),
        )
        with pytest.raises(ExaoneThrottledError):
            await agent.invoke_async("Say ok.")

    assert requests == 3


@pytest.mark.parametrize("use_exaone_strategy", [False, True])
async def test_retry_strategy_does_not_replay_after_stream_events_are_exposed(use_exaone_strategy: bool) -> None:
    requests = 0

    class Interrupted(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield events(chunk({"content": "partial"}), done=False)
            raise httpx.ConnectError("synthetic connection failure during response body")

        async def aclose(self) -> None:
            self.closed = True

    stream = Interrupted()

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent_options: dict[str, Any] = {}
        if use_exaone_strategy:
            agent_options["retry_strategy"] = ExaoneRetryStrategy(max_attempts=2, base_retry_delay=0, max_retry_delay=0)
        agent = Agent(
            model=ExaoneModel(api_key="test-only-key", http_client=client), callback_handler=None, **agent_options
        )
        observed = []
        with pytest.raises(httpx.ConnectError, match="synthetic connection failure") as caught:
            async for event in agent.stream_async("Say hello."):
                observed.append(event)

    assert requests == 1
    assert observed
    assert any("partial" in str(event) for event in observed)
    assert stream.closed
    assert "test-only-key" not in str(caught.value)
    assert getattr(caught.value, "_exaone_events_exposed", False)


async def test_exaone_retry_strategy_does_not_retry_model_generated_invalid_tool_arguments() -> None:
    requests = 0
    tool_calls: list[str] = []

    @tool
    def lookup(query: str) -> str:
        """Return a fixed lookup result."""
        tool_calls.append(query)
        return "result"

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return response(
            events(
                chunk({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "lookup", "arguments": "{"}}]}),
                chunk(finish="tool_calls"),
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = make_harness(
            model,
            [lookup],
            retry_strategy=ExaoneRetryStrategy(max_attempts=3, base_retry_delay=0, max_retry_delay=0),
        )
        with pytest.raises(ExaoneError, match="Invalid tool arguments"):
            await agent.invoke_async("Use the tool.")

    assert requests == 1
    assert tool_calls == []


async def test_exaone_retry_strategy_does_not_retry_malformed_reasoning_content() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return response(events(chunk({"reasoning_content": {"invalid": "private payload"}})))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = make_harness(
            model,
            retry_strategy=ExaoneRetryStrategy(max_attempts=3, base_retry_delay=0, max_retry_delay=0),
        )
        with pytest.raises(ExaoneError, match="Invalid reasoning_content delta") as caught:
            await agent.invoke_async("Continue thinking.")

    assert requests == 1
    assert "private payload" not in str(caught.value)


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


@pytest.mark.parametrize("arguments", ["", " \t\n"])
async def test_harness_executes_zero_argument_tool_with_empty_input(arguments: str) -> None:
    calls: list[str] = []
    requests: list[dict[str, Any]] = []

    @tool
    def no_arguments() -> str:
        """Return a fixed value without arguments."""
        calls.append("called")
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(
                events(
                    chunk(tool_delta("call_1", "no_arguments", arguments)),
                    chunk(finish="tool_calls"),
                )
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        await make_harness(model, [no_arguments]).invoke_async("Use the tool.")

    assert calls == ["called"]
    assert len(requests) == 2
    assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
    assert assistant["content"] == ""
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {}


async def test_harness_executes_tool_when_only_closing_object_brace_is_missing() -> None:
    calls: list[str] = []
    requests: list[dict[str, Any]] = []

    @tool
    def lookup(query: str) -> str:
        """Return the captured query."""
        calls.append(query)
        return query

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            malformed = '{"query":"EXAONE"'
            return response(
                events(
                    chunk(
                        {
                            "tool_calls": [
                                {"index": 0, "id": "lookup_1", "function": {"name": "lookup", "arguments": malformed}}
                            ]
                        }
                    ),
                    chunk(finish="tool_calls"),
                )
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        await make_harness(model, [lookup]).invoke_async("Look up EXAONE.")

    assert calls == ["EXAONE"]
    assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"query":"EXAONE"}'


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
async def test_harness_ignores_system_blocks_without_text(block: dict[str, Any]) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return response(events(chunk(finish="stop")))

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
        await agent.invoke_async("Hello.")
    assert len(requests) == 1
    assert [message for message in requests[0]["messages"] if message["role"] == "system"] == [
        {"role": "system", "content": "Synthetic instruction."}
    ]


async def test_harness_accepts_initial_split_utf8_bom() -> None:
    """A synthetic UTF-8 BOM can span HTTP byte chunks."""
    calls: list[bool] = []
    requests: list[dict[str, Any]] = []

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    class OneByteStream(httpx.AsyncByteStream):
        def __init__(self, data: bytes) -> None:
            self.data = data

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for byte in self.data:
                yield bytes([byte])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        data = (
            b"\xef\xbb\xbf" + events(chunk(tool_delta("call_1", "get_number")), chunk(finish="tool_calls"))
            if len(requests) == 1
            else events(chunk({"content": "7"}), chunk(finish="stop"))
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=OneByteStream(data))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_number]).invoke_async(
            "Get a number."
        )
    assert calls == [True]
    assert len(requests) == 2
    assert any(message.get("tool_call_id") == "call_1" for message in requests[1]["messages"])


@pytest.mark.parametrize("ending", [b"", b"\n", b"\r", b"\r\n"])
async def test_harness_does_not_execute_tool_when_done_event_is_incomplete(ending: bytes) -> None:
    """A synthetic EOF is not an SSE event delimiter."""
    calls: list[bool] = []
    requests = 0

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests > 1:
            return response(events(chunk({"content": "unexpected execution"}), chunk(finish="stop")))
        data = events(chunk(tool_delta("call_1", "get_number")), chunk(finish="tool_calls"), done=False)
        return response(data + b"data: [DONE]" + ending)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ExaoneError, match=r"without \[DONE\]"):
            await make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_number]).invoke_async(
                "Get a number."
            )
    assert requests == 1
    assert calls == []


@pytest.mark.parametrize("charset", ["utf-8", "iso-8859-1", "utf-16"])
@pytest.mark.parametrize("ending", [b"\n", b"\r", b"\r\n"])
@pytest.mark.parametrize("size", [1, 7])
async def test_harness_sse_utf8_ignores_charset_and_keeps_unicode(charset: str, ending: bytes, size: int) -> None:
    """Exercise synthetic SSE boundaries, including Unicode separators inside text."""
    original = "한글\ufeffmiddle\u2028line\u2029next\u0085end"

    class ByteChunks(httpx.AsyncByteStream):
        def __init__(self, data: bytes) -> None:
            self.data = data

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for index in range(0, len(self.data), size):
                yield self.data[index : index + size]

    def handler(request: httpx.Request) -> httpx.Response:
        data = events(chunk({"content": original}), chunk(finish="stop")).replace(b"\n", ending)
        return httpx.Response(
            200, headers={"content-type": f"text/event-stream; charset={charset}"}, stream=ByteChunks(data)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await make_harness(ExaoneModel(api_key="test-only-key", http_client=client)).invoke_async("Say text.")
    assert "".join(block.get("text", "") for block in result.message["content"]) == original


async def test_harness_many_interleaved_unicode_argument_fragments() -> None:
    """Synthetic tool arguments are reconstructed exactly before execution."""
    from itertools import zip_longest

    values = {"first": "한🙂" * 512, "second": "글🚀" * 384}
    calls: dict[str, str] = {}
    requests: list[dict[str, Any]] = []

    @tool
    def first_tool(value: str) -> str:
        """Capture the first argument."""
        calls["first"] = value
        return "first complete"

    @tool
    def second_tool(value: str) -> str:
        """Capture the second argument."""
        calls["second"] = value
        return "second complete"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) > 1:
            return response(events(chunk({"content": "done"}), chunk(finish="stop")))
        chunks = [
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "first_1", "function": {"name": "first_tool", "arguments": ""}},
                        {"index": 1, "id": "second_1", "function": {"name": "second_tool", "arguments": None}},
                    ]
                }
            )
        ]
        raw_first = json.dumps({"value": values["first"]}, ensure_ascii=False)
        raw_second = json.dumps({"value": values["second"]}, ensure_ascii=False)
        for first, second in zip_longest(raw_first, raw_second):
            parts = [
                {"index": index, "function": {"arguments": value}}
                for index, value in enumerate((first, second))
                if value is not None
            ]
            chunks.append(chunk({"tool_calls": parts}))
        chunks.append(chunk(finish="tool_calls"))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events(*chunks))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [first_tool, second_tool])
        await agent.invoke_async("Use both synthetic tools.")
    assert calls == values
    assert len(requests) == 2
    assert {message["tool_call_id"] for message in requests[1]["messages"] if message["role"] == "tool"} == {
        "first_1",
        "second_1",
    }


@pytest.mark.parametrize("invalid_arguments", ["{", "[]", '{"value":NaN}'])
async def test_harness_rejects_all_tools_when_any_arguments_are_invalid(invalid_arguments: str) -> None:
    calls: list[str] = []

    @tool
    def first_tool(value: int) -> str:
        """Capture the first argument."""
        calls.append("first")
        return str(value)

    @tool
    def second_tool(value: int) -> str:
        """Capture the second argument."""
        calls.append("second")
        return str(value)

    def handler(request: httpx.Request) -> httpx.Response:
        calls_delta = [
            {"index": 0, "id": "first_1", "function": {"name": "first_tool", "arguments": '{"value":1}'}},
            {"index": 1, "id": "second_1", "function": {"name": "second_tool", "arguments": invalid_arguments}},
        ]
        return response(events(chunk({"tool_calls": calls_delta}), chunk(finish="tool_calls")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [first_tool, second_tool])
        with pytest.raises(ExaoneError):
            await agent.invoke_async("Use both synthetic tools.")
    assert calls == []


@pytest.mark.parametrize(("initial_retries", "updated_retries"), [(0, 1), (1, 0)])
@pytest.mark.parametrize("failure", ["connect", "throttle"])
async def test_harness_config_update_does_not_change_inflight_retry_policy(
    initial_retries: int, updated_retries: int, failure: str
) -> None:
    import asyncio

    from strands_exaone import ExaoneThrottledError

    entered = asyncio.Event()
    release = asyncio.Event()
    calls: dict[str, list[tuple[str, float, float]]] = {"first": [], "second": []}
    phase = "first"

    async def handler(request: httpx.Request) -> httpx.Response:
        current = calls[phase]
        current.append(
            (request.url.host, json.loads(request.content)["temperature"], request.extensions["timeout"]["read"])
        )
        if len(current) == 1:
            if phase == "first":
                entered.set()
                await release.wait()
            if failure == "connect":
                raise httpx.ConnectError("synthetic connection failure", request=request)
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "synthetic"}})
        return response(events(chunk({"content": "ok"}), chunk(finish="stop")))

    error_type = httpx.ConnectError if failure == "connect" else ExaoneThrottledError
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(
            api_key="test-only-key",
            http_client=client,
            max_retries=initial_retries,
            base_retry_delay=0,
            max_retry_delay=0,
            temperature=0.4,
            base_url="https://old.example/v1",
            timeout=11,
        )
        first = asyncio.create_task(make_harness(model).invoke_async("First."))
        await asyncio.wait_for(entered.wait(), timeout=2)
        model.update_config(max_retries=updated_retries, temperature=0.7, base_url="https://new.example/v1", timeout=22)
        release.set()
        if initial_retries:
            await first
        else:
            with pytest.raises(error_type):
                await first
        phase = "second"
        if updated_retries:
            await make_harness(model).invoke_async("Second.")
        else:
            with pytest.raises(error_type):
                await make_harness(model).invoke_async("Second.")
    assert calls["first"] == [("old.example", 0.4, 11)] * (initial_retries + 1)
    assert calls["second"] == [("new.example", 0.7, 22)] * (updated_retries + 1)


async def test_harness_cancellation_still_interrupts_retry_after_config_update(monkeypatch: Any) -> None:
    import asyncio
    import threading

    import strands_exaone.model as model_module

    waiting = asyncio.Event()
    cancel = threading.Event()
    count = 0

    def delay(*args: Any, **kwargs: Any) -> float:
        waiting.set()
        return 30.0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(429, headers={"retry-after": "30"}, json={"error": {"message": "busy"}})

    monkeypatch.setattr(model_module, "retry_delay", delay)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client, max_retries=1, max_retry_delay=30)
        task = asyncio.create_task(make_harness(model).invoke_async("Hello.", cancel_signal=cancel))
        await asyncio.wait_for(waiting.wait(), timeout=2)
        model.update_config(max_retries=0, max_retry_delay=0)
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
    assert count == 1


@pytest.mark.parametrize(("initial_max_delay", "updated_max_delay"), [(0.0, 1.0), (1.0, 0.0)])
async def test_harness_config_update_does_not_change_inflight_retry_delay_ceiling(
    initial_max_delay: float, updated_max_delay: float
) -> None:
    import asyncio

    from strands_exaone import ExaoneThrottledError

    entered = asyncio.Event()
    release = asyncio.Event()
    requests: dict[str, int] = {"first": 0, "second": 0}
    phase = "first"

    async def handler(request: httpx.Request) -> httpx.Response:
        requests[phase] += 1
        if requests[phase] == 1:
            if phase == "first":
                entered.set()
                await release.wait()
            return httpx.Response(429, headers={"retry-after": "0.01"}, json={"error": {"message": "busy"}})
        return response(events(chunk({"content": "ok"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(
            api_key="test-only-key", http_client=client, max_retries=1, max_retry_delay=initial_max_delay
        )
        first = asyncio.create_task(make_harness(model).invoke_async("First."))
        await asyncio.wait_for(entered.wait(), timeout=2)
        model.update_config(max_retry_delay=updated_max_delay)
        release.set()
        if initial_max_delay:
            await first
        else:
            with pytest.raises(ExaoneThrottledError):
                await first
        phase = "second"
        if updated_max_delay:
            await make_harness(model).invoke_async("Second.")
        else:
            with pytest.raises(ExaoneThrottledError):
                await make_harness(model).invoke_async("Second.")
    assert requests == {"first": 2 if initial_max_delay else 1, "second": 2 if updated_max_delay else 1}


async def test_harness_preserves_nested_json_tool_result() -> None:
    expected = {"greeting": "안녕", "values": [0, False, None], "nested": {"key": "x\n🙂"}}
    requests: list[dict[str, Any]] = []

    @tool
    def get_json() -> dict[str, Any]:
        """Return a synthetic JSON object."""
        return expected

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(
                events(
                    chunk({"reasoning_content": "request synthetic data"}),
                    chunk(tool_delta("call_1", "get_json")),
                    chunk(finish="tool_calls"),
                )
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_json])
        await agent.invoke_async("Get the synthetic JSON.")
    assert len(requests) == 2
    result = next(message for message in requests[1]["messages"] if message.get("tool_call_id") == "call_1")
    assert json.loads(result["content"]) == expected
    assert any(message.get("reasoning_content") == "request synthetic data" for message in requests[1]["messages"])


async def test_harness_repeated_identical_finish_reason_executes_tool_once() -> None:
    calls: list[bool] = []
    requests = 0

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return response(
                events(
                    chunk(tool_delta("call_1", "get_number")), chunk(finish="tool_calls"), chunk(finish="tool_calls")
                )
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_number]).invoke_async(
            "Get a number."
        )
    assert calls == [True]
    assert requests == 2


async def test_harness_conflicting_finish_reasons_fail_before_tool_execution() -> None:
    calls: list[bool] = []

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        return response(events(chunk(tool_delta("call_1", "get_number")), chunk(finish="length"), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ExaoneError, match="Multiple finish reasons"):
            await make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_number]).invoke_async(
                "Get a number."
            )
    assert calls == []


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", '{"nested":[1,NaN]}'])
async def test_harness_rejects_non_json_number_constants_before_tool_execution(token: str) -> None:
    calls: list[float] = []
    requests = 0

    @tool
    def capture(value: float) -> str:
        """Record a synthetic numeric input."""
        calls.append(value)
        return "done"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests > 1:
            return response(events(chunk({"content": "unexpected"}), chunk(finish="stop")))
        call = {
            "tool_calls": [
                {"index": 0, "id": "capture_1", "function": {"name": "capture", "arguments": '{"value":' + token + "}"}}
            ]
        }
        return response(events(chunk(call), chunk(finish="tool_calls")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ExaoneError, match="Invalid tool arguments") as caught:
            await make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [capture]).invoke_async(
                "Use the tool."
            )
    assert calls == []
    assert requests == 1
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "raw_value",
    ['"NaN"', '"Infinity"', "0", "-2.5", "1e308", '{"label":"NaN","nested":[-1,0,4.2]}'],
)
async def test_harness_accepts_valid_json_numbers_and_constant_strings(raw_value: str) -> None:
    calls: list[Any] = []
    requests = 0

    @tool
    def capture(value: Any) -> str:
        """Record a synthetic JSON value."""
        calls.append(value)
        return "done"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests > 1:
            return response(events(chunk({"content": "done"}), chunk(finish="stop")))
        call = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "capture_1",
                    "function": {"name": "capture", "arguments": '{"value":' + raw_value + "}"},
                }
            ]
        }
        return response(events(chunk(call), chunk(finish="tool_calls")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [capture]).invoke_async(
            "Use the tool."
        )
    assert calls == [json.loads(raw_value)]
    assert requests == 2
