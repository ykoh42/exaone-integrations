"""Agent-level tests for opt-in malformed tool-call recovery."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from strands import Agent, tool
from strands.middleware import InvokeModelStage
from strands.session.file_session_manager import FileSessionManager
from strands.types.exceptions import EventLoopException, MaxTokensReachedException

from strands_exaone import ExaoneError, ExaoneModel
from strands_exaone.errors import MalformedToolCallError
from strands_exaone.recovery import recover_tool_calls

NAMED_TRANSFER = {"type": "function", "function": {"name": "transfer"}}


def completion(arguments: str | None, *, finish: str = "tool_calls", text: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if arguments is not None:
        message["tool_calls"] = [
            {"id": "call_1", "type": "function", "function": {"name": "transfer", "arguments": arguments}}
        ]
    return {
        "id": "fixture",
        "model": "fixture",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 5,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("arguments", "malformed"),
    [
        ('{"amount":123}', False),
        ('{"amount":123,}', True),
        ('{"amount":12', True),
        ('{\n"note":"😀```","amount":123,,}', True),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_agent_recovers_only_when_enabled(enabled: bool, arguments: str, malformed: bool, stream: bool) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []
    content: list[str] = []
    metadata: list[dict[str, Any]] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        number = len(requests)
        if number == 1:
            data = completion(arguments, text="original preamble" if malformed else None)
            if malformed:
                data["choices"][0]["message"]["reasoning_content"] = "synthetic thought"
        elif number == 2 and malformed:
            data = completion('{"amount":123}', text="private correction preamble")
            data["choices"][0]["message"]["reasoning_content"] = "private correction reasoning"
            data["choices"][0]["message"]["tool_calls"][0]["id"] = "correction-id"
        else:
            data = completion(None, finish="stop", text="done")
        if not stream:
            return httpx.Response(200, json=data)
        choice = data["choices"][0]
        delta = dict(choice["message"])
        for index, call in enumerate(delta.get("tool_calls", [])):
            call["index"] = index
        chunks = [
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
            {"choices": [], "usage": data["usage"]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    def callback(**event: Any) -> None:
        if event.get("data"):
            content.append(event["data"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=stream, metadata_callback=metadata.append)
        agent = Agent(model=model, tools=[transfer], system_prompt="synthetic system", callback_handler=callback)
        if enabled:
            agent.add_middleware(InvokeModelStage, recover_tool_calls)
        if malformed and not enabled:
            with pytest.raises(ExaoneError, match="Invalid tool arguments"):
                await agent.invoke_async("synthetic user request")
            assert executions == []
            assert len(requests) == 1
            return
        result = await agent.invoke_async("synthetic user request")

    assert executions == [123]
    assert all(
        [m for m in request["messages"] if m["role"] == "system"] == [{"role": "system", "content": "synthetic system"}]
        for request in requests
    )
    assert all(len([m for m in request["messages"] if m["role"] == "user"]) == 1 for request in requests)
    if malformed:
        assert len(requests) == 3
        assert [request.get("tool_choice") for request in requests] == [None, NAMED_TRANSFER, None]
        assert result.metrics.accumulated_usage["totalTokens"] == 36
        assert result.metrics.accumulated_usage["cacheReadInputTokens"] == 6
        assert metadata[0]["recovery"]["failed_attempts"][0]["usage"]["total_tokens"] == 12
        feedback = requests[1]["messages"][-1]
        assert feedback["role"] == "tool"
        assert feedback["tool_call_id"] == "call_1"
        header, diagnostic, instruction = feedback["content"].split("\n")
        assert header.startswith("Error: ") and "The previous tool call is a placeholder." in header
        with pytest.raises(json.JSONDecodeError) as parser_error:
            json.loads(arguments)
        error = parser_error.value
        assert json.loads(diagnostic) == {
            "raw_arguments": arguments,
            "parse_error": f"{error.msg} (line {error.lineno}, column {error.colno})",
        }
        assert instruction.startswith("Call transfer once")
        assert requests[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == "{}"
        assert requests[1]["messages"][-2]["reasoning_content"] == "synthetic thought"
        assert content == ["original preamble", "done"]
        followup = requests[2]["messages"]
        assert followup[-2]["content"] == "original preamble"
        assert followup[-2]["reasoning_content"] == "synthetic thought"
        assert followup[-2]["tool_calls"][0]["id"] == followup[-1]["tool_call_id"] == "call_1"
        assert "private correction" not in json.dumps(followup)
        assert "The previous tool call is a placeholder." not in json.dumps(requests[2]["messages"])
        assert all(
            call["function"]["arguments"] != "{}"
            for message in requests[2]["messages"]
            for call in message.get("tool_calls", [])
        )
    else:
        assert len(requests) == 2
        assert result.metrics.accumulated_usage["totalTokens"] == 24


@pytest.mark.parametrize(
    "scenario", ["fallback", "bad_request", "text", "wrong_tool", "multiple", "blank", "array", "length"]
)
async def test_only_a_completed_original_tool_counts_as_recovery(scenario: str) -> None:
    choices: list[Any] = []
    executions: list[int] = []
    visible: list[str] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(request: httpx.Request) -> httpx.Response:
        choices.append(json.loads(request.content).get("tool_choice"))
        attempt = len(choices)
        if attempt == 1:
            return httpx.Response(200, json=completion('{"amount":12'))
        if attempt == 2 and scenario in {"fallback", "bad_request"}:
            message = "Named tool choice is not supported" if scenario == "fallback" else "invalid request to model"
            return httpx.Response(400, json={"error": {"message": message}})
        if scenario == "text" or attempt == 4:
            return httpx.Response(200, json=completion(None, finish="stop", text="done"))
        arguments = {"blank": " ", "array": "[]"}.get(scenario, '{"amount":123}')
        result = completion(
            arguments, text="correction body", finish="length" if scenario == "length" else "tool_calls"
        )
        calls = result["choices"][0]["message"]["tool_calls"]
        if scenario == "wrong_tool":
            calls[0]["function"]["name"] = "other"
        if scenario == "multiple":
            calls.append({**calls[0], "id": "extra"})
        return httpx.Response(200, json=result)

    def callback(**event: Any) -> None:
        if event.get("data"):
            visible.append(event["data"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False)
        agent = Agent(model=model, tools=[transfer], callback_handler=callback, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        if scenario == "fallback":
            await agent.invoke_async("Record 123.")
            assert choices == [None, NAMED_TRANSFER, "required", None]
            assert executions == [123]
        else:
            error_type = MaxTokensReachedException if scenario == "length" else ExaoneError
            with pytest.raises(error_type):
                await agent.invoke_async("Record 123.")
            assert choices == [None, NAMED_TRANSFER]
            assert executions == []
            if scenario != "length":
                assert visible == []


@pytest.mark.parametrize("fallback", [False, True])
async def test_correction_never_exposes_a_wrong_tool_or_its_text(fallback: bool) -> None:
    choices: list[Any] = []
    executions: list[int] = []
    visible: list[str] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic transfer."""
        executions.append(amount)
        return "Recorded."

    @tool
    def other(amount: int) -> str:
        """Record an unrelated synthetic operation."""
        executions.append(amount)
        return "Recorded."

    def respond(request: httpx.Request) -> httpx.Response:
        choice = json.loads(request.content).get("tool_choice")
        choices.append(choice)
        if choice is None:
            result = completion('{"amount":12')
        elif fallback and choice == NAMED_TRANSFER:
            return httpx.Response(400, json={"error": {"message": "Named tool choice is not supported"}})
        else:
            result = completion('{"amount":123}', text="Unvalidated correction")
            result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "other"
        return httpx.Response(200, json=result)

    def callback(**event: Any) -> None:
        if event.get("data"):
            visible.append(event["data"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False)
        agent = Agent(model=model, tools=[transfer, other], callback_handler=callback, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)

        with pytest.raises(ExaoneError, match="original tool"):
            await agent.invoke_async("Record 123.")
        assert choices == [None, NAMED_TRANSFER, *(["required"] if fallback else [])]
        assert executions == []
        assert visible == []


@pytest.mark.parametrize("case", ["tool_only", "visible_text", "ambiguous_correction", "filtered_initial"])
async def test_live_recovery_only_exposes_valid_tools(case: str) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []
    visible: list[str] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def event(delta: dict[str, Any], finish: str | None = None) -> bytes:
        data = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(data)}\n\n".encode()

    class FirstResponse(httpx.AsyncByteStream):
        async def __aiter__(self):
            delta: dict[str, Any] = {
                "tool_calls": [
                    {"index": 0, "id": "failed", "function": {"name": "transfer", "arguments": '{"amount":12'}}
                ]
            }
            if case == "visible_text":
                delta["content"] = "Checking."
            yield event(delta)
            assert ("Checking." in visible) == (case == "visible_text")
            yield event({}, "tool_calls") + b"data: [DONE]\n\n"

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, stream=FirstResponse())
        if len(requests) == 2:
            delta = {
                "tool_calls": [
                    {"index": 0, "id": "corrected", "function": {"name": "transfer", "arguments": '{"amount":123}'}}
                ]
            }
            content = event(delta, "tool_calls")
            if case == "ambiguous_correction":
                content = event(delta) + event({"tool_calls": [{"index": 0, "id": "corrected"}]}, "tool_calls")
            return httpx.Response(200, content=content + b"data: [DONE]\n\n")
        return httpx.Response(200, content=event({"content": "Done."}, "stop") + b"data: [DONE]\n\n")

    def callback(**event: Any) -> None:
        if event.get("data"):
            visible.append(event["data"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client)
        agent = Agent(model=model, tools=[transfer], callback_handler=callback, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        if case.startswith("filtered"):

            async def filter_tools(context, next_fn):
                if case == "filtered_initial":
                    context = context.replace(tool_specs=[])
                async for event in next_fn(context):
                    yield event

            agent.add_middleware(InvokeModelStage, filter_tools)
        if case == "ambiguous_correction":
            with pytest.raises(ExaoneError, match="ambiguous"):
                await agent.invoke_async("Record 123.")
            assert len(requests) == 2
            assert executions == []
        elif case.startswith("filtered"):
            with pytest.raises(ExaoneError, match="Invalid tool arguments|does not allow the tool correction"):
                await agent.invoke_async("Record 123.")
            assert len(requests) == 1
            assert not requests[0].get("tools")
            assert executions == []
        else:
            await agent.invoke_async("Record 123.")
            assert len(requests) == 3
            assert requests[1]["tool_choice"] == NAMED_TRANSFER
            assert executions == [123]
            assert visible == (["Checking.", "Done."] if case == "visible_text" else ["Done."])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_recovery_blocks_refused_tool_even_with_text(stream: bool, enabled: bool) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            result = completion('{"amount":123}', text="Synthetic reply.")
            result["choices"][0]["message"]["refusal"] = "Declined."
        else:
            result = completion(None, text="Done.", finish="stop")
        if not stream:
            return httpx.Response(200, json=result)
        choice = result["choices"][0]
        delta = choice["message"]
        for index, call in enumerate(delta.get("tool_calls", [])):
            call["index"] = index
        frame = {"choices": [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}]}
        return httpx.Response(200, text=f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=stream, max_retries=0)
        agent = Agent(model=model, tools=[transfer], callback_handler=None, retry_strategy=None)
        if enabled:
            agent.add_middleware(InvokeModelStage, recover_tool_calls)
            with pytest.raises(ExaoneError, match="refusal text together with tool calls"):
                await agent.invoke_async("Record 123.")
            assert len(requests) == 1
            assert executions == []
        else:
            await agent.invoke_async("Record 123.")
            assert len(requests) == 2
            assert executions == [123]


@pytest.mark.parametrize("case", ["repeated", "truncated", "mixed", "missing_id", "refusal"])
async def test_ambiguous_or_repeated_failures_do_not_execute_tools(case: str) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        response = completion('{"amount":123,,}', finish="length" if case == "truncated" else "tool_calls")
        if case == "refusal":
            response["choices"][0]["message"].update({"content": "refused", "refusal": "refused"})
        calls = response["choices"][0]["message"]["tool_calls"]
        if case == "mixed":
            calls.append(
                {"id": "call_2", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":456}'}}
            )
        if case == "missing_id":
            calls[0]["id"] = ""
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False)
        agent = Agent(model=model, tools=[transfer], callback_handler=None, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        error_type = MaxTokensReachedException if case == "truncated" else ExaoneError
        with pytest.raises(error_type):
            await agent.invoke_async("synthetic user request")

    assert executions == []
    assert len(requests) == (2 if case == "repeated" else 1)


async def test_concurrent_agents_do_not_exchange_failed_arguments() -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    async def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        user = next(message["content"] for message in body["messages"] if message["role"] == "user")
        amount = 101 if "101" in user else 202
        results = [message for message in body["messages"] if message["role"] == "tool"]
        await asyncio.sleep(0)
        if not results:
            return httpx.Response(200, json=completion(f'{{"amount":{amount},,}}'))
        if results[0]["content"].startswith("Error: "):
            assert json.loads(results[0]["content"].split("\n")[1])["raw_arguments"] == f'{{"amount":{amount},,}}'
            return httpx.Response(200, json=completion(f'{{"amount":{amount}}}'))
        return httpx.Response(200, json=completion(None, finish="stop", text="done"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False)
        agents = [Agent(model=model, tools=[transfer], callback_handler=None) for _ in range(2)]
        for agent in agents:
            agent.add_middleware(InvokeModelStage, recover_tool_calls)
        await asyncio.gather(agents[0].invoke_async("Record 101"), agents[1].invoke_async("Record 202"))

    assert sorted(executions) == [101, 202]
    assert len(requests) == 6


@pytest.mark.parametrize("during_correction", [False, True])
async def test_cancelled_request_is_not_retried(during_correction: bool) -> None:
    started = asyncio.Event()
    requests = 0
    executions = []
    text = []

    @tool
    def transfer(amount: int) -> str:
        """Record the synthetic transfer."""
        executions.append(amount)
        return "done"

    async def transport(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if during_correction and requests == 1:
            return httpx.Response(200, json=completion('{"amount":123,}', text="original preamble"))
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    def callback(**event: Any) -> None:
        if event.get("data"):
            text.append(event["data"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False)
        agent = Agent(model=model, tools=[transfer], callback_handler=callback, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        task = asyncio.create_task(agent.invoke_async("synthetic user request"))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert requests == (2 if during_correction else 1)
    assert text == (["original preamble"] if during_correction else [])
    assert executions == []


async def test_buffered_delivery_honors_timeout_when_consumer_pauses() -> None:
    requests = 0

    def transport(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=completion(None, finish="stop", text="hello"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False, timeout=1)
        agent = Agent(model=model, callback_handler=None, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        events = agent.stream_async("synthetic user request")
        try:
            async for event in events:
                if event.get("data") == "hello":
                    await asyncio.sleep(1.1)
                    with pytest.raises((TimeoutError, EventLoopException), match="timeout|deadline|timed out"):
                        async for _ in events:
                            pass
                    break
            else:
                pytest.fail("Expected a streamed text event")
        finally:
            await events.aclose()

    assert requests == 1


async def test_named_tool_choice_is_preserved_for_correction() -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []
    stages = 0

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            response = completion('{"amount":123,,}', finish="stop")
        elif len(requests) == 2:
            response = completion('{"amount":123}', finish="stop")
        else:
            response = completion(None, finish="stop", text="done")
        return httpx.Response(200, json=response)

    def choose_first(context: Any) -> Any:
        nonlocal stages
        stages += 1
        return context.replace(tool_choice={"tool": {"name": "transfer"}}) if stages == 1 else context

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False)
        agent = Agent(model=model, tools=[transfer], callback_handler=None, retry_strategy=None)
        agent.add_middleware(InvokeModelStage.Input, choose_first)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        await agent.invoke_async("synthetic user request")

    expected = {"type": "function", "function": {"name": "transfer"}}
    assert requests[0]["tool_choice"] == requests[1]["tool_choice"] == expected
    assert executions == [123]


async def test_multimodal_user_and_system_blocks_are_preserved() -> None:
    requests: list[dict[str, Any]] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        return str(amount)

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            response = completion('{"amount":123,,}')
        elif len(requests) == 2:
            response = completion('{"amount":123}')
        else:
            response = completion(None, finish="stop", text="done")
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False, base_url="https://local.example/v1")
        agent = Agent(
            model=model,
            tools=[transfer],
            callback_handler=None,
            system_prompt=[{"text": "first synthetic instruction"}, {"text": "second synthetic instruction"}],
        )
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        await agent.invoke_async(
            [{"text": "synthetic image"}, {"image": {"format": "png", "source": {"bytes": b"synthetic"}}}]
        )

    for role in ("system", "user"):
        assert [message for message in requests[0]["messages"] if message["role"] == role] == [
            message for message in requests[1]["messages"] if message["role"] == role
        ]
    assert len(requests[0]["messages"][-1]["content"]) == 2


async def test_ambiguous_streamed_call_ids_are_not_retried() -> None:
    requests = 0
    executions: list[int] = []
    visible: list[str] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "discarded",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "transfer", "arguments": '{"amount":'},
                                }
                            ],
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"arguments": "123,,}"},
                                }
                            ]
                        },
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    def callback(**event: Any) -> None:
        if event.get("data"):
            visible.append(event["data"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=True)
        agent = Agent(model=model, tools=[transfer], callback_handler=callback, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        with pytest.raises(ExaoneError, match="ambiguous streamed ID"):
            await agent.invoke_async("synthetic user request")

    assert requests == 1
    assert executions == []
    assert "discarded" in visible


@pytest.mark.parametrize("field", ["arguments", "id", "name"])
async def test_does_not_retry_tool_fields_containing_encoded_api_key(field: str) -> None:
    requests = 0

    @tool
    def transfer(amount: str) -> str:
        """Record a synthetic amount."""
        pytest.fail("Invalid arguments must not execute a tool")

    def transport(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        arguments = '{"amount":"synthetic%20secret",,}' if field == "arguments" else '{"amount":123,,}'
        response = completion(arguments)
        call = response["choices"][0]["message"]["tool_calls"][0]
        if field == "id":
            call["id"] = "synthetic%20secret"
        if field == "name":
            call["function"]["name"] = "synthetic%20secret"
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="synthetic secret", http_client=client, stream=False)
        agent = Agent(model=model, tools=[transfer], callback_handler=None, retry_strategy=None)
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        with pytest.raises(ExaoneError, match="Invalid tool arguments") as caught:
            await agent.invoke_async("synthetic user request")

    assert requests == 1
    error: BaseException | None = caught.value
    while error is not None:
        assert not isinstance(error, MalformedToolCallError)
        assert "synthetic%20secret" not in str(error)
        error = error.__cause__ or error.__context__


@pytest.mark.parametrize("stream", [False, True])
async def test_successful_response_keeps_metadata_callback_order(stream: bool) -> None:
    async def run(enabled: bool) -> list[str]:
        order: list[str] = []

        def transport(_: httpx.Request) -> httpx.Response:
            data = completion(None, finish="stop", text="hello")
            if not stream:
                return httpx.Response(200, json=data)
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": data["usage"]},
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body)

        def callback(**event: Any) -> None:
            if event.get("data"):
                order.append("text")

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            model = ExaoneModel(
                api_key="fixture",
                http_client=client,
                stream=stream,
                metadata_callback=lambda _: order.append("metadata"),
            )
            agent = Agent(model=model, callback_handler=callback)
            if enabled:
                agent.add_middleware(InvokeModelStage, recover_tool_calls)
            await agent.invoke_async("synthetic user request")
        return order

    assert await run(True) == await run(False) == ["text", "metadata"]


async def test_recovered_session_can_reload_without_placeholder(tmp_path: Any) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=completion('{"amount":123,,}'))
        if len(requests) == 2:
            return httpx.Response(200, json=completion('{"amount":123}'))
        return httpx.Response(200, json=completion(None, finish="stop", text="done"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=False)

        def make_agent() -> Agent:
            agent = Agent(
                model=model,
                tools=[transfer],
                callback_handler=None,
                agent_id="recovery-agent",
                session_manager=FileSessionManager("recovery-session", str(tmp_path)),
            )
            agent.add_middleware(InvokeModelStage, recover_tool_calls)
            return agent

        await make_agent().invoke_async("Record 123")
        restored = make_agent()
        await restored.invoke_async("Continue")

    assert executions == [123]
    assert len(requests) == 4
    assert "The previous tool call is a placeholder." not in json.dumps(requests[3]["messages"])
    assert all("The previous tool call is a placeholder." not in path.read_text() for path in tmp_path.rglob("*.json"))
    assert all(
        call["function"]["arguments"] != "{}"
        for message in requests[3]["messages"]
        for call in message.get("tool_calls", [])
    )


@pytest.mark.parametrize("valid_first", [False, True])
async def test_mixed_streamed_calls_never_execute(valid_first: bool) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        calls = completion('{"amount":123,}')["choices"][0]["message"]["tool_calls"]
        calls.append(
            {"id": "call_2", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":456}'}}
        )
        if valid_first:
            calls.reverse()
        chunks = [
            {"choices": [{"index": 0, "delta": {"tool_calls": [{**call, "index": index}]}}]}
            for index, call in enumerate(calls)
        ]
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=True, max_retries=0)
        agent = Agent(
            model=model, tools=[transfer], system_prompt="synthetic system", callback_handler=None, retry_strategy=None
        )
        agent.add_middleware(InvokeModelStage, recover_tool_calls)
        with pytest.raises(ExaoneError):
            await agent.invoke_async("synthetic user request")

    assert executions == []
    assert len(requests) == 1
    assert [message["content"] for message in requests[0]["messages"] if message["role"] == "user"] == [
        "synthetic user request"
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_multiple_valid_calls_are_unchanged_by_optional_recovery(stream: bool) -> None:
    async def run(enabled: bool) -> tuple[list[dict[str, Any]], list[int]]:
        requests: list[dict[str, Any]] = []
        executions: list[int] = []

        @tool
        def transfer(amount: int) -> str:
            """Record a synthetic amount."""
            executions.append(amount)
            return "Recorded."

        def transport(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            data = completion('{"amount":123}') if len(requests) == 1 else completion(None, finish="stop", text="done")
            if len(requests) == 1:
                data["choices"][0]["message"]["tool_calls"].append(
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "transfer", "arguments": '{"amount":456}'},
                    }
                )
            if not stream:
                return httpx.Response(200, json=data)
            choice = data["choices"][0]
            delta = dict(choice["message"])
            if "tool_calls" in delta:
                delta["tool_calls"] = [{**call, "index": index} for index, call in enumerate(delta["tool_calls"])]
            chunks = [
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            model = ExaoneModel(api_key="fixture", http_client=client, stream=stream, max_retries=0)
            agent = Agent(
                model=model,
                tools=[transfer],
                system_prompt="synthetic system",
                callback_handler=None,
                retry_strategy=None,
            )
            if enabled:
                agent.add_middleware(InvokeModelStage, recover_tool_calls)
            await agent.invoke_async("synthetic user request")
        return requests, executions

    baseline_requests, baseline_executions = await run(False)
    guarded_requests, guarded_executions = await run(True)
    assert baseline_requests == guarded_requests
    assert sorted(baseline_executions) == sorted(guarded_executions) == [123, 456]
