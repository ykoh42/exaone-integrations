"""Strands must decide what to do with calls from a length-limited response."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
from strands import Agent, tool
from strands.hooks import AfterModelCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.types.exceptions import MaxTokensReachedException

from strands_exaone import ExaoneError, ExaoneModel


def response(stream: bool, arguments: list[str]) -> httpx.Response:
    calls = [
        {"id": f"call-{index}", "type": "function", "function": {"name": "lookup", "arguments": value}}
        for index, value in enumerate(arguments)
    ]
    if not stream:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"reasoning_content": "synthetic reason", "content": "partial", "tool_calls": calls},
                        "finish_reason": "length",
                    }
                ]
            },
        )
    deltas = [
        {"reasoning_content": "synthetic reason", "content": "partial"},
        {"tool_calls": [{"index": index, **call} for index, call in enumerate(calls)]},
    ]
    chunks = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
    raw = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=raw.encode())


def config(hosted: bool) -> dict[str, Any]:
    if hosted:
        return {"model_id": "k-exaone_v2.5"}
    return {"model_id": "EXAONE-4.5-33B", "base_url": "http://localhost:8000/v1"}


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ['{"value":12}', '{"value":12', '{"value":123'])
async def test_length_exposes_original_tool_fragments_and_max_tokens(
    hosted: bool, stream: bool, arguments: str
) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response(stream, [arguments]))) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=stream, **config(hosted))
        events = [event async for event in model.stream([{"role": "user", "content": [{"text": "test"}]}])]

    inputs = [
        event["contentBlockDelta"]["delta"]["toolUse"]["input"]
        for event in events
        if "toolUse" in event.get("contentBlockDelta", {}).get("delta", {})
    ]
    assert inputs == [arguments]
    assert events[-1]["messageStop"]["stopReason"] == "max_tokens"
    assert any(event.get("contentBlockDelta", {}).get("delta", {}).get("reasoningContent") for event in events)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("identities", [[("", "lookup")], [("call-1", "")], [("same", "a"), ("same", "b")]])
async def test_length_still_rejects_missing_or_duplicate_tool_identities(
    stream: bool, identities: list[tuple[str, str]]
) -> None:
    calls = [
        {"id": ident, "type": "function", "function": {"name": name, "arguments": "{"}} for ident, name in identities
    ]
    if stream:
        calls = [{"index": index, **call} for index, call in enumerate(calls)]
        chunks = [
            {"choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
        ]
        raw = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        payload = httpx.Response(200, content=raw.encode())
    else:
        payload = httpx.Response(200, json={"choices": [{"message": {"tool_calls": calls}, "finish_reason": "length"}]})
    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: payload)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=stream)
        with pytest.raises(ExaoneError, match="Incomplete tool call|Duplicate tool call IDs"):
            async for event in model.stream([{"role": "user", "content": [{"text": "test"}]}]):
                events.append(event)
    assert not any(event.get("contentBlockStart", {}).get("start", {}).get("toolUse") for event in events)


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
async def test_length_makes_real_agent_stop_without_executing_tool(hosted: bool, stream: bool, truncated: bool) -> None:
    arguments = ['{"value":12', '{"value":123}'] if truncated else ['{"value":12}', '{"value":123}']
    calls: list[int] = []
    before_tool: list[str] = []
    model_stops: list[tuple[str, dict[str, Any]]] = []
    requests = 0

    @tool
    def lookup(value: int) -> str:
        """Read a synthetic value."""
        calls.append(value)
        return "synthetic result"

    class Observe(HookProvider):
        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(BeforeToolCallEvent, self.before_tool)
            registry.add_callback(AfterModelCallEvent, self.after_model)

        def before_tool(self, event: BeforeToolCallEvent) -> None:
            before_tool.append(event.tool_use["name"])

        def after_model(self, event: AfterModelCallEvent) -> None:
            if event.stop_response is not None:
                model_stops.append((event.stop_response.stop_reason, event.stop_response.message))

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return response(stream, arguments)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=stream, **config(hosted))
        agent = Agent(model=model, tools=[lookup], hooks=[Observe()], callback_handler=None)
        with pytest.raises(MaxTokensReachedException):
            await agent.invoke_async("Use the tool.")

    assert requests == 1
    assert calls == []
    assert before_tool == []
    assert len(model_stops) == 1 and model_stops[0][0] == "max_tokens"
    tool_blocks = [block["toolUse"] for block in model_stops[0][1]["content"] if "toolUse" in block]
    assert len(tool_blocks) == 2
    assert tool_blocks[0]["input"] == ({} if truncated else {"value": 12})
    assert tool_blocks[1]["input"] == {"value": 123}
    recovered_blocks = agent.messages[-1]["content"]
    assert not any("toolUse" in block for block in recovered_blocks)
    assert sum("lookup" in block.get("text", "") for block in recovered_blocks) == 2
    assert any(block.get("text") == "partial" for block in recovered_blocks)
    assert any(block.get("reasoningContent") for block in recovered_blocks)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["valid_length", "invalid_length", "invalid_tool_calls"])
async def test_strands_sdk_logs_only_invalid_tool_input_it_receives(
    stream: bool, case: str, caplog: pytest.LogCaptureFixture
) -> None:
    raw = '{"value":12}' if case == "valid_length" else '{"value":"SYNTHETIC_ARGUMENT_' + "x" * 250
    payload = response(stream, [raw])
    if case == "invalid_tool_calls":
        payload = httpx.Response(200, content=payload.content.replace(b'"length"', b'"tool_calls"'))

    @tool
    def lookup(value: int) -> str:
        """Read a synthetic value."""
        pytest.fail("A truncated or invalid call must not execute")

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: payload)) as client:
        model = ExaoneModel(api_key="fixture", http_client=client, stream=stream)
        agent = Agent(model=model, tools=[lookup], callback_handler=None)
        expected = ExaoneError if case == "invalid_tool_calls" else MaxTokensReachedException
        with caplog.at_level(logging.WARNING, logger="strands.event_loop.streaming"), pytest.raises(expected):
            await agent.invoke_async("Use the tool.")

    records = [
        record
        for record in caplog.records
        if record.name == "strands.event_loop.streaming" and "failed to parse tool input json" in record.getMessage()
    ]
    if case == "invalid_length":
        assert len(records) == 1 and records[0].levelno == logging.WARNING
        assert raw[:200] in records[0].getMessage()
        assert raw[:201] not in records[0].getMessage()
    else:
        assert not records
