"""Replay pinned-source synthetic tool-choice fixtures through one Python provider."""

import asyncio
import functools
import json
import sys
from pathlib import Path

import httpx

FIXTURES = Path(__file__).with_name("choice-fixtures")
NAMED = {"type": "function", "function": {"name": "lookup"}}
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up synthetic data",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
    },
}


def response(fixture, stream):
    if stream:
        body = "".join(f"data: {json.dumps(item)}\n\n" for item in fixture["stream"])
        return httpx.Response(200, content=(body + "data: [DONE]\n\n").encode())
    return httpx.Response(200, json=fixture["nonstream"])


async def main(framework):
    paths = sorted(FIXTURES.glob("*.json"))
    assert len(paths) == 4, "expected four tool-choice fixtures"
    for path in paths:
        fixture = json.loads(path.read_text())
        choice = "named" if path.stem.endswith("-named") else "required"
        for stream in (False, True):
            requests = []

            def handler(request, requests=requests, fixture=fixture, stream=stream):
                requests.append(json.loads(request.content))
                return response(fixture, stream)

            if framework == "strands":
                from strands_exaone import ExaoneModel

                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    model = ExaoneModel(
                        api_key="EMPTY",
                        model_id="synthetic",
                        base_url="http://fixture/v1",
                        stream=stream,
                        http_client=client,
                    )
                    selected = (
                        {"tool": {"name": "lookup"}}
                        if choice == "named"
                        else {"any": {}}
                    )
                    events = [
                        event
                        async for event in model.stream(
                            [{"role": "user", "content": [{"text": "lookup"}]}],
                            tool_specs=[
                                {
                                    "name": "lookup",
                                    "description": "Look up synthetic data",
                                    "inputSchema": {
                                        "json": TOOL["function"]["parameters"]
                                    },
                                }
                            ],
                            tool_choice=selected,
                        )
                    ]
                starts = [
                    e["contentBlockStart"]["start"]["toolUse"]
                    for e in events
                    if "toolUse" in e.get("contentBlockStart", {}).get("start", {})
                ]
                parts = [
                    e["contentBlockDelta"]["delta"]["toolUse"]["input"]
                    for e in events
                    if "toolUse" in e.get("contentBlockDelta", {}).get("delta", {})
                ]
                assert starts == [{"name": "lookup", "toolUseId": "call_0"}], (
                    path,
                    stream,
                )
                assert json.loads("".join(parts)) == fixture["expected"], (path, stream)
                assert any(
                    e.get("messageStop", {}).get("stopReason") == "tool_use"
                    for e in events
                )
            elif framework == "langchain":
                from langchain_core.messages import HumanMessage
                from langchain_exaone import ChatEXAONE

                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    model = ChatEXAONE(
                        api_key="EMPTY",
                        model="synthetic",
                        base_url="http://fixture/v1",
                        http_client=client,
                        output_version="v1",
                    ).bind_tools(
                        [TOOL],
                        tool_choice="lookup" if choice == "named" else "required",
                    )
                    message = HumanMessage("lookup")
                    result = (
                        functools.reduce(lambda a, b: a + b, model.stream([message]))
                        if stream
                        else model.invoke([message])
                    )
                assert len(result.tool_calls) == 1, (path, stream)
                call = result.tool_calls[0]
                assert (call["name"], call["id"], call["args"]) == (
                    "lookup",
                    "call_0",
                    fixture["expected"],
                ), (path, stream)
            else:
                raise ValueError(framework)
            assert len(requests) == 1
            assert requests[0]["tool_choice"] == (
                NAMED if choice == "named" else "required"
            )
        print(f"{path.name}: {framework} streaming/nonstreaming passed")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
