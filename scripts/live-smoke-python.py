"""Opt-in live integration smoke check. Requires EXAONE_API_KEY in the environment."""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import time
from typing import Any

import httpx
import truststore
from pydantic import BaseModel
from strands import Agent, tool

from strands_exaone import ExaoneModel


class CountResult(BaseModel):
    count: int


class Probe:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.metadata: list[dict[str, Any]] = []
        self.last_request = 0.0
        self.lock = asyncio.Lock()
        self.tool_calls = 0

    async def on_request(self, request: httpx.Request) -> None:
        async with self.lock:
            remaining = 1.6 - (time.monotonic() - self.last_request)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self.last_request = time.monotonic()
        body = json.loads(request.content)
        messages = body.get("messages", [])
        self.requests.append(
            {
                "stream": body.get("stream"),
                "preserve_thinking": (body.get("chat_template_kwargs") or {}).get("preserve_thinking"),
                "reasoning_history": any(
                    m.get("role") == "assistant" and bool(m.get("reasoning_content")) for m in messages
                ),
                "tool_result": any(m.get("role") == "tool" for m in messages),
                "tool_result_count": sum(m.get("role") == "tool" for m in messages),
                "max_tool_calls": max((len(m.get("tool_calls") or []) for m in messages), default=0),
            }
        )

    def on_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.append(
            {
                "reasoning_tokens_present": isinstance((metadata.get("usage") or {}).get("reasoning_tokens"), int),
                "request_id_present": bool(metadata.get("request_id") or (metadata.get("headers") or {}).get("x-request-id")),
            }
        )

    def report(self, name: str, **fields: Any) -> None:
        print(json.dumps({"case": name, **fields}, ensure_ascii=False), flush=True)


async def main() -> None:
    if not os.environ.get("EXAONE_API_KEY"):
        raise SystemExit("Set EXAONE_API_KEY to run this opt-in live test")
    probe = Probe()

    @tool
    def get_test_value() -> str:
        """Return the fixed test value seven."""
        probe.tool_calls += 1
        return "7"

    parallel_calls: list[str] = []

    @tool
    def first_number() -> str:
        """Return the first fixed test number."""
        parallel_calls.append("first")
        return "3"

    @tool
    def second_number() -> str:
        """Return the second fixed test number."""
        parallel_calls.append("second")
        return "4"

    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    async with httpx.AsyncClient(verify=context, event_hooks={"request": [probe.on_request]}, timeout=120.0) as client:
        model = ExaoneModel(
            reasoning_effort="low", preserve_thinking=True, max_tokens=512, metadata_callback=probe.on_metadata,
            http_client=client,
        )
        messages = [{"role": "user", "content": [{"text": "Say a short greeting."}]}]
        stream_events = [event async for event in model.stream(messages)]
        stream_text = any("text" in event.get("contentBlockDelta", {}).get("delta", {}) for event in stream_events)
        stream_reasoning = any("reasoningContent" in event.get("contentBlockDelta", {}).get("delta", {}) for event in stream_events)
        if not (stream_text and stream_reasoning and any("messageStop" in event for event in stream_events)):
            raise AssertionError("Streaming text/reasoning/stop did not all arrive")
        probe.report("stream", text=stream_text, reasoning=stream_reasoning)

        model.update_config(stream=False)
        nonstream_events = [event async for event in model.stream(messages)]
        if not any("messageStop" in event for event in nonstream_events):
            raise AssertionError("Non-streaming completion did not finish")
        probe.report("nonstream", finished=True)
        model.update_config(stream=True)

        conversation = Agent(model=model, callback_handler=None)
        await conversation.invoke_async("Remember the code word amber. Reply with one short sentence.")
        await conversation.invoke_async("Repeat the code word I asked you to remember.")
        if not probe.requests[-1]["reasoning_history"] or not probe.requests[-1]["preserve_thinking"]:
            raise AssertionError("Assistant reasoning history was not forwarded")
        probe.report("reasoning_history", forwarded=True)

        tool_agent = Agent(model=model, tools=[get_test_value], callback_handler=None)
        await tool_agent.invoke_async(
            "First call the get_test_value tool, then report the returned value. Do not guess.",
            limits={"turns": 4},
        )
        if not probe.tool_calls or not any(item["tool_result"] for item in probe.requests):
            raise AssertionError("Strands tool invocation did not reach a follow-up request")
        probe.report("agent_tool_roundtrip", tool_calls=probe.tool_calls, tool_result_forwarded=True)

        parallel_model = ExaoneModel(
            reasoning_effort="low", preserve_thinking=True, max_tokens=512,
            params={"parallel_tool_calls": True}, http_client=client,
        )
        parallel_agent = Agent(model=parallel_model, tools=[first_number, second_number], callback_handler=None)
        await parallel_agent.invoke_async(
            "You must call both first_number and second_number before adding their returned values. Do not guess.",
            limits={"turns": 4},
        )
        if set(parallel_calls) != {"first", "second"} or not any(
            item["max_tool_calls"] >= 2 and item["tool_result_count"] >= 2 for item in probe.requests
        ):
            raise AssertionError("Parallel tool calls were not both forwarded")
        probe.report("parallel_tool_agent", both_tools_called=True, results_forwarded=2)

        events = [event async for event in model.structured_output(
            CountResult, [{"role": "user", "content": [{"text": "Return the value count as 3."}]}]
        )]
        outputs = [event.get("output") for event in events if "output" in event]
        if not outputs or outputs[-1].count != 3:
            raise AssertionError("Structured output did not validate the requested value")
        probe.report("structured_output", validated=True)

        if not any(item["reasoning_tokens_present"] for item in probe.metadata):
            raise AssertionError("Provider-specific reasoning token metadata was missing")
        probe.report("metadata", requests=len(probe.requests), reasoning_tokens=True,
                     request_ids=all(item["request_id_present"] for item in probe.metadata))


if __name__ == "__main__":
    asyncio.run(main())
