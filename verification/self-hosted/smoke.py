"""Opt-in, serial provider checks against an already-running Chat Completions server."""

import argparse
import asyncio
import base64
import functools
import importlib.metadata
import json
import os
import re
import sys

import httpx

IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAIAAABvFaqvAAAAIElEQVR4nGP4z8BAFUQdU0YNGjVo1KBRg0YNGjWIIgQABMo93z5bbNQAAAAASUVORK5CYII="
)
current_case = "setup"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=("strands", "langchain"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", required=True, choices=("on", "off"))
    parser.add_argument(
        "--server", default="unknown", choices=("vllm", "sglang", "unknown")
    )
    parser.add_argument("--server-version", default="unknown")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--reasoning-parser", default="unknown")
    parser.add_argument("--tool-parser", default="unknown")
    parser.add_argument(
        "--server-options", default="", help="Non-secret server settings to record"
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Check an EXAONE 4.5 image-capable endpoint",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


class Audit:
    def __init__(self):
        self.requests = []

    async def observe(self, request):
        body = json.loads(request.content)
        messages = body.get("messages", [])
        calls = {
            call.get("id")
            for m in messages
            if m.get("role") == "assistant"
            for call in m.get("tool_calls") or []
            if (call.get("function") or {}).get("name") == "lookup"
        }
        self.requests.append(
            {
                "stream": body.get("stream"),
                "reasoning_history": any(
                    m.get("role") == "assistant"
                    and bool(m.get("reasoning_content") or m.get("reasoning"))
                    for m in messages
                ),
                "tool_result": any(
                    m.get("role") == "tool"
                    and m.get("content") == "7"
                    and m.get("tool_call_id") in calls
                    for m in messages
                ),
            }
        )


def report(name):
    print(json.dumps({"case": name, "status": "passed"}), flush=True)


def check(condition):
    if not condition:
        raise AssertionError("smoke assertion failed")


async def strands(args, client, audit):
    global current_case
    from strands import Agent, tool
    from strands_exaone import ExaoneModel

    model = ExaoneModel(
        api_key=os.environ.get("SELF_HOSTED_API_KEY", "EMPTY"),
        model_id=args.model,
        base_url=args.base_url,
        enable_thinking=args.thinking == "on",
        max_tokens=args.max_tokens,
        include_usage=False,
        http_client=client,
    )
    user = [
        {"role": "user", "content": [{"text": "Reply with the single word READY."}]}
    ]
    for stream in (False, True):
        current_case = "text_stream" if stream else "text_nonstream"
        model.update_config(stream=stream)
        events = [event async for event in model.stream(user)]
        deltas = [
            event.get("contentBlockDelta", {}).get("delta", {}) for event in events
        ]
        check(any(d.get("text", "").strip() for d in deltas))
        check(
            any(
                event.get("messageStop", {}).get("stopReason") == "end_turn"
                for event in events
            )
        )
        check(audit.requests[-1]["stream"] is stream)
        reasoning_present = any(
            d.get("reasoningContent", {}).get("text") for d in deltas
        )
        check(bool(reasoning_present) == (args.thinking == "on"))
        report("text_stream" if stream else "text_nonstream")

    model.update_config(stream=True)
    current_case = "multiturn"
    conversation = Agent(model=model, callback_handler=None, retry_strategy=None)
    first = await conversation.invoke_async("Remember the word amber. Reply briefly.")
    check(first.stop_reason == "end_turn")
    answer = await conversation.invoke_async("Repeat the word I asked you to remember.")
    check(answer.stop_reason == "end_turn")
    check(
        "amber"
        in "".join(block.get("text", "") for block in answer.message["content"]).lower()
    )
    if args.thinking == "on":
        check(audit.requests[-1]["reasoning_history"])
    report("multiturn")

    used = []

    @tool
    def lookup(value: int) -> str:
        """Return a fixed test value."""
        used.append(value)
        return "7"

    for stream in (False, True):
        current_case = "tool_stream" if stream else "tool_nonstream"
        model.update_config(stream=stream)
        used.clear()
        agent = Agent(
            model=model, tools=[lookup], callback_handler=None, retry_strategy=None
        )
        start = len(audit.requests)
        result = await agent.invoke_async(
            "Call lookup with value 7, then report its result.", limits={"turns": 4}
        )
        check(result.stop_reason == "end_turn")
        check(used == [7] and any(r["tool_result"] for r in audit.requests[start:]))
        report("tool_stream" if stream else "tool_nonstream")

    if args.image:
        current_case = "image"
        image_agent = Agent(model=model, callback_handler=None, retry_strategy=None)
        result = await image_agent.invoke_async(
            [
                {"text": "What is the main color of this image? Answer in English."},
                {"image": {"format": "png", "source": {"bytes": IMAGE}}},
            ]
        )
        check(result.stop_reason == "end_turn")
        check(
            bool(
                re.search(
                    r"\bred\b",
                    "".join(
                        block.get("text", "") for block in result.message["content"]
                    ),
                    re.IGNORECASE,
                )
            )
        )
        report("image")


async def langchain(args, client, audit):
    global current_case
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_exaone import ChatEXAONE

    model = ChatEXAONE(
        api_key=os.environ.get("SELF_HOSTED_API_KEY", "EMPTY"),
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        model_kwargs={
            "chat_template_kwargs": {"enable_thinking": args.thinking == "on"}
        },
        http_async_client=client,
    )
    user = HumanMessage("Reply with the single word READY.")
    for stream in (False, True):
        current_case = "text_stream" if stream else "text_nonstream"
        if stream:
            chunks = [chunk async for chunk in model.astream([user])]
            check(bool(chunks))
            result = functools.reduce(lambda a, b: a + b, chunks)
        else:
            result = await model.ainvoke([user])
        check(bool(result.text.strip()) and audit.requests[-1]["stream"] is stream)
        check(result.response_metadata.get("finish_reason") == "stop")
        check(
            bool(result.additional_kwargs.get("reasoning_content"))
            == (args.thinking == "on")
        )
        report("text_stream" if stream else "text_nonstream")

    current_case = "multiturn"
    first_user = HumanMessage("Remember the word amber. Reply briefly.")
    first = await model.ainvoke([first_user])
    check(first.response_metadata.get("finish_reason") == "stop")
    answer = await model.ainvoke(
        [first_user, first, HumanMessage("Repeat the word I asked you to remember.")]
    )
    check(answer.response_metadata.get("finish_reason") == "stop")
    check("amber" in answer.text.lower())
    if args.thinking == "on":
        check(audit.requests[-1]["reasoning_history"])
    report("multiturn")

    @tool
    def lookup(value: int) -> str:
        """Return a fixed test value."""
        return "7"

    tool_user = HumanMessage("Call lookup with value 7, then report its result.")
    bound = model.bind_tools([lookup])
    for stream in (False, True):
        current_case = "tool_stream" if stream else "tool_nonstream"
        if stream:
            chunks = [chunk async for chunk in bound.astream([tool_user])]
            check(bool(chunks))
            call = functools.reduce(lambda a, b: a + b, chunks)
        else:
            call = await bound.ainvoke([tool_user])
        check(call.response_metadata.get("finish_reason") == "tool_calls")
        if stream:
            check(len(call.tool_call_chunks) == 1)
            check(json.loads(call.tool_call_chunks[0]["args"]) == {"value": 7})
        check(len(call.tool_calls) == 1 and call.tool_calls[0]["name"] == "lookup")
        check(call.tool_calls[0]["args"] == {"value": 7})
        tool_result = lookup.invoke(call.tool_calls[0]["args"])
        result = await model.ainvoke(
            [
                tool_user,
                call,
                ToolMessage(tool_result, tool_call_id=call.tool_calls[0]["id"]),
            ]
        )
        check(result.response_metadata.get("finish_reason") == "stop")
        check(audit.requests[-1]["tool_result"])
        report("tool_stream" if stream else "tool_nonstream")

    if args.image:
        current_case = "image"
        image = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What is the main color of this image? Answer in English.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(IMAGE).decode()
                    },
                },
            ]
        )
        result = await model.ainvoke([image])
        check(result.response_metadata.get("finish_reason") == "stop")
        check(bool(re.search(r"\bred\b", result.text, re.IGNORECASE)))
        report("image")


async def main():
    args = arguments()
    print(
        json.dumps(
            {
                "framework": args.framework,
                "language": "python",
                "server": args.server,
                "server_version": args.server_version,
                "model": args.model,
                "model_revision": args.model_revision,
                "reasoning_parser": args.reasoning_parser,
                "tool_parser": args.tool_parser,
                "thinking": args.thinking,
                "server_options": args.server_options,
                "provider_version": importlib.metadata.version(
                    f"{args.framework}-exaone"
                ),
                "sdk_version": importlib.metadata.version(
                    "strands-agents"
                    if args.framework == "strands"
                    else "langchain-core"
                ),
                "image_requested": args.image,
            }
        ),
        flush=True,
    )
    audit = Audit()
    async with httpx.AsyncClient(
        event_hooks={"request": [audit.observe]}, timeout=120
    ) as client:
        await (
            strands(args, client, audit)
            if args.framework == "strands"
            else langchain(args, client, audit)
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001 - do not print SDK exception text that may include response data.
        print(
            json.dumps(
                {
                    "case": current_case,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)
