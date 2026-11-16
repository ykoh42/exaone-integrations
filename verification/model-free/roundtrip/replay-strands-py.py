import asyncio, json, sys
from pathlib import Path
import httpx
from strands import Agent, tool
from strands_exaone import ExaoneModel

paths = sorted(Path(sys.argv[1]).glob("*.json"))
assert {p.name for p in paths} == {
    "k2-vllm.json",
    "k2-sglang.json",
    "45-vllm.json",
    "45-sglang.json",
}, "generate all four fixtures first"


async def main():
    for path in paths:
        f = json.loads(path.read_text())
        expected = f["expected"]
        for stream in (False, True):
            requests = []
            used = []

            @tool
            def lookup(
                text: str,
                count: int,
                ratio: float,
                enabled: bool,
                nested: dict,
                values: list,
            ) -> str:
                """Use synthetic arguments."""
                used.append(
                    dict(
                        text=text,
                        count=count,
                        ratio=ratio,
                        enabled=enabled,
                        nested=nested,
                        values=values,
                    )
                )
                return "synthetic result"

            def handler(request):
                requests.append(json.loads(request.content))
                if len(requests) == 1:
                    if stream:
                        raw = (
                            "".join(
                                "data: " + json.dumps(e, ensure_ascii=False) + "\n\n"
                                for e in f["stream"]
                            )
                            + "data: [DONE]\n\n"
                        )
                        return httpx.Response(200, content=raw.encode())
                    return httpx.Response(200, json=f["nonstream"])
                if stream:
                    return httpx.Response(
                        200,
                        content=b'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                    )
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "index": 0,
                                "message": {"content": "done"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                model = ExaoneModel(
                    api_key="EMPTY",
                    model_id="synthetic",
                    base_url="http://server.example/v1",
                    http_client=client,
                    stream=stream,
                )
                agent = Agent(
                    model=model,
                    tools=[lookup],
                    callback_handler=None,
                    retry_strategy=None,
                )
                try:
                    await agent.invoke_async("Use lookup.", limits={"turns": 3})
                except Exception as exc:
                    raise RuntimeError(f"{path.stem} stream={stream}") from exc
            assert used == [expected], (path.stem, stream, used)
            assert len(requests) == 2, (path.stem, stream, len(requests))
            assistant = next(
                m for m in requests[1]["messages"] if m["role"] == "assistant"
            )
            assert assistant.get("content") == "synthetic answer"
            assert assistant.get("reasoning_content") == "synthetic reasoning", (
                path.stem,
                stream,
                assistant,
            )
            assert any(
                m.get("tool_call_id") == "call_0"
                and m.get("content") == "synthetic result"
                for m in requests[1]["messages"]
            ), (path.stem, stream)
        print(
            path.stem
            + ": Strands Python SDK tool/reasoning round trip, stream + nonstream OK",
            flush=True,
        )


asyncio.run(main())
