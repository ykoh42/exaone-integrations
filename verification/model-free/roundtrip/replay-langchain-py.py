import functools, json, sys
from pathlib import Path
import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_exaone import ChatEXAONE

paths = sorted(Path(sys.argv[1]).glob("*.json"))
assert {p.name for p in paths} == {
    "k2-vllm.json",
    "k2-sglang.json",
    "45-vllm.json",
    "45-sglang.json",
}, "generate all four fixtures first"
for path in paths:
    f = json.loads(path.read_text())
    expected = f["expected"]
    for stream in (False, True):
        requests = []

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

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            model = ChatEXAONE(
                api_key="EMPTY",
                model="synthetic",
                base_url="http://server.example/v1",
                http_client=client,
                output_version="v1",
            )
            user = HumanMessage("Use lookup.")
            try:
                first = (
                    functools.reduce(lambda a, b: a + b, model.stream([user]))
                    if stream
                    else model.invoke([user])
                )
                assert len(first.tool_calls) == 1
                assert first.tool_calls[0]["name"] == "lookup"
                assert first.tool_calls[0]["args"] == expected, (
                    path.stem,
                    stream,
                    first.tool_calls,
                )
                model.invoke(
                    [
                        user,
                        first,
                        ToolMessage("synthetic result", tool_call_id="call_0"),
                    ]
                )
            except Exception as exc:
                raise RuntimeError(f"{path.stem} stream={stream}") from exc
        assert len(requests) == 2
        assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
        assert assistant.get("content") == "synthetic answer"
        assert assistant.get("reasoning_content") == "synthetic reasoning", (
            path.stem,
            stream,
            assistant,
        )
        assert any(
            m.get("tool_call_id") == "call_0" and m.get("content") == "synthetic result"
            for m in requests[1]["messages"]
        )
    print(
        path.stem
        + ": LangChain Python tool/reasoning round trip, stream + nonstream OK",
        flush=True,
    )
