"""Exercise the smoke checks against a local, deterministic HTTP fixture."""

import argparse
import base64
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    fault = ""

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        thinking = body.get("chat_template_kwargs", {}).get("enable_thinking", False)
        stream = body.get("stream", False)
        user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        content = user.get("content", "")
        text = (
            content
            if isinstance(content, str)
            else " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        )
        if "main color" in text:
            assert [p.get("type") for p in content] == ["text", "image_url"]
            url = content[1]["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            assert base64.b64decode(url.split(",", 1)[1]) == base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAIAAABvFaqvAAAAIElEQVR4nGP4z8BAFUQdU0YNGjVo1KBRg0YNGjWIIgQABMo93z5bbNQAAAAASUVORK5CYII="
            )
            answer = "red"
        elif "Repeat the word" in text:
            answer = "amber"
        elif any(m.get("role") == "tool" for m in messages):
            answer = "7"
        else:
            answer = "READY"
        calls = None
        if body.get("tools") and not any(m.get("role") == "tool" for m in messages):
            value = 8 if self.fault == "tool" else 7
            arguments = json.dumps({"value": value})
            if stream and self.fault == "json-stream":
                arguments = '{"value": 7'
            calls = [
                {
                    "id": "call_fixture",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": arguments,
                    },
                }
            ]
        reasoning = (
            "Synthetic reasoning."
            if (
                (thinking and self.fault != "reasoning")
                or self.fault == "unexpected-reasoning"
            )
            else None
        )
        finish = "tool_calls" if calls else "stop"
        if calls and (
            (self.fault == "length-stream" and stream)
            or (self.fault == "length-nonstream" and not stream)
        ):
            finish = "length"
        message = {"role": "assistant", "content": None if calls else answer}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        if stream:
            deltas = [{"role": "assistant"}]
            if reasoning:
                deltas.append({"reasoning_content": reasoning})
            if calls:
                arguments = calls[0]["function"]["arguments"]
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": calls[0]["id"],
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": arguments[:3],
                                },
                            }
                        ]
                    }
                )
                deltas.extend(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": arguments[i : i + 3]},
                            }
                        ]
                    }
                    for i in range(3, len(arguments), 3)
                )
            else:
                deltas.append({"content": answer})
            events = [
                {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                for d in deltas
            ]
            events.append(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            )
            data = (
                "".join("data: " + json.dumps(e) + "\n\n" for e in events)
                + "data: [DONE]\n\n"
            ).encode()
            content_type = "text/event-stream"
        else:
            data = json.dumps(
                {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}
            ).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strands-python", required=True)
    parser.add_argument("--langchain-python", required=True)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    common = [
        "--base-url",
        base,
        "--model",
        "synthetic",
        "--server",
        "vllm",
        "--server-version",
        "fixture",
        "--model-revision",
        "fixture",
        "--reasoning-parser",
        "fixture",
        "--tool-parser",
        "fixture",
        "--image",
    ]
    env = {**os.environ, "SELF_HOSTED_API_KEY": "EMPTY"}
    try:
        for language, framework, executable, script in [
            ("python", "strands", args.strands_python, "smoke.py"),
            ("python", "langchain", args.langchain_python, "smoke.py"),
            ("typescript", "strands", args.node, "smoke.mjs"),
            ("typescript", "langchain", args.node, "smoke.mjs"),
        ]:
            if language == "python":
                source = root.parents[1] / "packages" / f"{framework}-exaone-py" / "src"
                env["PYTHONPATH"] = str(source)
            else:
                env.pop("PYTHONPATH", None)
            for thinking, fault in [
                ("on", ""),
                ("off", ""),
                ("on", "reasoning"),
                ("off", "unexpected-reasoning"),
                ("on", "tool"),
                ("on", "length-nonstream"),
                ("on", "length-stream"),
                ("on", "json-stream"),
            ]:
                Handler.fault = fault
                result = subprocess.run(
                    [
                        executable,
                        str(root / script),
                        "--framework",
                        framework,
                        "--thinking",
                        thinking,
                        *common,
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                expected = bool(fault)
                if (result.returncode != 0) != expected:
                    raise AssertionError(
                        f"{language} {framework} thinking={thinking} fault={fault or 'none'}: "
                        f"{result.stdout} {result.stderr}"
                    )
                if not fault:
                    rows = [json.loads(line) for line in result.stdout.splitlines()]
                    cases = {r.get("case") for r in rows}
                    assert {
                        "text_stream",
                        "text_nonstream",
                        "multiturn",
                        "tool_stream",
                        "tool_nonstream",
                        "image",
                    } <= cases
                else:
                    error = json.loads(result.stderr.splitlines()[-1])
                    assert error["status"] == "failed"
                    expected_case = {
                        "reasoning": "text_nonstream",
                        "unexpected-reasoning": "text_nonstream",
                        "tool": "tool_nonstream",
                        "length-nonstream": "tool_nonstream",
                        "length-stream": "tool_stream",
                        "json-stream": "tool_stream",
                    }[fault]
                    assert error["case"] == expected_case
                print(
                    f"{language} {framework} thinking={thinking} fault={fault or 'none'}: OK",
                    flush=True,
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
