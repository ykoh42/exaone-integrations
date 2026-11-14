"""Offline checks for the package's Harness example."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from harness_support import chunk, events, response

from strands_exaone import ExaoneModel

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "harness" / "agent.py"


@pytest.mark.parametrize("web_search", [False, True])
async def test_harness_example_explicitly_enables_file_read_and_opt_in_search(tmp_path: Path, web_search: bool) -> None:
    path = tmp_path / "합성 파일.txt"
    path.write_text("SYNTHETIC_EXAMPLE_CONTENT\n")
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "read_1",
                        "function": {"name": "read", "arguments": json.dumps({"path": str(path)}, ensure_ascii=False)},
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    module = runpy.run_path(str(EXAMPLE))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = module["make_agent"](model=model, allow_file_read=True, web_search=web_search)
        assert agent.model is model
        await agent.invoke_async("Read the synthetic file.")
    assert len(requests) == 2
    names = {spec["function"]["name"] for spec in requests[0]["tools"]}
    assert "read" in names
    assert ("web_search" in names) is web_search
    assert not {"shell", "write", "edit"}.intersection(names)
    tool_result = next(message for message in requests[1]["messages"] if message.get("tool_call_id") == "read_1")
    assert "SYNTHETIC_EXAMPLE_CONTENT" in tool_result["content"]


async def test_harness_example_does_not_advertise_or_read_file_by_default(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-default-disabled.txt"
    path.write_text("SYNTHETIC_DEFAULT_FILE_CONTENT\n")
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        names = {spec["function"]["name"] for spec in payload["tools"]}
        assert "read" not in names
        return response(events(chunk({"content": "File reading is disabled by default."}), chunk(finish="stop")))

    module = runpy.run_path(str(EXAMPLE))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = module["make_agent"](model=model)
        await agent.invoke_async(f"Read this file: {path}")

    assert len(requests) == 1
    assert "SYNTHETIC_DEFAULT_FILE_CONTENT" not in json.dumps(requests)
    assert "read" not in {spec["function"]["name"] for spec in requests[0]["tools"]}


def test_harness_example_help_needs_no_credentials() -> None:
    result = subprocess.run([sys.executable, str(EXAMPLE), "--help"], check=True, capture_output=True, text=True)
    assert "--web-search" in result.stdout
    assert "--allow-file-read" in result.stdout


def test_harness_example_uses_default_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAONE_BASE_URL", "https://ignored.example/v1")
    monkeypatch.setenv("EXAONE_API_KEY", "test-only-key")
    module = runpy.run_path(str(EXAMPLE))
    agent = module["make_agent"]()
    assert agent.model.get_config()["base_url"] == "https://api.exaone.ai/v1"


def test_harness_example_uses_local_compatible_endpoint(tmp_path: Path) -> None:
    """Run the custom endpoint example against a local synthetic SSE server."""
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    path = tmp_path / "테스트 파일.txt"
    path.write_text("LOCAL_SYNTHETIC_CONTENT\n")
    requests: list[dict[str, Any]] = []
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            paths.append(self.path)
            requests.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            if len(requests) == 1:
                call = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "read_1",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps({"path": str(path)}, ensure_ascii=False),
                            },
                        }
                    ]
                }
                payload = events(chunk(call), chunk(finish="tool_calls"))
            else:
                payload = events(chunk({"content": "LOCAL_FINAL_ANSWER"}), chunk(finish="stop"))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env["EXAONE_API_KEY"] = "synthetic-test-key"
        env["EXAONE_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
        env["NO_PROXY"] = "127.0.0.1,localhost"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; from agent import make_agent; from strands_exaone import ExaoneModel; "
                "model = ExaoneModel(base_url=os.environ['EXAONE_BASE_URL']); "
                "print(make_agent(model=model, allow_file_read=True)('Read the synthetic file.'))",
            ],
            cwd=EXAMPLE.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result.stdout.strip() == "LOCAL_FINAL_ANSWER"
    assert paths == ["/v1/chat/completions", "/v1/chat/completions"]
    assert len(requests) == 2
    tool_result = next(message for message in requests[1]["messages"] if message.get("tool_call_id") == "read_1")
    assert "LOCAL_SYNTHETIC_CONTENT" in tool_result["content"]
