"""Integration tests using the Strands harness and a mock EXAONE transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from harness_support import chunk, events, response
from strands_harness import create_harness

from strands_exaone import ExaoneModel


async def test_harness_builtin_tools_with_synthetic_responses(tmp_path: Any) -> None:
    """Exercise local built-ins through the real harness with mock model output."""
    import shlex

    path = tmp_path / "sample.txt"
    requests: list[dict[str, Any]] = []
    steps = [
        ("write", {"path": str(path), "content": "alpha\nneedle original\n"}),
        ("read", {"path": str(path)}),
        ("edit", {"path": str(path), "old_str": "needle original", "new_str": "needle revised"}),
        ("shell", {"command": f"grep -n needle {shlex.quote(str(path))}"}),
        ("todo_write", {"todos": [{"content": "Check file", "activeForm": "Checking file", "status": "in_progress"}]}),
        ("programmatic_tool_caller", {"code": f"result = await read(path={str(path)!r})\nprint(result)"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        index = len(requests) - 1
        if index == len(steps):
            return response(events(chunk({"content": "done"}), chunk(finish="stop")))
        name, arguments = steps[index]
        call = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{index}",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ]
        }
        return response(events(chunk({"reasoning_content": f"step {index}"}), chunk(call), chunk(finish="tool_calls")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        agent = create_harness(
            model=model,
            builtin_tools=["read", "write", "edit", "shell", "programmatic_tool_caller"],
            builtin_plugins=["todos"],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Update and inspect the local file.")

    assert path.read_text() == "alpha\nneedle revised\n"
    assert len(requests) == len(steps) + 1
    messages = requests[-1]["messages"]
    results = {message["tool_call_id"]: message["content"] for message in messages if message["role"] == "tool"}
    assert set(results) == {f"call_{index}" for index in range(len(steps))}
    assert "needle original" in str(results["call_1"])
    assert "needle revised" in str(results["call_3"])
    assert "needle revised" in str(results["call_5"])
    assert [message["reasoning_content"] for message in messages if message.get("reasoning_content")] == [
        f"step {index}" for index in range(len(steps))
    ]


async def test_harness_read_image_as_text_with_media_disabled(tmp_path: Any) -> None:
    """The text-only model can receive the harness's description of a file."""
    import base64

    path = tmp_path / "pixel.png"
    path.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6n8AAAAASUVORK5CYII=")
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "read_1",
                        "function": {"name": "read", "arguments": json.dumps({"path": str(path)})},
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "I cannot inspect the image."}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools={"*": False, "read": {"media": False}},
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Read the image.")
    assert len(requests) == 2
    result = next(message for message in requests[-1]["messages"] if message.get("tool_call_id") == "read_1")
    assert isinstance(result["content"], str)
    assert result["content"]


@pytest.mark.parametrize("approved", [True, False])
async def test_harness_tool_intervention_preserves_reasoning(tmp_path: Any, approved: bool) -> None:
    import shlex

    from strands.vended_interventions.hitl import HumanInTheLoop

    marker = tmp_path / "marker"
    decisions: list[bool] = []
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": f"touch {shlex.quote(str(marker))}"}),
                        },
                    }
                ]
            }
            return response(events(chunk({"reasoning_content": "synthetic"}), chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    def ask(prompt: str, **kwargs: Any) -> str:
        decisions.append(approved)
        return "yes" if approved else "no"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["shell"],
            builtin_plugins=[],
            interventions=HumanInTheLoop(ask=ask),
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Run the local test.")
    assert decisions == [approved]
    assert marker.exists() is approved
    assert len(requests) == 2
    messages = requests[-1]["messages"]
    assert any(message.get("reasoning_content") == "synthetic" for message in messages)
    assert any(message.get("tool_call_id") == "call_1" for message in messages)


async def test_harness_subagent_uses_same_mock_provider() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "delegate_1",
                        "function": {
                            "name": "subagent",
                            "arguments": json.dumps({"task": "Return CHILD_SYNTHETIC.", "tools": []}),
                        },
                    }
                ]
            }
            return response(
                events(chunk({"reasoning_content": "delegate task"}), chunk(call), chunk(finish="tool_calls"))
            )
        if any(
            "CHILD_SYNTHETIC" in str(message.get("content", ""))
            for message in requests[-1]["messages"]
            if message["role"] == "user"
        ):
            return response(
                events(
                    chunk({"reasoning_content": "CHILD_REASONING"}),
                    chunk({"content": "CHILD_SYNTHETIC"}),
                    chunk(finish="stop"),
                )
            )
        return response(events(chunk({"content": "Parent finished."}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["subagent"],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Delegate the test task.")
    assert len(requests) >= 3
    assert any("CHILD_SYNTHETIC" in str(request["messages"]) for request in requests[1:])
    assert any(
        "CHILD_SYNTHETIC" in str(message.get("content", ""))
        for request in requests[1:]
        for message in request["messages"]
        if message.get("role") == "tool"
    )
    assert all(
        "CHILD_REASONING" not in str(message.get("content", ""))
        for request in requests[1:]
        for message in request["messages"]
        if message.get("role") == "tool"
    )
    assert any(
        message.get("reasoning_content") == "delegate task"
        for request in requests[1:]
        for message in request["messages"]
    )


async def test_harness_web_fetch_with_synthetic_page_and_summarizer(monkeypatch: Any) -> None:
    import importlib

    fetch_module = importlib.import_module("strands_harness.tools.web_fetch")
    requests: list[dict[str, Any]] = []

    async def mock_fetch(url: str) -> tuple[str, str]:
        assert url == "https://example.com/synthetic"
        return url, "The SYNTHETIC_PAGE value is blue."

    monkeypatch.setattr(fetch_module, "_fetch_direct", mock_fetch)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "fetch_1",
                        "function": {
                            "name": "web_fetch",
                            "arguments": json.dumps(
                                {"url": "https://example.com/synthetic", "prompt": "What is the value?"}
                            ),
                        },
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        if len(requests) == 2:
            return response(
                events(
                    chunk({"reasoning_content": "SUMMARIZER_REASONING"}),
                    chunk({"content": "The value is blue."}),
                    chunk(finish="stop"),
                )
            )
        return response(events(chunk({"content": "blue"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools={"*": False, "web_fetch": {"transport": "direct"}},
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Check the synthetic page.")
    assert len(requests) == 3
    assert "SYNTHETIC_PAGE" in str(requests[1]["messages"])
    result = next(message for message in requests[2]["messages"] if message.get("tool_call_id") == "fetch_1")
    assert "The value is blue." in result["content"]
    assert "SYNTHETIC_PAGE" not in result["content"]
    assert "SUMMARIZER_REASONING" not in result["content"]


async def test_harness_default_environment_and_tools_with_mock_model(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Follow the synthetic ENVIRONMENT_MARKER instruction.\n")
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = "ok" if len(requests) == 1 else "[]"
        return response(events(chunk({"content": content}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(model=ExaoneModel(api_key="test-only-key", http_client=client), callback_handler=None)
        await agent.invoke_async("Say hello.")
        assert agent.memory_manager is not None
        await agent.memory_manager.flush()
    assert "ENVIRONMENT_MARKER" in str(requests[0]["messages"])
    names = {spec["function"]["name"] for spec in requests[0]["tools"]}
    assert {
        "shell",
        "read",
        "write",
        "edit",
        "web_fetch",
        "programmatic_tool_caller",
        "subagent",
        "todo_write",
    } <= names
    assert "web_search" not in names


async def test_harness_local_mcp_tool_round_trip(tmp_path: Any) -> None:
    import sys

    server = tmp_path / "mock_mcp_server.py"
    server.write_text(
        "from mcp.server.mcpserver import MCPServer\n"
        "server = MCPServer('test')\n"
        "@server.tool()\n"
        "def add_numbers(a: int, b: int) -> str:\n"
        "    return str(a + b)\n"
        "if __name__ == '__main__':\n"
        "    server.run('stdio')\n"
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            names = {spec["function"]["name"] for spec in requests[0]["tools"]}
            assert "local_add_numbers" in names
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "local_add_numbers", "arguments": '{"a":3,"b":4}'},
                    }
                ]
            }
            return response(events(chunk({"reasoning_content": "synthetic"}), chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            mcp_servers={"local": {"command": sys.executable, "args": [str(server)]}},
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Add two test values.")
    assert len(requests) == 2
    messages = requests[1]["messages"]
    assert any(message.get("reasoning_content") == "synthetic" for message in messages)
    assert any(message.get("tool_call_id") == "call_1" and "7" in message["content"] for message in messages)


async def test_harness_multifile_unicode_paths_recovers_from_missing_file(tmp_path: Any) -> None:
    """Exercise synthetic file work through the real harness, including a failed read."""
    import shlex

    workspace = tmp_path / "한글 작업 공간"
    workspace.mkdir()
    first, second = workspace / "파일 하나.txt", workspace / "파일 둘.txt"
    requests: list[dict[str, Any]] = []
    steps = [
        [("missing", "read", {"path": str(first)})],
        [
            ("write_a", "write", {"path": str(first), "content": "찾을 값: 이전 하나\n"}),
            ("write_b", "write", {"path": str(second), "content": "찾을 값: 이전 둘\n"}),
        ],
        [
            ("edit_a", "edit", {"path": str(first), "old_str": "이전 하나", "new_str": "완료 하나"}),
            ("edit_b", "edit", {"path": str(second), "old_str": "이전 둘", "new_str": "완료 둘"}),
        ],
        [("search", "shell", {"command": f"grep -n '찾을 값' {shlex.quote(str(first))} {shlex.quote(str(second))}"})],
        [
            (
                "program",
                "programmatic_tool_caller",
                {
                    "code": "\n".join(
                        [
                            f"a = await read(path={str(first)!r})",
                            f"b = await read(path={str(second)!r})",
                            "print(a)",
                            "print(b)",
                        ]
                    )
                },
            )
        ],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        index = len(requests) - 1
        if index == len(steps):
            return response(events(chunk({"content": "done"}), chunk(finish="stop")))
        calls = [
            {
                "index": pos,
                "id": ident,
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
            for pos, (ident, name, arguments) in enumerate(steps[index])
        ]
        return response(
            events(
                chunk({"reasoning_content": f"step {index}"}), chunk({"tool_calls": calls}), chunk(finish="tool_calls")
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["read", "write", "edit", "shell", "programmatic_tool_caller"],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Perform the synthetic local file task.")
    assert first.read_text() == "찾을 값: 완료 하나\n"
    assert second.read_text() == "찾을 값: 완료 둘\n"
    assert len(requests) == len(steps) + 1
    results = {
        message["tool_call_id"]: message["content"] for message in requests[-1]["messages"] if message["role"] == "tool"
    }
    assert set(results) == {ident for step in steps for ident, _, _ in step}
    assert any(word in results["missing"].lower() for word in ("not found", "no such file", "does not exist"))
    assert "완료 하나" in results["search"] and "완료 둘" in results["search"]
    assert "완료 하나" in results["program"] and "완료 둘" in results["program"]


async def test_harness_large_file_read_and_offloaded_search(tmp_path: Any) -> None:
    """Exercise synthetic large tool arguments, offload, and retrieval."""
    import re

    path = tmp_path / "큰 문서.txt"
    text = (
        "\n".join(
            f"행 {index:04d} " + "가" * 80 + (" TARGET_SYNTHETIC" if index == 200 else "") for index in range(400)
        )
        + "\n"
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        match len(requests):
            case 1:
                ident, name, arguments = "write_1", "write", {"path": str(path), "content": text}
            case 2:
                ident, name, arguments = "read_1", "read", {"path": str(path)}
            case 3:
                result = next(
                    message for message in requests[-1]["messages"] if message.get("tool_call_id") == "read_1"
                )
                match = re.search(r"\[Stored references:\]\s+(\S+)\s+\(text,", result["content"])
                assert match is not None
                ident, name, arguments = (
                    "retrieve_1",
                    "retrieve_offloaded_content",
                    {"reference": match.group(1), "pattern": "TARGET_SYNTHETIC", "context_lines": 0},
                )
            case 4:
                return response(events(chunk({"content": "done"}), chunk(finish="stop")))
            case _:
                raise AssertionError("Unexpected mock request")
        call = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": ident,
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ]
        }
        return response(events(chunk(call), chunk(finish="tool_calls")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["read", "write"],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Write, inspect, and search a large synthetic file.")
    assert path.read_text() == text
    assert len(requests) == 4
    results = {
        message["tool_call_id"]: message["content"] for message in requests[-1]["messages"] if message["role"] == "tool"
    }
    assert "offloaded" in results["read_1"].lower()
    assert "TARGET_SYNTHETIC" in results["retrieve_1"]
    assert "행 0200" in results["retrieve_1"]


async def test_harness_parallel_background_shell_and_file_write(tmp_path: Any) -> None:
    """Exercise built-in foreground and background tools in one synthetic turn."""
    import shlex

    shell_path = tmp_path / "백그라운드 결과.txt"
    write_path = tmp_path / "일반 결과.txt"
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            calls = [
                {
                    "index": 0,
                    "id": "shell_1",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps(
                            {
                                "command": f"printf '%s' 'BACKGROUND_SYNTHETIC' > {shlex.quote(str(shell_path))}",
                                "_background_execution": True,
                            }
                        ),
                    },
                },
                {
                    "index": 1,
                    "id": "write_1",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps(
                            {"path": str(write_path), "content": "FOREGROUND_SYNTHETIC"}, ensure_ascii=False
                        ),
                    },
                },
            ]
            return response(
                events(
                    chunk({"reasoning_content": "run both"}), chunk({"tool_calls": calls}), chunk(finish="tool_calls")
                )
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["shell", "write"],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Run two local synthetic file tasks.")
    assert shell_path.read_text() == "BACKGROUND_SYNTHETIC"
    assert write_path.read_text() == "FOREGROUND_SYNTHETIC"
    history = [message for request in requests[1:] for message in request["messages"]]
    assert any(message.get("tool_call_id") == "write_1" for message in history)
    assert any(
        message.get("tool_call_id") == "shell_1" and "background" in message["content"].lower() for message in history
    )
    assert any(message.get("reasoning_content") == "run both" for message in history)


async def test_harness_programmatic_tool_recovers_from_read_failure(tmp_path: Any) -> None:
    """Recover from a failed nested tool and continue with a valid file."""
    existing = tmp_path / "있는 파일.txt"
    missing = tmp_path / "없는 파일.txt"
    existing.write_text("VALID_SYNTHETIC_VALUE\n")
    requests: list[dict[str, Any]] = []
    program = (
        f"try:\n    await read(path={str(missing)!r})\n"
        "except RuntimeError:\n    print('RECOVERED_SYNTHETIC')\n"
        f"print(await read(path={str(existing)!r}))"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "program_1",
                        "function": {
                            "name": "programmatic_tool_caller",
                            "arguments": json.dumps({"code": program}, ensure_ascii=False),
                        },
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            builtin_tools=["read", "programmatic_tool_caller"],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        await agent.invoke_async("Inspect synthetic files and recover from missing data.")
    assert len(requests) == 2
    result = next(message for message in requests[-1]["messages"] if message.get("tool_call_id") == "program_1")
    assert "RECOVERED_SYNTHETIC" in result["content"]
    assert "VALID_SYNTHETIC_VALUE" in result["content"]
