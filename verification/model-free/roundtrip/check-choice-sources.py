"""Regenerate synthetic K-EXAONE 2.0 parser fixtures and compare them with CI snapshots.

Only use trusted checkouts: this probe executes selected code from them.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import difflib
import hashlib
import inspect
import json
import os
import runpy
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

REVISIONS = {
    "sglang": "918653d22b2f1ae853b81fc0f5539b7c9311487e",
    "vllm": "33fa995c084e1bf652b4a20130ed0f04d99c9381",
}
TOKENIZER_SHA256 = "b657dadb90bf2855dcd931c78bf04f32874c2ece3a14ed1bd398eef7f8f23d98"
FIXTURES = Path(__file__).with_name("choice-fixtures")
EXPECTED = {"text": 'hello 한글 "quoted" \\path'}
RAW_TOOL = (
    "<tool_call>\n<function=lookup>\n<parameter=text>\n"
    + EXPECTED["text"]
    + "\n</parameter>\n</function>\n</tool_call>"
)
NAMED = {"type": "function", "function": {"name": "lookup"}}


def check_source(server: str, root: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISIONS[server]:
        raise ValueError(f"Unexpected {server} revision: {revision}")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise ValueError(f"The {server} checkout must be clean")


def envelope(server: str, mode: str, arguments: str, frames: list[dict]) -> dict:
    finish = "stop" if server == "vllm" and mode == "named" else "tool_calls"
    message = {
        "role": "assistant",
        "content": "synthetic answer",
        (
            "reasoning" if server == "vllm" else "reasoning_content"
        ): "synthetic reasoning",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": arguments},
            }
        ],
    }
    return {
        "name": f"k2-{server}",
        "expected": EXPECTED,
        "nonstream": {
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]
        },
        "stream": frames
        + [{"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}],
    }


def extract_methods(
    path: Path, source_class: str, names: set[str], target: str, ns: dict, prefix=()
) -> None:
    tree = ast.parse(path.read_text())
    source = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == source_class
    )
    methods = [
        copy.deepcopy(node)
        for node in source.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    if {node.name for node in methods} != names:
        raise ValueError(f"Missing expected methods in {path}")
    cls = ast.ClassDef(
        name=target,
        bases=[],
        keywords=[],
        body=[*copy.deepcopy(prefix), *methods],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)  # noqa: S102


async def sglang(root: Path) -> dict[str, dict]:
    package = types.ModuleType("sglang")
    package.__path__ = [str(root / "python/sglang")]
    sys.modules["sglang"] = package
    from sglang.srt.entrypoints.openai import protocol
    from sglang.srt.function_call.json_array_parser import JsonArrayParser
    from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

    class UnexpectedJsonArray(JsonArrayParser):
        def __init__(self, *args, **kwargs):
            raise AssertionError("Expected the native structural-tag parser")

    ns = {
        **vars(protocol),
        "inspect": inspect,
        "json": json,
        "time": time,
        "JsonArrayParser": UnexpectedJsonArray,
        "Qwen3CoderDetector": Qwen3CoderDetector,
        "envs": SimpleNamespace(
            SGLANG_TOOL_STRICT_LEVEL=SimpleNamespace(get=lambda: 0)
        ),
    }
    mapping = ast.Assign(
        targets=[ast.Name(id="ToolCallParserEnum", ctx=ast.Store())],
        value=ast.Dict(
            keys=[ast.Constant("qwen3_coder")],
            values=[ast.Name(id="Qwen3CoderDetector", ctx=ast.Load())],
        ),
    )
    extract_methods(
        root / "python/sglang/srt/function_call/function_call_parser.py",
        "FunctionCallParser",
        {"__init__", "has_tool_call", "parse_non_stream", "parse_stream_chunk"},
        "FunctionCallParser",
        ns,
        [mapping],
    )
    extract_methods(
        root / "python/sglang/srt/entrypoints/openai/serving_chat.py",
        "OpenAIServingChat",
        {"_process_tool_call_stream", "_process_tool_calls"},
        "Serving",
        ns,
    )

    class Server(ns["Serving"]):
        def __init__(self):
            self.tool_call_parser = "qwen3_coder"
            self.tokenizer_manager = SimpleNamespace(tokenizer=None)

        def _effective_tools(self, request):
            return request.tools

        def _process_tool_call_id(self, call, history_count):
            return "call_0"

        def _get_history_tool_calls_cnt(self, request):
            return 0

    result = {}
    for mode in ("named", "required"):
        request = protocol.ChatCompletionRequest.model_validate(
            {
                "model": "synthetic",
                "messages": [{"role": "user", "content": "before"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": NAMED if mode == "named" else "required",
            }
        )
        server = Server()
        full = server._process_tool_calls(
            RAW_TOOL,
            request.tools,
            {"type": "stop", "matched": None},
            request.tool_choice,
        )
        if not full.tool_calls or len(full.tool_calls) != 1:
            raise AssertionError(f"{mode}: expected one nonstream tool call")
        call = full.tool_calls[0]
        if (
            call.function.name != "lookup"
            or json.loads(call.function.arguments) != EXPECTED
        ):
            raise AssertionError(f"{mode}: unexpected nonstream tool call")
        if full.finish_reason["type"] != "tool_calls":
            raise AssertionError(f"{mode}: unexpected finish reason")
        parsers, has_tools, frames = {}, {}, []
        for offset in range(0, len(RAW_TOOL), 3):
            async for frame in server._process_tool_call_stream(
                0,
                RAW_TOOL[offset : offset + 3],
                parsers,
                {"meta_info": {"id": "synthetic"}},
                request,
                has_tools,
            ):
                chunk = json.loads(frame.removeprefix("data: ").strip())
                chunk["created"] = 0
                frames.append(chunk)
        if not parsers[0].detector.supports_structural_tag() or not has_tools.get(0):
            raise AssertionError(f"{mode}: expected structural-tag tool parsing")
        result[mode] = envelope("sglang", mode, call.function.arguments, frames)
    return result


def vllm(root: Path, tokenizer_path: Path) -> dict[str, dict]:
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer does not match the pinned model revision")
    from transformers import PreTrainedTokenizerFast

    os.environ["VLLM_PATH"] = str(root)
    os.environ["VLLM_MODERN"] = "0"
    parser_class = runpy.run_path(str(Path(__file__).with_name("vllm_qwen.py")))[
        "Qwen3Parser"
    ]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tools = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )
        )
    ]
    raw = "synthetic reasoning</think>synthetic answer" + RAW_TOOL
    result = {}
    for mode in ("named", "required"):
        choice = (
            SimpleNamespace(type="function", function=SimpleNamespace(name="lookup"))
            if mode == "named"
            else "required"
        )
        request = SimpleNamespace(
            tools=tools, tool_choice=choice, include_reasoning=True, messages=[]
        )
        parser = parser_class(
            tokenizer, tools, chat_template_kwargs={"enable_thinking": True}
        )
        reason, content, calls = parser.parse(raw, request, enable_auto_tools=True)
        if reason != "synthetic reasoning" or content != "synthetic answer":
            raise AssertionError(f"{mode}: unexpected reasoning or content")
        if (
            len(calls) != 1
            or calls[0].name != "lookup"
            or json.loads(calls[0].arguments) != EXPECTED
        ):
            raise AssertionError(f"{mode}: unexpected nonstream tool call")
        parser = parser_class(
            tokenizer, tools, chat_template_kwargs={"enable_thinking": True}
        )
        frames = []
        for offset in range(0, len(raw), 3):
            part = parser.parse_delta(
                raw[offset : offset + 3], [], request, finished=offset + 3 >= len(raw)
            )
            if not part:
                continue
            delta = {}
            if part.reasoning:
                delta["reasoning_content"] = part.reasoning
            if part.content:
                delta["content"] = part.content
            if part.tool_calls:
                delta["tool_calls"] = [
                    item.model_dump(exclude_none=True) for item in part.tool_calls
                ]
            if delta:
                frames.append(
                    {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                )
        result[mode] = envelope("vllm", mode, calls[0].arguments, frames)
    return result


def compare(server: str, generated: dict[str, dict]) -> None:
    for mode, actual in generated.items():
        path = FIXTURES / f"k2-{server}-{mode}.json"
        expected = json.loads(path.read_text())
        if actual != expected:
            before = json.dumps(expected, ensure_ascii=False, indent=2).splitlines()
            after = json.dumps(actual, ensure_ascii=False, indent=2).splitlines()
            diff = "\n".join(
                difflib.unified_diff(
                    before, after, fromfile=str(path), tofile="regenerated", lineterm=""
                )
            )
            raise AssertionError(f"{path.name} does not match:\n{diff}")
        print(f"{path.name}: regenerated output matches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", choices=REVISIONS)
    parser.add_argument("checkout", type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, help="Pinned K-EXAONE 2.0 tokenizer (vLLM only)"
    )
    args = parser.parse_args()
    root = args.checkout.resolve()
    check_source(args.server, root)
    if args.server == "sglang":
        generated = asyncio.run(sglang(root))
    else:
        if args.tokenizer is None:
            parser.error("--tokenizer is required for vLLM")
        generated = vllm(root, args.tokenizer)
    compare(args.server, generated)


if __name__ == "__main__":
    main()
