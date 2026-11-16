import ast
import copy
import json
import os
from json import JSONDecodeError
from pathlib import Path

import regex


class Item:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, **_):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ToolParser:
    def __init__(self, tokenizer, tools=None):
        self.model_tokenizer = tokenizer
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []


class Logger:
    def error(self, *a, **k):
        raise RuntimeError("parser error")

    def exception(self, *a, **k):
        raise RuntimeError("parser error")


namespace = {
    "re": regex,
    "json": json,
    "JSONDecodeError": JSONDecodeError,
    "ToolParser": ToolParser,
    "logger": Logger(),
    "is_mistral_tokenizer": lambda _: False,
    "make_tool_call_id": lambda: "synthetic-id",
    "DeltaFunctionCall": Item,
    "DeltaMessage": Item,
    "DeltaToolCall": Item,
    "ExtractedToolCallInformation": Item,
    "FunctionCall": Item,
    "ToolCall": Item,
}


def load(path, names):
    tree = ast.parse(Path(path).read_text())
    nodes = [
        copy.deepcopy(n)
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)  # noqa: S102 -- run the pinned parser


load(
    os.path.join(os.environ["VLLM_PATH"], "vllm/tool_parsers/utils.py"),
    {"is_complete_json", "partial_tag_overlap"},
)
load(
    os.path.join(os.environ["VLLM_PATH"], "vllm/tool_parsers/hermes_tool_parser.py"),
    {"Hermes2ProToolParser"},
)
Parser = namespace["Hermes2ProToolParser"]
for case in (
    {"text": "hello", "count": 23, "nested": {"ok": True}},
    {"text": "", "count": 0, "nested": {}},
):
    raw = (
        "<tool_call>"
        + json.dumps(
            {"name": "lookup", "arguments": case},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "</tool_call>"
    )
    parsed = Parser(object()).extract_tool_calls(raw, None)
    assert (
        parsed.tools_called
        and len(parsed.tool_calls) == 1
        and json.loads(parsed.tool_calls[0].function.arguments) == case
    )
    for width in (1, 2, 3, 5, 13, 64, len(raw)):
        parser = Parser(object())
        old = ""
        calls = []
        for off in range(0, len(raw), width):
            delta = raw[off : off + width]
            current = old + delta
            result = parser.extract_tool_calls_streaming(
                old, current, delta, [], [], [], None
            )
            if result:
                calls.extend(result.tool_calls)
            old = current
        name = ""
        args = ""
        for call in calls:
            name += call.function.get("name", "")
            args += call.function.get("arguments", "")
        assert name == "lookup" and json.loads(args) == case, (width, name, args)
print(
    "vLLM Hermes: two synthetic calls, non-streaming and seven chunk widths passed (isolated parser)"
)
