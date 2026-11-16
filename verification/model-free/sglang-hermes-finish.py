import ast
import copy
import os
import sys
import types
from pathlib import Path

root = os.environ["SGLANG_PATH"]
p = types.ModuleType("sglang")
p.__path__ = [root + "/python/sglang"]
sys.modules["sglang"] = p
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.hermes_detector import HermesDetector

raw = '<tool_call>{"name":"lookup","arguments":{"value":7,"text":"hello"}}</tool_call>'
tools = [
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
]
detector = HermesDetector()
result = detector.parse_streaming_increment(raw, tools)
assert (
    len(result.calls) == 1
    and result.calls[0].name == "lookup"
    and not result.calls[0].parameters
)
path = root + "/python/sglang/srt/entrypoints/openai/serving_chat.py"
tree = ast.parse(Path(path).read_text())
fn = next(
    n
    for cls in tree.body
    if isinstance(cls, ast.ClassDef)
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "_check_for_unstreamed_tool_args"
)
fn = copy.deepcopy(fn)
fn.decorator_list = []
fn.args.defaults = []
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        fn,
    ],
    type_ignores=[],
)
ns = {}
exec(compile(ast.fix_missing_locations(module), path, "exec"), ns)  # noqa: S102 -- explicitly run the pinned server code
args = [None, detector, {}, None, 0] + [None] * (len(fn.args.args) - 5)
assert ns["_check_for_unstreamed_tool_args"](*args) is None
print(
    "full single chunk: name emitted, empty arguments, isolated finish check has no remainder"
)
