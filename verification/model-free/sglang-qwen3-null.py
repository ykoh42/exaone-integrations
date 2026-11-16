import json
import os
import sys
import types
from pathlib import Path

root = os.environ["SGLANG_PATH"]
p = types.ModuleType("sglang")
p.__path__ = [root + "/python/sglang"]
sys.modules["sglang"] = p
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

tools = [
    Tool.model_validate(
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        }
    )
]
raw = "<tool_call>\n<function=lookup>\n<parameter=text>\nnull\n</parameter>\n</function>\n</tool_call>"
r = Qwen3CoderDetector().detect_and_parse(raw, tools)
value = json.loads(r.calls[0].parameters)["text"]
assert value is None
for width in (1, 2, 3, 5, 13, 64):
    d = Qwen3CoderDetector()
    calls = []
    for i in range(0, len(raw), width):
        calls.extend(d.parse_streaming_increment(raw[i : i + width], tools).calls)
    payload = "".join(c.parameters for c in calls)
    assert json.loads(payload)["text"] is None
print(
    Path(root).name
    + ': string parameter "null" becomes JSON null in nonstreaming and 6 streaming fragment sizes'
)
