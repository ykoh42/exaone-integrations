"""Load the pinned vLLM Qwen3 parser with CPU-only dependency stubs."""

from __future__ import annotations
import ast, json, logging, math, os, sys, types
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field

root = os.environ["VLLM_PATH"]


def package(name, path=None):
    m = types.ModuleType(name)
    m.__path__ = [path] if path else []
    sys.modules[name] = m
    return m


package("vllm", root + "/vllm")
package("vllm.parser", root + "/vllm/parser")
package("vllm.parser.engine", root + "/vllm/parser/engine")
package("vllm.tool_parsers", root + "/vllm/tool_parsers")
chat = types.ModuleType("vllm.entrypoints.chat_utils")
chat.get_tool_call_id_type = lambda x: "random"
chat.make_tool_call_id = lambda **kw: "call_" + str(kw.get("idx", 0))
sys.modules[chat.__name__] = chat
logger = types.ModuleType("vllm.logger")
logger.init_logger = logging.getLogger
sys.modules[logger.__name__] = logger
abstract = types.ModuleType("vllm.parser.abstract_parser")


@dataclass
class StreamState:
    tool_call_id_type: str = "random"
    history_tool_call_cnt: int = 0
    history_tool_call_cnt_initialized: bool = False


class Parser:
    def _initialize_history_tool_call_cnt(self, request):
        self._stream_state.history_tool_call_cnt_initialized = True


abstract.Parser = Parser
abstract.StreamState = StreamState
sys.modules[abstract.__name__] = abstract
protocol = types.ModuleType("vllm.entrypoints.openai.engine.protocol")


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    id: str | None = None
    type: str | None = None
    index: int
    function: DeltaFunctionCall | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = "call_0"
    type: str = "function"
    function: FunctionCall


class ExtractedToolCallInformation(BaseModel):
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None


for c in (
    DeltaFunctionCall,
    DeltaToolCall,
    DeltaMessage,
    FunctionCall,
    ToolCall,
    ExtractedToolCallInformation,
):
    setattr(protocol, c.__name__, c)
sys.modules[protocol.__name__] = protocol
if os.environ.get("VLLM_MODERN") == "1":
    package("vllm.entrypoints")
    package("vllm.entrypoints.generate")
    package("vllm.entrypoints.generate.base")
    sys.modules["vllm.entrypoints.generate.base.protocol"] = protocol
    package("vllm.renderers")
    sys.modules["vllm.renderers.chat_utils"] = chat
    package("vllm.v1")
    token_buffer = types.ModuleType("vllm.v1.token_buffer")
    token_buffer.rfind_tokens = lambda ids, targets: next(
        (i for i in range(len(ids) - 1, -1, -1) if ids[i] in targets), -1
    )
    sys.modules[token_buffer.__name__] = token_buffer
utils = types.ModuleType("vllm.tool_parsers.utils")
tree = ast.parse(Path(root + "/vllm/tool_parsers/utils.py").read_text())
wanted = {"_is_json_finite", "extract_types_from_schema", "coerce_to_schema_type"}
nodes = [
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name in wanted
    or isinstance(n, ast.AnnAssign)
    and getattr(n.target, "id", None) == "_TYPE_ALIASES"
]
utils.__dict__.update({"json": json, "math": math})
mod = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *nodes,
    ],
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(mod), "utils-subset", "exec"), utils.__dict__)
utils.find_tool_name = lambda tools, name: any(
    x.function.name == name for x in (tools or [])
)
utils.find_tool_properties = lambda tools, name: next(
    (
        x.function.parameters["properties"]
        for x in tools or []
        if x.function.name == name
    ),
    {},
)
sys.modules[utils.__name__] = utils
from vllm.parser.qwen3 import Qwen3Parser as Qwen3Parser
