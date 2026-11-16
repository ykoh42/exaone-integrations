"""Generate synthetic wire fixtures from pinned, isolated serving parsers.

Each invocation loads only one server checkout. No server or model is started.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from transformers import PreTrainedTokenizerFast

EXPECTED = {
    "text": 'hello 한글 "quoted" \\path',
    "count": 23,
    "ratio": 1.25,
    "enabled": True,
    "nested": {"ok": True, "n": 3},
    "values": [1, "둘", False],
}
SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "enabled": {"type": "boolean"},
        "nested": {"type": "object"},
        "values": {"type": "array"},
    },
}
REASONING = "synthetic reasoning"
CONTENT = "synthetic answer"
REVISIONS = {
    ("k2", "vllm"): "33fa995c084e1bf652b4a20130ed0f04d99c9381",
    ("k2", "sglang"): "918653d22b2f1ae853b81fc0f5539b7c9311487e",
    ("45", "vllm"): "4bdc8a788d2e2ce9165d552b3d4d8b72604626bf",
    ("45", "sglang"): "6757c9f904cdb8ae9028a394a2108d079b9e088c",
}
TOKENIZER_SHA256 = {
    "k2": "b657dadb90bf2855dcd931c78bf04f32874c2ece3a14ed1bd398eef7f8f23d98",
    "45": "0bd798efa30739e209d51f36cfc2f0a636711e37ad9d69b77e4e5c8ca5f09fab",
}


def xml_tool():
    def value(item):
        return (
            item
            if isinstance(item, str)
            else json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        )

    return (
        "<tool_call>\n<function=lookup>\n"
        + "".join(
            f"<parameter={key}>\n{value(item)}\n</parameter>\n"
            for key, item in EXPECTED.items()
        )
        + "</function>\n</tool_call>"
    )


def json_tool():
    return (
        "<tool_call>"
        + json.dumps(
            {"name": "lookup", "arguments": EXPECTED},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "</tool_call>"
    )


def envelope(deltas, message, model, server):
    return {
        "name": f"{model}-{server}",
        "expected": EXPECTED,
        "nonstream": {
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]
        },
        "stream": [
            *(
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                for delta in deltas
            ),
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ],
    }


def sglang(model, root, tokenizer_file):
    package = types.ModuleType("sglang")
    package.__path__ = [str(root / "python/sglang")]
    sys.modules["sglang"] = package
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.hermes_detector import HermesDetector
    from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
    from sglang.srt.parser.reasoning_parser import Qwen3Detector

    tools = [
        Tool.model_validate(
            {"type": "function", "function": {"name": "lookup", "parameters": SCHEMA}}
        )
    ]
    detector = Qwen3CoderDetector if model == "k2" else HermesDetector
    raw = (
        REASONING
        + "</think>"
        + CONTENT
        + (xml_tool() if model == "k2" else json_tool())
    )
    parsed_reasoning = Qwen3Detector(force_reasoning=True).detect_and_parse(raw)
    parsed = detector().detect_and_parse(parsed_reasoning.normal_text, tools)
    assert parsed_reasoning.reasoning_text == REASONING
    assert len(parsed.calls) == 1 and parsed.calls[0].name == "lookup"
    assert json.loads(parsed.calls[0].parameters) == EXPECTED
    message = {
        "role": "assistant",
        "content": parsed.normal_text,
        "reasoning_content": parsed_reasoning.reasoning_text,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": parsed.calls[0].name,
                    "arguments": parsed.calls[0].parameters,
                },
            }
        ],
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    ids = tokenizer.encode(raw, add_special_tokens=False)
    reason_parser, tool_parser = Qwen3Detector(force_reasoning=True), detector()
    previous, deltas = "", []
    for end in range(3, len(ids) + 3, 3):
        current = tokenizer.decode(ids[:end])
        assert current.startswith(previous)
        part = reason_parser.parse_streaming_increment(current[len(previous) :])
        previous = current
        tools_part = tool_parser.parse_streaming_increment(part.normal_text, tools)
        delta = {}
        if part.reasoning_text:
            delta["reasoning_content"] = part.reasoning_text
        if tools_part.normal_text:
            delta["content"] = tools_part.normal_text
        if tools_part.calls:
            delta["tool_calls"] = []
            for call in tools_part.calls:
                function = {}
                item = {"index": call.tool_index, "function": function}
                if call.name:
                    item.update(id="call_0", type="function")
                    function["name"] = call.name
                if call.parameters:
                    function["arguments"] = call.parameters
                delta["tool_calls"].append(item)
        if delta:
            deltas.append(delta)
    return envelope(deltas, message, model, "sglang")


def vllm(model, root, tokenizer_file):
    os.environ["VLLM_PATH"] = str(root)
    os.environ["VLLM_MODERN"] = "1" if model == "45" else "0"
    parser_class = runpy.run_path(str(Path(__file__).with_name("vllm_qwen.py")))[
        "Qwen3Parser"
    ]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    tools = [
        SimpleNamespace(function=SimpleNamespace(name="lookup", parameters=SCHEMA))
    ]
    request = SimpleNamespace(
        tools=tools, tool_choice="auto", include_reasoning=True, messages=[]
    )
    reasoning_text = REASONING + "</think>" + CONTENT
    if model == "k2":
        raw = reasoning_text + xml_tool()
        parser = parser_class(
            tokenizer, tools, chat_template_kwargs={"enable_thinking": True}
        )
        reason, content, calls = parser.parse(raw, request, enable_auto_tools=True)
        assert (
            len(calls) == 1
            and calls[0].name == "lookup"
            and json.loads(calls[0].arguments) == EXPECTED
        )
        message_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": calls[0].name,
                    "arguments": calls[0].arguments,
                },
            }
        ]
        parser = parser_class(
            tokenizer, tools, chat_template_kwargs={"enable_thinking": True}
        )
        deltas = []
        for offset in range(0, len(raw), 13):
            part = parser.parse_delta(
                raw[offset : offset + 13], [], request, finished=offset + 13 >= len(raw)
            )
            if part:
                deltas.append(part.model_dump(exclude_none=True))
    else:
        # The 4.5 path uses vLLM's Qwen3 reasoning and Hermes tool parsers.
        parser = parser_class(
            tokenizer, tools, chat_template_kwargs={"enable_thinking": True}
        )
        reason, content, _ = parser.parse(
            reasoning_text, request, enable_auto_tools=True
        )
        parser = parser_class(
            tokenizer, tools, chat_template_kwargs={"enable_thinking": True}
        )
        reason_delta = parser.parse_delta(reasoning_text, [], request, finished=True)
        hermes = runpy.run_path(str(Path(__file__).parents[1] / "vllm-hermes.py"))[
            "Parser"
        ]
        raw = json_tool()
        parsed = hermes(object()).extract_tool_calls(raw, None)
        assert parsed.tools_called and len(parsed.tool_calls) == 1
        call = parsed.tool_calls[0].function
        assert call.name == "lookup" and json.loads(call.arguments) == EXPECTED
        message_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
        ]
        deltas = [reason_delta.model_dump(exclude_none=True)]
        stream_parser, previous = hermes(object()), ""
        for offset in range(0, len(raw), 13):
            piece = raw[offset : offset + 13]
            current = previous + piece
            parsed_delta = stream_parser.extract_tool_calls_streaming(
                previous, current, piece, [], [], [], None
            )
            if parsed_delta:
                entries = []
                for item in parsed_delta.tool_calls:
                    entries.append(
                        {
                            "index": item.index,
                            "function": item.function,
                            **(
                                {"id": "call_0", "type": "function"}
                                if item.function.get("name")
                                else {}
                            ),
                        }
                    )
                if entries:
                    deltas.append({"tool_calls": entries})
            previous = current
    assert reason == REASONING and content == CONTENT
    message = {
        "role": "assistant",
        "content": content,
        "reasoning": reason,
        "tool_calls": message_calls,
    }
    return envelope(deltas, message, model, "vllm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server", choices=["vllm", "sglang"])
    parser.add_argument("model", choices=["k2", "45"])
    parser.add_argument("checkout", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISIONS[args.model, args.server]:
        parser.error(f"unexpected {args.server} revision: {revision}")
    if (
        hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
        != TOKENIZER_SHA256[args.model]
    ):
        parser.error("tokenizer does not match the pinned model revision")
    fixture = globals()[args.server](
        args.model, args.checkout.resolve(), args.tokenizer.resolve()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{fixture['name']}.json").write_text(
        json.dumps(fixture, ensure_ascii=False)
    )
    print(f"{fixture['name']}: synthetic parser fixture written")


if __name__ == "__main__":
    main()
