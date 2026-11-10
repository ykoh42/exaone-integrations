"""Translate Strands conversation history to EXAONE Chat Completions."""

from __future__ import annotations

import json
from typing import Any, cast

from strands.types.content import ContentBlock, Message, Messages, SystemContentBlock
from strands.types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse


def _text_of_result(result: ToolResult) -> str:
    parts: list[str] = []
    for item in result.get("content", []):
        if "text" in item:
            parts.append(item["text"])
        elif "json" in item:
            parts.append(json.dumps(item["json"], ensure_ascii=False, separators=(",", ":")))
        else:
            raise ValueError("EXAONE supports text and JSON tool results only")
    text = "\n".join(parts)
    return f"[ERROR] {text}" if result.get("status") == "error" else text


def _tool_call(tool: ToolUse) -> dict[str, Any]:
    if tool.get("reasoningSignature"):
        raise ValueError("EXAONE does not support tool reasoning signatures")
    return {
        "id": tool["toolUseId"],
        "type": "function",
        "function": {
            "name": tool["name"],
            "arguments": json.dumps(tool["input"], ensure_ascii=False, separators=(",", ":")),
        },
    }


def _reasoning_text(block: ContentBlock) -> str:
    reasoning = block["reasoningContent"]
    if reasoning.get("redactedContent") is not None:
        raise ValueError("EXAONE does not support redacted reasoning")
    value = reasoning.get("reasoningText") or {}
    if value.get("signature"):
        raise ValueError("EXAONE does not support reasoning signatures")
    return value.get("text", "")


def _message_parts(message: Message) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    role = message["role"]
    for block in message["content"]:
        if "text" in block:
            text.append(block["text"])
        elif "reasoningContent" in block and role == "assistant":
            reasoning.append(_reasoning_text(block))
        elif "toolUse" in block and role == "assistant":
            calls.append(_tool_call(block["toolUse"]))
        elif "toolResult" in block and role == "user":
            result = block["toolResult"]
            results.append({"role": "tool", "tool_call_id": result["toolUseId"], "content": _text_of_result(result)})
        else:
            raise ValueError(f"Unsupported EXAONE content block for {role}: {', '.join(block)}")
    return text, reasoning, calls, results


def _system_messages(
    system_prompt: str | None, system_prompt_content: list[SystemContentBlock] | None
) -> list[dict[str, Any]]:
    if system_prompt_content is not None:
        if any(set(block) != {"text"} for block in system_prompt_content):
            raise ValueError("EXAONE supports text-only system prompt blocks")
        return [{"role": "system", "content": block["text"]} for block in system_prompt_content]
    return [{"role": "system", "content": system_prompt}] if system_prompt is not None else []


def _format_message(message: Message) -> list[dict[str, Any]]:
    role = message["role"]
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported Strands message role: {role}")
    text, reasoning, calls, results = _message_parts(message)
    if role == "assistant":
        assistant: dict[str, Any] = {"role": "assistant", "content": "".join(text) if text else None}
        if reasoning:
            assistant["reasoning_content"] = "".join(reasoning)
        if calls:
            assistant["tool_calls"] = calls
        return [assistant]
    if text:
        return [*results, {"role": "user", "content": "".join(text)}]
    if not results:
        raise ValueError("User message must contain text or a tool result")
    return results


def format_messages(
    messages: Messages,
    system_prompt: str | None,
    system_prompt_content: list[SystemContentBlock] | None,
) -> list[dict[str, Any]]:
    """Build ordered wire messages without losing thinking or tool result IDs."""
    output = _system_messages(system_prompt, system_prompt_content)
    for message in messages:
        output.extend(_format_message(message))
    return output


def format_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert Strands tool specifications to OpenAI-compatible functions."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["inputSchema"]["json"],
            },
        }
        for spec in specs
    ]


def format_tool_choice(choice: ToolChoice) -> str | dict[str, Any]:
    """Map a Strands tool choice to the Chat Completions wire format."""
    if "auto" in choice:
        return "auto"
    if "any" in choice:
        return "required"
    if "tool" in choice:
        return {"type": "function", "function": {"name": cast(dict[str, Any], choice)["tool"]["name"]}}
    raise ValueError("Unsupported tool choice")
