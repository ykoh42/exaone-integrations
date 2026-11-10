"""Decode OpenAI-style SSE into the sequential events Strands requires."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .errors import ExaoneError


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Read SSE records across byte, UTF-8, line and event boundaries."""
    data: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
        elif line == "data":
            data.append("")
    if data:
        yield "\n".join(data)


@dataclass
class _Tool:
    ident: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamDecoder:
    """Accumulate interleaved tool calls while emitting text and reasoning deltas."""

    active: str | None = None
    tools: dict[int, _Tool] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    done: bool = False

    def _switch(self, kind: str) -> Iterator[dict[str, Any]]:
        if self.active != kind:
            if self.active is not None:
                yield {"contentBlockStop": {}}
            yield {"contentBlockStart": {"start": {}}}
            self.active = kind

    def _text_events(self, delta: dict[str, Any]) -> Iterator[dict[str, Any]]:
        reasoning = delta.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ExaoneError("Invalid reasoning_content delta")
        if reasoning:
            yield from self._switch("reasoning")
            yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": reasoning}}}}
        text = delta.get("content")
        if text is not None and not isinstance(text, str):
            raise ExaoneError("Invalid content delta")
        if text:
            yield from self._switch("text")
            yield {"contentBlockDelta": {"delta": {"text": text}}}

    def _store_tool(self, fragment: dict[str, Any]) -> None:
        index = fragment.get("index")
        if not isinstance(index, int) or index < 0:
            raise ExaoneError("Tool delta is missing a valid index")
        if fragment.get("type") not in (None, "function"):
            raise ExaoneError("Unsupported tool call type")
        item = self.tools.setdefault(index, _Tool())
        ident = fragment.get("id")
        function = fragment.get("function") or {}
        if ident and ident != item.ident:
            item.ident += ident
        name = function.get("name")
        if name and name != item.name:
            item.name += name
        if function.get("arguments"):
            item.arguments += function["arguments"]

    def feed(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Consume one stream chunk (or an equivalent synthetic non-stream chunk)."""
        if payload.get("usage") is not None:
            if not isinstance(payload["usage"], dict):
                raise ExaoneError("Invalid usage payload")
            self.usage = payload["usage"]
        choices = payload.get("choices") or []
        if len(choices) > 1 or any(c.get("index", 0) != 0 for c in choices):
            raise ExaoneError("Multiple choices are not supported by the Strands Model interface")
        if choices:
            yield from self._feed_choice(choices[0])

    def _feed_choice(self, choice: dict[str, Any]) -> Iterator[dict[str, Any]]:
        delta = choice.get("delta") or {}
        yield from self._text_events(delta)
        for fragment in delta.get("tool_calls") or []:
            self._store_tool(fragment)
        finish = choice.get("finish_reason")
        if finish:
            if self.finish_reason is not None:
                raise ExaoneError("Multiple finish reasons received")
            self.finish_reason = finish

    def _tool_events(self) -> Iterator[dict[str, Any]]:
        ids: set[str] = set()
        for index in sorted(self.tools):
            item = self.tools[index]
            if not item.ident or not item.name or not item.arguments:
                raise ExaoneError("Incomplete tool call in EXAONE response")
            if item.ident in ids:
                raise ExaoneError("Duplicate tool call IDs in EXAONE response")
            ids.add(item.ident)
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, ValueError) as exc:
                raise ExaoneError("Invalid tool arguments in EXAONE response") from exc
            if not isinstance(arguments, dict):
                raise ExaoneError("Tool arguments must be a JSON object")
        for index in sorted(self.tools):
            item = self.tools[index]
            yield {"contentBlockStart": {"start": {"toolUse": {"name": item.name, "toolUseId": item.ident}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": item.arguments}}}}
            yield {"contentBlockStop": {}}

    def finish(self, *, require_done: bool) -> Iterator[dict[str, Any]]:
        """Complete the message only after an unambiguous end of stream."""
        if require_done and not self.done:
            raise ExaoneError("EXAONE stream ended without [DONE]")
        reasons = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "content_filtered",
        }
        if self.finish_reason not in reasons:
            raise ExaoneError("EXAONE response ended without a supported finish reason")
        if bool(self.tools) != (self.finish_reason == "tool_calls"):
            raise ExaoneError("Tool calls and finish reason are inconsistent")
        if self.active is not None:
            yield {"contentBlockStop": {}}
            self.active = None
        yield from self._tool_events()
        yield {"messageStop": {"stopReason": reasons[self.finish_reason]}}


def nonstream_chunk(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a complete Chat Completions response into a decoder input."""
    choices = data.get("choices") or []
    if len(choices) != 1:
        raise ExaoneError("Expected exactly one completion choice")
    choice = choices[0]
    message = choice.get("message") or {}
    calls = []
    for index, call in enumerate(message.get("tool_calls") or []):
        calls.append({**call, "index": index})
    return {
        "choices": [
            {
                "index": choice.get("index", 0),
                "delta": {**message, "tool_calls": calls},
                "finish_reason": choice.get("finish_reason"),
            }
        ],
        "usage": data.get("usage"),
    }
