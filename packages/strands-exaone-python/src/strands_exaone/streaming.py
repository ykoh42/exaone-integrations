"""Decode OpenAI-style SSE into the sequential events Strands requires."""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, NoReturn

import httpx

from .errors import ExaoneError, redact


async def _sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Split UTF-8 text only at SSE CR, LF and CRLF line endings."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    pending: list[str] = []
    skip_lf = False
    async for chunk in response.aiter_bytes():
        text = decoder.decode(chunk)
        start = 0
        for match in re.finditer(r"[\r\n]", text):
            end = match.start()
            char = text[end]
            if skip_lf and end == start and char == "\n":
                skip_lf = False
                start = end + 1
                continue
            pending.append(text[start:end])
            yield "".join(pending)
            pending.clear()
            skip_lf = char == "\r"
            start = end + 1
        if start < len(text):
            skip_lf = False
            pending.append(text[start:])


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Read SSE records across byte, UTF-8, line and event boundaries."""
    data: list[str] = []
    async for line in _sse_lines(response):
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
        elif line == "data":
            data.append("")


@dataclass
class _Tool:
    ident: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError("Non-standard JSON numeric constant")


def _validate_tool_arguments(raw: str) -> None:
    """Require JSON object arguments without retaining invalid data in an error."""
    try:
        arguments = json.loads(raw, parse_constant=_reject_json_constant)
    except (TypeError, ValueError):
        # Raise outside the handler so tool arguments are not retained as exception context.
        pass
    else:
        if isinstance(arguments, dict):
            return
        raise ExaoneError("Tool arguments must be a JSON object")
    raise ExaoneError("Invalid tool arguments in EXAONE response")


def _tool_text(value: Any) -> str:
    """Accept optional string fragments and reject invalid JSON types."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ExaoneError("Invalid EXAONE tool-call delta: expected a string")
    return value


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

    def _store_tool(self, fragment: Any) -> None:
        if not isinstance(fragment, dict):
            raise ExaoneError("Invalid EXAONE tool-call delta: expected an object")
        index = fragment.get("index")
        if type(index) is not int or index < 0:
            raise ExaoneError("Tool delta is missing a valid index")
        if fragment.get("type") not in (None, "function"):
            raise ExaoneError("Unsupported tool call type")
        item = self.tools.setdefault(index, _Tool())
        function = fragment.get("function")
        if function is None:
            function = {}
        if not isinstance(function, dict):
            raise ExaoneError("Invalid EXAONE tool-call delta: expected a function object")
        item.ident += _tool_text(fragment.get("id"))
        item.name += _tool_text(function.get("name"))
        arguments = _tool_text(function.get("arguments"))
        if arguments:
            item.arguments.append(arguments)

    def feed(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Consume one stream chunk (or an equivalent synthetic non-stream chunk)."""
        if payload.get("usage") is not None:
            if not isinstance(payload["usage"], dict):
                raise ExaoneError("Invalid usage payload")
            self.usage = payload["usage"]
        choices = payload.get("choices")
        if choices is None:
            choices = []
        if not isinstance(choices, list) or any(not isinstance(choice, dict) for choice in choices):
            raise ExaoneError("Invalid EXAONE response choices")
        if any(type(choice.get("index", 0)) is not int for choice in choices):
            raise ExaoneError("Invalid EXAONE response choice index")
        if len(choices) > 1 or any(choice.get("index", 0) != 0 for choice in choices):
            raise ExaoneError("Multiple choices are not supported by the Strands Model interface")
        if choices:
            yield from self._feed_choice(choices[0])

    def _feed_choice(self, choice: dict[str, Any]) -> Iterator[dict[str, Any]]:
        delta = choice.get("delta")
        if delta is None:
            delta = {}
        if not isinstance(delta, dict):
            raise ExaoneError("Invalid EXAONE choice delta")
        yield from self._text_events(delta)
        calls = delta.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ExaoneError("Invalid EXAONE tool-call delta: expected an array")
            for fragment in calls:
                self._store_tool(fragment)
        finish = choice.get("finish_reason")
        if finish:
            if self.finish_reason is not None and self.finish_reason != finish:
                raise ExaoneError("Multiple finish reasons received")
            self.finish_reason = finish

    def _tool_events(self) -> Iterator[dict[str, Any]]:
        ids: set[str] = set()
        validated: list[tuple[_Tool, str]] = []
        for index in sorted(self.tools):
            item = self.tools[index]
            arguments = "".join(item.arguments)
            if not item.ident or not item.name or not arguments:
                raise ExaoneError("Incomplete tool call in EXAONE response")
            if item.ident in ids:
                raise ExaoneError("Duplicate tool call IDs in EXAONE response")
            ids.add(item.ident)
            _validate_tool_arguments(arguments)
            validated.append((item, arguments))
        for item, arguments in validated:
            yield {"contentBlockStart": {"start": {"toolUse": {"name": item.name, "toolUseId": item.ident}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": arguments}}}}
            yield {"contentBlockStop": {}}

    def finish(self, *, require_done: bool, secret: str = "") -> Iterator[dict[str, Any]]:
        """Complete the message only after an unambiguous end of stream."""
        if require_done and not self.done:
            raise ExaoneError("EXAONE stream ended without [DONE]")
        reasons = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "content_filtered",
        }
        if not isinstance(self.finish_reason, str) or self.finish_reason not in reasons:
            reason = redact(self.finish_reason, secret) if isinstance(self.finish_reason, str) else None
            raise ExaoneError(f"EXAONE response ended with unsupported finish reason: {reason!r}", finish_reason=reason)
        if bool(self.tools) != (self.finish_reason == "tool_calls"):
            raise ExaoneError("Tool calls and finish reason are inconsistent")
        if self.active is not None:
            yield {"contentBlockStop": {}}
            self.active = None
        yield from self._tool_events()
        yield {"messageStop": {"stopReason": reasons[self.finish_reason]}}


def nonstream_chunk(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a complete Chat Completions response into a decoder input."""
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ExaoneError("Expected exactly one completion choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ExaoneError("Invalid EXAONE response choice")
    message = choice.get("message")
    if message is None:
        message = {}
    if not isinstance(message, dict):
        raise ExaoneError("Invalid EXAONE response message")
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        tool_calls = []
    if not isinstance(tool_calls, list):
        raise ExaoneError("Invalid EXAONE tool-call delta: expected an array")
    calls = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            raise ExaoneError("Invalid EXAONE tool-call delta: expected an object")
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
