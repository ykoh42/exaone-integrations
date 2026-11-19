"""Decode OpenAI-style SSE into the sequential events Strands requires."""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, NoReturn

import httpx

from .errors import ExaoneError, MalformedToolCallError, redact


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
    id_fragments: int = 0


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError("Non-standard JSON numeric constant")


def _json_syntax_diagnostic(arguments: str) -> str | None:
    try:
        json.loads(arguments, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        return f"{error.msg} (line {error.lineno}, column {error.colno})"
    except ValueError:
        return None
    return None


def _raise_malformed_tool_call(item: _Tool, arguments: str, usage: dict[str, Any] | None, secret: str) -> None:
    if not arguments.strip():
        return
    detail = _json_syntax_diagnostic(arguments)
    if detail is None:
        return
    if item.id_fragments > 1:
        raise ExaoneError("Cannot recover a tool call with ambiguous streamed ID fragments")
    if any(redact(value, secret) != value for value in (arguments, item.name, item.ident)):
        raise ExaoneError("Invalid tool arguments in EXAONE response")
    failure = MalformedToolCallError(item.ident, item.name, arguments, detail)
    failure.usage = {
        name: value
        for name, value in (usage or {}).items()
        if name in {"prompt_tokens", "completion_tokens", "total_tokens"} and type(value) is int and value >= 0
    }
    details = (usage or {}).get("prompt_tokens_details")
    if isinstance(details, dict):
        failure.usage["prompt_tokens_details"] = {
            name: value
            for name, value in details.items()
            if name in {"cached_tokens", "cache_write_tokens"} and type(value) is int and value >= 0
        }
    raise failure


def _validate_tool_arguments(raw: str) -> None:
    """Require nonblank arguments to be a JSON object."""
    if not raw.strip():
        return
    try:
        arguments = json.loads(raw, parse_constant=_reject_json_constant)
    except ValueError:
        pass
    else:
        if isinstance(arguments, dict):
            return
        raise ExaoneError("Tool arguments must be a JSON object")
    # Raise outside the handler so tool arguments are not retained as exception context.
    raise ExaoneError("Invalid tool arguments in EXAONE response")


def _tool_text(value: Any) -> str:
    """Accept optional string fragments and reject invalid JSON types."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ExaoneError("Invalid EXAONE tool-call delta: expected a string")
    return value


def _reasoning_text(delta: dict[str, Any]) -> str | None:
    """Accept the reasoning field used by EXAONE, SGLang, and vLLM."""
    content = delta.get("reasoning_content")
    reasoning = delta.get("reasoning")
    for name, value in (("reasoning_content", content), ("reasoning", reasoning)):
        if value is not None and not isinstance(value, str):
            raise ExaoneError(f"Invalid {name} delta")
    if content and reasoning and content != reasoning:
        raise ExaoneError("Conflicting reasoning and reasoning_content values")
    return content or reasoning or None


@dataclass
class StreamDecoder:
    """Accumulate interleaved tool calls while emitting text and reasoning deltas."""

    named_tool_choice: bool = False
    diagnose_tool_calls: bool = False
    recovery_tool_name: str | None = None
    recoverable_tools: frozenset[str] = field(default_factory=frozenset)
    active: str | None = None
    tools: dict[int, _Tool] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    done: bool = False
    refusal: str = ""
    has_non_whitespace_content: bool = False
    _recovery_text: list[str] = field(default_factory=list)
    _recovery_reasoning: list[str] = field(default_factory=list)

    def _switch(self, kind: str) -> Iterator[dict[str, Any]]:
        if self.active != kind:
            if self.active is not None:
                yield {"contentBlockStop": {}}
            yield {"contentBlockStart": {"start": {}}}
            self.active = kind

    def _text_events(self, delta: dict[str, Any]) -> Iterator[dict[str, Any]]:
        reasoning = _reasoning_text(delta)
        if reasoning:
            if self.recoverable_tools:
                self._recovery_reasoning.append(reasoning)
            yield from self._switch("reasoning")
            yield {"contentBlockDelta": {"delta": {"reasoningContent": {"text": reasoning}}}}
        text = delta.get("content")
        if text is not None and not isinstance(text, str):
            raise ExaoneError("Invalid content delta")
        if text:
            if self.recoverable_tools:
                self._recovery_text.append(text)
            if text.strip():
                self.has_non_whitespace_content = True
            yield from self._switch("text")
            yield {"contentBlockDelta": {"delta": {"text": text}}}
        refusal = delta.get("refusal")
        if refusal is not None and not isinstance(refusal, str):
            raise ExaoneError("Invalid refusal delta")
        if refusal:
            self.refusal += refusal

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
        ident = _tool_text(fragment.get("id"))
        if ident:
            item.id_fragments += 1
        item.ident += ident
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

    def _tool_events(self, *, truncated: bool = False, secret: str = "") -> Iterator[dict[str, Any]]:
        if self.diagnose_tool_calls and self.refusal and self.tools:
            raise ExaoneError("EXAONE returned refusal text together with tool calls")
        ids: set[str] = set()
        validated: list[tuple[_Tool, str]] = []
        for index in sorted(self.tools):
            item = self.tools[index]
            arguments = "".join(item.arguments)
            if not item.ident or not item.name:
                raise ExaoneError("Incomplete tool call in EXAONE response")
            if item.ident in ids:
                raise ExaoneError("Duplicate tool call IDs in EXAONE response")
            ids.add(item.ident)
            # The SDK must see a length stop before deciding what to do with a
            # truncated call. Preserve its fragments verbatim, without repair.
            if not truncated:
                if item.name in self.recoverable_tools and len(self.tools) == 1:
                    _raise_malformed_tool_call(item, arguments, self.usage, secret)
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
        named_tool_stop = bool(self.tools) and self.named_tool_choice and self.finish_reason == "stop"
        if self.finish_reason != "length" and bool(self.tools) != (
            self.finish_reason == "tool_calls" or named_tool_stop
        ):
            raise ExaoneError("Tool calls and finish reason are inconsistent")
        if self.recovery_tool_name is not None:
            self._validate_replacement(named_tool_stop)
        if self.active is not None:
            yield {"contentBlockStop": {}}
            self.active = None
        if self.refusal and not self.has_non_whitespace_content:
            if self.tools:
                raise ExaoneError("EXAONE returned refusal text together with tool calls")
            yield from self._switch("text")
            yield {"contentBlockDelta": {"delta": {"text": self.refusal}}}
            yield {"contentBlockStop": {}}
            self.active = None
        yield from self._tool_events(truncated=self.finish_reason == "length", secret=secret)
        yield {"messageStop": {"stopReason": "tool_use" if named_tool_stop else reasons[self.finish_reason]}}

    def recovery_messages(self, error: MalformedToolCallError) -> list[dict[str, Any]]:
        assistant = {
            "role": "assistant",
            "content": "".join(self._recovery_text),
            "tool_calls": [
                {"id": error.call_id, "type": "function", "function": {"name": error.name, "arguments": "{}"}}
            ],
        }
        if self._recovery_reasoning:
            assistant["reasoning_content"] = "".join(self._recovery_reasoning)
        diagnostic = json.dumps(
            {"raw_arguments": error.arguments, "parse_error": error.detail}, ensure_ascii=False, separators=(",", ":")
        )
        detail = (
            "Error: Invalid tool-call JSON. Tool not executed. The previous tool call is a placeholder.\n"
            f"{diagnostic}\nCall {error.name} once with a valid JSON object."
        )
        return [assistant, {"role": "tool", "tool_call_id": error.call_id, "content": detail}]

    def replace_tool_arguments(self, correction: StreamDecoder, *, require_done: bool, secret: str) -> None:
        if correction.finish_reason == "length" and (correction.done or not require_done):
            self.tools.clear()
            self.finish_reason = "length"
            return
        for _ in correction.finish(require_done=require_done, secret=secret):
            pass
        next(iter(self.tools.values())).arguments = next(iter(correction.tools.values())).arguments

    def _validate_replacement(self, named_tool_stop: bool) -> None:
        if self.finish_reason == "length":
            return
        if self.refusal or (self.finish_reason != "tool_calls" and not named_tool_stop) or len(self.tools) != 1:
            raise ExaoneError("EXAONE recovery must return exactly one completed tool call")
        call = next(iter(self.tools.values()))
        if call.id_fragments > 1:
            raise ExaoneError("EXAONE recovery returned an ambiguous streamed tool call ID")
        if call.name != self.recovery_tool_name or not "".join(call.arguments).strip():
            raise ExaoneError("EXAONE recovery must return the original tool with a JSON object")


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
