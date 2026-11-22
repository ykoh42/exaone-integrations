"""Internal, invocation-local guard for optional agent recovery."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, quote_plus

from langchain_core.messages import AIMessage


@dataclass
class RecoveryState:
    """Per-invocation failure detail shared only with the recovery middleware."""

    failure: tuple[MalformedToolArguments, AIMessage] | None = field(default=None, repr=False)
    expected_tool: str | None = None
    required: bool = False


recovery_state: ContextVar[RecoveryState | None] = ContextVar("exaone_tool_call_recovery", default=None)


def recoverable_tools(request: Mapping[str, Any], state: RecoveryState) -> frozenset[str]:
    """Use the final serialized tool policy, after the SDK has applied middleware."""
    tools = request.get("tools")
    names = frozenset(
        tool["function"]["name"]
        for tool in (tools if isinstance(tools, list) else [])
        if isinstance(tool, Mapping)
        and isinstance(tool.get("function"), Mapping)
        and isinstance(tool["function"].get("name"), str)
    )
    choice = request.get("tool_choice")
    if state.expected_tool is not None:
        wanted = "required" if state.required else {"type": "function", "function": {"name": state.expected_tool}}
        if state.expected_tool not in names or choice != wanted:
            raise ValueError("The effective EXAONE request does not allow the tool correction")
    if choice in (None, "auto", "required"):
        return names
    if isinstance(choice, Mapping) and isinstance(function := choice.get("function"), Mapping):
        name = function.get("name")
        if choice.get("type") == "function" and isinstance(name, str) and name in names:
            return frozenset({name})
    return frozenset()


def _contains_secret(text: str, secret: str) -> bool:
    if not secret:
        return False
    try:
        variants = {secret, quote(secret, safe=""), quote_plus(secret, safe="")}
    except UnicodeError:
        variants = {secret}
    for variant in variants:
        encoded_pattern = re.sub(
            r"%[0-9A-F]{2}", lambda match: f"(?i:{match.group()})", re.escape(variant), flags=re.IGNORECASE
        )
        if re.search(encoded_pattern, text):
            return True
    return False


class MalformedToolArguments(ValueError):
    """A single malformed tool invocation, withheld from the SDK."""

    def __init__(self, message: AIMessage, detail: str, secret: str) -> None:
        super().__init__("EXAONE returned malformed tool arguments")
        self.detail = detail
        usage = message.response_metadata.get("usage")
        self.usage = _numeric_usage(usage, secret)


def _numeric_usage(value: Any, secret: str) -> dict[str, Any] | None:
    """Retain numeric accounting fields without copying arbitrary metadata."""
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        if type(value.get(name)) is int and value[name] >= 0:
            result[name] = value[name]
    for name in ("prompt_tokens_details", "completion_tokens_details"):
        details = value.get(name)
        if isinstance(details, dict):
            safe = {
                key: item
                for key, item in details.items()
                if isinstance(key, str) and not _contains_secret(key, secret) and type(item) is int and item >= 0
            }
            if safe:
                result[name] = safe
    return result or None


def attach_recovery_usage(error: Exception, failed: Any, correction: Any, secret: str) -> None:
    """Keep received accounting on a failed correction without changing the error type."""
    original = _numeric_usage(failed, secret)
    retried = _numeric_usage(correction, secret)
    if original is None and retried is None:
        return
    usage = {}
    if original is not None:
        usage["failed_usage"] = original
    if retried is not None:
        usage["correction_usage"] = retried
    try:
        error.exaone_recovery_usage = usage  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass


@dataclass
class CorrectionFailureUsage:
    """Remember the last reported usage before correction validation can fail."""

    failed: Any
    secret: str = field(repr=False)
    correction: dict[str, Any] | None = None

    def record(self, payload: Any) -> None:
        if isinstance(payload, dict):
            usage = _numeric_usage(payload.get("usage"), self.secret)
            if usage is not None:
                self.correction = usage

    def attach(self, error: Exception) -> None:
        attach_recovery_usage(error, self.failed, self.correction, self.secret)


def _has_malformed_json(arguments: list[Any]) -> bool:
    """Detect a syntax error without retaining decoder exceptions or input."""
    return any(_syntax_diagnostic(raw) is not None for raw in arguments)


def _syntax_diagnostic(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        json.loads(raw, parse_constant=reject_json_constant)
    except json.JSONDecodeError as error:
        return f"{error.msg} (line {error.lineno}, column {error.colno})"
    except ValueError:
        pass
    else:
        return None
    raise ValueError("EXAONE tool arguments contain an unsupported JSON value")


def check_tool_completion(has_calls: bool, finish: Any, refusal: Any) -> None:
    """Allow tool execution only for a completed response without a refusal."""
    if not has_calls:
        return
    if finish == "length":
        raise ValueError("EXAONE tool call was truncated by the output length limit")
    if refusal or finish not in ("tool_calls", "stop"):
        raise ValueError("EXAONE returned tool calls without a completed tool response")


def check_message(message: AIMessage, secret: str, *, tools: frozenset[str], allow_retry: bool = True) -> None:
    """Signal a recoverable syntax error without exposing it as a model result."""
    invalid = message.invalid_tool_calls
    check_tool_completion(
        bool(invalid or message.tool_calls),
        message.response_metadata.get("finish_reason"),
        message.additional_kwargs.get("refusal"),
    )
    if len(invalid) + len(message.tool_calls) > 1 and _has_malformed_json([call.get("args") for call in invalid]):
        raise ValueError("EXAONE returned malformed arguments in a multi-tool response")
    if len(invalid) != 1 or message.tool_calls:
        return
    call = invalid[0]
    raw, ident, name = call.get("args"), call.get("id"), call.get("name")
    detail = _syntax_diagnostic(raw)
    if detail is None:
        return
    if not allow_retry:
        raise ValueError("EXAONE returned malformed arguments outside a retryable response")
    if not ident or not name:
        raise ValueError("Cannot recover malformed tool arguments without a call ID and name")
    if any(_contains_secret(value, secret) for value in (raw, ident, name) if isinstance(value, str)):
        raise ValueError("Cannot retry malformed tool arguments containing a credential")
    if name not in tools:
        raise ValueError("The effective EXAONE request does not allow the tool correction")
    state = recovery_state.get()
    if state is None:
        return
    error = MalformedToolArguments(message, detail, secret)
    state.failure = error, message
    raise error from None


def reject_json_constant(value: str) -> None:
    """Reject non-standard numeric literals accepted by Python's default decoder."""
    raise ValueError("Non-standard JSON constant")


def check_replacement(name: str, calls: list[dict[str, Any]], finish: Any, refusal: Any) -> None:
    """Require exactly one completed call to the failed tool, with strict JSON."""
    if refusal or finish not in ("tool_calls", "stop") or len(calls) != 1:
        raise ValueError("EXAONE recovery must return exactly one completed tool call")
    call = calls[0]
    if call.get("name") == name and call.get("id") and isinstance(call.get("args"), str):
        try:
            parsed = json.loads(call["args"], parse_constant=reject_json_constant)
        except ValueError:
            pass
        else:
            if isinstance(parsed, dict):
                return
    raise ValueError("EXAONE recovery must return the original tool with a valid JSON object")


def check_completed_message(message: AIMessage, name: str) -> None:
    calls = message.additional_kwargs.get("tool_calls", [])
    check_replacement(
        name,
        [
            {"id": call.get("id"), "name": call["function"].get("name"), "args": call["function"].get("arguments")}
            for call in calls
        ],
        message.response_metadata.get("finish_reason"),
        message.additional_kwargs.get("refusal"),
    )


def _incomplete_recovery_stream(name: str | None) -> None:
    if name is not None:
        raise ValueError("EXAONE recovery returned incomplete tool call fragments")


class RecoveryStream:
    """Pass text through while withholding tool frames until the stream is complete."""

    def __init__(self, state: RecoveryState, tools: frozenset[str]) -> None:
        self.state = state
        self.tools = tools
        self.exposed = False
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.leading: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] = []
        self.refusal = False
        self.finish_reason: str | None = None

    def accept(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        choice, delta = _stream_parts(payload)
        self.refusal = self.refusal or bool(delta.get("refusal"))
        if self._record_finish(choice.get("finish_reason")):
            choice = {**choice}
            del choice["finish_reason"]
            payload = {**payload, "choices": [choice]}
        self._remember_text(delta)
        has_tools = bool(delta.get("tool_calls"))
        has_output = any(delta.get(field) for field in ("content", "reasoning_content", "reasoning", "refusal"))
        if not has_output:
            if has_tools or self.pending or payload.get("usage") is not None:
                self.pending.append(payload)
            elif self.exposed or payload.get("error") is not None:
                return [payload]
            else:
                self.leading.append(payload)
            return []
        if has_tools or payload.get("usage") is not None or (self.pending and choice.get("finish_reason") is not None):
            payload, tools = _split_recovery_payload(payload, choice, delta)
            self.pending.append(tools)
        self.exposed = True
        leading, self.leading = self.leading, []
        return [*leading, payload]

    def _remember_text(self, delta: dict[str, Any]) -> None:
        if not self.tools or self.state.expected_tool is not None:
            return
        if isinstance(content := delta.get("content"), str) and content:
            self.content.append(content)
        if isinstance(thought := delta.get("reasoning_content") or delta.get("reasoning"), str) and thought:
            self.reasoning.append(thought)

    def finish(self, complete: bool) -> list[dict[str, Any]]:
        if not complete:
            raise ValueError("EXAONE event stream ended before [DONE]")
        return [*self.leading, *self.pending]

    def _record_finish(self, finish: str | None) -> bool:
        """Record the finish reason and report repetitions before the SDK merges chunks."""
        if finish is None:
            return False
        if self.finish_reason is None:
            self.finish_reason = finish
            return False
        if self.finish_reason != finish:
            raise ValueError("EXAONE returned conflicting stream finish reasons")
        return True


def _stream_parts(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = payload.get("choices")
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        choice = choices[0]
        if isinstance(delta := choice.get("delta"), dict):
            return choice, delta
    return {}, {}


def _split_recovery_payload(
    payload: dict[str, Any], choice: dict[str, Any], delta: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    early_choice = {**choice, "delta": {key: value for key, value in delta.items() if key != "tool_calls"}}
    early_choice.pop("finish_reason", None)
    early_choice.pop("matched_stop", None)
    early = {**payload, "choices": [early_choice]}
    early.pop("usage", None)
    tool_delta = {"tool_calls": delta["tool_calls"]} if "tool_calls" in delta else {}
    tool_choice = {**choice, "delta": tool_delta}
    tool_choice.pop("logprobs", None)
    return early, {**payload, "choices": [tool_choice]}


def check_stream(payloads: list[dict[str, Any]], secret: str, router: RecoveryStream) -> str | None:
    """Check complete streamed tool arguments, not intermediate fragments."""
    calls: dict[int, dict[str, str]] = {}
    id_fragments: dict[int, int] = {}
    finish, refusal, expected = router.finish_reason, router.refusal, router.state.expected_tool
    usage: dict[str, Any] | None = None
    for payload in payloads:
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            return _incomplete_recovery_stream(expected)
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index") not in (None, 0):
            return _incomplete_recovery_stream(expected)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return _incomplete_recovery_stream(expected)
        fragments = delta.get("tool_calls") or []
        if not isinstance(fragments, list):
            return _incomplete_recovery_stream(expected)
        for fragment in fragments:
            if not isinstance(fragment, dict) or type(fragment.get("index")) is not int or fragment["index"] < 0:
                raise ValueError("EXAONE streamed tool call is missing a valid index")
            if fragment.get("type") not in (None, "function"):
                return _incomplete_recovery_stream(expected)
            function = fragment.get("function") or {}
            if not isinstance(function, dict):
                return _incomplete_recovery_stream(expected)
            parts = (fragment.get("id") or "", function.get("name") or "", function.get("arguments") or "")
            if any(not isinstance(part, str) for part in parts):
                return _incomplete_recovery_stream(expected)
            if parts[0]:
                id_fragments[fragment["index"]] = id_fragments.get(fragment["index"], 0) + 1
            entry = calls.setdefault(fragment["index"], {"id": "", "name": "", "args": ""})
            for key, value in zip(("id", "name", "args"), parts, strict=True):
                entry[key] += value
    check_tool_completion(bool(calls), finish, refusal)
    if expected is not None:
        if any(count > 1 for count in id_fragments.values()):
            raise ValueError("EXAONE recovery returned an ambiguous streamed tool call ID")
        check_replacement(expected, list(calls.values()), finish, refusal)
        return next(iter(calls.values()))["args"]
    if any(count > 1 and _has_malformed_json([calls[index]["args"]]) for index, count in id_fragments.items()):
        raise ValueError("EXAONE returned malformed arguments with an ambiguous streamed tool call ID")
    if len(calls) > 1 and _has_malformed_json([call["args"] for call in calls.values()]):
        raise ValueError("EXAONE returned malformed arguments in a multi-tool response")
    if len(calls) != 1:
        return
    call = next(iter(calls.values()))
    message = AIMessage(
        content="".join(router.content),
        additional_kwargs={"reasoning_content": "".join(router.reasoning)} if router.reasoning else {},
        response_metadata={"finish_reason": finish, "usage": usage},
        invalid_tool_calls=[{"id": call["id"], "name": call["name"], "args": call["args"], "error": None}],
    )
    check_message(message, secret, tools=router.tools)
    return None


def correction_prompt(name: str, raw: str, diagnostic: str) -> str:
    data = json.dumps({"raw_arguments": raw, "parse_error": diagnostic}, ensure_ascii=False, separators=(",", ":"))
    return (
        "Error: Invalid tool-call JSON. Tool not executed. The previous tool call is a placeholder.\n"
        f"{data}\nCall {name} once with a valid JSON object."
    )


def correction_request(body: dict[str, Any], failed: AIMessage, error: MalformedToolArguments) -> dict[str, Any]:
    call = failed.invalid_tool_calls[0]
    name, ident = call["name"], call["id"]
    assistant = {
        "role": "assistant",
        "content": failed.content,
        "tool_calls": [{"id": ident, "type": "function", "function": {"name": name, "arguments": "{}"}}],
    }
    if reasoning := failed.additional_kwargs.get("reasoning_content"):
        assistant["reasoning_content"] = reasoning
    feedback = {"role": "tool", "tool_call_id": ident, "content": correction_prompt(name, call["args"], error.detail)}
    return {
        **body,
        "messages": [*body["messages"], assistant, feedback],
        "tool_choice": {"type": "function", "function": {"name": name}},
    }


def replace_arguments(payloads: list[dict[str, Any]], call: Any, arguments: str) -> None:
    """Replace the original fragments while keeping their original call ID and envelope."""
    first = True
    for payload in payloads:
        _, delta = _stream_parts(payload)
        if calls := delta.pop("tool_calls", None):
            if first:
                first = False
                delta["tool_calls"] = [
                    {
                        "id": call["id"],
                        "index": calls[0]["index"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": arguments},
                    }
                ]
