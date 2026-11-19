"""Opt-in agent middleware for malformed tool-call arguments."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from copy import copy
from typing import Any

from strands.middleware import InvokeModelContext, ModelStopReason, TypedEvent

from .errors import ExaoneError, MalformedToolCallError
from .model import ExaoneModel

logger = logging.getLogger(__name__)


def _named_choice_unsupported(error: ExaoneError) -> bool:
    if error.status_code not in {400, 422, 501}:
        return False
    if error.code == "unsupported_tool_choice" or (error.param == "tool_choice" and error.code == "unsupported_value"):
        return True
    message = str(error).lower()
    return bool(re.search(r"named (?:function|tool)|(?:function|object)[- ](?:type|valued) tool_choice", message)) and (
        "not support" in message or "unsupported" in message
    )


def _failed_message(error: MalformedToolCallError) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for event in error.buffered_events:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if delta.get("text"):
            content.append({"text": delta["text"]})
        reasoning = delta.get("reasoningContent", {})
        if reasoning.get("text"):
            content.append({"reasoningContent": {"reasoningText": {"text": reasoning["text"]}}})
    content.append({"toolUse": {"toolUseId": error.call_id, "name": error.name, "input": {}}})
    return {"role": "assistant", "content": content}


def _validate_correction(events: list[TypedEvent], name: str) -> bool:
    stop = next((event for event in reversed(events) if isinstance(event, ModelStopReason)), None)
    if stop is not None and stop.stop_reason in {"cancelled", "max_tokens"}:
        return False
    if stop is not None and stop.stop_reason == "tool_use":
        calls = [block["toolUse"] for block in stop.message["content"] if "toolUse" in block]
        call = calls[0] if len(calls) == 1 else None
        if call and call["name"] == name and call["toolUseId"] and isinstance(call["input"], dict):
            return True
    raise ExaoneError("EXAONE recovery must return exactly one completed tool call for the original tool")


async def _correct_tool_call(
    context: InvokeModelContext, model: ExaoneModel, error: MalformedToolCallError, next_fn: Callable[..., Any]
) -> AsyncIterator[Any]:
    failed = _failed_message(error)
    diagnostic = json.dumps(
        {"raw_arguments": error.arguments, "parse_error": error.detail}, ensure_ascii=False, separators=(",", ":")
    )
    detail = (
        "Error: Invalid tool-call JSON. Tool not executed. The previous tool call is a placeholder.\n"
        f"{diagnostic}\n"
        f"Call {error.name} once with a valid JSON object."
    )
    result = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": error.call_id, "status": "error", "content": [{"text": detail}]}}],
    }
    model._recovery_failed_usage = error.usage
    model._recovery_tool_name = error.name
    model._recovery_tool_id = error.call_id
    retry = context.replace(messages=[*context.messages, failed, result], tool_choice={"tool": {"name": error.name}})
    events: list[TypedEvent] = []
    try:
        async for event in next_fn(retry):
            events.append(event)
    except ExaoneError as rejection:
        if events or getattr(rejection, "_exaone_events_exposed", False) or not _named_choice_unsupported(rejection):
            raise
        model._recovery_requires_any = True
        async for event in next_fn(retry.replace(tool_choice={"any": {}})):
            events.append(event)
    recovered = _validate_correction(events, error.name)
    for event in events:
        yield event
    if recovered:
        logger.info("Recovered malformed tool call; failed request usage: %s", error.usage)


async def recover_tool_calls(
    context: InvokeModelContext, next_fn: Callable[..., Any], *, buffer_response: bool = False
) -> AsyncIterator[Any]:
    """Retry one malformed tool call through a matching error result.

    Register with ``agent.add_middleware(InvokeModelStage, recover_tool_calls)``.
    Text and reasoning stream as usual unless `buffer_response` is enabled.
    """
    if type(context.model) is not ExaoneModel:
        async for event in next_fn(context):
            yield event
        return
    model = copy(context.model)
    model._recovery_enabled = True
    model._recovery_buffer_response = buffer_response
    guarded = context.replace(model=model)
    emitted = False
    try:
        async for event in next_fn(guarded):
            emitted = True
            yield event
        return
    except MalformedToolCallError as error:
        if emitted:
            raise
        async for event in _correct_tool_call(guarded, model, error, next_fn):
            yield event
