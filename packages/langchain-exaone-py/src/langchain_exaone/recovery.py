"""Optional LangChain agent middleware for malformed EXAONE tool arguments.

This module requires the LangChain agent package; importing the model itself
continues to require only langchain-core.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from langchain_exaone._recovery import (
    MalformedToolArguments,
    RecoveryState,
    check_tool_completion,
    recovery_state,
)
from langchain_exaone.chat_models import ExaoneAPIError


def _named_choice_unsupported(error: ExaoneAPIError) -> bool:
    if error.status_code not in {400, 422, 501}:
        return False
    if error.code == "unsupported_tool_choice" or (error.param == "tool_choice" and error.code == "unsupported_value"):
        return True
    message = str(error).lower()
    return bool(re.search(r"named (?:function|tool)|(?:function|object)[- ](?:type|valued) tool_choice", message)) and (
        "not support" in message or "unsupported" in message
    )


def _check_agent_response(response: ModelResponse[Any]) -> ModelResponse[Any]:
    """Check completion again before the SDK can execute a cached tool call."""
    if isinstance(response, ModelResponse):
        for message in response.result:
            if isinstance(message, AIMessage) and message.response_metadata.get("model_provider") == "exaone":
                check_tool_completion(
                    bool(message.tool_calls or message.invalid_tool_calls),
                    message.response_metadata.get("finish_reason"),
                    message.additional_kwargs.get("refusal"),
                )
    return response


def _retry_request(request: ModelRequest[Any], failed: AIMessage, error: MalformedToolArguments) -> ModelRequest[Any]:
    """Build a paired tool-error exchange without changing stored agent history."""
    call = failed.invalid_tool_calls[0]
    name, ident, raw = call["name"], call["id"], call["args"]
    additional = {key: value for key, value in failed.additional_kwargs.items() if key != "tool_calls"}
    placeholder = failed.model_copy(
        update={
            "additional_kwargs": additional,
            "tool_calls": [{"id": ident, "name": name, "args": {}, "type": "tool_call"}],
            "invalid_tool_calls": [],
        }
    )
    diagnostic = json.dumps(
        {"raw_arguments": raw, "parse_error": error.detail}, ensure_ascii=False, separators=(",", ":")
    )
    feedback = ToolMessage(
        content=(
            "Error: Invalid tool-call JSON. Tool not executed. The previous tool call is a placeholder.\n"
            f"{diagnostic}\n"
            f"Call {name} once with a valid JSON object."
        ),
        tool_call_id=ident,
        name=name,
        status="error",
    )
    return request.override(
        messages=[*request.messages, placeholder, feedback],
        tool_choice={"type": "function", "function": {"name": name}},
    )


def _record_recovery(response: ModelResponse[Any], error: MalformedToolArguments, expected: str) -> ModelResponse[Any]:
    """Attach failed-attempt accounting without altering the successful usage."""
    messages = list(_check_agent_response(response).result)
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        if message.response_metadata.get("model_provider") != "exaone":
            raise ValueError("EXAONE recovery requires a response from an EXAONE model")
        if message.invalid_tool_calls or len(message.tool_calls) != 1 or message.tool_calls[0]["name"] != expected:
            raise ValueError("EXAONE recovery did not return the original tool call")
        metadata = dict(message.response_metadata)
        if "exaone_tool_call_recovery" in metadata:
            return response
        recovery: dict[str, Any] = {"attempts": 1}
        if error.usage is not None:
            recovery["failed_usage"] = error.usage
        metadata["exaone_tool_call_recovery"] = recovery
        messages[index] = message.model_copy(update={"response_metadata": metadata})
        return replace(response, result=messages)
    raise ValueError("EXAONE recovery did not return a tool-call message")


class ToolCallRecoveryMiddleware(AgentMiddleware):
    """Retry one malformed JSON tool call before its output reaches the agent.

    The default forwards text as it arrives and only retries before exposure.
    Set `buffer_response=True` to withhold the full response before validation.
    """

    def __init__(self, *, buffer_response: bool = False) -> None:
        """Select whether to hold text until the model response passes validation."""
        self.buffer_response = buffer_response

    def wrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], ModelResponse]
    ) -> ModelResponse:
        """Wrap one synchronous model invocation."""
        state = RecoveryState(buffer_response=self.buffer_response)
        token = recovery_state.set(state)
        try:
            try:
                return _check_agent_response(handler(request))
            except MalformedToolArguments as error:
                if state.failure is None or state.failure[0] is not error:
                    raise
                failed = state.failure[1]
                retry = _retry_request(request, failed, error)
                state.expected_tool = failed.invalid_tool_calls[0]["name"]
                try:
                    response = handler(retry)
                except ExaoneAPIError as rejection:
                    if not _named_choice_unsupported(rejection):
                        raise
                    state.required = True
                    response = handler(retry.override(tool_choice="required"))
                return _record_recovery(response, error, state.expected_tool)
        finally:
            recovery_state.reset(token)

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Wrap one asynchronous model invocation."""
        state = RecoveryState(buffer_response=self.buffer_response)
        token = recovery_state.set(state)
        try:
            try:
                return _check_agent_response(await handler(request))
            except MalformedToolArguments as error:
                if state.failure is None or state.failure[0] is not error:
                    raise
                failed = state.failure[1]
                retry = _retry_request(request, failed, error)
                state.expected_tool = failed.invalid_tool_calls[0]["name"]
                try:
                    response = await handler(retry)
                except ExaoneAPIError as rejection:
                    if not _named_choice_unsupported(rejection):
                        raise
                    state.required = True
                    response = await handler(retry.override(tool_choice="required"))
                return _record_recovery(response, error, state.expected_tool)
        finally:
            recovery_state.reset(token)
