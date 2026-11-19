"""Chat model support for the EXAONE API."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction
from operator import itemgetter
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.model_profile import ModelProfile
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import add_usage
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import ConfigDict, Field, SecretStr, ValidationInfo, field_validator

from langchain_exaone._profiles import get_model_profile
from langchain_exaone._recovery import (
    RecoveryStream,
    check_completed_message,
    check_message,
    check_stream,
    recoverable_tools,
    recovery_state,
    reject_json_constant,
)

_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
_SAFE_HEADERS = {"content-type", "retry-after", "retry-after-ms", "x-request-id"}
_HTTPX_ERRORS = (httpx.HTTPError, httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError)


def _new_metadata_fields(value: Mapping[str, Any], seen: dict[str, Any]) -> dict[str, Any]:
    """Retain new metadata fields without repeating unchanged snapshot values."""
    result: dict[str, Any] = {}
    for key, item in value.items():
        previous = seen.get(key)
        if isinstance(item, Mapping) and isinstance(previous, dict):
            nested = _new_metadata_fields(item, previous)
            if nested:
                result[key] = nested
        elif key not in seen or previous != item:
            result[key] = item
            seen[key] = deepcopy(item)
    return result


def _is_hosted_default(data: Mapping[str, Any]) -> bool:
    return data.get("model") == "k-exaone_v2.5" and data.get("base_url", "").rstrip("/") == "https://api.exaone.ai/v1"


class ExaoneAPIError(Exception):
    """An EXAONE API error that does not retain its authenticated request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str | None = None,
        code: Any = None,
        param: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize an API error without retaining authenticated request data.

        Args:
            message: Sanitized error message returned to the caller.
            status_code: HTTP status code from the EXAONE endpoint.
            error_type: Optional API error type.
            code: Optional API error code.
            param: Optional request parameter associated with the error.
            headers: Allowlisted diagnostic and rate-limit response headers.
        """
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.headers = dict(headers or {})


def _safe_headers(headers: Mapping[str, str], api_key: str) -> dict[str, str]:
    return {
        name.lower(): _redact_secret(value, api_key)
        for name, value in headers.items()
        if name.lower() in _SAFE_HEADERS
        or name.lower().startswith(("x-second-rate-limit", "x-minute-rate-limit", "x-day-rate-limit"))
    }


def _redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        try:
            variants = {secret, quote(secret, safe=""), quote_plus(secret, safe="")}
        except UnicodeError:
            variants = {secret}
        for variant in sorted(variants, key=len, reverse=True):
            if variant:
                value = value.replace(variant, "[REDACTED]")
                encoded_pattern = re.sub(
                    r"%[0-9A-F]{2}", lambda match: f"(?i:{match.group()})", re.escape(variant), flags=re.IGNORECASE
                )
                value = re.sub(encoded_pattern, "[REDACTED]", value)
        return value
    if isinstance(value, Mapping):
        return {_redact_secret(key, secret): _redact_secret(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    return value


def _copy_transport_error(error: Exception, api_key: str) -> Exception:
    """Copy an HTTPX error without retaining its authenticated request or cause."""
    raw_message = error.args[0] if error.args and isinstance(error.args[0], str) else "HTTP transport error"
    message = _redact_secret(raw_message, api_key)
    if isinstance(error, httpx.HTTPStatusError):
        safe_request = _safe_request(error, api_key) or httpx.Request("POST", "https://api.exaone.ai/")
        safe_response = httpx.Response(error.response.status_code, request=safe_request)
        return httpx.HTTPStatusError(message, request=safe_request, response=safe_response)

    for error_type in type(error).__mro__:
        if getattr(httpx, error_type.__name__, None) is not error_type:
            continue
        if error_type in (httpx.RequestNotRead, httpx.ResponseNotRead, httpx.StreamClosed, httpx.StreamConsumed):
            return error_type()
        if issubclass(error_type, httpx.RequestError):
            return error_type(message, request=_safe_request(error, api_key))
        if issubclass(error_type, (httpx.HTTPError, httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError)):
            return error_type(message)
    return httpx.HTTPError(message)


def _safe_request(error: Exception, api_key: str) -> httpx.Request | None:
    try:
        request = error.request  # type: ignore[attr-defined]
    except Exception:
        return None
    if not isinstance(request, httpx.Request):
        return None
    safe_url = request.url.copy_with(username=None, password=None, query=None, fragment=None)
    return httpx.Request(request.method, _redact_secret(str(safe_url), api_key))


def _safe_transport_errors(method: Any) -> Any:
    """Replace propagated HTTPX errors after the request scope has unwound."""
    if isasyncgenfunction(method):

        @wraps(method)
        async def wrap_async_generator(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            safe_error: Exception | None = None
            try:
                async for item in method(self, *args, **kwargs):
                    yield item
            except _HTTPX_ERRORS as error:
                safe_error = _copy_transport_error(error, self.api_key.get_secret_value())
            if safe_error is not None:
                raise safe_error

        return wrap_async_generator

    if isgeneratorfunction(method):

        @wraps(method)
        def wrap_generator(self: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
            safe_error: Exception | None = None
            try:
                yield from method(self, *args, **kwargs)
            except _HTTPX_ERRORS as error:
                safe_error = _copy_transport_error(error, self.api_key.get_secret_value())
            if safe_error is not None:
                raise safe_error

        return wrap_generator

    if iscoroutinefunction(method):

        @wraps(method)
        async def wrap_coroutine(self: Any, *args: Any, **kwargs: Any) -> Any:
            safe_error: Exception | None = None
            try:
                return await method(self, *args, **kwargs)
            except _HTTPX_ERRORS as error:
                safe_error = _copy_transport_error(error, self.api_key.get_secret_value())
            assert safe_error is not None
            raise safe_error

        return wrap_coroutine

    @wraps(method)
    def wrap_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
        safe_error: Exception | None = None
        try:
            return method(self, *args, **kwargs)
        except _HTTPX_ERRORS as error:
            safe_error = _copy_transport_error(error, self.api_key.get_secret_value())
        assert safe_error is not None
        raise safe_error

    return wrap_sync


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Honor the longer valid retry-after hint, including EXAONE milliseconds."""
    hints: list[float] = []
    for name, divisor in (("retry-after", 1.0), ("retry-after-ms", 1000.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(raw)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                value = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                continue
        if value >= 0.0:
            hints.append(value / divisor)
    return max(hints) if hints else None


def _api_error(status: int, headers: Mapping[str, str], body: bytes, api_key: str) -> ExaoneAPIError:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        payload = {}
    envelope = payload if isinstance(payload, dict) else {}
    nested_error = envelope.get("error")
    error = nested_error if isinstance(nested_error, dict) else envelope
    message = str(
        error.get("message")
        or error.get("description")
        or envelope.get("message")
        or envelope.get("description")
        or f"EXAONE API request failed with HTTP {status}"
    )
    message = _redact_secret(message, api_key)
    code = error.get("code")
    if code is None:
        code = error.get("result_code")
    if code is None:
        code = envelope.get("result_code")
    return ExaoneAPIError(
        message,
        status_code=status,
        error_type=_redact_secret(error["type"], api_key) if isinstance(error.get("type"), str) else None,
        code=_redact_secret(code, api_key),
        param=_redact_secret(error.get("param"), api_key),
        headers=_safe_headers(headers, api_key),
    )


def _load_response_json(payload: bytes | str, description: str) -> Any:
    """Decode JSON without retaining an invalid response body in the exception."""
    valid = True
    try:
        result = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        valid = False
        result = None
    if not valid:
        raise ValueError(f"EXAONE returned malformed JSON in the {description}.")
    return result


def _image_part(block: Mapping[str, Any]) -> dict[str, Any]:
    if block.get("type") == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, Mapping) and isinstance(image_url.get("url"), str):
            return {"type": "image_url", "image_url": dict(image_url)}
    elif isinstance(block.get("url"), str):
        return {"type": "image_url", "image_url": {"url": block["url"]}}
    else:
        data = block.get("base64", block.get("data"))
        mime = block.get("mime_type")
        if isinstance(data, str) and isinstance(mime, str) and mime.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
    raise TypeError("EXAONE image blocks require a URL or base64 data with an image MIME type")


def _content_parts(message: BaseMessage) -> tuple[str | list[dict[str, Any]], str | None, list[dict[str, Any]]]:
    content = message.content
    text: list[str] = []
    content_parts: list[dict[str, Any]] = []
    has_image = False
    reasoning: list[str] = []
    refusals: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, str):
        text.append(content)
    elif isinstance(content, Sequence):
        for block in content:
            if isinstance(block, str):
                text.append(block)
                content_parts.append({"type": "text", "text": block})
            elif isinstance(block, Mapping):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text.append(block["text"])
                    content_parts.append({"type": "text", "text": block["text"]})
                elif isinstance(message, HumanMessage) and block.get("type") in {"image", "image_url"}:
                    content_parts.append(_image_part(block))
                    has_image = True
                elif (
                    isinstance(message, AIMessage)
                    and block.get("type") == "reasoning"
                    and isinstance(block.get("reasoning"), str)
                ):
                    reasoning.append(block["reasoning"])
                elif (
                    isinstance(message, AIMessage)
                    and block.get("type") == "refusal"
                    and isinstance(block.get("refusal"), str)
                ):
                    refusals.append(block["refusal"])
                elif isinstance(message, AIMessage) and block.get("type") == "tool_call":
                    name, args, call_id = block.get("name"), block.get("args"), block.get("id")
                    if (
                        not isinstance(name, str)
                        or not isinstance(args, dict)
                        or (call_id is not None and not isinstance(call_id, str))
                    ):
                        raise TypeError(
                            "EXAONE tool call blocks require a name, object arguments, and optional string id"
                        )
                    tool_calls.append({"id": call_id, "name": name, "args": args})
                else:
                    raise TypeError(f"EXAONE does not support message block type {block.get('type')!r}")
            else:
                raise TypeError("EXAONE supports only text and reasoning content blocks")
    else:
        raise TypeError("EXAONE message content must be text or text/reasoning blocks")
    if isinstance(message, AIMessage):
        if message.response_metadata.get("output_version") != "v1" and not refusals:
            prior_refusal = message.additional_kwargs.get("refusal")
            if isinstance(prior_refusal, str):
                refusals.append(prior_refusal)
        if not "".join(text).strip() and refusals:
            text = refusals
        prior = message.additional_kwargs.get("reasoning_content")
        if (
            message.response_metadata.get("output_version") != "v1"
            and not reasoning
            and isinstance(prior, str)
            and prior
        ):
            reasoning.append(prior)
    return content_parts if has_image else "".join(text), "".join(reasoning) or None, tool_calls


def _message_to_api(message: BaseMessage) -> dict[str, Any]:
    content, reasoning, block_tool_calls = _content_parts(message)
    if isinstance(message, SystemMessage):
        return _named_message(message, "system", content)
    if isinstance(message, HumanMessage):
        return _named_message(message, "user", content)
    if isinstance(message, ToolMessage):
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}
    if isinstance(message, FunctionMessage):
        return {"role": "function", "name": message.name, "content": content}
    if isinstance(message, ChatMessage):
        return _named_message(message, message.role, content)
    if isinstance(message, AIMessage):
        result = _named_message(message, "assistant", content)
        if reasoning is not None:
            result["reasoning_content"] = reasoning
        raw_calls = message.additional_kwargs.get("tool_calls")
        normalized = (
            message.response_metadata.get("model_provider") == "exaone" and "tool_calls" in message.model_fields_set
        )
        calls = message.tool_calls if normalized else message.tool_calls or block_tool_calls
        if calls or message.invalid_tool_calls:
            originals = raw_calls if isinstance(raw_calls, list) else []
            result["tool_calls"] = [_tool_call_to_api(call, originals, invalid=False) for call in calls] + [
                _tool_call_to_api(call, originals, invalid=True) for call in message.invalid_tool_calls
            ]
        elif not normalized and isinstance(raw_calls, list) and raw_calls:
            result["tool_calls"] = raw_calls
        function_call = message.additional_kwargs.get("function_call")
        if isinstance(function_call, Mapping):
            result["function_call"] = function_call
        return result
    raise TypeError(f"EXAONE does not support LangChain message type {message.type!r}")


def _named_message(message: BaseMessage, role: str, content: str | list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"role": role, "content": content}
    if message.name is not None:
        result["name"] = message.name
    return result


def _tool_call_to_api(call: Mapping[str, Any], originals: list[Any], *, invalid: bool) -> dict[str, Any]:
    call_id, name, args = call.get("id"), call.get("name"), call.get("args")
    original = next(
        (item for item in originals if call_id is not None and isinstance(item, Mapping) and item.get("id") == call_id),
        {},
    )
    function = original.get("function")
    function = dict(function) if isinstance(function, Mapping) else {}
    if not isinstance(name, str) or (not isinstance(args, str) if invalid else not isinstance(args, dict)):
        raise TypeError("EXAONE tool calls require a name and valid arguments")
    encoded = args if invalid else _tool_arguments(args, function.get("arguments"))
    return {
        **original,
        "id": call_id,
        "type": "function",
        "function": {**function, "name": name, "arguments": encoded},
    }


def _tool_arguments(arguments: dict[str, Any], original: Any) -> str:
    if isinstance(original, str):
        try:
            if json.dumps(json.loads(original), sort_keys=True, ensure_ascii=False) == json.dumps(
                arguments, sort_keys=True, ensure_ascii=False
            ):
                return original
        except ValueError:
            pass
    return json.dumps(arguments, ensure_ascii=False)


def _usage_metadata(usage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not usage:
        return None
    values = (usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))
    if any(type(value) is not int or value < 0 for value in values):
        return None
    result: dict[str, Any] = {"input_tokens": values[0], "output_tokens": values[1], "total_tokens": values[2]}
    input_details = usage.get("prompt_tokens_details")
    if isinstance(input_details, Mapping):
        mapped_input = {
            target: value
            for source, target in (
                ("cached_tokens", "cache_read"),
                ("cache_write_tokens", "cache_creation"),
            )
            if type(value := input_details.get(source)) is int and value >= 0
        }
        if mapped_input:
            result["input_token_details"] = mapped_input
    details = usage.get("completion_tokens_details")
    reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens")
    if type(reasoning_tokens) is int and reasoning_tokens >= 0:
        result["output_token_details"] = {"reasoning": reasoning_tokens}
    return result


def _tool_calls(raw_calls: Any, *, strict_json: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    if raw_calls is None:
        return valid, invalid
    if not isinstance(raw_calls, list):
        raise ValueError("EXAONE response tool_calls must be an array")
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise ValueError("EXAONE response tool call entries must be objects")
        if raw.get("type") is not None and raw["type"] != "function":
            raise ValueError("EXAONE returned an unsupported tool call type")
        if not isinstance(raw.get("function"), Mapping):
            raise ValueError("EXAONE response tool call function must be an object")
        function = raw["function"]
        name, arguments, call_id = function.get("name"), function.get("arguments", ""), raw.get("id")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise ValueError("EXAONE response tool call name and arguments must be strings")
        if call_id is not None and not isinstance(call_id, str):
            raise ValueError("EXAONE response tool call id must be a string")
        try:
            parsed = json.loads(arguments, parse_constant=reject_json_constant if strict_json else None)
        except json.JSONDecodeError as exc:
            # JSONDecodeError.msg and the location do not include the input text.
            error = f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
            invalid.append({"name": name, "args": arguments, "id": call_id, "error": error})
            continue
        except ValueError:
            invalid.append({"name": name, "args": arguments, "id": call_id, "error": "Arguments were not valid JSON."})
            continue
        if not isinstance(parsed, dict):
            invalid.append(
                {"name": name, "args": arguments, "id": call_id, "error": "Arguments must be a JSON object."}
            )
            continue
        valid.append({"name": name, "args": parsed, "id": call_id, "type": "tool_call"})
    return valid, invalid


def _response_reasoning(source: Mapping[str, Any], context: str) -> str | None:
    """Read equivalent EXAONE and vLLM reasoning fields without silently losing data."""
    original, alternate = source.get("reasoning_content"), source.get("reasoning")
    for name, value in (("reasoning_content", original), ("reasoning", alternate)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"EXAONE {context} {name} must be a string")
    if original and alternate and original != alternate:
        raise ValueError(f"EXAONE {context} has conflicting reasoning and reasoning_content fields")
    return original or alternate


def _assistant_message(
    choice: Mapping[str, Any], response: Mapping[str, Any], api_key: str, *, strict_json: bool = False
) -> AIMessage:
    raw = choice.get("message")
    if not isinstance(raw, Mapping):
        raise ValueError("EXAONE response choice is missing its assistant message")
    content, refusal = raw.get("content"), raw.get("refusal")
    if content is not None and not isinstance(content, str):
        raise ValueError("EXAONE assistant content must be a string")
    reasoning = _response_reasoning(raw, "assistant")
    if refusal is not None and not isinstance(refusal, str):
        raise ValueError("EXAONE assistant refusal must be a string")
    additional: dict[str, Any] = {}
    if isinstance(reasoning, str) and reasoning:
        additional["reasoning_content"] = reasoning
    if isinstance(refusal, str):
        additional["refusal"] = refusal.replace(api_key, "[REDACTED]")
    for name in ("annotations", "audio"):
        if raw.get(name) is not None:
            additional[name] = raw[name]
    if isinstance(raw.get("tool_calls"), list):
        additional["tool_calls"] = raw["tool_calls"]
    if isinstance(raw.get("function_call"), Mapping):
        additional["function_call"] = raw["function_call"]
    valid, invalid = _tool_calls(raw.get("tool_calls"), strict_json=strict_json)
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else None
    metadata = {
        "model_provider": "exaone",
        "model_name": response.get("model"),
        "id": response.get("id"),
        "created": response.get("created"),
        "finish_reason": choice.get("finish_reason"),
        "matched_stop": choice.get("matched_stop"),
        "logprobs": choice.get("logprobs"),
        "usage": usage,
        "object": response.get("object"),
        "metadata": response.get("metadata"),
        "moderation": response.get("moderation"),
        "service_tier": response.get("service_tier"),
        "system_fingerprint": response.get("system_fingerprint"),
        "result_code": _redact_secret(response.get("result_code"), api_key),
        "description": _redact_secret(response.get("description"), api_key),
    }
    return AIMessage(
        content=content or "",
        additional_kwargs=additional,
        response_metadata=metadata,
        usage_metadata=_usage_metadata(usage),
        tool_calls=valid,
        invalid_tool_calls=invalid,
    )


def _decode_sse(lines: Iterator[str], complete: list[bool] | None = None) -> Iterator[dict[str, Any]]:
    pending: list[str] = []
    for line in lines:
        if not line:
            if pending:
                payload = "\n".join(pending)
                pending.clear()
                if payload == "[DONE]":
                    if complete is not None:
                        complete[0] = True
                    return
                data = _load_response_json(payload, "event stream")
                if not isinstance(data, dict):
                    raise ValueError("EXAONE stream event must be a JSON object")
                yield data
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip())
    if pending:
        payload = "\n".join(pending)
        if payload == "[DONE]" and complete is not None:
            complete[0] = True
        if payload != "[DONE]":
            data = _load_response_json(payload, "event stream")
            if not isinstance(data, dict):
                raise ValueError("EXAONE stream event must be a JSON object")
            yield data


async def _decode_sse_async(lines: Any, complete: list[bool] | None = None) -> AsyncIterator[dict[str, Any]]:
    pending: list[str] = []
    async for line in lines:
        if not line:
            if pending:
                payload = "\n".join(pending)
                pending.clear()
                if payload == "[DONE]":
                    if complete is not None:
                        complete[0] = True
                    return
                data = _load_response_json(payload, "event stream")
                if not isinstance(data, dict):
                    raise ValueError("EXAONE stream event must be a JSON object")
                yield data
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip())
    if pending:
        payload = "\n".join(pending)
        if payload == "[DONE]" and complete is not None:
            complete[0] = True
        if payload != "[DONE]":
            data = _load_response_json(payload, "event stream")
            if not isinstance(data, dict):
                raise ValueError("EXAONE stream event must be a JSON object")
            yield data


class ChatEXAONE(BaseChatModel):
    """LangChain chat model for the K-EXAONE 2.5 Chat Completions API.

    The API key defaults to the ``EXAONE_API_KEY`` environment variable. The
    model preserves EXAONE reasoning on assistant messages so it can be sent
    back when the caller includes those messages in later turns.

    Example:
        .. code-block:: python

            from langchain_exaone import ChatEXAONE

            model = ChatEXAONE()
            response = model.invoke("Hello")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model: str = Field(default="k-exaone_v2.5", description="EXAONE model ID.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("EXAONE_API_KEY", "")),
        repr=False,
        description="EXAONE API key. Defaults to the EXAONE_API_KEY environment variable.",
    )
    base_url: str = Field(
        default="https://api.exaone.ai/v1",
        description="Chat Completions API base URL.",
    )
    temperature: float | None = Field(default=1.0, description="Sampling temperature.")
    top_p: float | None = Field(default=0.95, description="Nucleus sampling value.")
    presence_penalty: float | None = Field(default=0.0, description="Presence penalty.")
    max_tokens: int | None = Field(default=None, description="Maximum generated tokens, including reasoning tokens.")
    top_k: int = Field(default=-1, description="Top-k sampling value; -1 disables top-k filtering.")
    reasoning_effort: str | float | None = Field(
        default=None,
        description="EXAONE reasoning effort. Omit to use the service default.",
    )
    preserve_thinking: bool = Field(
        default=True,
        description="Preserve EXAONE reasoning across turns.",
    )
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional EXAONE chat-template options.",
    )
    timeout: float | None = Field(
        default=None,
        description="HTTP timeout in seconds. Omit for no provider-imposed timeout.",
    )
    max_retries: int = Field(default=2, description="Maximum retries before output is returned.")
    max_retry_delay: float = Field(default=60.0, description="Maximum retry delay in seconds.")
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional endpoint-specific request fields. Reserved fields are provider-controlled.",
    )
    http_client: httpx.Client | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Optional synchronous HTTPX client for connection reuse or custom transport.",
    )
    http_async_client: httpx.AsyncClient | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Optional asynchronous HTTPX client for connection reuse or custom transport.",
    )

    def _resolve_model_profile(self) -> ModelProfile | None:
        """Supply the default profile through LangChain's model-resolution hook."""
        if not _is_hosted_default({"model": self.model, "base_url": self.base_url}):
            return None
        return get_model_profile(self.model)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
        except ValueError as error:
            raise ValueError("base_url must be a valid absolute URL") from error
        if not hostname or parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not include URL credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not include a query or fragment")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, value: int | None, info: ValidationInfo) -> int | None:
        if value is not None and value < 0:
            raise ValueError("max_tokens must be non-negative")
        if value is not None and value > 524_288 and _is_hosted_default(info.data):
            raise ValueError("max_tokens cannot exceed the EXAONE context limit of 524288")
        return value

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("temperature must be a finite non-negative number")
        return value

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("top_p must be between 0 and 1")
        return value

    @field_validator("presence_penalty")
    @classmethod
    def _check_presence_penalty(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not -2 <= value <= 2):
            raise ValueError("presence_penalty must be between -2 and 2")
        return value

    @field_validator("max_retries")
    @classmethod
    def _check_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_retries must be non-negative")
        return value

    @field_validator("timeout")
    @classmethod
    def _check_timeout(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout must be a finite positive number")
        return value

    @field_validator("max_retry_delay")
    @classmethod
    def _check_max_retry_delay(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("max_retry_delay must be a finite non-negative number")
        return value

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, value: int, info: ValidationInfo) -> int:
        if value < -1 or (value == 0 and _is_hosted_default(info.data)):
            raise ValueError("top_k is not supported by the selected endpoint")
        return value

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _check_reasoning_effort(cls, value: Any, info: ValidationInfo) -> Any:
        if value is not None and not (isinstance(value, str) or (type(value) in (int, float) and math.isfinite(value))):
            raise ValueError("reasoning_effort must be a string or finite number")
        if value is not None and _is_hosted_default(info.data) and value not in ("low", "medium", "high", "max"):
            raise ValueError("reasoning_effort must be one of low, medium, high, or max for the EXAONE API")
        return value

    @property
    def _llm_type(self) -> str:
        return "exaone-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "base_url": self.base_url}
        state = recovery_state.get()
        if state is not None:
            # Cached streams may no longer carry the original tool argument fragments.
            params["_exaone_tool_call_recovery"] = (2, state.expected_tool, state.required)
        return params

    def _key(self) -> str:
        key = self.api_key.get_secret_value()
        if not key:
            raise ValueError("An EXAONE API key is required. Set EXAONE_API_KEY or pass api_key.")
        return key

    def _body(
        self, messages: list[BaseMessage], stop: list[str] | None, stream: bool, kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        serialized_messages = [_message_to_api(message) for message in messages]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": serialized_messages,
            "stream": stream,
        }
        for name in ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens", "reasoning_effort"):
            value = getattr(self, name)
            if value is not None:
                body[name] = value
        if stream:
            body["stream_options"] = {"include_usage": True}
        body.update({name: value for name, value in self.model_kwargs.items() if value is not None})
        body.update({name: value for name, value in kwargs.items() if value is not None})
        if stop is not None:
            body["stop"] = stop
        template_kwargs = dict(self.chat_template_kwargs)
        for source in (self.model_kwargs, kwargs):
            custom_template_kwargs = source.get("chat_template_kwargs")
            if isinstance(custom_template_kwargs, Mapping):
                template_kwargs.update(custom_template_kwargs)
        body["chat_template_kwargs"] = {**template_kwargs, "preserve_thinking": self.preserve_thinking}
        body["model"] = kwargs.get("model") or self.model
        body["messages"] = serialized_messages
        body["stream"] = stream
        state = recovery_state.get()
        if state is not None and state.expected_tool is not None:
            recoverable_tools(body, state)
        return body

    @contextmanager
    def _sync_client(self) -> Iterator[httpx.Client]:
        if self.http_client is not None:
            yield self.http_client
        else:
            with httpx.Client(timeout=self.timeout) as client:
                yield client

    @asynccontextmanager
    async def _async_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.http_async_client is not None:
            yield self.http_async_client
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                yield client

    def _raise_api_error(self, response: httpx.Response, api_key: str) -> ExaoneAPIError:
        body = response.content if response.is_stream_consumed else response.read()
        return _api_error(response.status_code, response.headers, body, api_key)

    async def _araise_api_error(self, response: httpx.Response, api_key: str) -> ExaoneAPIError:
        body = response.content if response.is_stream_consumed else await response.aread()
        return _api_error(response.status_code, response.headers, body, api_key)

    def _delay(self, error: ExaoneAPIError, attempt: int) -> float | None:
        hint = _retry_after(error.headers)
        if hint is not None:
            return hint if hint <= self.max_retry_delay else None
        ceiling = min(self.max_retry_delay, 1.0 * 2 ** min(attempt, 20))
        if ceiling < 1.0:
            return None
        return random.uniform(1.0, ceiling)

    def _combine_llm_outputs(self, llm_outputs: list[dict[str, Any] | None]) -> dict[str, Any]:
        """Aggregate request-level usage once per prompt for LangChain's generate result."""
        total: dict[str, Any] | None = None
        models: set[str] = set()
        for output in llm_outputs:
            if output is None:
                continue
            model = output.get("model")
            if isinstance(model, str):
                models.add(model)
            raw_usage = output.get("usage")
            usage = _usage_metadata(raw_usage) if isinstance(raw_usage, Mapping) else None
            if usage is not None:
                total = add_usage(total, usage)
        result: dict[str, Any] = {}
        if len(models) == 1:
            result["model_name"] = next(iter(models))
        if total is not None:
            token_usage = {
                "prompt_tokens": total["input_tokens"],
                "completion_tokens": total["output_tokens"],
                "total_tokens": total["total_tokens"],
            }
            reasoning = total.get("output_token_details", {}).get("reasoning")
            if reasoning is not None:
                token_usage["reasoning_tokens"] = reasoning
            prompt_details = total.get("input_token_details", {})
            mapped_prompt = {
                target: prompt_details[source]
                for source, target in (
                    ("cache_read", "cached_tokens"),
                    ("cache_creation", "cache_write_tokens"),
                )
                if source in prompt_details
            }
            if mapped_prompt:
                token_usage["prompt_tokens_details"] = mapped_prompt
            result["token_usage"] = token_usage
        return result

    def _parse_result(self, payload: Any, api_key: str, request: Mapping[str, Any]) -> ChatResult:
        if not isinstance(payload, Mapping):
            raise ValueError("EXAONE response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("EXAONE response did not contain choices")
        if any(not isinstance(choice, Mapping) for choice in choices):
            raise ValueError("EXAONE response choices must be objects")
        state = recovery_state.get()
        generations = [
            ChatGeneration(
                message=self._standard_refusal(
                    _assistant_message(choice, payload, api_key, strict_json=state is not None)
                ),
                generation_info={
                    key: choice.get(key)
                    for key in ("finish_reason", "matched_stop", "logprobs")
                    if choice.get(key) is not None
                },
            )
            for choice in choices
        ]
        if state is not None:
            if state.expected_tool is not None:
                if len(generations) != 1:
                    raise ValueError("EXAONE recovery must return exactly one completion choice")
                check_completed_message(generations[0].message, state.expected_tool)
            else:
                tools = recoverable_tools(request, state)
                for generation in generations:
                    check_message(generation.message, api_key, tools=tools, allow_retry=len(generations) == 1)
        return ChatResult(
            generations=generations, llm_output={"model": payload.get("model"), "usage": payload.get("usage")}
        )

    def _standard_refusal(self, message: AIMessage) -> AIMessage:
        refusal = message.additional_kwargs.get("refusal")
        if self.output_version == "v1" and isinstance(refusal, str) and refusal:
            message.content = [*message.content_blocks, {"type": "refusal", "refusal": refusal}]
            message.response_metadata["output_version"] = "v1"
        return message

    @_safe_transport_errors
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        key, body = self._key(), self._body(messages, stop, False, kwargs)
        with self._sync_client() as client:
            for attempt in range(self.max_retries + 1):
                response = client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                )
                if response.is_error:
                    error = self._raise_api_error(response, key)
                    response.close()
                    if error.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                        delay = self._delay(error, attempt)
                        if delay is not None:
                            time.sleep(delay)
                            continue
                    raise error
                try:
                    payload = _load_response_json(response.content, "response body")
                finally:
                    response.close()
                return self._parse_result(payload, key, body)
        raise RuntimeError("EXAONE retry loop exited without a response")

    @_safe_transport_errors
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        key, body = self._key(), self._body(messages, stop, False, kwargs)
        async with self._async_client() as client:
            for attempt in range(self.max_retries + 1):
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                )
                if response.is_error:
                    error = await self._araise_api_error(response, key)
                    await response.aclose()
                    if error.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                        delay = self._delay(error, attempt)
                        if delay is not None:
                            await asyncio.sleep(delay)
                            continue
                    raise error
                try:
                    payload = _load_response_json(response.content, "response body")
                finally:
                    await response.aclose()
                return self._parse_result(payload, key, body)
        raise RuntimeError("EXAONE retry loop exited without a response")

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | Mapping[str, Any] | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Bind LangChain tools using the EXAONE Chat Completions schema."""
        formatted = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        if tool_choice is True or tool_choice == "any":
            tool_choice = "required"
        elif tool_choice is False:
            tool_choice = None
        if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def with_structured_output(self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any) -> Any:
        """Return schema-constrained output using a LangChain structured-output method."""
        method = kwargs.pop("method", None)
        if method is None:
            method = "function_calling"
        strict = kwargs.pop("strict", None)
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        if strict is not None and not isinstance(strict, bool):
            raise TypeError("strict must be a boolean or None")
        if method not in ("function_calling", "json_mode", "json_schema"):
            raise ValueError(f"Unsupported structured output method: {method}")
        if strict and method == "json_mode":
            raise ValueError("strict is not supported with the json_mode method")
        if method == "function_calling":
            if strict is None:
                return super().with_structured_output(schema, include_raw=include_raw)
            tool = convert_to_openai_tool(schema, strict=strict)
            llm = self.bind_tools(
                [tool],
                tool_choice="any",
                ls_structured_output_format={"kwargs": {"method": "function_calling"}, "schema": schema},
            )
            parser = (
                PydanticToolsParser(tools=[schema], first_tool_only=True)
                if isinstance(schema, type) and is_basemodel_subclass(schema)
                else JsonOutputKeyToolsParser(key_name=tool["function"]["name"], first_tool_only=True)
            )
        elif method == "json_mode":
            response_format: dict[str, Any] = {"type": "json_object"}
        else:
            if (
                isinstance(schema, dict)
                and schema.get("type") != "function"
                and not ("name" in schema and "parameters" in schema)
            ):
                name = schema.get("title", "extract")
                parameters = schema
            else:
                function = convert_to_openai_tool(schema)["function"]
                name = function["name"]
                parameters = function["parameters"]
            json_schema: dict[str, Any] = {"name": name, "schema": parameters}
            if strict is not None:
                json_schema["strict"] = strict
            response_format = {"type": "json_schema", "json_schema": json_schema}

        if method != "function_calling":
            llm = self.bind(response_format=response_format)
            parser = (
                PydanticOutputParser(pydantic_object=schema)
                if isinstance(schema, type) and is_basemodel_subclass(schema)
                else JsonOutputParser()
            )
        if not include_raw:
            return llm | parser
        parser_assign = RunnablePassthrough.assign(parsed=itemgetter("raw") | parser, parsing_error=lambda _: None)
        parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
        return RunnableMap(raw=llm) | parser_assign.with_fallbacks([parser_none], exception_key="parsing_error")

    @_safe_transport_errors
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        key, body = self._key(), self._body(messages, stop, True, kwargs)
        with self._sync_client() as client:
            for attempt in range(self.max_retries + 1):
                with client.stream(
                    "POST",
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                ) as response:
                    if response.is_error:
                        error = self._raise_api_error(response, key)
                        if error.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                            delay = self._delay(error, attempt)
                            if delay is not None:
                                time.sleep(delay)
                                continue
                        raise error
                    seen_metadata: dict[str, Any] = {}
                    complete = [False]
                    payloads = _decode_sse(response.iter_lines(), complete)
                    state = recovery_state.get()
                    if state is not None:
                        router = RecoveryStream(state, recoverable_tools(body, state))
                        payloads = self._guard_stream(payloads, complete, router, key)
                    for payload in payloads:
                        for chunk in self._stream_payload(payload, run_manager, key):
                            self._deduplicate_stream_metadata(chunk, seen_metadata)
                            yield chunk
                    return
        raise RuntimeError("EXAONE retry loop exited without a response")

    @_safe_transport_errors
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        key, body = self._key(), self._body(messages, stop, True, kwargs)
        async with self._async_client() as client:
            for attempt in range(self.max_retries + 1):
                async with client.stream(
                    "POST",
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                ) as response:
                    if response.is_error:
                        error = await self._araise_api_error(response, key)
                        if error.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                            delay = self._delay(error, attempt)
                            if delay is not None:
                                await asyncio.sleep(delay)
                                continue
                        raise error
                    seen_metadata: dict[str, Any] = {}
                    complete = [False]
                    payloads = _decode_sse_async(response.aiter_lines(), complete)
                    state = recovery_state.get()
                    if state is not None:
                        router = RecoveryStream(state, recoverable_tools(body, state))
                        payloads = self._aguard_stream(payloads, complete, router, key)
                    async for payload in payloads:
                        for chunk in self._stream_payload(payload, None, key):
                            self._deduplicate_stream_metadata(chunk, seen_metadata)
                            yield chunk
                            if run_manager is not None:
                                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    return
        raise RuntimeError("EXAONE retry loop exited without a response")

    def _guard_stream(
        self, payloads: Iterator[dict[str, Any]], complete: list[bool], router: RecoveryStream, key: str
    ) -> Iterator[dict[str, Any]]:
        for payload in payloads:
            yield from router.accept(payload)
        yield from self._validate_held_payloads(router.finish(complete[0]), router, key)

    async def _aguard_stream(
        self, payloads: AsyncIterator[dict[str, Any]], complete: list[bool], router: RecoveryStream, key: str
    ) -> AsyncIterator[dict[str, Any]]:
        async for payload in payloads:
            for item in router.accept(payload):
                yield item
        for item in self._validate_held_payloads(router.finish(complete[0]), router, key):
            yield item

    def _validate_held_payloads(
        self, payloads: list[dict[str, Any]], router: RecoveryStream, key: str
    ) -> list[dict[str, Any]]:
        for payload in payloads:
            list(self._stream_payload(payload, None, key))
        check_stream(payloads, key, router)
        return payloads

    @staticmethod
    def _deduplicate_stream_metadata(chunk: ChatGenerationChunk, seen: dict[str, Any]) -> None:
        """Avoid merging identical response-level fields repeatedly across chunks."""
        metadata = chunk.message.response_metadata
        for name in (
            "model_name",
            "id",
            "created",
            "object",
            "metadata",
            "moderation",
            "service_tier",
            "system_fingerprint",
            "result_code",
            "description",
        ):
            value = metadata.get(name)
            if value is None:
                continue
            if name == "metadata" and isinstance(value, Mapping) and isinstance(seen.get(name), dict):
                remaining = _new_metadata_fields(value, seen[name])
                if remaining:
                    metadata[name] = remaining
                else:
                    metadata.pop(name)
            elif name in seen and seen[name] == value:
                metadata.pop(name)
            else:
                seen[name] = deepcopy(value)

    def _stream_payload(
        self, payload: Mapping[str, Any], run_manager: CallbackManagerForLLMRun | None, api_key: str
    ) -> Iterator[ChatGenerationChunk]:
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise ValueError("EXAONE stream event choices must be a list")
        response_fields = {
            key: _redact_secret(payload[key], api_key)
            for key in (
                "created",
                "object",
                "metadata",
                "moderation",
                "service_tier",
                "system_fingerprint",
                "result_code",
                "description",
            )
            if payload.get(key) is not None
        }
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            chunk = AIMessageChunk(
                content="", response_metadata={**response_fields, "usage": usage}, usage_metadata=_usage_metadata(usage)
            )
            yield ChatGenerationChunk(message=chunk, text="")
        if not choices:
            return
        if len(choices) > 1:
            raise ValueError("EXAONE returned multiple streamed choices; LangChain supports one choice per run")
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise ValueError("EXAONE stream choice must be an object")
            index = choice.get("index")
            if index is not None and (type(index) is not int or index != 0):
                raise ValueError(
                    "EXAONE returned a non-zero streamed choice index; LangChain supports one choice per run"
                )
            if not isinstance(choice.get("delta"), Mapping):
                raise ValueError("EXAONE stream choice is missing its delta object")
            delta = choice["delta"]
            text = delta.get("content")
            if text is None:
                text = ""
            reasoning = _response_reasoning(delta, "stream delta")
            refusal = delta.get("refusal")
            if not isinstance(text, str) or (refusal is not None and not isinstance(refusal, str)):
                raise ValueError("EXAONE stream delta content, reasoning, and refusal must be text")
            tool_chunks = []
            raw_tool_calls = delta.get("tool_calls")
            if raw_tool_calls is not None and not isinstance(raw_tool_calls, list):
                raise ValueError("EXAONE stream delta tool_calls must be an array")
            for tool in raw_tool_calls or []:
                if not isinstance(tool, Mapping):
                    raise ValueError("EXAONE stream tool call entries must be objects")
                if tool.get("type") is not None and tool["type"] != "function":
                    raise ValueError("EXAONE returned an unsupported tool call type")
                function = tool.get("function")
                if function is None:
                    function = {}
                elif not isinstance(function, Mapping):
                    raise ValueError("EXAONE stream tool call function must be an object")
                name = function.get("name")
                arguments = function.get("arguments", "")
                call_id = tool.get("id")
                tool_index = tool.get("index")
                if name is not None and not isinstance(name, str):
                    raise ValueError("EXAONE stream tool call name must be text")
                if arguments is not None and not isinstance(arguments, str):
                    raise ValueError("EXAONE stream tool call arguments must be text")
                if call_id is not None and not isinstance(call_id, str):
                    raise ValueError("EXAONE stream tool call id must be a string")
                if tool_index is not None and type(tool_index) is not int:
                    raise ValueError("EXAONE stream tool call index must be an integer")
                tool_chunks.append(
                    {
                        "name": name,
                        "args": arguments,
                        "id": call_id,
                        "index": tool_index,
                    }
                )
            additional = {"reasoning_content": reasoning} if reasoning else {}
            if isinstance(refusal, str):
                additional["refusal"] = refusal.replace(api_key, "[REDACTED]")
            response_metadata = {
                "model_provider": "exaone",
                "model_name": payload.get("model"),
                "id": payload.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "matched_stop": choice.get("matched_stop"),
                **response_fields,
            }
            chunk_content: str | list[dict[str, Any]] = text
            if self.output_version == "v1":
                blocks: list[dict[str, Any]] = []
                if reasoning:
                    blocks.append({"type": "reasoning", "reasoning": reasoning, "index": 0})
                if text:
                    blocks.append({"type": "text", "text": text, "index": 1})
                if isinstance(refusal, str) and refusal:
                    blocks.append({"type": "refusal", "refusal": refusal.replace(api_key, "[REDACTED]"), "index": 2})
                for tool_chunk in tool_chunks:
                    index = tool_chunk.get("index")
                    blocks.append(
                        {**tool_chunk, "type": "tool_call_chunk", "index": index + 3 if index is not None else 3}
                    )
                chunk_content = blocks
                response_metadata["output_version"] = "v1"
            chunk = AIMessageChunk(
                content=chunk_content,
                additional_kwargs=additional,
                response_metadata=response_metadata,
                tool_call_chunks=tool_chunks,
            )
            generation_info = {
                key: choice[key] for key in ("finish_reason", "matched_stop", "logprobs") if choice.get(key) is not None
            }
            generation = ChatGenerationChunk(message=chunk, text=text, generation_info=generation_info)
            yield generation
            if run_manager is not None:
                run_manager.on_llm_new_token(text, chunk=generation)
