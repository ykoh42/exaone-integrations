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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction
from typing import Any, Literal
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
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, SecretStr, field_validator

from langchain_exaone._profiles import get_model_profile

_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
_SAFE_HEADERS = {"content-type", "retry-after", "retry-after-ms", "x-request-id"}
_HTTPX_ERRORS = (httpx.HTTPError, httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError)


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


def _content_parts(message: BaseMessage) -> tuple[str, str | None, list[dict[str, Any]]]:
    content = message.content
    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, str):
        text.append(content)
    elif isinstance(content, Sequence):
        for block in content:
            if isinstance(block, str):
                text.append(block)
            elif isinstance(block, Mapping):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text.append(block["text"])
                elif (
                    isinstance(message, AIMessage)
                    and block.get("type") == "reasoning"
                    and isinstance(block.get("reasoning"), str)
                ):
                    reasoning.append(block["reasoning"])
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
        prior = message.additional_kwargs.get("reasoning_content")
        if not reasoning and isinstance(prior, str) and prior:
            reasoning.append(prior)
    return "".join(text), "".join(reasoning) or None, tool_calls


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


def _named_message(message: BaseMessage, role: str, content: str) -> dict[str, Any]:
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
    details = usage.get("completion_tokens_details")
    reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens")
    if type(reasoning_tokens) is int and reasoning_tokens >= 0:
        result["output_token_details"] = {"reasoning": reasoning_tokens}
    return result


def _tool_calls(raw_calls: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    if raw_calls is None:
        return valid, invalid
    if not isinstance(raw_calls, list):
        raise ValueError("EXAONE response tool_calls must be an array")
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise ValueError("EXAONE response tool call entries must be objects")
        if not isinstance(raw.get("function"), Mapping):
            raise ValueError("EXAONE response tool call function must be an object")
        function = raw["function"]
        name, arguments, call_id = function.get("name"), function.get("arguments", ""), raw.get("id")
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise ValueError("EXAONE response tool call name and arguments must be strings")
        if call_id is not None and not isinstance(call_id, str):
            raise ValueError("EXAONE response tool call id must be a string")
        try:
            parsed = json.loads(arguments)
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


def _assistant_message(choice: Mapping[str, Any], response: Mapping[str, Any], api_key: str) -> AIMessage:
    raw = choice.get("message")
    if not isinstance(raw, Mapping):
        raise ValueError("EXAONE response choice is missing its assistant message")
    content, refusal = raw.get("content"), raw.get("refusal")
    if content is not None and not isinstance(content, str):
        raise ValueError("EXAONE assistant content must be a string")
    if (
        (content is None or isinstance(content, str) and not content.strip())
        and isinstance(refusal, str)
        and refusal.strip()
    ):
        content = refusal
    additional: dict[str, Any] = {}
    reasoning = raw.get("reasoning_content")
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
    valid, invalid = _tool_calls(raw.get("tool_calls"))
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
    }
    return AIMessage(
        content=content or "",
        additional_kwargs=additional,
        response_metadata=metadata,
        usage_metadata=_usage_metadata(usage),
        tool_calls=valid,
        invalid_tool_calls=invalid,
    )


def _decode_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    pending: list[str] = []
    for line in lines:
        if not line:
            if pending:
                payload = "\n".join(pending)
                pending.clear()
                if payload != "[DONE]":
                    data = _load_response_json(payload, "event stream")
                    if not isinstance(data, dict):
                        raise ValueError("EXAONE stream event must be a JSON object")
                    yield data
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip())
    if pending:
        payload = "\n".join(pending)
        if payload != "[DONE]":
            data = _load_response_json(payload, "event stream")
            if not isinstance(data, dict):
                raise ValueError("EXAONE stream event must be a JSON object")
            yield data


async def _decode_sse_async(lines: Any) -> AsyncIterator[dict[str, Any]]:
    pending: list[str] = []
    async for line in lines:
        if not line:
            if pending:
                payload = "\n".join(pending)
                pending.clear()
                if payload != "[DONE]":
                    data = _load_response_json(payload, "event stream")
                    if not isinstance(data, dict):
                        raise ValueError("EXAONE stream event must be a JSON object")
                    yield data
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip())
    if pending:
        payload = "\n".join(pending)
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
        description="Chat Completions API base URL. Remote endpoints must use HTTPS.",
    )
    temperature: float | None = Field(default=1.0, description="Sampling temperature.")
    top_p: float | None = Field(default=0.95, description="Nucleus sampling value.")
    presence_penalty: float | None = Field(default=0.0, description="Presence penalty.")
    max_tokens: int | None = Field(default=None, description="Maximum generated tokens, including reasoning tokens.")
    top_k: int = Field(default=-1, description="Top-k sampling value; -1 disables top-k filtering.")
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = Field(
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
        return get_model_profile(self.model)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
        except ValueError as error:
            raise ValueError("base_url must be a valid absolute URL") from error
        loopback = hostname in {"localhost", "127.0.0.1", "::1"}
        if not hostname or (parsed.scheme != "https" and not (parsed.scheme == "http" and loopback)):
            raise ValueError("base_url must use HTTPS, except for loopback HTTP test servers")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not include URL credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not include a query or fragment")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= 524_288:
            raise ValueError("max_tokens must be between 0 and 524288")
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
    def _check_top_k(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("top_k must be -1 or a positive integer")
        return value

    @property
    def _llm_type(self) -> str:
        return "exaone-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url}

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
        if stop is not None:
            body["stop"] = stop
        if stream:
            body["stream_options"] = {"include_usage": True}
        body.update({name: value for name, value in self.model_kwargs.items() if value is not None})
        body.update({name: value for name, value in kwargs.items() if value is not None})
        template_kwargs = dict(self.chat_template_kwargs)
        for source in (self.model_kwargs, kwargs):
            custom_template_kwargs = source.get("chat_template_kwargs")
            if isinstance(custom_template_kwargs, Mapping):
                template_kwargs.update(custom_template_kwargs)
        body["chat_template_kwargs"] = {**template_kwargs, "preserve_thinking": self.preserve_thinking}
        body["model"] = kwargs.get("model") or self.model
        body["messages"] = serialized_messages
        body["stream"] = stream
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
            result["token_usage"] = token_usage
        return result

    def _parse_result(self, payload: Any, api_key: str) -> ChatResult:
        if not isinstance(payload, Mapping):
            raise ValueError("EXAONE response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("EXAONE response did not contain choices")
        if any(not isinstance(choice, Mapping) for choice in choices):
            raise ValueError("EXAONE response choices must be objects")
        generations = [
            ChatGeneration(
                message=_assistant_message(choice, payload, api_key),
                generation_info={
                    key: choice.get(key)
                    for key in ("finish_reason", "matched_stop", "logprobs")
                    if choice.get(key) is not None
                },
            )
            for choice in choices
        ]
        return ChatResult(
            generations=generations, llm_output={"model": payload.get("model"), "usage": payload.get("usage")}
        )

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
                return self._parse_result(payload, key)
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
                return self._parse_result(payload, key)
        raise RuntimeError("EXAONE retry loop exited without a response")

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | Mapping[str, Any] | bool | None = None, **kwargs: Any
    ) -> Any:
        """Bind LangChain tools using the EXAONE Chat Completions schema."""
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is True or tool_choice == "any":
            tool_choice = "required"
        elif tool_choice is False:
            tool_choice = None
        if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

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
                    for payload in _decode_sse(response.iter_lines()):
                        yield from self._stream_payload(payload, run_manager, key)
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
                    async for payload in _decode_sse_async(response.aiter_lines()):
                        for chunk in self._stream_payload(payload, None, key):
                            yield chunk
                            if run_manager is not None:
                                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    return
        raise RuntimeError("EXAONE retry loop exited without a response")

    def _stream_payload(
        self, payload: Mapping[str, Any], run_manager: CallbackManagerForLLMRun | None, api_key: str
    ) -> Iterator[ChatGenerationChunk]:
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise ValueError("EXAONE stream event choices must be a list")
        if not choices:
            usage = payload.get("usage")
            chunk = AIMessageChunk(
                content="", response_metadata={"usage": usage}, usage_metadata=_usage_metadata(usage)
            )
            yield ChatGenerationChunk(message=chunk, text="")
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
            reasoning = delta.get("reasoning_content")
            refusal = delta.get("refusal")
            if (
                not isinstance(text, str)
                or (reasoning is not None and not isinstance(reasoning, str))
                or (refusal is not None and not isinstance(refusal, str))
            ):
                raise ValueError("EXAONE stream delta content, reasoning, and refusal must be text")
            if not text.strip() and isinstance(refusal, str) and refusal.strip():
                text = refusal
            tool_chunks = []
            raw_tool_calls = delta.get("tool_calls")
            if raw_tool_calls is not None and not isinstance(raw_tool_calls, list):
                raise ValueError("EXAONE stream delta tool_calls must be an array")
            for tool in raw_tool_calls or []:
                if not isinstance(tool, Mapping):
                    raise ValueError("EXAONE stream tool call entries must be objects")
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
            }
            chunk_content: str | list[dict[str, Any]] = text
            if self.output_version == "v1":
                blocks: list[dict[str, Any]] = []
                if reasoning:
                    blocks.append({"type": "reasoning", "reasoning": reasoning, "index": 0})
                if text:
                    blocks.append({"type": "text", "text": text, "index": 1})
                for tool_chunk in tool_chunks:
                    index = tool_chunk.get("index")
                    blocks.append(
                        {**tool_chunk, "type": "tool_call_chunk", "index": index + 2 if index is not None else 2}
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
