"""EXAONE model provider.

- Docs: https://docs.exaone.ai/en/docs/k-exaone-2.5
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import ssl
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing, asynccontextmanager, suppress
from copy import deepcopy
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar, cast
from urllib.parse import urlsplit

import httpx
import truststore
from pydantic import BaseModel, ValidationError
from strands.models import ModelInfo
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.event_loop import Usage
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import override

from .errors import ExaoneError, redact, response_error
from .formatting import format_messages, format_tool_choice, format_tools
from .metadata import ResponseMetadata
from .retry import RETRY_STATUSES, retry_delay
from .streaming import StreamDecoder, nonstream_chunk, sse_data

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
P = ParamSpec("P")
MetadataCallback = Callable[[dict[str, Any]], Any]
MAX_CONTEXT_TOKENS = 524_288
_SUPPORTED_CONFIG = {
    "model_id",
    "base_url",
    "stream",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "reasoning_effort",
    "preserve_thinking",
    "enable_thinking",
    "include_usage",
    "response_format",
    "timeout",
    "params",
    "context_window_limit",
    "model_info",
    "max_retries",
    "base_retry_delay",
    "max_retry_delay",
}
_MANAGED_PARAMS = {"model", "messages", "stream", "tools", "tool_choice"}


def _validate_config_keys(config: dict[str, Any]) -> None:
    unknown = set(config) - _SUPPORTED_CONFIG
    if unknown:
        raise ValueError(f"Unknown model configuration fields: {', '.join(sorted(unknown))}")


def _validate_config(config: dict[str, Any]) -> None:
    _validate_config_keys(config)
    _validate_identity(config)
    _validate_boolean_options(config)
    _validate_response_format(config)
    _validate_model_info(config)
    _validate_params(config)
    _validate_numbers(config)


def _validate_identity(config: dict[str, Any]) -> None:
    if not isinstance(config.get("model_id"), str) or not config["model_id"]:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(config.get("base_url"), str) or not config["base_url"]:
        raise ValueError("base_url must be a non-empty string")
    _validate_base_url(config["base_url"])


def _validate_base_url(value: str) -> None:
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


def _validate_boolean_options(config: dict[str, Any]) -> None:
    for field in ("stream", "preserve_thinking", "enable_thinking", "include_usage"):
        if config.get(field) is not None and not isinstance(config[field], bool):
            raise ValueError(f"{field} must be a boolean")


def _validate_response_format(config: dict[str, Any]) -> None:
    if config.get("response_format") is not None and not isinstance(config["response_format"], dict):
        raise ValueError("response_format must be a dictionary")


def _validate_model_info(config: dict[str, Any]) -> None:
    model_info = config.get("model_info")
    if model_info is not None and not isinstance(model_info, ModelInfo):
        raise ValueError("model_info must be a ModelInfo or None")
    if model_info is not None:
        for field in ("context_window", "max_output_tokens"):
            value = getattr(model_info.limits, field)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"model_info.limits.{field} must be a positive integer")


def _validate_params(config: dict[str, Any]) -> None:
    params = config.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    if reserved := set(params) & _MANAGED_PARAMS:
        raise ValueError(f"params cannot override managed fields: {', '.join(sorted(reserved))}")
    if type(params.get("n", 1)) is not int or params.get("n", 1) != 1:
        raise ValueError("Strands supports exactly one completion choice")
    if "chat_template_kwargs" in params and not isinstance(params["chat_template_kwargs"], dict):
        raise ValueError("chat_template_kwargs must be a dictionary")
    options = params.get("stream_options")
    if options is not None and not isinstance(options, dict):
        raise ValueError("stream_options must be a dictionary or None")


def _apply_stream_options(body: dict[str, Any], *, stream: bool, include_usage: bool | None) -> None:
    if not stream:
        body.pop("stream_options", None)
        return
    options = dict(body.get("stream_options") or {})
    if include_usage is not None:
        options["include_usage"] = include_usage
    options.setdefault("include_usage", True)
    body["stream_options"] = options


def _validate_numbers(config: dict[str, Any]) -> None:
    _validate_sampling_numbers(config)
    _validate_token_limits(config)
    _validate_reasoning_effort(config)
    _validate_timeout(config)
    _validate_retry_config(config)


def _validate_sampling_numbers(config: dict[str, Any]) -> None:
    for field, minimum, maximum in (("temperature", 0, None), ("top_p", 0, 1), ("presence_penalty", -2, 2)):
        value = config.get(field)
        _validate_number(field, value, minimum, maximum)

    for field in ("max_tokens", "top_k"):
        value = config.get(field)
        if value is not None and type(value) is not int:
            raise ValueError(f"{field} must be an integer")

    if config.get("top_k") is not None and config["top_k"] != -1 and config["top_k"] < 1:
        raise ValueError("top_k must be -1 or a positive integer")


def _validate_token_limits(config: dict[str, Any]) -> None:
    value = config.get("context_window_limit")
    if value is not None and type(value) is not int:
        raise ValueError("context_window_limit must be an integer")
    if config.get("max_tokens") is not None and config["max_tokens"] < 0:
        raise ValueError("max_tokens cannot be negative")
    if (
        config.get("max_tokens") is not None
        and config["model_id"] == "k-exaone_v2.5"
        and config["max_tokens"] > MAX_CONTEXT_TOKENS
    ):
        raise ValueError(f"max_tokens cannot exceed the EXAONE context limit of {MAX_CONTEXT_TOKENS}")
    if config.get("context_window_limit") is not None and config["context_window_limit"] <= 0:
        raise ValueError("context_window_limit must be a positive integer")


def _validate_reasoning_effort(config: dict[str, Any]) -> None:
    effort = config.get("reasoning_effort")
    if effort is not None and effort not in ("low", "medium", "high", "max"):
        raise ValueError("reasoning_effort must be one of low, medium, high, or max")


def _validate_timeout(config: dict[str, Any]) -> None:
    timeout = config.get("timeout")
    if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be non-negative and finite, or None")


def _validate_retry_config(config: dict[str, Any]) -> None:
    if type(config.get("max_retries")) is not int or config["max_retries"] < 0:
        raise ValueError("max_retries must be a non-negative integer")
    for field in ("base_retry_delay", "max_retry_delay"):
        if config.get(field) is None:
            raise ValueError(f"{field} must be non-negative and finite")
        _validate_number(field, config.get(field), 0, None)


def _validate_number(field: str, value: Any, minimum: int, maximum: int | None) -> None:
    if value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {field}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {field}")


async def _cancellable(awaitable: Awaitable[R], signal: threading.Event | None, deadline: float | None = None) -> R:
    """Abort an in-flight operation when cancelled or when its total deadline expires."""
    if signal is None and deadline is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            _check_cancel(signal, deadline)
            wait_timeout = 0.05 if signal is not None else None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("EXAONE request timed out")
                wait_timeout = remaining if wait_timeout is None else min(wait_timeout, remaining)
            await asyncio.wait({task}, timeout=wait_timeout)
        _check_cancel(signal, deadline)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _check_cancel(signal: threading.Event | None, deadline: float | None = None) -> None:
    if signal is not None and signal.is_set():
        raise asyncio.CancelledError
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("EXAONE request timed out")


def _operation_deadline(timeout: float | None) -> float | None:
    return None if timeout is None else time.monotonic() + timeout


def _copy_transport_error(error: Exception, secret: str) -> Exception:
    """Copy into the nearest standard HTTPX type without retaining the original."""
    message = error.args[0] if error.args and isinstance(error.args[0], str) else "HTTP transport error"
    if isinstance(error, httpx.HTTPStatusError):
        safe_error = httpx.HTTPError(redact(message, secret))
        if isinstance(error.response, httpx.Response):
            safe_error.status_code = error.response.status_code  # type: ignore[attr-defined]
        return safe_error
    for error_type in type(error).__mro__:
        if getattr(httpx, error_type.__name__, None) is not error_type:
            continue
        if error_type in (httpx.RequestNotRead, httpx.ResponseNotRead, httpx.StreamClosed, httpx.StreamConsumed):
            return error_type()
        if issubclass(error_type, (httpx.HTTPError, httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError)):
            return error_type(redact(message, secret))
    return httpx.HTTPError(redact(message, secret))


def _safe_transport_errors(
    method: Callable[Concatenate[ExaoneModel, P], AsyncGenerator[R, None]],
) -> Callable[Concatenate[ExaoneModel, P], AsyncGenerator[R, None]]:
    """Remove credentials and the original request from HTTPX request errors."""

    @wraps(method)
    async def wrapped(self: ExaoneModel, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[R, None]:
        safe_error: Exception | None = None
        try:
            async with aclosing(method(self, *args, **kwargs)) as events:
                async for event in events:
                    yield event
        except (httpx.HTTPError, httpx.InvalidURL, httpx.CookieConflict, httpx.StreamError) as error:
            safe_error = _copy_transport_error(error, self._api_key)
        if safe_error is not None:
            # Raise outside the handler so the original error is not retained as context.
            raise safe_error

    return wrapped


def _safe_headers(headers: httpx.Headers, key: str) -> dict[str, str]:
    names = {"x-request-id", "retry-after", "retry-after-ms", "content-type"}
    return {
        name.lower(): value.replace(key, "[REDACTED]")
        for name, value in headers.items()
        if name.lower() in names
        or name.lower().startswith(("x-second-rate-limit", "x-minute-rate-limit", "x-day-rate-limit"))
    }


def _standard_usage(usage: dict[str, Any] | None) -> Usage | None:
    if not usage:
        return None
    names = {"prompt_tokens": "inputTokens", "completion_tokens": "outputTokens", "total_tokens": "totalTokens"}
    if any(type(usage.get(name)) is not int or not 0 <= usage[name] <= 2**53 - 1 for name in names):
        return None
    return {
        "inputTokens": usage["prompt_tokens"],
        "outputTokens": usage["completion_tokens"],
        "totalTokens": usage["total_tokens"],
    }


def _parse_response(payload: bytes | str, *, stream: bool = False) -> dict[str, Any]:
    """Decode a response object without attaching the raw payload to an error."""
    try:
        data = json.loads(payload)
    except ValueError:
        # Raise outside the handler so the raw JSON is not retained as exception context.
        pass
    else:
        if isinstance(data, dict):
            return data
        raise ExaoneError("Invalid EXAONE event stream payload" if stream else "Invalid EXAONE response payload")
    raise ExaoneError("Malformed JSON in EXAONE event stream" if stream else "Malformed JSON in EXAONE response")


def _check_response_error(data: dict[str, Any], response: httpx.Response, key: str) -> None:
    """Preserve a non-null error envelope even when the HTTP status is successful."""
    if data.get("error") is not None:
        raise response_error(response.status_code, data, _safe_headers(response.headers, key), key)


def _validate_output(output_model: type[T], content: str) -> T:
    """Validate output without exposing its contents in validation errors."""
    try:
        return output_model.model_validate_json(content)
    except ValidationError:
        # Validation errors can contain generated text; do not retain them as context.
        pass
    raise ExaoneError("Structured output did not match the requested schema")


class ExaoneModel(Model):
    """EXAONE model provider implementation.

    ``api_key`` defaults to the ``EXAONE_API_KEY`` environment variable. The optional
    ``metadata_callback`` receives response metadata and raw usage, including EXAONE's
    ``reasoning_tokens``, which Strands does not standardize. Metadata can include
    generated tokens and other sensitive content.

    Provider-level retries are disabled by default; set ``max_retries`` to opt in.
    Retries occur only before response events are exposed. A supplied ``http_client``
    stays caller-owned and is not closed by this model.

    Requests use EXAONE's recommended agent settings when no value is supplied:
    ``temperature=1.0``, ``top_p=0.95``, and ``preserve_thinking=True``. Named
    options take precedence over ``params``, followed by these defaults.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = "k-exaone_v2.5",
        base_url: str = "https://api.exaone.ai/v1",
        stream: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        reasoning_effort: str | None = None,
        preserve_thinking: bool | None = None,
        enable_thinking: bool | None = None,
        include_usage: bool | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,
        params: dict[str, Any] | None = None,
        context_window_limit: int | None = 524288,
        model_info: ModelInfo | None = None,
        metadata_callback: MetadataCallback | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 0,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
    ) -> None:
        """Initialize the EXAONE model provider.

        Args:
            api_key: API key. Defaults to the ``EXAONE_API_KEY`` environment variable.
            model_id: EXAONE model identifier.
            base_url: Base URL for the EXAONE API, including ``/v1``.
            stream: Whether to request a streaming Chat Completions response.
            max_tokens: Maximum generated tokens, including reasoning tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
            presence_penalty: Penalty for tokens already present in the generated text.
            reasoning_effort: EXAONE reasoning effort.
            preserve_thinking: Whether EXAONE should preserve previous reasoning in its input.
                Reasoning present in the conversation is forwarded independently of this setting.
            enable_thinking: EXAONE reasoning setting. False does not guarantee that
                response content contains no reasoning.
            include_usage: Whether to request usage in a streaming response. Defaults to true.
            response_format: Output format for ordinary model calls.
            timeout: Deadline for requests, retry waits, response reads, asynchronous metadata callbacks, and
                pauses between yielded events while the caller consumes a stream. An uncooperative custom
                transport may delay cleanup beyond the deadline. Defaults to no provider-imposed timeout.
            params: Additional EXAONE Chat Completions parameters.
            context_window_limit: Model context window in tokens.
            model_info: Overrides the model facts exposed to Strands. This follows the SDK merge
                contract and takes precedence over the context_window_limit shorthand.
            metadata_callback: Callback for response metadata after a successful response.
                It is not called if the response fails or is interrupted before completion. Metadata may contain
                generated tokens and other sensitive content even after API key redaction.
            http_client: Caller-owned async client; the provider does not close it.
            max_retries: Maximum number of retries before a response is exposed.
            base_retry_delay: Minimum and initial exponential retry delay, in seconds.
            max_retry_delay: Maximum permitted retry delay, in seconds.

        Raises:
            ValueError: If the API key or any model configuration is invalid.
        """
        self._api_key = os.environ.get("EXAONE_API_KEY", "") if api_key is None else api_key
        if not self._api_key:
            raise ValueError("EXAONE API key is required. Set the EXAONE_API_KEY environment variable or pass api_key.")
        self._callback = metadata_callback
        self._http_client = http_client
        self._config: dict[str, Any] = {
            "model_id": model_id,
            "base_url": base_url,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "reasoning_effort": reasoning_effort,
            "preserve_thinking": preserve_thinking,
            "enable_thinking": enable_thinking,
            "include_usage": include_usage,
            "response_format": deepcopy(response_format),
            "timeout": timeout,
            "params": deepcopy({} if params is None else params),
            "context_window_limit": context_window_limit,
            "model_info": deepcopy(model_info),
            "max_retries": max_retries,
            "base_retry_delay": base_retry_delay,
            "max_retry_delay": max_retry_delay,
        }
        _validate_config(self._config)

    @override
    def get_config(self) -> dict[str, Any]:
        """Get the model configuration without credentials or client objects.

        Returns:
            A copy of the EXAONE model configuration.
        """
        return deepcopy(self._config)

    @override
    def update_config(self, **model_config: Any) -> None:
        """Update the model configuration for requests that have not started.

        Args:
            **model_config: Configuration overrides.

        Raises:
            ValueError: If an override is invalid. Existing settings remain unchanged.
        """
        _validate_config_keys(model_config)
        updated = {**self._config, **deepcopy(model_config)}
        _validate_config(updated)
        self._config = updated

    @asynccontextmanager
    async def _client(self, config: dict[str, Any]) -> AsyncIterator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        async with httpx.AsyncClient(verify=context, timeout=config["timeout"]) as client:
            yield client

    def _request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt: str | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
        *,
        stream: bool,
        response_format: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self._config if config is None else config
        params = deepcopy(config.get("params") or {})
        body: dict[str, Any] = {
            **params,
            "model": config["model_id"],
            "messages": format_messages(messages, system_prompt, system_prompt_content),
            "stream": stream,
        }
        for name in ("max_tokens", "temperature", "top_p", "top_k", "presence_penalty", "reasoning_effort"):
            if config[name] is not None:
                body[name] = config[name]
        body.setdefault("temperature", 1.0)
        body.setdefault("top_p", 0.95)
        thinking = dict(body.get("chat_template_kwargs") or {})
        for name in ("preserve_thinking", "enable_thinking"):
            if config[name] is not None:
                thinking[name] = config[name]
        thinking.setdefault("preserve_thinking", True)
        body["chat_template_kwargs"] = thinking
        _apply_stream_options(body, stream=stream, include_usage=config["include_usage"])
        if tool_specs:
            body["tools"] = format_tools(tool_specs)
        if tool_choice is not None:
            body["tool_choice"] = format_tool_choice(tool_choice)
        selected_format = response_format if response_format is not None else config["response_format"]
        if selected_format is not None:
            body["response_format"] = deepcopy(selected_format)
        return body

    async def _send(
        self,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        signal: threading.Event | None,
        config: dict[str, Any],
        deadline: float | None,
    ) -> httpx.Response:
        request_options = {"timeout": config["timeout"]} if config["timeout"] is not None else {}
        request = client.build_request(
            "POST",
            config["base_url"].rstrip("/") + "/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            **request_options,
        )
        for attempt in range(config["max_retries"] + 1):
            try:
                response = await _cancellable(client.send(request, stream=True), signal, deadline)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if not await self._retry_connection(attempt, signal, config, deadline):
                    raise
                continue
            if response.is_success:
                return response
            if await self._retry_response(response, attempt, signal, config, deadline):
                continue
            await self._raise_response_error(response, signal, deadline)
        raise AssertionError("Retry loop ended without a response")

    async def _retry_connection(
        self, attempt: int, signal: threading.Event | None, config: dict[str, Any], deadline: float | None
    ) -> bool:
        if attempt >= config["max_retries"]:
            return False
        delay = retry_delay({}, attempt, config["base_retry_delay"], config["max_retry_delay"])
        if delay is None:
            return False
        await _cancellable(asyncio.sleep(delay), signal, deadline)
        return True

    async def _retry_response(
        self,
        response: httpx.Response,
        attempt: int,
        signal: threading.Event | None,
        config: dict[str, Any],
        deadline: float | None,
    ) -> bool:
        if response.status_code not in RETRY_STATUSES or attempt >= config["max_retries"]:
            return False
        delay = retry_delay(response.headers, attempt, config["base_retry_delay"], config["max_retry_delay"])
        if delay is None:
            return False
        await response.aclose()
        await _cancellable(asyncio.sleep(delay), signal, deadline)
        return True

    async def _raise_response_error(
        self, response: httpx.Response, signal: threading.Event | None, deadline: float | None
    ) -> None:
        try:
            payload = await _cancellable(response.aread(), signal, deadline)
            try:
                data = json.loads(payload)
            except ValueError:
                data = payload.decode("utf-8", errors="replace")
            raise response_error(
                response.status_code, data, _safe_headers(response.headers, self._api_key), self._api_key
            )
        finally:
            await response.aclose()

    async def _metadata(
        self,
        callback: MetadataCallback | None,
        response: httpx.Response,
        usage: dict[str, Any] | None,
        metadata: ResponseMetadata,
        signal: threading.Event | None,
        deadline: float | None,
    ) -> None:
        _check_cancel(signal, deadline)
        if callback is not None:
            headers = _safe_headers(response.headers, self._api_key)
            try:
                result = callback(
                    {
                        "status_code": response.status_code,
                        "request_id": headers.get("x-request-id"),
                        "headers": headers,
                        "usage": redact(deepcopy(usage), self._api_key),
                        "response": redact(deepcopy(metadata.response), self._api_key),
                    }
                )
                if inspect.isawaitable(result):
                    await _cancellable(result, signal, deadline)
            except Exception as error:
                _check_cancel(signal, deadline)
                if isinstance(error, (ModelThrottledException, ContextWindowOverflowException)):
                    raise ExaoneError("EXAONE metadata callback failed") from error
                raise
        _check_cancel(signal, deadline)

    async def _decode_stream(
        self,
        response: httpx.Response,
        decoder: StreamDecoder,
        signal: threading.Event | None,
        metadata: ResponseMetadata,
        deadline: float | None,
    ) -> AsyncIterator[dict[str, Any]]:
        iterator = sse_data(response).__aiter__()
        while True:
            try:
                raw = await _cancellable(iterator.__anext__(), signal, deadline)
            except StopAsyncIteration:
                return
            if raw == "[DONE]":
                decoder.done = True
                return
            if not raw:
                continue
            chunk = _parse_response(raw, stream=True)
            _check_response_error(chunk, response, self._api_key)
            metadata.add(chunk)
            for event in decoder.feed(chunk):
                yield event

    async def _decode_nonstream(
        self,
        response: httpx.Response,
        decoder: StreamDecoder,
        signal: threading.Event | None,
        metadata: ResponseMetadata,
        deadline: float | None,
    ) -> AsyncIterator[dict[str, Any]]:
        data = _parse_response(await _cancellable(response.aread(), signal, deadline))
        _check_response_error(data, response, self._api_key)
        metadata.add(data)
        for event in decoder.feed(nonstream_chunk(data)):
            yield event

    @override
    @_safe_transport_errors
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        cancel_signal: threading.Event | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a conversation with the EXAONE model.

        Args:
            messages: Conversation messages to send to the model.
            tool_specs: Tool specifications available to the model.
            system_prompt: System prompt to provide to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt blocks, when supplied by Strands.
            cancel_signal: Signal used to cancel in-flight requests and retry waits.
            **kwargs: Additional arguments, including an optional ``metadata_callback``.

        Yields:
            Strands model events translated from the EXAONE response.

        Raises:
            ExaoneError: If the service returns an error or an invalid response.
            ExaoneThrottledError: If the service returns HTTP 429 after retries.
            ExaoneContextWindowError: If the input exceeds the model context window.
            TimeoutError: If the configured operation deadline expires.
        """
        config = self._config
        streaming = config["stream"]
        body = self._request(
            messages, tool_specs, system_prompt, system_prompt_content, tool_choice, stream=streaming, config=config
        )
        callback = kwargs.get("metadata_callback", self._callback)
        decoder = StreamDecoder()
        metadata = ResponseMetadata(enabled=callback is not None)
        started = time.monotonic()
        deadline = _operation_deadline(config["timeout"])
        async with self._client(config) as client:
            response = await self._send(client, body, cancel_signal, config, deadline)
            try:
                _check_cancel(cancel_signal, deadline)
                yield {"messageStart": {"role": "assistant"}}
                if streaming:
                    events = self._decode_stream(response, decoder, cancel_signal, metadata, deadline)
                else:
                    events = self._decode_nonstream(response, decoder, cancel_signal, metadata, deadline)
                async for event in events:
                    _check_cancel(cancel_signal, deadline)
                    yield cast(StreamEvent, event)
                for event in decoder.finish(require_done=streaming, secret=self._api_key):
                    _check_cancel(cancel_signal, deadline)
                    yield cast(StreamEvent, event)
                usage = _standard_usage(decoder.usage)
                latency_ms = round((time.monotonic() - started) * 1000) if usage is not None else None
                await self._metadata(callback, response, decoder.usage, metadata, cancel_signal, deadline)
                if usage is not None:
                    _check_cancel(cancel_signal, deadline)
                    assert latency_ms is not None
                    yield {"metadata": {"usage": usage, "metrics": {"latencyMs": latency_ms}}}
            finally:
                await response.aclose()

    @override
    @_safe_transport_errors
    async def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Get structured output from the EXAONE model.

        Args:
            output_model: Pydantic model used to generate the response schema.
            prompt: Conversation messages to send to the model.
            system_prompt: System prompt to provide to the model.
            **kwargs: Additional arguments, including cancellation and metadata callbacks.

        Yields:
            The validated output model in a Strands structured-output event.

        Raises:
            ExaoneError: If the response is invalid or does not finish normally.
            ExaoneThrottledError: If the service returns HTTP 429 after retries.
            ExaoneContextWindowError: If the input exceeds the model context window.
            TimeoutError: If the configured operation deadline expires.
        """
        config = self._config
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": output_model.__name__, "schema": output_model.model_json_schema()},
        }
        body = self._request(prompt, None, system_prompt, None, None, stream=False, response_format=fmt, config=config)
        signal = kwargs.get("cancel_signal")
        deadline = _operation_deadline(config["timeout"])
        async with self._client(config) as client:
            response = await self._send(client, body, signal, config, deadline)
            try:
                data = _parse_response(await _cancellable(response.aread(), signal, deadline))
                _check_response_error(data, response, self._api_key)
                choices = data.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ExaoneError("Expected exactly one structured output choice")
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ExaoneError("Invalid structured output choice")
                if choice.get("finish_reason") != "stop":
                    reason = choice.get("finish_reason")
                    reason = redact(reason, self._api_key) if isinstance(reason, str) else None
                    raise ExaoneError(f"Structured output did not finish normally: {reason!r}", finish_reason=reason)
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise ExaoneError("Structured output did not include a message")
                content = message.get("content")
                refusal = message.get("refusal")
                if (
                    isinstance(refusal, str)
                    and refusal.strip()
                    and (not isinstance(content, str) or not content.strip())
                ):
                    raise ExaoneError(
                        "EXAONE refused the structured output request",
                        body={"refusal": redact(refusal, self._api_key)},
                    )
                if not isinstance(content, str):
                    raise ExaoneError("Structured output did not include JSON text")
                result = _validate_output(output_model, content)
                callback = kwargs.get("metadata_callback", self._callback)
                metadata = ResponseMetadata(enabled=callback is not None)
                metadata.add(data)
                await self._metadata(callback, response, data.get("usage"), metadata, signal, deadline)
                _check_cancel(signal, deadline)
                yield {"output": result}
            finally:
                await response.aclose()
