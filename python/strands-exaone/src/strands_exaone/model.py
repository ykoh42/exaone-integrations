"""Strands model provider for the EXAONE Chat Completions API."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import ssl
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from typing import Any, TypeVar

import httpx
import truststore
from pydantic import BaseModel
from strands.models.model import Model

from .errors import ExaoneError, redact, response_error
from .formatting import format_messages, format_tool_choice, format_tools
from .streaming import StreamDecoder, nonstream_chunk, sse_data

T = TypeVar("T", bound=BaseModel)
MetadataCallback = Callable[[dict[str, Any]], Any]
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
}
_MANAGED_PARAMS = {"model", "messages", "stream", "stream_options", "tools", "tool_choice", "response_format"}


def _validate_config(config: dict[str, Any]) -> None:
    unknown = set(config) - _SUPPORTED_CONFIG
    if unknown:
        raise ValueError(f"Unknown model configuration fields: {', '.join(sorted(unknown))}")
    if not isinstance(config.get("model_id"), str) or not config["model_id"]:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(config.get("base_url"), str) or not config["base_url"]:
        raise ValueError("base_url must be a non-empty string")
    for field in ("stream", "preserve_thinking", "enable_thinking", "include_usage"):
        if config.get(field) is not None and not isinstance(config[field], bool):
            raise ValueError(f"{field} must be a boolean")
    if config.get("response_format") is not None and not isinstance(config["response_format"], dict):
        raise ValueError("response_format must be a dictionary")
    _validate_params(config)
    _validate_numbers(config)


def _validate_params(config: dict[str, Any]) -> None:
    params = config.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    if reserved := set(params) & _MANAGED_PARAMS:
        raise ValueError(f"params cannot override managed fields: {', '.join(sorted(reserved))}")
    if type(params.get("n", 1)) is not int or params.get("n", 1) != 1:
        raise ValueError("Strands supports exactly one completion choice")
    if "chat_template_kwargs" in params and not isinstance(params["chat_template_kwargs"], dict):
        raise ValueError("chat_template_kwargs must be a dictionary")


def _validate_numbers(config: dict[str, Any]) -> None:
    for field, minimum, maximum in (("temperature", 0, None), ("top_p", 0, 1), ("presence_penalty", -2, 2)):
        value = config.get(field)
        _validate_number(field, value, minimum, maximum)
    for field in ("max_tokens", "top_k", "context_window_limit"):
        value = config.get(field)
        if value is not None and type(value) is not int:
            raise ValueError(f"{field} must be an integer")
    if config.get("max_tokens") is not None and config["max_tokens"] < 0:
        raise ValueError("max_tokens cannot be negative")
    if config.get("top_k") is not None and config["top_k"] != -1 and config["top_k"] < 1:
        raise ValueError("top_k must be -1 or a positive integer")
    if (
        type(config.get("timeout")) not in (int, float)
        or not math.isfinite(config["timeout"])
        or config["timeout"] <= 0
    ):
        raise ValueError("timeout must be positive and finite")


def _validate_number(field: str, value: Any, minimum: int, maximum: int | None) -> None:
    if value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {field}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {field}")


async def _cancellable(awaitable: Any, signal: threading.Event | None) -> Any:
    """Abort in-flight HTTP reads when a Strands cancellation signal is set."""
    if signal is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            if signal.is_set():
                raise asyncio.CancelledError
            await asyncio.wait({task}, timeout=0.05)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _safe_headers(headers: httpx.Headers, key: str) -> dict[str, str]:
    names = {"x-request-id", "retry-after", "retry-after-ms", "content-type"}
    return {
        name.lower(): value.replace(key, "[REDACTED]")
        for name, value in headers.items()
        if name.lower() in names
        or name.lower().startswith(("x-second-rate-limit", "x-minute-rate-limit", "x-day-rate-limit"))
    }


def _standard_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not usage:
        return None
    names = {"prompt_tokens": "inputTokens", "completion_tokens": "outputTokens", "total_tokens": "totalTokens"}
    if any(not isinstance(usage.get(name), int) for name in names):
        return None
    return {target: usage[source] for source, target in names.items()}


class ExaoneModel(Model):
    """EXAONE chat model with separate reasoning history and tool-call streaming.

    ``api_key`` defaults to the ``EXAONE_API_KEY`` environment variable. The optional
    ``metadata_callback`` receives request-specific HTTP headers and raw usage,
    including EXAONE's ``reasoning_tokens``, which Strands does not standardize.
    A supplied ``http_client`` stays caller-owned and is not closed by this model.
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
        timeout: float = 120.0,
        params: dict[str, Any] | None = None,
        context_window_limit: int = 524288,
        metadata_callback: MetadataCallback | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = os.environ.get("EXAONE_API_KEY", "") if api_key is None else api_key
        if not self._api_key:
            raise ValueError("Set EXAONE_API_KEY or pass api_key")
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
            "params": deepcopy(params or {}),
            "context_window_limit": context_window_limit,
        }
        _validate_config(self._config)

    def get_config(self) -> dict[str, Any]:
        """Return generation settings without credentials or client objects."""
        return deepcopy(self._config)

    def update_config(self, **model_config: Any) -> None:
        """Validate and atomically replace the specified generation settings."""
        updated = {**self._config, **deepcopy(model_config)}
        _validate_config(updated)
        self._config = updated

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
            return
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        async with httpx.AsyncClient(verify=context, timeout=self._config["timeout"]) as client:
            yield client

    def _request(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None,
        system_prompt: str | None,
        system_prompt_content: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None,
        *,
        stream: bool,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self._config
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
        thinking = dict(body.get("chat_template_kwargs") or {})
        for name in ("preserve_thinking", "enable_thinking"):
            if config[name] is not None:
                thinking[name] = config[name]
        if thinking:
            body["chat_template_kwargs"] = thinking
        if stream and config["include_usage"] is not None:
            body["stream_options"] = {"include_usage": config["include_usage"]}
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
    ) -> httpx.Response:
        request = client.build_request(
            "POST",
            self._config["base_url"].rstrip("/") + "/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            timeout=self._config["timeout"],
        )
        response = await _cancellable(client.send(request, stream=True), signal)
        if response.is_success:
            return response
        try:
            payload = await _cancellable(response.aread(), signal)
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
    ) -> None:
        if callback:
            headers = _safe_headers(response.headers, self._api_key)
            result = callback(
                {
                    "status_code": response.status_code,
                    "request_id": headers.get("x-request-id"),
                    "headers": headers,
                    "usage": redact(deepcopy(usage), self._api_key),
                }
            )
            if inspect.isawaitable(result):
                await result

    async def _decode_stream(
        self, response: httpx.Response, decoder: StreamDecoder, signal: threading.Event | None
    ) -> AsyncIterator[dict[str, Any]]:
        iterator = sse_data(response).__aiter__()
        while True:
            try:
                raw = await _cancellable(iterator.__anext__(), signal)
            except StopAsyncIteration:
                return
            if raw == "[DONE]":
                decoder.done = True
                return
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except ValueError as exc:
                raise ExaoneError("Malformed JSON in EXAONE event stream") from exc
            if not isinstance(chunk, dict):
                raise ExaoneError("Invalid EXAONE event stream payload")
            if "error" in chunk:
                raise response_error(
                    response.status_code, chunk, _safe_headers(response.headers, self._api_key), self._api_key
                )
            for event in decoder.feed(chunk):
                yield event

    async def _decode_nonstream(
        self, response: httpx.Response, decoder: StreamDecoder, signal: threading.Event | None
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            data = json.loads(await _cancellable(response.aread(), signal))
        except ValueError as exc:
            raise ExaoneError("Malformed JSON in EXAONE response") from exc
        if not isinstance(data, dict):
            raise ExaoneError("Invalid EXAONE response payload")
        for event in decoder.feed(nonstream_chunk(data)):
            yield event

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: dict[str, Any] | None = None,
        system_prompt_content: list[dict[str, Any]] | None = None,
        cancel_signal: threading.Event | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Emit Strands model events and forward request-specific metadata."""
        streaming = self._config["stream"]
        body = self._request(messages, tool_specs, system_prompt, system_prompt_content, tool_choice, stream=streaming)
        callback = kwargs.get("metadata_callback", self._callback)
        decoder = StreamDecoder()
        started = time.monotonic()
        async with self._client() as client:
            response = await self._send(client, body, cancel_signal)
            try:
                yield {"messageStart": {"role": "assistant"}}
                if streaming:
                    events = self._decode_stream(response, decoder, cancel_signal)
                else:
                    events = self._decode_nonstream(response, decoder, cancel_signal)
                async for event in events:
                    yield event
                for event in decoder.finish(require_done=streaming):
                    yield event
                await self._metadata(callback, response, decoder.usage)
                usage = _standard_usage(decoder.usage)
                if usage is not None:
                    latency_ms = round((time.monotonic() - started) * 1000)
                    yield {"metadata": {"usage": usage, "metrics": {"latencyMs": latency_ms}}}
            finally:
                await response.aclose()

    async def structured_output(
        self, output_model: type[T], prompt: list[dict[str, Any]], system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Request and validate structured JSON using the model's schema support."""
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": output_model.__name__, "schema": output_model.model_json_schema()},
        }
        body = self._request(prompt, None, system_prompt, None, None, stream=False, response_format=fmt)
        signal = kwargs.get("cancel_signal")
        async with self._client() as client:
            response = await self._send(client, body, signal)
            try:
                data = json.loads(await _cancellable(response.aread(), signal))
                choices = data.get("choices") or []
                if len(choices) != 1:
                    raise ExaoneError("Expected exactly one structured output choice")
                choice = choices[0]
                if choice.get("finish_reason") != "stop":
                    raise ExaoneError("Structured output did not finish normally")
                content = (choice.get("message") or {}).get("content")
                if not isinstance(content, str):
                    raise ExaoneError("Structured output did not include JSON text")
                result = output_model.model_validate_json(content)
                await self._metadata(kwargs.get("metadata_callback", self._callback), response, data.get("usage"))
                yield {"output": result}
            finally:
                await response.aclose()
