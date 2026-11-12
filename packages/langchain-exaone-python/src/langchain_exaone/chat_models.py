"""Chat model support for the EXAONE API."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
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
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, SecretStr, field_validator

_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
_SAFE_HEADERS = {"content-type", "retry-after", "retry-after-ms", "x-request-id"}


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
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.headers = dict(headers or {})


def _safe_headers(headers: Mapping[str, str], api_key: str) -> dict[str, str]:
    return {
        name.lower(): value.replace(api_key, "[REDACTED]")
        for name, value in headers.items()
        if name.lower() in _SAFE_HEADERS
        or name.lower().startswith(("x-second-rate-limit", "x-minute-rate-limit", "x-day-rate-limit"))
    }


def _redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, Mapping):
        return {_redact_secret(key, secret): _redact_secret(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    return value


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
    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        error = {}
    message = str(error.get("message") or error.get("description") or f"EXAONE API request failed with HTTP {status}")
    message = message.replace(api_key, "[REDACTED]")
    return ExaoneAPIError(
        message,
        status_code=status,
        error_type=_redact_secret(error["type"], api_key) if isinstance(error.get("type"), str) else None,
        code=_redact_secret(error.get("code") if error.get("code") is not None else error.get("result_code"), api_key),
        param=_redact_secret(error.get("param"), api_key),
        headers=_safe_headers(headers, api_key),
    )


def _content_parts(message: BaseMessage) -> tuple[str, str | None]:
    content = message.content
    text: list[str] = []
    reasoning: list[str] = []
    if isinstance(content, str):
        text.append(content)
    elif isinstance(content, Sequence):
        for block in content:
            if isinstance(block, str):
                text.append(block)
            elif isinstance(block, Mapping):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text.append(block["text"])
                elif block.get("type") == "reasoning" and isinstance(block.get("reasoning"), str):
                    reasoning.append(block["reasoning"])
                else:
                    raise TypeError(f"EXAONE does not support message block type {block.get('type')!r}")
            else:
                raise TypeError("EXAONE supports only text and reasoning content blocks")
    else:
        raise TypeError("EXAONE message content must be text or text/reasoning blocks")
    if isinstance(message, AIMessage):
        prior = message.additional_kwargs.get("reasoning_content")
        if isinstance(prior, str) and prior:
            reasoning.insert(0, prior)
    return "".join(text), "".join(reasoning) or None


def _message_to_api(message: BaseMessage) -> dict[str, Any]:
    content, reasoning = _content_parts(message)
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": content}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}
    if isinstance(message, FunctionMessage):
        return {"role": "function", "name": message.name, "content": content}
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": content}
    if isinstance(message, AIMessage):
        result: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning is not None:
            result["reasoning_content"] = reasoning
        raw_calls = message.additional_kwargs.get("tool_calls")
        if isinstance(raw_calls, list):
            result["tool_calls"] = raw_calls
        elif message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"], ensure_ascii=False)},
                }
                for call in message.tool_calls
            ]
        function_call = message.additional_kwargs.get("function_call")
        if isinstance(function_call, Mapping):
            result["function_call"] = function_call
        return result
    raise TypeError(f"EXAONE does not support LangChain message type {message.type!r}")


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
    for raw in raw_calls if isinstance(raw_calls, list) else []:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("function"), Mapping):
            continue
        function = raw["function"]
        name, arguments, call_id = function.get("name"), function.get("arguments", ""), raw.get("id")
        if not isinstance(name, str) or not isinstance(arguments, str):
            continue
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
                    data = json.loads(payload)
                    if not isinstance(data, dict):
                        raise ValueError("EXAONE stream event must be a JSON object")
                    yield data
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip())
    if pending:
        payload = "\n".join(pending)
        if payload != "[DONE]":
            data = json.loads(payload)
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
                    data = json.loads(payload)
                    if not isinstance(data, dict):
                        raise ValueError("EXAONE stream event must be a JSON object")
                    yield data
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip())
    if pending:
        payload = "\n".join(pending)
        if payload != "[DONE]":
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("EXAONE stream event must be a JSON object")
            yield data


class ChatEXAONE(BaseChatModel):
    """LangChain chat model for the K-EXAONE 2.5 API."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model: str = "k-exaone_v2.5"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.environ.get("EXAONE_API_KEY", "")), repr=False)
    base_url: str = "https://api.exaone.ai/v1"
    temperature: float | None = 1.0
    top_p: float | None = 0.95
    presence_penalty: float | None = 0.0
    max_tokens: int | None = None
    top_k: int = -1
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = None
    preserve_thinking: bool = True
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    max_retries: int = 2
    max_retry_delay: float = 60.0
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    http_client: httpx.Client | None = Field(default=None, exclude=True, repr=False)
    http_async_client: httpx.AsyncClient | None = Field(default=None, exclude=True, repr=False)

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

    def _parse_result(self, payload: Any, api_key: str) -> ChatResult:
        if not isinstance(payload, Mapping):
            raise ValueError("EXAONE response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("EXAONE response did not contain choices")
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
            if isinstance(choice, Mapping)
        ]
        if not generations:
            raise ValueError("EXAONE response choices were invalid")
        return ChatResult(
            generations=generations, llm_output={"model": payload.get("model"), "usage": payload.get("usage")}
        )

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
                payload = response.json()
                response.close()
                return self._parse_result(payload, key)
        raise RuntimeError("EXAONE retry loop exited without a response")

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
                payload = response.json()
                await response.aclose()
                return self._parse_result(payload, key)
        raise RuntimeError("EXAONE retry loop exited without a response")

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Bind LangChain tools using the EXAONE Chat Completions schema."""
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice == "any":
            tool_choice = "required"
        if isinstance(tool_choice, str) and tool_choice not in {"auto", "none", "required"}:
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

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
            for tool in delta.get("tool_calls") or []:
                function = tool.get("function") or {}
                tool_chunks.append(
                    {
                        "name": function.get("name"),
                        "args": function.get("arguments", ""),
                        "id": tool.get("id"),
                        "index": tool.get("index"),
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
            chunk = AIMessageChunk(
                content=text,
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
