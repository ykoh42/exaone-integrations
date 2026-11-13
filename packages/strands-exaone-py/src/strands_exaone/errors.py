"""Provider errors with safe, request-specific response details."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, quote_plus

from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


class ExaoneError(Exception):
    """EXAONE request failed or returned an invalid stream."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: Any = None,
        error_type: str | None = None,
        param: Any = None,
        request_id: str | None = None,
        headers: dict[str, str] | None = None,
        body: Any = None,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.error_type = error_type
        self.param = param
        self.request_id = request_id
        self.headers = headers or {}
        self.body = body
        self.finish_reason = finish_reason


class ExaoneThrottledError(ModelThrottledException):
    """HTTP 429 with the sanitized response attached as ``details``."""

    def __init__(self, details: ExaoneError) -> None:
        super().__init__(str(details))
        self.details = details
        self.status_code = details.status_code
        self.error_type = details.error_type
        self.request_id = details.request_id
        self.headers = details.headers
        self.body = details.body
        self.code = details.code
        self.param = details.param


class ExaoneContextWindowError(ContextWindowOverflowException):
    """The service explicitly reported that the context window was exceeded."""

    def __init__(self, details: ExaoneError) -> None:
        super().__init__(str(details))
        self.details = details
        self.status_code = details.status_code
        self.error_type = details.error_type
        self.request_id = details.request_id
        self.headers = details.headers
        self.body = details.body
        self.code = details.code
        self.param = details.param


def redact(value: Any, secret: str) -> Any:
    """Remove a credential from a response before exposing it."""
    return _redact_value(value, _secret_patterns(secret))


def _redact_value(value: Any, patterns: tuple[tuple[str, re.Pattern[str] | None], ...]) -> Any:
    if isinstance(value, str):
        for literal, pattern in patterns:
            value = value.replace(literal, "[REDACTED]")
            if pattern is not None:
                value = pattern.sub("[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {_redact_value(key, patterns): _redact_value(item, patterns) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, patterns) for item in value]
    return value


def _secret_patterns(secret: str) -> tuple[tuple[str, re.Pattern[str] | None], ...]:
    if not secret:
        return ()
    try:
        variants = {secret, quote(secret, safe=""), quote_plus(secret, safe="")}
    except UnicodeError:
        variants = {secret}
    patterns = []
    for variant in sorted(variants, key=len, reverse=True):
        if re.search(r"%[0-9A-F]{2}", variant, re.IGNORECASE):
            escaped = re.escape(variant)
            encoded_pattern = re.sub(
                r"%[0-9A-F]{2}", lambda match: f"(?i:{match.group()})", escaped, flags=re.IGNORECASE
            )
            patterns.append((variant, re.compile(encoded_pattern)))
        else:
            patterns.append((variant, None))
    return tuple(patterns)


def response_error(status: int, body: Any, headers: dict[str, str], secret: str) -> Exception:
    """Map an error envelope without conflating arbitrary bad requests with context overflow."""
    body, headers = redact([body, headers], secret)
    envelope = body if isinstance(body, dict) else {}
    nested = envelope.get("error", {})
    if not isinstance(nested, dict):
        nested = {}
    fallback = f"HTTP {status}"
    if 200 <= status < 300:
        fallback = "EXAONE returned an error in the response body"
    message = nested.get("message") or envelope.get("description") or fallback
    code = nested["code"] if "code" in nested else envelope.get("result_code")
    details = ExaoneError(
        str(message),
        status_code=status,
        code=code,
        error_type=nested.get("type") if isinstance(nested.get("type"), str) else None,
        param=nested.get("param"),
        request_id=headers.get("x-request-id"),
        headers=headers,
        body=body,
    )
    if status == 429:
        return ExaoneThrottledError(details)
    if status in {400, 413} and nested.get("code") == "context_length_exceeded":
        return ExaoneContextWindowError(details)
    return details
