"""Provider errors with safe, request-specific response details."""

from __future__ import annotations

from typing import Any

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
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def response_error(status: int, body: Any, headers: dict[str, str], secret: str) -> Exception:
    """Map an error envelope without conflating arbitrary bad requests with context overflow."""
    body = redact(body, secret)
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
