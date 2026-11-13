"""Credential redaction regression tests."""

from urllib.parse import quote, quote_plus

import httpx

from strands_exaone.errors import ExaoneError, redact, response_error
from strands_exaone.model import _safe_headers


def test_redact_covers_literal_and_url_encoded_credentials() -> None:
    api_key = "synthetic+reserved/key with spaces"
    encoded_values = (
        api_key,
        quote(api_key, safe=""),
        quote_plus(api_key, safe=""),
        quote(api_key, safe="").replace("%2F", "%2f"),
    )

    for value in encoded_values:
        assert redact({value: value}, api_key) == {"[REDACTED]": "[REDACTED]"}

    different_value = quote(api_key, safe="").replace("%2F", "%3F")
    assert redact({different_value: different_value}, api_key) == {different_value: different_value}


def test_redaction_returns_independent_nested_containers() -> None:
    source = {"items": [{"value": "safe"}]}

    redacted = redact(source, "synthetic-key")
    redacted["items"][0]["value"] = "changed"

    assert source == {"items": [{"value": "safe"}]}


def test_safe_rate_headers_redact_url_encoded_credentials() -> None:
    api_key = "synthetic+reserved/key with spaces"
    headers = httpx.Headers({"x-request-id": quote_plus(api_key, safe="")})

    assert _safe_headers(headers, api_key) == {"x-request-id": "[REDACTED]"}


def test_response_error_redacts_credentials_from_body_and_headers() -> None:
    api_key = "synthetic+reserved/key with spaces"
    encoded_key = quote_plus(api_key, safe="")

    error = response_error(
        400,
        {"description": f"Rejected {encoded_key}"},
        {"x-request-id": encoded_key},
        api_key,
    )

    assert isinstance(error, ExaoneError)
    assert str(error) == "Rejected [REDACTED]"
    assert error.body == {"description": "Rejected [REDACTED]"}
    assert error.request_id == "[REDACTED]"
    assert error.headers == {"x-request-id": "[REDACTED]"}
