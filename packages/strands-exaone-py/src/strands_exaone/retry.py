"""Bounded retry delays for requests that have not exposed a response."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _parse_retry_after_seconds(value: str | None, *, milliseconds: bool = False) -> float | None:
    """Convert a numeric or HTTP-date retry hint to seconds."""
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        if milliseconds:
            return None
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            number = max(0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
    # A numeric header can overflow float to +inf. Keep that as an unbounded
    # delay so callers reject the retry instead of falling back to a short wait.
    if math.isinf(number):
        return number if number > 0 and any(char.isdigit() for char in value) else None
    if math.isnan(number) or number < 0:
        return None
    return number / 1000 if milliseconds else number


def retry_delay(headers: Mapping[str, str], attempt: int, base: float, maximum: float) -> float | None:
    """Prefer the millisecond hint and respect the configured retry cap."""
    milliseconds = _parse_retry_after_seconds(headers.get("retry-after-ms"), milliseconds=True)
    hint = milliseconds if milliseconds is not None else _parse_retry_after_seconds(headers.get("retry-after"))
    if hint is not None:
        return hint if hint <= maximum else None
    delay_ceiling = min(maximum, base * 2 ** min(attempt, 60))
    if delay_ceiling < base:
        return None
    return random.uniform(base, delay_ceiling)
