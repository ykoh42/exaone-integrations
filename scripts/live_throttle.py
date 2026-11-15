"""Retry HTTP 429 responses before any response body reaches a live comparison client."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx


def retry_after_seconds(headers: httpx.Headers) -> float:
    """Wait at least ten seconds and honor the longer valid server hint."""
    delays = [10.0]
    for name, divisor in (("retry-after", 1), ("retry-after-ms", 1000)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw) / divisor
        except ValueError:
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(raw)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)  # noqa: UP017  # Python 3.10
                value = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())  # noqa: UP017  # Python 3.10
            except (TypeError, ValueError, OverflowError):
                continue
        if not math.isnan(value) and value >= 0:
            delays.append(value)
    return max(delays)


class Retry429Transport(httpx.AsyncBaseTransport):
    """Retry only 429 responses, before the SDK can observe any response bytes."""

    def __init__(self, transport: httpx.AsyncBaseTransport, *, max_retries: int = 20) -> None:
        self._transport = transport
        self._max_retries = max_retries

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # The comparison sends JSON; buffer it so each HTTP attempt has the same body.
        await request.aread()
        for attempt in range(self._max_retries + 1):
            response = await self._transport.handle_async_request(request)
            if response.status_code != 429 or attempt == self._max_retries:
                return response
            delay = retry_after_seconds(response.headers)
            if not math.isfinite(delay):
                return response
            await response.aclose()
            await asyncio.sleep(delay)
        raise RuntimeError("429 retry loop exited without a response")

    async def aclose(self) -> None:
        await self._transport.aclose()
