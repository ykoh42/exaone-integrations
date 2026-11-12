"""EXAONE-aware retry policy for the public Strands retry extension point."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx
from strands import ModelRetryStrategy
from strands.hooks import AfterInvocationEvent, AfterModelCallEvent, HookRegistry
from strands.types.exceptions import ModelThrottledException

from .errors import ExaoneError, ExaoneThrottledError
from .retry import RETRY_STATUSES, retry_delay


class ExaoneRetryStrategy(ModelRetryStrategy):
    """Honor EXAONE Retry-After headers in Strands Agents and Harness.

    Pass one instance to ``retry_strategy`` instead of relying on Strands'
    default strategy, which uses its own backoff and cannot read EXAONE headers.
    Keep ``ExaoneModel(max_retries=0)`` to avoid nested retry budgets.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
    ) -> None:
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        for name, value in (("base_retry_delay", base_retry_delay), ("max_retry_delay", max_retry_delay)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")
        super().__init__(max_attempts=max_attempts, initial_delay=base_retry_delay, max_delay=max_retry_delay)
        self._max_attempts = max_attempts
        self._base_retry_delay = float(base_retry_delay)
        self._max_retry_delay = float(max_retry_delay)
        self._retry_counts: dict[int, int] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register retry callbacks through Strands' public hook registry."""
        registry.add_callback(AfterModelCallEvent, self._handle_after_model_call)
        registry.add_callback(AfterInvocationEvent, self._handle_after_invocation)

    async def _handle_after_model_call(self, event: AfterModelCallEvent) -> None:
        if event.retry:
            return
        if event.stop_response is not None:
            self._retry_counts.pop(id(event.invocation_state), None)
            return

        error = event.exception
        api_error = error if isinstance(error, (ExaoneError, ExaoneThrottledError)) else None
        headers = api_error.headers if api_error is not None else {}
        status_code = api_error.status_code if api_error is not None else None
        retryable_status = status_code in RETRY_STATUSES
        retryable_transport = isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
        if not retryable_status and not retryable_transport and not isinstance(error, ModelThrottledException):
            return

        invocation_id = id(event.invocation_state)
        attempt = self._retry_counts.get(invocation_id, 0)
        if attempt >= self._max_attempts - 1:
            return
        delay = retry_delay(headers, attempt, self._base_retry_delay, self._max_retry_delay)
        if delay is None:
            return

        await asyncio.sleep(delay)
        self._retry_counts[invocation_id] = attempt + 1
        event.retry = True

    async def _handle_after_invocation(self, event: AfterInvocationEvent) -> None:
        self._retry_counts.pop(id(event.invocation_state), None)
