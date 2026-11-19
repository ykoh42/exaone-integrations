"""Opt-in agent middleware for malformed EXAONE tool-call arguments."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from copy import copy
from typing import Any

from strands.middleware import InvokeModelContext

from .model import ExaoneModel


async def recover_tool_calls(context: InvokeModelContext, next_fn: Callable[..., Any]) -> AsyncIterator[Any]:
    """Enable correction within the model stream for this invocation."""
    if type(context.model) is ExaoneModel:
        model = copy(context.model)
        model._recover_tool_calls = True
        context = context.replace(model=model)
    async for event in next_fn(context):
        yield event
