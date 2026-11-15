"""Exercise the public Strands retry hook lifecycle."""

import httpx
from strands.hooks import AfterModelCallEvent, HookRegistry

from strands_exaone import ExaoneRetryStrategy


async def test_retry_budget_resets_on_a_model_call_without_an_exception() -> None:
    strategy = ExaoneRetryStrategy(max_attempts=2, base_retry_delay=0, max_retry_delay=0)
    registry = HookRegistry()
    strategy.register_hooks(registry)
    invocation_state = {}

    first = AfterModelCallEvent(agent=None, invocation_state=invocation_state, exception=httpx.ConnectError("test"))
    await registry.invoke_callbacks_async(first)
    assert first.retry is True

    await registry.invoke_callbacks_async(AfterModelCallEvent(agent=None, invocation_state=invocation_state))

    subsequent = AfterModelCallEvent(
        agent=None, invocation_state=invocation_state, exception=httpx.ConnectError("test")
    )
    await registry.invoke_callbacks_async(subsequent)
    assert subsequent.retry is True
