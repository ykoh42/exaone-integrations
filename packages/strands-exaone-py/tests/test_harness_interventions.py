"""Synthetic intervention and nested-tool tests using the real Strands Harness."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from harness_support import chunk, events, response, tool_delta
from strands import tool
from strands_harness import create_harness

from strands_exaone import ExaoneModel


@pytest.mark.parametrize("decision", ["no_approval_needed", "approve", "deny"])
async def test_harness_smart_intervention_classifier_and_resume(decision: str) -> None:
    """The smart preset uses a synthetic structured-output classification call."""
    requests: list[dict[str, Any]] = []
    calls: list[bool] = []

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        names = {spec["function"]["name"] for spec in body.get("tools", [])}
        if "_RiskDecision" in names:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "risk_1",
                        "function": {
                            "name": "_RiskDecision",
                            "arguments": json.dumps(
                                {"requires_approval": decision != "no_approval_needed", "reason": "synthetic"}
                            ),
                        },
                    }
                ]
            }
            return response(events(chunk(call), chunk(finish="tool_calls")))
        if len(requests) == 1:
            return response(
                events(
                    chunk({"reasoning_content": "parent synthetic reasoning"}),
                    chunk(tool_delta("parent_1", "get_number")),
                    chunk(finish="tool_calls"),
                )
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[get_number],
            interventions="smart",
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        result = await agent.invoke_async("Get a synthetic number.")
        if decision != "no_approval_needed":
            assert result.stop_reason == "interrupt"
            assert len(result.interrupts) == 1
            assert calls == []
            result = await agent.invoke_async(
                [
                    {
                        "interruptResponse": {
                            "interruptId": result.interrupts[0].id,
                            "response": "yes" if decision == "approve" else "no",
                        }
                    }
                ]
            )
    assert result.stop_reason == "end_turn"
    assert calls == ([] if decision == "deny" else [True])
    assert len(requests) == 3
    classifier = requests[1]
    assert classifier["stream"] is True
    assert any(spec["function"]["name"] == "_RiskDecision" for spec in classifier["tools"])
    messages = requests[2]["messages"]
    assert any(message.get("reasoning_content") == "parent synthetic reasoning" for message in messages)
    assert any(message.get("tool_call_id") == "parent_1" for message in messages)


@pytest.mark.parametrize("mode", ["allow", "hook_deny", "interrupt"])
async def test_harness_nested_programmatic_tool_approval_and_hooks(mode: str) -> None:
    """Nested calls use SDK hooks; human interrupts cannot be prompted inside Monty."""
    from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry
    from strands.vended_interventions.hitl import HumanInTheLoop

    requests: list[dict[str, Any]] = []
    calls: list[str] = []
    observed: list[tuple[str, str]] = []

    @tool
    def safe_inner(value: str) -> str:
        """Return a synthetic value."""
        calls.append(value)
        return "INNER_RESULT"

    class ToolHook(HookProvider):
        def register_hooks(self, registry: HookRegistry) -> None:
            registry.add_callback(BeforeToolCallEvent, self.before)

        def before(self, event: BeforeToolCallEvent) -> None:
            observed.append((event.tool_use["name"], event.tool_use["toolUseId"]))
            if mode == "hook_deny" and event.tool_use["name"] == "safe_inner":
                event.cancel_tool = "SYNTHETIC_HOOK_DENIAL"

    program = (
        "try:\n"
        "    result = await safe_inner(value='synthetic')\n"
        "    print(result)\n"
        "except RuntimeError:\n"
        "    print('INNER_BLOCKED')\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            call = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "program_1",
                        "function": {"name": "programmatic_tool_caller", "arguments": json.dumps({"code": program})},
                    }
                ]
            }
            return response(
                events(
                    chunk({"reasoning_content": "parent synthetic reasoning"}), chunk(call), chunk(finish="tool_calls")
                )
            )
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    allowed = ["programmatic_tool_caller"] + ([] if mode == "interrupt" else ["safe_inner"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[safe_inner],
            builtin_tools=["programmatic_tool_caller"],
            builtin_plugins=[],
            interventions=HumanInTheLoop(allowed_tools=allowed),
            hooks=[ToolHook()],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        result = await agent.invoke_async("Run a synthetic nested tool.")
    assert result.stop_reason == "end_turn"
    assert calls == (["synthetic"] if mode == "allow" else [])
    assert ("programmatic_tool_caller", "program_1") in observed
    assert any(name == "safe_inner" and ident.startswith("tooluse_safe_inner_") for name, ident in observed)
    assert len(requests) == 2
    messages = requests[1]["messages"]
    tool_results = [message for message in messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_results] == ["program_1"]
    assert ("INNER_RESULT" if mode == "allow" else "INNER_BLOCKED") in tool_results[0]["content"]
    assert any(message.get("reasoning_content") == "parent synthetic reasoning" for message in messages)


async def test_smart_classifier_forces_structured_tool_after_plain_response() -> None:
    """The classifier's auxiliary agent can require an extra forced-tool request."""
    requests: list[dict[str, Any]] = []
    calls: list[bool] = []

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        match len(requests):
            case 1:
                return response(
                    events(
                        chunk({"reasoning_content": "parent reason"}),
                        chunk(tool_delta("parent_1", "get_number")),
                        chunk(finish="tool_calls"),
                    )
                )
            case 2:
                return response(
                    events(
                        chunk({"reasoning_content": "classifier reason"}),
                        chunk({"content": "Synthetic classification draft."}),
                        chunk(finish="stop"),
                    )
                )
            case 3:
                call = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "risk_1",
                            "function": {
                                "name": "_RiskDecision",
                                "arguments": '{"requires_approval":false,"reason":"synthetic"}',
                            },
                        }
                    ]
                }
                return response(events(chunk(call), chunk(finish="tool_calls")))
            case 4:
                return response(events(chunk({"content": "done"}), chunk(finish="stop")))
            case _:
                raise AssertionError("Unexpected synthetic request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[get_number],
            interventions="smart",
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        result = await agent.invoke_async("Get a synthetic number.")
    assert result.stop_reason == "end_turn"
    assert calls == [True]
    assert len(requests) == 4
    assert requests[2]["tool_choice"] == {"type": "function", "function": {"name": "_RiskDecision"}}
    assert any(message.get("reasoning_content") == "classifier reason" for message in requests[2]["messages"])
    assert any(message.get("reasoning_content") == "parent reason" for message in requests[3]["messages"])
    assert any(message.get("tool_call_id") == "parent_1" for message in requests[3]["messages"])
    assert "classifier reason" not in str(requests[3]["messages"])


async def test_smart_classifier_failure_requires_approval() -> None:
    requests = 0
    calls: list[bool] = []

    @tool
    def get_number() -> str:
        """Return a synthetic number."""
        calls.append(True)
        return "7"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return response(events(chunk(tool_delta("parent_1", "get_number")), chunk(finish="tool_calls")))
        if requests == 2:
            return httpx.Response(500, json={"error": {"message": "synthetic classifier failure"}})
        return response(events(chunk({"content": "done"}), chunk(finish="stop")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = create_harness(
            model=ExaoneModel(api_key="test-only-key", http_client=client),
            tools=[get_number],
            interventions="smart",
            builtin_tools=[],
            builtin_plugins=[],
            session=False,
            memory=False,
            skills=False,
            caching=False,
            callback_handler=None,
            retry_strategy=None,
        )
        first = await agent.invoke_async("Get a synthetic number.")
        assert first.stop_reason == "interrupt"
        assert calls == []
        result = await agent.invoke_async(
            [{"interruptResponse": {"interruptId": first.interrupts[0].id, "response": "no"}}]
        )
    assert result.stop_reason == "end_turn"
    assert calls == []
    assert requests == 3
