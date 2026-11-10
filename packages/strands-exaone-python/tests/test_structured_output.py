"""Synthetic structured-output integration tests for EXAONE and Strands Harness."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from harness_support import chunk, events, make_harness, response, tool_delta
from pydantic import BaseModel, Field, field_validator
from strands import tool

from strands_exaone import ExaoneError, ExaoneModel


class Observation(BaseModel):
    item_name: str = Field(alias="itemName")
    quantity: int
    tags: list[str] = Field(default_factory=list)

    @field_validator("quantity")
    @classmethod
    def positive_quantity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("quantity must be positive")
        return value


class Report(BaseModel):
    report_title: str = Field(alias="reportTitle")
    observations: list[Observation]
    note: str | None = None


@pytest.mark.parametrize(("include_note", "note"), [(False, None), (True, None), (True, "선택 메모")])
async def test_native_structured_output_nested_aliases_and_validator(include_note: bool, note: str | None) -> None:
    requests: list[dict[str, Any]] = []
    payload = {
        "reportTitle": "합성 보고서",
        "observations": [
            {"itemName": "첫째", "quantity": 2, "tags": ["가", "나"]},
            {"itemName": "둘째", "quantity": 5, "tags": []},
        ],
    }
    if include_note:
        payload["note"] = note

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        output = [
            event
            async for event in model.structured_output(Report, [{"role": "user", "content": [{"text": "Report."}]}])
        ]
    assert output == [{"output": Report.model_validate(payload)}]
    request = requests[0]
    assert request["stream"] is False
    assert "stream_options" not in request
    assert request["response_format"]["json_schema"]["name"] == "Report"
    assert request["response_format"]["json_schema"]["schema"] == Report.model_json_schema()


async def test_native_structured_output_custom_validation_failure_is_sanitized() -> None:
    payload = {"reportTitle": "SYNTHETIC_PRIVATE", "observations": [{"itemName": "sample", "quantity": -1, "tags": []}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ExaoneModel(api_key="test-only-key", http_client=client)
        with pytest.raises(ExaoneError, match="Structured output did not match") as caught:
            _ = [
                event
                async for event in model.structured_output(Report, [{"role": "user", "content": [{"text": "Report."}]}])
            ]
    assert "SYNTHETIC_PRIVATE" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


async def test_harness_forced_structured_tool_nested_failure_then_retry() -> None:
    requests: list[dict[str, Any]] = []
    calls: list[bool] = []

    @tool
    def get_quantity() -> int:
        """Return a synthetic quantity."""
        calls.append(True)
        return 7

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        match len(requests):
            case 1:
                return response(
                    events(
                        chunk({"reasoning_content": "first find quantity"}),
                        chunk(tool_delta("quantity_1", "get_quantity")),
                        chunk(finish="tool_calls"),
                    )
                )
            case 2:
                return response(
                    events(
                        chunk({"reasoning_content": "draft"}),
                        chunk({"content": "The quantity is 7."}),
                        chunk(finish="stop"),
                    )
                )
            case 3 | 4:
                payload = {
                    "reportTitle": "Synthetic",
                    "observations": [
                        {"itemName": "sample", "quantity": -1 if len(requests) == 3 else 7, "tags": ["one", "two"]}
                    ],
                    "note": None,
                }
                call = {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": f"report_{len(requests)}",
                            "function": {"name": "Report", "arguments": json.dumps(payload)},
                        }
                    ]
                }
                return response(
                    events(
                        chunk({"reasoning_content": f"format attempt {len(requests)}"}),
                        chunk(call),
                        chunk(finish="tool_calls"),
                    )
                )
            case 5:
                return response(events(chunk({"content": "The quantity was 7."}), chunk(finish="stop")))
            case _:
                raise AssertionError("Unexpected mock request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent = make_harness(ExaoneModel(api_key="test-only-key", http_client=client), [get_quantity])
        result = await agent.invoke_async(
            "Get a structured report.", structured_output_model=Report, limits={"structured_output_attempts": 2}
        )
        assert result.structured_output == Report.model_validate(
            {
                "reportTitle": "Synthetic",
                "observations": [{"itemName": "sample", "quantity": 7, "tags": ["one", "two"]}],
                "note": None,
            }
        )
        await agent.invoke_async("What was the quantity?")
    assert calls == [True]
    assert len(requests) == 5
    forced = requests[2]
    assert forced["tool_choice"] == {"type": "function", "function": {"name": "Report"}}
    assert {spec["function"]["name"] for spec in forced["tools"]} == {"Report"}
    schema = forced["tools"][0]["function"]["parameters"]
    assert "reportTitle" in schema["properties"]
    assert all(name in json.dumps(schema) for name in ("itemName", "quantity", "tags"))
    assert any(message.get("tool_call_id") == "quantity_1" for message in forced["messages"])
    assert any(message.get("reasoning_content") == "first find quantity" for message in forced["messages"])
    retry = requests[3]["messages"]
    assert any(message.get("tool_call_id") == "report_3" and "quantity" in message["content"] for message in retry)
    assert any(message.get("reasoning_content") == "format attempt 4" for message in requests[4]["messages"])
