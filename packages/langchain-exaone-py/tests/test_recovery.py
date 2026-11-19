"""Agent-level contract for opt-in malformed tool call recovery."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import AgentMiddleware  # noqa: E402
from langchain.agents.structured_output import ToolStrategy  # noqa: E402
from langchain_core.caches import InMemoryCache  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from langchain_exaone import ChatEXAONE  # noqa: E402
from langchain_exaone._recovery import MalformedToolArguments  # noqa: E402
from langchain_exaone.recovery import ToolCallRecoveryMiddleware  # noqa: E402

NAMED_TRANSFER = {"type": "function", "function": {"name": "transfer"}}


def completion(arguments: str | None, *, finish: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": None if arguments is not None else "Done."}
    if arguments is not None:
        message["tool_calls"] = [
            {"id": "call_bad", "type": "function", "function": {"name": "transfer", "arguments": arguments}}
        ]
    return {
        "id": "synthetic",
        "model": "k-exaone_v2.5",
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish or ("tool_calls" if arguments else "stop")}
        ],
    }


def sse(
    data: dict[str, Any], *, split_id: bool = False, include_failure_text: bool = True, repeated_finish: bool = False
) -> bytes:
    """Emit representative content and split tool arguments before stop."""
    choice = data["choices"][0]
    message = choice["message"]
    payloads = []
    if message.get("tool_calls"):
        calls = message["tool_calls"]
        has_failure = include_failure_text and any("123," in call["function"]["arguments"] for call in calls)
        payloads.append(
            {
                "index": 0,
                "delta": {
                    "content": "hidden-failure " if has_failure else "",
                    **({"reasoning_content": "synthetic reasoning"} if has_failure else {}),
                },
            }
        )
        for index, call in enumerate(calls):
            args = call["function"]["arguments"]
            ident = call["id"]
            split = len(ident) // 2 if split_id else len(ident)
            payloads.append(
                {
                    "index": 0,
                    "delta": {
                        **({"refusal": message["refusal"]} if index == 0 and "refusal" in message else {}),
                        "tool_calls": [
                            {
                                "index": index,
                                "id": ident[:split],
                                "type": "function",
                                "function": {"name": call["function"]["name"], "arguments": args[:5]},
                            }
                        ],
                    },
                }
            )
            payloads.append(
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": index,
                                **({"id": ident[split:]} if split_id else {}),
                                "function": {"arguments": args[5:]},
                            }
                        ]
                    },
                }
            )
    else:
        payloads.append({"index": 0, "delta": {"content": message["content"]}})
    if repeated_finish:
        payloads[-1]["finish_reason"] = choice["finish_reason"]
    payloads.append({"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]})
    records = [{"id": "synthetic", "choices": [item]} for item in payloads]
    if "usage" in data:
        records.append({"id": "synthetic", "choices": [], "usage": data["usage"]})
    return b"".join(("data: " + json.dumps(item) + "\n\n").encode() for item in records) + b"data: [DONE]\n\n"


def run_agent(
    first: dict[str, Any],
    *,
    enabled: bool = True,
    stream: bool = False,
    repeat: bool = False,
    strict: bool = False,
    named: bool = False,
    api_key: str = "fixture",
    complete_stream: bool = True,
    stream_mode: str = "messages",
    split_id: bool = False,
    correction: dict[str, Any] | None = None,
    named_rejection: str | None = None,
    buffer_response: bool = True,
    include_failure_text: bool = True,
    observe_first_frame: bool = False,
    repeated_finish: bool = False,
    inner: AgentMiddleware | None = None,
) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []
    output: list[Any] = []
    observed_errors: list[BaseException] = []
    observed_tokens: list[str] = []

    class Observer(BaseCallbackHandler):
        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            observed_errors.append(error)

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            observed_tokens.append(token)

    class StrictChoice(AgentMiddleware):
        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            return handler(
                request.override(
                    tool_choice="transfer" if named else "required",
                    model_settings={**request.model_settings, **({"strict": True} if strict else {})},
                )
            )

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 2 and named_rejection is not None:
            return httpx.Response(400, json={"error": {"message": named_rejection}})
        if len(requests) == 1 or (repeat and len(requests) == 2):
            result = first
        elif len(requests) == (3 if named_rejection else 2) and "123," in str(
            first["choices"][0]["message"].get("tool_calls")
        ):
            result = correction if correction is not None else completion('{"amount":123}')
        else:
            result = completion(None)
        if body["stream"]:
            content = sse(
                result,
                split_id=split_id,
                include_failure_text=include_failure_text,
                repeated_finish=repeated_finish and len(requests) == 1,
            )
            if not complete_stream:
                content = content.removesuffix(b"data: [DONE]\n\n")
            if observe_first_frame and len(requests) == 1:
                first_frame, separator, rest = content.partition(b"\n\n")

                class FirstResponse(httpx.SyncByteStream):
                    def __iter__(self):
                        yield first_frame + separator
                        assert ("hidden-failure" in "".join(observed_tokens)) == include_failure_text
                        yield rest

                return httpx.Response(200, stream=FirstResponse(), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=result)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key=api_key, http_client=client, max_retries=0)
        agent = create_agent(
            model=model,
            tools=[transfer],
            system_prompt="A synthetic system prompt.",
            middleware=([StrictChoice()] if strict or named else [])
            + ([ToolCallRecoveryMiddleware(buffer_response=buffer_response)] if enabled else [])
            + ([inner] if inner is not None else []),
        )
        try:
            if stream:
                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": "Record 123."}]},
                    config={"callbacks": [Observer()]},
                    stream_mode=stream_mode,
                ):
                    output.append(chunk)
            else:
                output = [
                    agent.invoke(
                        {"messages": [{"role": "user", "content": "Record 123."}]}, config={"callbacks": [Observer()]}
                    )
                ]
            error = None
        except Exception as exc:
            error = exc
    return {
        "requests": requests,
        "executions": executions,
        "output": output,
        "error": error,
        "observed_errors": observed_errors,
        "observed_tokens": observed_tokens,
    }


def agent_messages(output: list[Any], *, stream: bool) -> list[Any]:
    """Read SDK-returned agent messages from invocation or graph updates."""
    if not stream:
        return output[0]["messages"]
    return [message for update in output for node in update.values() for message in node.get("messages", [])]


@pytest.mark.parametrize("visible_text", [False, True])
def test_live_recovery_only_retries_before_text_is_exposed(visible_text: bool) -> None:
    result = run_agent(
        completion('{"amount":123,}'),
        stream=True,
        buffer_response=False,
        include_failure_text=visible_text,
        observe_first_frame=True,
    )
    if visible_text:
        assert "after output was exposed" in str(result["error"])
        assert "hidden-failure" in str(result["output"])
        assert all(not message.tool_call_chunks for message, _ in result["output"])
        assert len(result["requests"]) == 1
        assert result["executions"] == []
    else:
        assert result["error"] is None
        assert [request.get("tool_choice") for request in result["requests"]] == [None, NAMED_TRANSFER, None]
        assert result["executions"] == [123]


@pytest.mark.parametrize("stream", [False, True])
def test_recovery_delivers_error_to_model_without_changing_user_or_system(stream: bool) -> None:
    arguments = '{"amount":123,}'
    with pytest.raises(json.JSONDecodeError) as parser_error:
        json.loads(arguments)
    result = run_agent(completion(arguments), stream=stream)
    assert result["error"] is None
    assert result["executions"] == [123]
    assert len(result["requests"]) == 3
    assert [request.get("tool_choice") for request in result["requests"]] == [None, NAMED_TRANSFER, None]
    assert {key: value for key, value in result["requests"][0].items() if key not in {"messages", "tool_choice"}} == {
        key: value for key, value in result["requests"][1].items() if key not in {"messages", "tool_choice"}
    }
    for request in result["requests"]:
        assert [m["content"] for m in request["messages"] if m["role"] == "system"] == ["A synthetic system prompt."]
        assert [m["content"] for m in request["messages"] if m["role"] == "user"] == ["Record 123."]
    feedback = result["requests"][1]["messages"]
    assert feedback[-2]["tool_calls"][0]["id"] == "call_bad"
    assert feedback[-2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert feedback[-1]["role"] == "tool" and feedback[-1]["tool_call_id"] == "call_bad"
    header, diagnostic, instruction = feedback[-1]["content"].split("\n")
    assert header.startswith("Error: ") and "The previous tool call is a placeholder." in header
    error = parser_error.value
    assert json.loads(diagnostic) == {
        "raw_arguments": arguments,
        "parse_error": f"{error.msg} (line {error.lineno}, column {error.colno})",
    }
    assert instruction.startswith("Call transfer once")
    assert "hidden-failure" not in str(result["output"])
    assert "hidden-failure" not in "".join(result["observed_tokens"])
    assert len(result["observed_errors"]) == 1
    assert isinstance(result["observed_errors"][0], MalformedToolArguments)
    assert result["observed_errors"][0].usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert not hasattr(result["observed_errors"][0], "message")
    assert "hidden-failure" not in repr(result["observed_errors"][0])
    if not stream:
        assert "The previous tool call is a placeholder." not in str(result["output"])
    else:
        assert feedback[-2]["content"] == "hidden-failure "
        assert feedback[-2]["reasoning_content"] == "synthetic reasoning"


@pytest.mark.parametrize("stream", [False, True])
def test_normal_request_and_output_are_unchanged(stream: bool) -> None:
    baseline = run_agent(completion('{"amount":123}'), enabled=False, stream=stream)
    guarded = run_agent(completion('{"amount":123}'), stream=stream, buffer_response=False)
    assert baseline["error"] is None and guarded["error"] is None
    assert baseline["requests"] == guarded["requests"]
    assert baseline["executions"] == guarded["executions"] == [123]
    assert baseline["observed_tokens"] == guarded["observed_tokens"]
    assert baseline["observed_errors"] == guarded["observed_errors"] == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "scenario", ["filtered_initial", "filtered_correction", "overridden_choice", "routed_model", "routed_mixed"]
)
def test_recovery_uses_the_final_model_request(stream: bool, scenario: str) -> None:
    class Downstream(AgentMiddleware):
        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            if scenario.startswith("routed"):
                return handler(request.override(model=request.model.model_copy()))
            if scenario == "filtered_initial" or (scenario == "filtered_correction" and request.tool_choice):
                return handler(request.override(tools=[]))
            if scenario == "overridden_choice" and request.tool_choice:
                return handler(request.override(tool_choice="none"))
            return handler(request)

    first = completion('{"amount":123,}')
    if scenario == "routed_mixed":
        valid_call = completion('{"amount":123}')["choices"][0]["message"]["tool_calls"][0]
        first["choices"][0]["message"]["tool_calls"].append({**valid_call, "id": "call_valid"})
    result = run_agent(first, stream=stream, inner=Downstream(), include_failure_text=False)
    if scenario == "routed_model":
        assert result["error"] is None
        assert [request.get("tool_choice") for request in result["requests"]] == [None, NAMED_TRANSFER, None]
        assert result["executions"] == [123]
    else:
        reason = "multi-tool" if scenario == "routed_mixed" else "does not allow the tool correction"
        assert reason in str(result["error"])
        assert len(result["requests"]) == 1
        assert bool(result["requests"][0].get("tools")) == (scenario != "filtered_initial")
        assert result["executions"] == []


@pytest.mark.parametrize("malformed", [False, True])
def test_tool_strategy_keeps_sdk_owned_tool_choice(malformed: bool) -> None:
    class Amount(BaseModel):
        value: int

    choices = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        choices.append(body["tool_choice"])
        response = completion('{"value":3,}' if malformed else '{"value":3}')
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = body["tools"][0]["function"]["name"]
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="fixture-key", http_client=client, max_retries=0)
        agent = create_agent(
            model=model, tools=[], response_format=ToolStrategy(Amount), middleware=[ToolCallRecoveryMiddleware()]
        )
        data = {"messages": [{"role": "user", "content": "Return a synthetic amount."}]}
        if malformed:
            with pytest.raises(ValueError, match="does not allow the tool correction"):
                agent.invoke(data)
        else:
            assert agent.invoke(data)["structured_response"] == Amount(value=3)
    assert choices == ["required"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("include_usage", [False, True])
def test_recovered_agent_message_exposes_failed_usage_separately(stream: bool, include_usage: bool) -> None:
    first = completion('{"amount":123,}')
    if not include_usage:
        del first["usage"]
    result = run_agent(first, stream=stream, stream_mode="updates")
    assert result["error"] is None
    messages = agent_messages(result["output"], stream=stream)
    recovered = [m for m in messages if isinstance(m, AIMessage) and "exaone_tool_call_recovery" in m.response_metadata]
    assert len(recovered) == 1
    expected: dict[str, Any] = {"attempts": 1}
    if include_usage:
        expected["failed_usage"] = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert recovered[0].response_metadata["exaone_tool_call_recovery"] == expected
    assert recovered[0].usage_metadata == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
    assert result["executions"] == [123]


@pytest.mark.parametrize("stream", [False, True])
def test_normal_agent_message_has_no_recovery_metadata(stream: bool) -> None:
    result = run_agent(completion('{"amount":123}'), stream=stream, stream_mode="updates")
    assert result["error"] is None
    messages = agent_messages(result["output"], stream=stream)
    assert not any(
        "exaone_tool_call_recovery" in message.response_metadata
        for message in messages
        if isinstance(message, AIMessage)
    )


@pytest.mark.parametrize("stream", [False, True])
def test_repeated_malformed_call_stops_without_running_tool(stream: bool) -> None:
    result = run_agent(completion('{"amount":123,}'), stream=stream, repeat=True)
    assert "valid JSON object" in str(result["error"])
    assert len(result["requests"]) == 2
    assert result["executions"] == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ['{"amount":123', '{"amount":123}'])
def test_length_stop_does_not_trigger_retry_or_tool(stream: bool, arguments: str) -> None:
    result = run_agent(completion(arguments, finish="length"), stream=stream)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert isinstance(result["error"], ValueError)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish,refusal", [(None, False), ("content_filter", False), ("tool_calls", True)])
def test_tool_requires_completed_response(stream: bool, finish: str | None, refusal: bool) -> None:
    first = completion('{"amount":123}')
    first["choices"][0]["finish_reason"] = finish
    if refusal:
        first["choices"][0]["message"]["refusal"] = "Declined."
    result = run_agent(first, stream=stream, buffer_response=False)
    assert "without a completed tool response" in str(result["error"])
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert all(not message.tool_call_chunks for message, _ in result["output"])
    if stream and refusal:
        assert "Declined." in str(result["output"])


@pytest.mark.parametrize("buffer_response", [False, True])
def test_repeated_finish_is_delivered_once_for_a_completed_tool(buffer_response: bool) -> None:
    result = run_agent(completion('{"amount":123}'), stream=True, repeated_finish=True, buffer_response=buffer_response)
    assert result["error"] is None
    assert len(result["requests"]) == 2
    assert result["executions"] == [123]
    reasons = [message.response_metadata.get("finish_reason") for message, _ in result["output"]]
    assert reasons.count("tool_calls") == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_values_fail_before_tool_execution(stream: bool, constant: str) -> None:
    result = run_agent(completion(f'{{"amount":{constant}}}'), stream=stream, buffer_response=False)
    assert "unsupported JSON value" in str(result["error"])
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert result["output"] == []


@pytest.mark.parametrize("unsafe", ["length", "mixed", "alternate", None])
def test_validates_all_nonstream_choices(unsafe: str | None) -> None:
    first = completion('{"amount":123}', finish="length" if unsafe == "length" else "tool_calls")
    alternate = {**completion(None)["choices"][0], "index": 1}
    malformed = completion('{"amount":123,}')["choices"][0]
    if unsafe == "mixed":
        bad_call = malformed["message"]["tool_calls"][0]
        first["choices"][0]["message"]["tool_calls"].append({**bad_call, "id": "malformed"})
    if unsafe == "alternate":
        alternate = {**malformed, "index": 1}
    first["choices"].append(alternate)
    result = run_agent(first)
    if unsafe:
        assert isinstance(result["error"], ValueError)
        assert len(result["requests"]) == 1
        assert result["executions"] == []
    else:
        assert result["error"] is None
        assert len(result["requests"]) == 2
        assert result["executions"] == [123]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_recovery_reuses_only_validated_cache_entries(asynchronous: bool) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        last = body["messages"][-1]
        if last["role"] == "user":
            finish = "length" if last["content"] == "Check a truncated response." else "tool_calls"
            return httpx.Response(200, json=completion('{"amount":123}', finish=finish))
        return httpx.Response(200, json=completion(None))

    async def run(agent: Any, prompt: str) -> None:
        data = {"messages": [{"role": "user", "content": prompt}]}
        if asynchronous:
            await agent.ainvoke(data)
        else:
            agent.invoke(data)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_client:
            model = ChatEXAONE(
                api_key="fixture", http_client=client, http_async_client=async_client, cache=InMemoryCache()
            )
            plain = create_agent(model=model, tools=[transfer])
            guarded = create_agent(model=model, tools=[transfer], middleware=[ToolCallRecoveryMiddleware()])
            await run(plain, "Check a truncated response.")
            assert executions == [123]
            with pytest.raises(ValueError, match="truncated"):
                await run(guarded, "Check a truncated response.")
            assert executions == [123]
            assert len(requests) == 3

            await run(guarded, "Check a completed response.")
            count = len(requests)
            await run(guarded, "Check a completed response.")
            assert all(request["messages"][-1]["role"] == "tool" for request in requests[count:])
            assert executions == [123, 123, 123]
            assert all("_exaone_tool_call_recovery" not in body for body in requests)


def test_incomplete_stream_does_not_retry_or_emit_partial_output() -> None:
    result = run_agent(completion('{"amount":123,}'), stream=True, complete_stream=False)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert isinstance(result["error"], ValueError)
    assert "[DONE]" in str(result["error"])
    assert "hidden-failure" not in str(result["output"])
    assert "hidden-failure" not in "".join(result["observed_tokens"])


def test_malformed_stream_with_multiple_id_fragments_is_not_retried() -> None:
    result = run_agent(completion('{"amount":123,}'), stream=True, split_id=True)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert result["output"] == []
    assert result["observed_tokens"] == []
    assert isinstance(result["error"], ValueError)
    assert "ambiguous" in str(result["error"])
    assert result["error"].__context__ is None


def test_valid_stream_with_multiple_id_fragments_keeps_existing_behavior() -> None:
    first = completion('{"amount":123}')
    baseline = run_agent(first, enabled=False, stream=True, split_id=True)
    guarded = run_agent(first, stream=True, split_id=True)
    assert type(baseline["error"]) is type(guarded["error"])
    assert str(baseline["error"]) == str(guarded["error"])
    assert baseline["requests"] == guarded["requests"]
    assert baseline["executions"] == guarded["executions"]
    assert baseline["observed_tokens"] == guarded["observed_tokens"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("credential", ["fixture +/", "fixture%20%2B%2F", "fixture+%2B%2F", "fixture%20%2b%2f"])
def test_does_not_resend_raw_arguments_containing_credentials(stream: bool, credential: str) -> None:
    raw = json.dumps({"amount": 123, "value": credential})[:-1] + ",}"
    result = run_agent(completion(raw), api_key="fixture +/", stream=stream)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert isinstance(result["error"], ValueError)
    assert result["error"].__context__ is None
    assert credential not in str(result["error"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field", ["id", "name"])
@pytest.mark.parametrize("credential", ["fixture +/", "fixture%20%2B%2F", "fixture+%2B%2F", "fixture%20%2b%2f"])
def test_does_not_resend_call_id_or_name_containing_credentials(stream: bool, field: str, credential: str) -> None:
    first = completion('{"amount":123,}')
    call = first["choices"][0]["message"]["tool_calls"][0]
    if field == "id":
        call["id"] = f"call-{credential}"
    else:
        call["function"]["name"] = f"transfer-{credential}"
    result = run_agent(first, api_key="fixture +/", stream=stream)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert result["output"] == []
    assert result["observed_tokens"] == []
    assert isinstance(result["error"], ValueError)
    assert result["error"].__context__ is None
    assert credential not in str(result["error"])
    assert all(credential not in repr(error) for error in result["observed_errors"])


@pytest.mark.parametrize("stream", [False, True])
def test_failed_usage_omits_credential_in_detail_keys(stream: bool) -> None:
    first = completion('{"amount":123,}')
    first["usage"]["prompt_tokens_details"] = {"fixture%20%2b%2f": 4, "cached_tokens": 2}
    result = run_agent(first, api_key="fixture +/", stream=stream, stream_mode="updates")
    assert result["error"] is None
    assert result["executions"] == [123]
    expected = {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    assert result["observed_errors"][0].usage == expected
    messages = agent_messages(result["output"], stream=stream)
    recovered = [m for m in messages if isinstance(m, AIMessage) and "exaone_tool_call_recovery" in m.response_metadata]
    assert len(recovered) == 1
    assert recovered[0].response_metadata["exaone_tool_call_recovery"]["failed_usage"] == expected


def test_retry_names_tool_and_preserves_strict_schema() -> None:
    result = run_agent(completion('{"amount":123,}'), strict=True)
    assert result["error"] is None
    assert result["executions"] == [123]
    assert result["requests"][0]["tool_choice"] == "required"
    assert result["requests"][1]["tool_choice"] == NAMED_TRANSFER
    assert result["requests"][0]["tools"] == result["requests"][1]["tools"]
    assert result["requests"][0]["tools"][0]["function"]["strict"] is True


@pytest.mark.parametrize("stream", [False, True])
def test_required_fallback_needs_explicit_named_choice_rejection(stream: bool) -> None:
    first = completion('{"amount":123,}')
    recovered = run_agent(first, stream=stream, named_rejection="Named tool choice is not supported")
    assert recovered["error"] is None
    requests = recovered["requests"]
    assert [request.get("tool_choice") for request in requests] == [None, NAMED_TRANSFER, "required", None]
    assert requests[1]["messages"] == requests[2]["messages"]
    assert recovered["executions"] == [123]

    failed = run_agent(first, stream=stream, named_rejection="invalid request to model")
    assert failed["error"] is not None
    assert len(failed["requests"]) == 2
    assert failed["executions"] == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["text", "wrong_tool", "multiple", "blank", "array", "nonstandard_json"])
def test_invalid_corrections_never_execute(case: str, stream: bool) -> None:
    corrected = completion(
        {
            "text": None,
            "blank": " ",
            "array": "[]",
            "nonstandard_json": '{"amount":NaN}',
        }.get(case, '{"amount":123}')
    )
    if case in {"wrong_tool", "multiple"}:
        calls = corrected["choices"][0]["message"]["tool_calls"]
        if case == "wrong_tool":
            calls[0]["function"]["name"] = "other"
        else:
            calls.append({**calls[0], "id": "extra"})
    result = run_agent(completion('{"amount":123,}'), stream=stream, correction=corrected)
    assert result["error"] is not None
    assert result["observed_tokens"] == []
    assert result["executions"] == []
    assert len(result["requests"]) == 2


@pytest.mark.parametrize("stream", [False, True])
def test_named_tool_stop_finish_recovery_and_normal_path(stream: bool) -> None:
    expected_choice = {"type": "function", "function": {"name": "transfer"}}
    valid = completion('{"amount":123}', finish="stop")
    baseline = run_agent(valid, enabled=False, stream=stream, named=True)
    guarded = run_agent(valid, stream=stream, named=True)
    assert baseline["error"] is None and guarded["error"] is None
    assert baseline["requests"] == guarded["requests"]
    assert baseline["executions"] == guarded["executions"] == [123]
    assert baseline["observed_tokens"] == guarded["observed_tokens"]
    assert baseline["requests"][0]["tool_choice"] == expected_choice

    recovered = run_agent(completion('{"amount":123,}', finish="stop"), stream=stream, named=True)
    assert recovered["error"] is None
    assert recovered["executions"] == [123]
    assert len(recovered["requests"]) == 3
    assert recovered["requests"][0]["tool_choice"] == recovered["requests"][1]["tool_choice"] == expected_choice
    messages = recovered["requests"][1]["messages"]
    assert messages[-2]["tool_calls"][0]["id"] == "call_bad"
    assert messages[-1]["role"] == "tool" and messages[-1]["tool_call_id"] == "call_bad"
    assert json.loads(messages[-1]["content"].split("\n")[1])["raw_arguments"] == '{"amount":123,}'
    assert "hidden-failure" not in str(recovered["output"])


def test_recovery_delimits_multiline_arguments_and_uses_original_character_location() -> None:
    raw = '{\n"note":"😀```","amount":123,,}'
    result = run_agent(completion(raw), stream=True)
    assert result["error"] is None
    feedback = result["requests"][1]["messages"][-1]["content"]
    _, diagnostic, _ = feedback.split("\n")
    assert json.loads(diagnostic)["raw_arguments"] == raw
    assert json.loads(diagnostic)["parse_error"].endswith("(line 2, column 28)")
    assert result["executions"] == [123]


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_valid_and_invalid_calls_fail_before_tool_execution(stream: bool) -> None:
    first = completion('{"amount":123,}')
    first["choices"][0]["message"]["tool_calls"].append(
        {"id": "call_good", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":123}'}}
    )
    result = run_agent(first, stream=stream)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert isinstance(result["error"], ValueError)
    assert "multi-tool" in str(result["error"])
    assert result["error"].__context__ is None
    assert result["output"] == []
    assert result["observed_tokens"] == []


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_calls_without_middleware_keep_existing_sdk_behavior(stream: bool) -> None:
    first = completion('{"amount":123,}')
    first["choices"][0]["message"]["tool_calls"].append(
        {"id": "call_good", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":456}'}}
    )
    result = run_agent(first, enabled=False, stream=stream)
    assert result["error"] is None
    assert len(result["requests"]) == 3
    assert sorted(result["executions"]) == [123, 456]


@pytest.mark.parametrize("stream", [False, True])
def test_multiple_malformed_calls_fail_before_tool_execution(stream: bool) -> None:
    first = completion('{"amount":123,}')
    first["choices"][0]["message"]["tool_calls"].append(
        {"id": "call_other", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":456,}'}}
    )
    result = run_agent(first, stream=stream)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert isinstance(result["error"], ValueError)
    assert "multi-tool" in str(result["error"])
    assert result["output"] == []
    assert result["observed_tokens"] == []


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_call_failure_does_not_expose_credentials(stream: bool) -> None:
    credential = "fixture%20%2B%2F"
    raw = json.dumps({"amount": 123, "value": credential})[:-1] + ",}"
    first = completion(raw)
    first["choices"][0]["message"]["tool_calls"].append(
        {"id": "call_good", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":456}'}}
    )
    result = run_agent(first, api_key="fixture +/", stream=stream)
    assert len(result["requests"]) == 1
    assert result["executions"] == []
    assert isinstance(result["error"], ValueError)
    assert result["error"].__context__ is None
    assert credential not in str(result["error"])
    assert all(credential not in repr(error) for error in result["observed_errors"])


@pytest.mark.parametrize("stream", [False, True])
def test_multiple_valid_calls_keep_native_behavior(stream: bool) -> None:
    first = completion('{"amount":123}')
    first["choices"][0]["message"]["tool_calls"].append(
        {"id": "call_good", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":456}'}}
    )
    baseline = run_agent(first, enabled=False, stream=stream)
    guarded = run_agent(first, stream=stream)
    assert baseline["error"] is None and guarded["error"] is None
    assert baseline["requests"] == guarded["requests"]
    assert sorted(baseline["executions"]) == sorted(guarded["executions"]) == [123, 456]
    assert baseline["observed_tokens"] == guarded["observed_tokens"]


async def test_cancellation_does_not_retry_or_execute_tool() -> None:
    requests: list[httpx.Request] = []
    executions: list[int] = []
    received = asyncio.Event()

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    async def cancel(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        received.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancel)) as client:
        agent = create_agent(
            model=ChatEXAONE(api_key="fixture", http_async_client=client, max_retries=0),
            tools=[transfer],
            middleware=[ToolCallRecoveryMiddleware()],
        )
        task = asyncio.create_task(agent.ainvoke({"messages": [{"role": "user", "content": "Record 123."}]}))
        await received.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(requests) == 1
    assert executions == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("malformed", [False, True])
async def test_async_multiple_tool_calls_validate_before_execution(stream: bool, malformed: bool) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []
    output: list[Any] = []
    first = completion('{"amount":123,}' if malformed else '{"amount":123}')
    first["choices"][0]["message"]["tool_calls"].append(
        {"id": "call_good", "type": "function", "function": {"name": "transfer", "arguments": '{"amount":456}'}}
    )

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        data = first if len(requests) == 1 else completion(None)
        if body["stream"]:
            return httpx.Response(200, content=sse(data), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        agent = create_agent(
            model=ChatEXAONE(api_key="fixture", http_async_client=client, max_retries=0),
            tools=[transfer],
            middleware=[ToolCallRecoveryMiddleware()],
        )
        error = None
        try:
            if stream:
                async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": "Record values."}]}, stream_mode="messages"
                ):
                    output.append(chunk)
            else:
                output.append(await agent.ainvoke({"messages": [{"role": "user", "content": "Record values."}]}))
        except Exception as exc:
            error = exc

    if malformed:
        assert len(requests) == 1
        assert executions == []
        assert bool(output) is stream
        assert all(not message.tool_call_chunks for message, _ in output)
        assert isinstance(error, ValueError)
        assert "multi-tool" in str(error)
        assert error.__context__ is None
    else:
        assert error is None
        assert len(requests) == 2
        assert sorted(executions) == [123, 456]


@pytest.mark.parametrize("malformed", [False, True])
async def test_async_stream_with_multiple_id_fragments(malformed: bool) -> None:
    first = completion('{"amount":123,}' if malformed else '{"amount":123}')

    async def run(enabled: bool) -> dict[str, Any]:
        requests: list[dict[str, Any]] = []
        executions: list[int] = []
        output: list[Any] = []

        @tool
        def transfer(amount: int) -> str:
            """Record a synthetic amount."""
            executions.append(amount)
            return "Recorded."

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            result = first if len(requests) == 1 else completion(None)
            return httpx.Response(
                200, content=sse(result, split_id=True), headers={"content-type": "text/event-stream"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            agent = create_agent(
                model=ChatEXAONE(api_key="fixture", http_async_client=client, max_retries=0),
                tools=[transfer],
                middleware=[ToolCallRecoveryMiddleware()] if enabled else [],
            )
            error = None
            try:
                async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": "Record 123."}]}, stream_mode="messages"
                ):
                    output.append(chunk)
            except Exception as exc:
                error = exc
        return {"requests": requests, "executions": executions, "output": output, "error": error}

    result = await run(True)
    if malformed:
        assert len(result["requests"]) == 1
        assert result["executions"] == []
        assert "hidden-failure" in str(result["output"])
        assert all(not message.tool_call_chunks for message, _ in result["output"])
        assert isinstance(result["error"], ValueError)
        assert "ambiguous" in str(result["error"])
    else:
        baseline = await run(False)
        assert type(baseline["error"]) is type(result["error"])
        assert str(baseline["error"]) == str(result["error"])
        assert baseline["requests"] == result["requests"]
        assert baseline["executions"] == result["executions"]
        assert [chunk[0].content for chunk in baseline["output"]] == [chunk[0].content for chunk in result["output"]]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", ["tool_calls", "stop"])
async def test_async_agent_recovers_without_changing_input_messages(stream: bool, finish: str) -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []
    output: list[Any] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        data = (
            completion('{"amount":123,}', finish=finish)
            if len(requests) == 1
            else (completion('{"amount":123}') if len(requests) == 2 else completion(None))
        )
        if body["stream"]:
            return httpx.Response(200, content=sse(data), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=data)

    class NamedChoice(AgentMiddleware):
        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            return await handler(request.override(tool_choice="transfer"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        agent = create_agent(
            model=ChatEXAONE(api_key="fixture", http_async_client=client, max_retries=0),
            tools=[transfer],
            system_prompt="A synthetic system prompt.",
            middleware=([NamedChoice()] if finish == "stop" else []) + [ToolCallRecoveryMiddleware()],
        )
        if stream:
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": "Record 123."}]}, stream_mode="updates"
            ):
                output.append(chunk)
        else:
            output.append(await agent.ainvoke({"messages": [{"role": "user", "content": "Record 123."}]}))
    assert len(requests) == 3
    assert executions == [123]
    assert all(
        [m["content"] for m in req["messages"] if m["role"] == "system"] == ["A synthetic system prompt."]
        for req in requests
    )
    assert all([m["content"] for m in req["messages"] if m["role"] == "user"] == ["Record 123."] for req in requests)
    assert requests[1]["messages"][-1]["tool_call_id"] == "call_bad"
    if finish == "stop":
        expected_choice = {"type": "function", "function": {"name": "transfer"}}
        assert requests[0]["tool_choice"] == requests[1]["tool_choice"] == expected_choice
    assert "hidden-failure" not in str(output)
    messages = agent_messages(output, stream=stream)
    recovered = [m for m in messages if isinstance(m, AIMessage) and "exaone_tool_call_recovery" in m.response_metadata]
    assert len(recovered) == 1
    assert recovered[0].response_metadata["exaone_tool_call_recovery"] == {
        "attempts": 1,
        "failed_usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    assert recovered[0].usage_metadata == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}


async def test_concurrent_agents_do_not_share_recovery_state() -> None:
    requests: dict[str, list[dict[str, Any]]] = {"first": [], "second": []}
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    async def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        requests[user].append(body)
        await asyncio.sleep(0)
        count = len(requests[user])
        if user == "first" and count == 1:
            result = completion('{"amount":123,}')
        elif (user == "first" and count == 2) or (user == "second" and count == 1):
            result = completion('{"amount":123}')
        else:
            result = completion(None)
        return httpx.Response(200, json=result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        agent = create_agent(
            model=ChatEXAONE(api_key="fixture", http_async_client=client, max_retries=0),
            tools=[transfer],
            middleware=[ToolCallRecoveryMiddleware()],
        )
        await asyncio.gather(
            agent.ainvoke({"messages": [{"role": "user", "content": "first"}]}),
            agent.ainvoke({"messages": [{"role": "user", "content": "second"}]}),
        )
    assert len(requests["first"]) == 3
    assert len(requests["second"]) == 2
    assert executions == [123, 123]
    assert requests["first"][1]["messages"][-1]["role"] == "tool"
    assert not any(m["role"] == "tool" for m in requests["second"][0]["messages"])


def test_checkpoint_resume_does_not_store_or_resend_failed_placeholder() -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        data = (
            completion('{"amount":123,}')
            if len(requests) == 1
            else completion('{"amount":123}')
            if len(requests) == 2
            else completion(None)
        )
        return httpx.Response(200, json=data)

    config = {"configurable": {"thread_id": "synthetic-checkpoint"}}
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        agent = create_agent(
            model=ChatEXAONE(api_key="fixture", http_client=client, max_retries=0),
            tools=[transfer],
            system_prompt="A synthetic system prompt.",
            middleware=[ToolCallRecoveryMiddleware()],
            checkpointer=MemorySaver(),
        )
        first = agent.invoke({"messages": [{"role": "user", "content": "Record 123."}]}, config=config)
        saved = agent.get_state(config).values["messages"]
        second = agent.invoke({"messages": [{"role": "user", "content": "Continue."}]}, config=config)
        resumed = agent.get_state(config).values["messages"]

    assert len(requests) == 4
    assert executions == [123]
    assert [m["content"] for m in requests[3]["messages"] if m["role"] == "user"] == ["Record 123.", "Continue."]
    assert [m["content"] for m in requests[3]["messages"] if m["role"] == "system"] == ["A synthetic system prompt."]
    assert "The previous tool call is a placeholder." not in str(requests[3]["messages"])
    assert '{"amount":123,}' not in str(requests[3]["messages"])
    assert all(
        "The previous tool call is a placeholder." not in str(value) for value in [first, saved, second, resumed]
    )
    assert all('{"amount":123,}' not in str(value) for value in [first, saved, second, resumed])
    successful_calls = [
        call
        for message in requests[3]["messages"]
        for call in message.get("tool_calls", [])
        if call["id"] == "call_bad"
    ]
    assert len(successful_calls) == 1
    assert json.loads(successful_calls[0]["function"]["arguments"]) == {"amount": 123}


async def test_async_checkpoint_resume_does_not_store_or_resend_failed_placeholder() -> None:
    requests: list[dict[str, Any]] = []
    executions: list[int] = []

    @tool
    def transfer(amount: int) -> str:
        """Record a synthetic amount."""
        executions.append(amount)
        return "Recorded."

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        data = (
            completion('{"amount":123,}')
            if len(requests) == 1
            else completion('{"amount":123}')
            if len(requests) == 2
            else completion(None)
        )
        return httpx.Response(200, json=data)

    config = {"configurable": {"thread_id": "synthetic-async-checkpoint"}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        agent = create_agent(
            model=ChatEXAONE(api_key="fixture", http_async_client=client, max_retries=0),
            tools=[transfer],
            system_prompt="A synthetic system prompt.",
            middleware=[ToolCallRecoveryMiddleware()],
            checkpointer=MemorySaver(),
        )
        first = await agent.ainvoke({"messages": [{"role": "user", "content": "Record 123."}]}, config=config)
        saved = (await agent.aget_state(config)).values["messages"]
        second = await agent.ainvoke({"messages": [{"role": "user", "content": "Continue."}]}, config=config)
        resumed = (await agent.aget_state(config)).values["messages"]

    assert len(requests) == 4
    assert executions == [123]
    assert [m["content"] for m in requests[3]["messages"] if m["role"] == "user"] == ["Record 123.", "Continue."]
    assert [m["content"] for m in requests[3]["messages"] if m["role"] == "system"] == ["A synthetic system prompt."]
    assert "The previous tool call is a placeholder." not in str(requests[3]["messages"])
    assert '{"amount":123,}' not in str(requests[3]["messages"])
    assert all(
        "The previous tool call is a placeholder." not in str(value) for value in [first, saved, second, resumed]
    )
    assert all('{"amount":123,}' not in str(value) for value in [first, saved, second, resumed])
    successful_calls = [
        call
        for message in requests[3]["messages"]
        for call in message.get("tool_calls", [])
        if call["id"] == "call_bad"
    ]
    assert len(successful_calls) == 1
    assert json.loads(successful_calls[0]["function"]["arguments"]) == {"amount": 123}
