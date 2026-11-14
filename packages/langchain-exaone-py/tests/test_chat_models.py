"""Synthetic protocol tests for the EXAONE LangChain chat model."""

import json
from typing import Any
from urllib.parse import quote, quote_plus

import httpx
import pytest
from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from langchain_exaone import ChatEXAONE, ExaoneAPIError


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example/v1",
        "ftp://api.example/v1",
        "https://user:password@api.example/v1",
        "https://api.example/v1?token=secret",
        "https://api.example/v1#fragment",
    ],
)
def test_rejects_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(ValidationError, match="base_url"):
        ChatEXAONE(api_key="synthetic-key", base_url=base_url)


@pytest.mark.parametrize("hostname", ["localhost", "127.0.0.1", "[::1]"])
def test_allows_loopback_http_for_local_test_servers(hostname: str) -> None:
    model = ChatEXAONE(api_key="synthetic-key", base_url=f"http://{hostname}:8000/v1")

    assert model.base_url == f"http://{hostname}:8000/v1"


def _completion(*, content: str = "4", reasoning: str = "Let me calculate.") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "k-exaone_v2.5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                "finish_reason": "stop",
                "matched_stop": None,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "total_tokens": 9,
            "reasoning_tokens": 3,
        },
    }


def test_invoke_maps_reasoning_usage_and_recommended_defaults() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    model = ChatEXAONE(
        api_key="synthetic-key",
        http_client=client,
        chat_template_kwargs={"instance_option": "value"},
        model_kwargs={"chat_template_kwargs": {"model_option": "value"}},
    )
    try:
        response = model.invoke("What is 2 + 2?", chat_template_kwargs={"call_option": "value"})
        model.invoke("Use the alternate model.", model="alternate-model")
    finally:
        client.close()

    assert response.content == "4"
    assert response.additional_kwargs["reasoning_content"] == "Let me calculate."
    assert response.usage_metadata == {
        "input_tokens": 4,
        "output_tokens": 5,
        "total_tokens": 9,
        "output_token_details": {"reasoning": 3},
    }
    assert requests[0]["temperature"] == 1.0
    assert requests[0]["top_p"] == 0.95
    assert requests[0]["top_k"] == -1
    assert requests[0]["presence_penalty"] == 0.0
    assert requests[0]["chat_template_kwargs"] == {
        "instance_option": "value",
        "model_option": "value",
        "call_option": "value",
        "preserve_thinking": True,
    }
    assert requests[1]["model"] == "alternate-model"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_generate_preserves_multiple_candidates(asynchronous: bool) -> None:
    payload = _completion(content="First", reasoning="Plan A")
    payload["choices"].append(
        {
            "index": 1,
            "message": {"role": "assistant", "content": "Second", "reasoning_content": "Plan B"},
            "finish_reason": "length",
        }
    )
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_client:
            model = ChatEXAONE(
                api_key="synthetic-key", http_client=client, http_async_client=async_client, model_kwargs={"n": 2}
            )
            messages = [[HumanMessage(content="Question")]]
            result = await model.agenerate(messages) if asynchronous else model.generate(messages)

    assert requests[0]["n"] == 2
    assert [item.message.content for item in result.generations[0]] == ["First", "Second"]
    assert [item.message.additional_kwargs["reasoning_content"] for item in result.generations[0]] == [
        "Plan A",
        "Plan B",
    ]
    assert [item.generation_info["finish_reason"] for item in result.generations[0]] == ["stop", "length"]
    assert result.llm_output == {
        "model_name": "k-exaone_v2.5",
        "token_usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9, "reasoning_tokens": 3},
    }


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_generate_aggregates_usage_once_per_prompt(asynchronous: bool) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_completion()))
    with httpx.Client(transport=transport) as client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = ChatEXAONE(api_key="synthetic-key", http_client=client, http_async_client=async_client)
            messages = [[HumanMessage(content="First")], [HumanMessage(content="Second")]]
            result = await model.agenerate(messages) if asynchronous else model.generate(messages)

    assert result.llm_output == {
        "model_name": "k-exaone_v2.5",
        "token_usage": {"prompt_tokens": 8, "completion_tokens": 10, "total_tokens": 18, "reasoning_tokens": 6},
    }


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_generate_rejects_malformed_candidate(asynchronous: bool) -> None:
    payload = _completion()
    payload["choices"].append(None)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    with httpx.Client(transport=transport) as client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = ChatEXAONE(api_key="synthetic-key", http_client=client, http_async_client=async_client)
            with pytest.raises(ValueError, match="choices must be objects"):
                if asynchronous:
                    await model.agenerate([[HumanMessage(content="Question")]])
                else:
                    model.generate([[HumanMessage(content="Question")]])


def test_retries_default_to_two_and_can_be_disabled() -> None:
    assert ChatEXAONE(api_key="synthetic-key").max_retries == 2
    assert ChatEXAONE(api_key="synthetic-key", max_retries=0).max_retries == 0


@pytest.mark.parametrize("choice,expected", [(True, "required"), (False, None), ("any", "required")])
def test_bind_tools_normalizes_langchain_tool_choice(choice: bool | str, expected: str | None) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client)
        model.bind_tools([tool], tool_choice=choice).invoke("Use the tool.")

    assert requests[0].get("tool_choice") == expected
    if expected is None:
        assert "tool_choice" not in requests[0]


def test_reasoning_effort_medium_is_sent_on_the_wire() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        ChatEXAONE(api_key="synthetic-key", http_client=client, reasoning_effort="medium").invoke("hello")
    finally:
        client.close()
    assert requests[0]["reasoning_effort"] == "medium"


def test_model_kwargs_cannot_replace_serialized_messages_or_stream_flag() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    model = ChatEXAONE(
        api_key="synthetic-key",
        http_client=client,
        model_kwargs={
            "messages": [{"role": "user", "content": "replacement"}],
            "stream": True,
        },
    )
    try:
        model.invoke("serialized input")
    finally:
        client.close()
    assert requests[0]["messages"] == [{"role": "user", "content": "serialized input"}]
    assert requests[0]["stream"] is False


def test_api_key_is_not_exposed_in_model_repr() -> None:
    synthetic_key = "synthetic-repr-key"
    assert synthetic_key not in repr(ChatEXAONE(api_key=synthetic_key))


@pytest.mark.asyncio
async def test_default_timeout_is_unbounded_for_sync_and_async_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_timeouts: list[tuple[str, float | None]] = []
    original_client = httpx.Client
    original_async_client = httpx.AsyncClient

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        observed_timeouts.append(("sync", kwargs.get("timeout")))
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_completion()))
        return original_client(*args, transport=transport, **kwargs)

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        observed_timeouts.append(("async", kwargs.get("timeout")))
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=_completion()))
        return original_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("langchain_exaone.chat_models.httpx.Client", client_factory)
    monkeypatch.setattr("langchain_exaone.chat_models.httpx.AsyncClient", async_client_factory)
    model = ChatEXAONE(api_key="synthetic-key")
    assert model.timeout is None
    with pytest.raises(ValidationError, match="timeout must be a finite positive number"):
        ChatEXAONE(api_key="synthetic-key", timeout=0)

    model.invoke("sync")
    await model.ainvoke("async")
    assert observed_timeouts == [("sync", None), ("async", None)]


def test_assistant_reasoning_and_tool_history_are_sent_back() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    model = ChatEXAONE(api_key="synthetic-key", http_client=client)
    assistant = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "first turn", "tool_calls": [{"id": "call-1"}]},
        tool_calls=[{"name": "lookup", "args": {"city": "Seoul"}, "id": "call-1"}],
    )
    history = [HumanMessage("Check Seoul."), assistant, ToolMessage("Sunny", tool_call_id="call-1")]
    try:
        model.invoke(history)
    finally:
        client.close()

    assistant_request = requests[0]["messages"][1]
    assert assistant_request["reasoning_content"] == "first turn"
    assert assistant_request["tool_calls"] == [
        {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"city": "Seoul"}'}}
    ]
    assert requests[0]["messages"][2] == {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"}


def test_named_chat_messages_are_preserved_in_requests() -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    messages = [
        SystemMessage("Instructions", name="policy"),
        HumanMessage("Hello", name="user_one"),
        AIMessage("Reply", name="assistant_one"),
        ChatMessage(content="Other", role="user", name="user_two"),
    ]
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        ChatEXAONE(api_key="synthetic-key", http_client=client).invoke(messages)

    assert [message.get("name") for message in requests[0]["messages"]] == [
        "policy",
        "user_one",
        "assistant_one",
        "user_two",
    ]


def test_edited_tool_call_is_used_for_conversation_history() -> None:
    payload = _completion()
    payload["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call-1",
            "type": "function",
            "provider_extra": {"tag": "fixture"},
            "function": {"name": "lookup", "arguments": '{"key":"old"}', "extra": "fixture"},
        }
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client)
        first = model.invoke("Use lookup.")
        edited = first.model_copy(update={"tool_calls": [{"name": "lookup", "args": {"key": "new"}, "id": "call-1"}]})
        model.invoke([edited])

    tool_call = requests[1]["messages"][0]["tool_calls"][0]
    assert tool_call["id"] == "call-1"
    assert json.loads(tool_call["function"]["arguments"]) == {"key": "new"}
    assert tool_call["provider_extra"] == {"tag": "fixture"}
    assert tool_call["function"]["extra"] == "fixture"


@pytest.mark.parametrize(("old_value", "new_value"), [(1, True), (0, False)])
def test_tool_call_history_preserves_json_value_types(old_value: int, new_value: bool) -> None:
    payload = _completion()
    payload["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"value": old_value})},
        }
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client)
        first = model.invoke("Use lookup.")
        first.tool_calls[0]["args"] = {"value": new_value}
        model.invoke([first])

    args = json.loads(requests[1]["messages"][0]["tool_calls"][0]["function"]["arguments"])
    assert args["value"] is new_value


def test_tool_call_history_respects_removed_calls_and_keeps_invalid_calls() -> None:
    payload = _completion()
    payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"a"}'}},
        {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"b"}'}},
        {"id": "bad", "type": "function", "function": {"name": "lookup", "arguments": '{"key":'}},
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client)
        first = model.invoke("Use lookup.")
        edited = first.model_copy(update={"tool_calls": first.tool_calls[:1]})
        model.invoke([edited])

    calls = requests[1]["messages"][0]["tool_calls"]
    assert [call["id"] for call in calls] == ["a", "bad"]
    assert calls[1]["function"]["arguments"] == '{"key":'


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_tool_call_history_does_not_restore_all_removed_calls(output_version: str) -> None:
    payload = _completion()
    payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client, output_version=output_version)
        first = model.invoke("Use lookup.")
        model.invoke([first.model_copy(update={"tool_calls": []})])

    assert "tool_calls" not in requests[1]["messages"][0]


def test_v1_output_can_be_reused_as_conversation_history() -> None:
    payload = _completion(content="Calling tool", reasoning="Need a lookup")
    payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"Seoul"}'}}
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client, output_version="v1")
        first = model.invoke("Lookup Seoul.")
        model.invoke([first, ToolMessage("Sunny", tool_call_id="call-1")])

    assert requests[1]["messages"][0] == {
        "role": "assistant",
        "content": "Calling tool",
        "reasoning_content": "Need a lookup",
        "tool_calls": payload["choices"][0]["message"]["tool_calls"],
    }


def test_v1_content_blocks_alone_can_supply_assistant_tool_calls() -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "Need a lookup"},
            {"type": "tool_call", "id": "call-1", "name": "lookup", "args": {"city": "Seoul"}},
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        ChatEXAONE(api_key="synthetic-key", http_client=client).invoke([message])

    assistant = requests[0]["messages"][0]
    assert assistant["reasoning_content"] == "Need a lookup"
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Seoul"}


def test_v1_stream_keeps_reasoning_and_tool_blocks_separate_when_combined() -> None:
    frames = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Think"}}]},
        {"choices": [{"index": 0, "delta": {"content": "Calling"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    stream = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, content=stream) if body["stream"] else httpx.Response(200, json=_completion())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client, output_version="v1")
        chunks = list(model.stream("Question"))
        result = chunks[0]
        for chunk in chunks[1:]:
            result += chunk
        result.tool_calls[0]["args"] = {"edited": True}
        model.invoke([result, ToolMessage("Done", tool_call_id="call-1")])
        model.invoke([result.model_copy(update={"tool_calls": []})])

    blocks = result.content
    assert isinstance(blocks, list)
    assert any(block.get("type") == "reasoning" and block.get("reasoning") == "Think" for block in blocks)
    assert any(block.get("type") == "text" and block.get("text") == "Calling" for block in blocks)
    assert any(block.get("type") == "tool_call" and block.get("id") == "call-1" for block in blocks)
    assert result.tool_calls[0]["name"] == "lookup"
    assert requests[1]["messages"][0]["reasoning_content"] == "Think"
    assert requests[1]["messages"][0]["tool_calls"][0]["id"] == "call-1"
    assert json.loads(requests[1]["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {"edited": True}
    assert "tool_calls" not in requests[2]["messages"][0]


def test_stream_message_supports_langchain_serialization() -> None:
    frame = {"choices": [{"index": 0, "delta": {"reasoning_content": "Think"}}]}
    stream = f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=stream))) as client:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client, output_version="v1")
        chunk = next(iter(model.stream("Question")))

    restored = load(dumpd(chunk), allowed_objects="messages")
    assert restored.content == chunk.content
    assert restored.additional_kwargs == chunk.additional_kwargs


def test_mixed_string_and_text_content_blocks_are_serialized() -> None:
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        ChatEXAONE(api_key="synthetic-key", http_client=client).invoke(
            [HumanMessage(content=["before ", {"type": "text", "text": "middle"}, " after"])]
        )
    finally:
        client.close()

    assert requests[0]["messages"] == [{"role": "user", "content": "before middle after"}]


def test_tool_calls_are_returned_as_langchain_tool_calls() -> None:
    payload = _completion()
    payload["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Need a weather tool.",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Seoul"}'}}
        ],
    }
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    try:
        response = ChatEXAONE(api_key="synthetic-key", http_client=client).invoke("Weather?")
    finally:
        client.close()
    assert response.tool_calls == [{"name": "weather", "args": {"city": "Seoul"}, "id": "call-1", "type": "tool_call"}]
    assert response.additional_kwargs["reasoning_content"] == "Need a weather tool."


@pytest.mark.parametrize(
    ("tool_calls", "error_message"),
    [
        ({"call-1": {}}, "tool_calls must be an array"),
        ([None], "tool call entries must be objects"),
        ([{"function": "weather"}], "tool call function must be an object"),
    ],
)
def test_invoke_rejects_malformed_tool_call_shapes(tool_calls: Any, error_message: str) -> None:
    payload = _completion()
    payload["choices"][0]["message"] = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    try:
        with pytest.raises(ValueError, match=error_message):
            ChatEXAONE(api_key="synthetic-key", http_client=client).invoke("hello")
    finally:
        client.close()


@pytest.mark.parametrize("operation", ["stream", "astream"])
@pytest.mark.parametrize(
    ("tool_calls", "error_message"),
    [
        ({"call-1": {}}, "stream delta tool_calls must be an array"),
        ([None], "stream tool call entries must be objects"),
        ([{"function": "weather"}], "stream tool call function must be an object"),
    ],
)
@pytest.mark.asyncio
async def test_stream_rejects_malformed_tool_call_shapes(
    operation: str,
    tool_calls: Any,
    error_message: str,
) -> None:
    event = {"choices": [{"index": 0, "delta": {"tool_calls": tool_calls}}]}
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    sync_client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body.encode())))
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body.encode()))
    )
    model = ChatEXAONE(api_key="synthetic-key", http_client=sync_client, http_async_client=async_client)
    try:
        with pytest.raises(ValueError, match=error_message):
            if operation == "stream":
                list(model.stream("hello"))
            else:
                _ = [chunk async for chunk in model.astream("hello")]
    finally:
        sync_client.close()
        await async_client.aclose()


@pytest.mark.parametrize("content", [" \n\t", None])
def test_refusal_replaces_empty_content_and_keeps_metadata(content: str | None) -> None:
    payload = _completion(content="")
    payload["choices"][0]["message"]["content"] = content
    payload["choices"][0]["message"]["refusal"] = "This request is refused."
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    try:
        response = ChatEXAONE(api_key="synthetic-key", http_client=client).invoke("hello")
    finally:
        client.close()
    assert response.content == "This request is refused."
    assert response.additional_kwargs["refusal"] == "This request is refused."


def test_streamed_tool_call_fragments_form_a_langchain_tool_call() -> None:
    events = [
        {
            "id": "tool-stream",
            "model": "k-exaone_v2.5",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "weather", "arguments": '{"city":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "tool-stream",
            "model": "k-exaone_v2.5",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Seoul"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=body.encode())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        chunks = list(ChatEXAONE(api_key="synthetic-key", http_client=client).bind_tools([tool]).stream("Weather?"))
    finally:
        client.close()
    result = chunks[0]
    for chunk in chunks[1:]:
        result += chunk
    assert result.tool_calls == [{"name": "weather", "args": {"city": "Seoul"}, "id": "call-1", "type": "tool_call"}]
    assert requests[0]["tools"][0]["function"]["name"] == "weather"
    assert "tool_choice" not in requests[0]


def test_structured_output_uses_langchain_tool_contract() -> None:
    class Answer(BaseModel):
        result: str

    payload = _completion()
    payload["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "Answer", "arguments": '{"result":"four"}'},
            }
        ],
    }
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        response = (
            ChatEXAONE(api_key="synthetic-key", http_client=client).with_structured_output(Answer).invoke("2 + 2?")
        )
    finally:
        client.close()
    assert response == Answer(result="four")
    assert requests[0]["tools"][0]["function"]["name"] == "Answer"


def test_stream_preserves_reasoning_text_and_usage_chunks() -> None:
    events = [
        {
            "id": "stream-1",
            "model": "k-exaone_v2.5",
            "choices": [{"index": 0, "delta": {"reasoning_content": "reason "}, "finish_reason": None}],
        },
        {
            "id": "stream-1",
            "model": "k-exaone_v2.5",
            "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop", "matched_stop": None}],
        },
        {
            "id": "stream-1",
            "model": "k-exaone_v2.5",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body.encode())))
    try:
        chunks = list(ChatEXAONE(api_key="synthetic-key", http_client=client).stream("question"))
    finally:
        client.close()
    final = chunks[0]
    for chunk in chunks[1:]:
        final = final + chunk
    assert final.content == "answer"
    assert final.additional_kwargs["reasoning_content"] == "reason "
    assert any(chunk.additional_kwargs.get("reasoning_content") == "reason " for chunk in chunks)
    assert any(chunk.content == "answer" for chunk in chunks)
    assert any(chunk.usage_metadata == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3} for chunk in chunks)


def test_streamed_refusal_replaces_empty_text_and_keeps_refusal_metadata() -> None:
    event = {
        "id": "refusal-stream",
        "model": "k-exaone_v2.5",
        "choices": [
            {
                "index": 0,
                "delta": {"content": " \n", "refusal": "This request is refused."},
                "finish_reason": "stop",
            }
        ],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)))
    try:
        chunks = list(ChatEXAONE(api_key="synthetic-key", http_client=client).stream("hello"))
    finally:
        client.close()

    final = chunks[0]
    for chunk in chunks[1:]:
        final = final + chunk
    assert final.content == "This request is refused."
    assert final.additional_kwargs["refusal"] == "This request is refused."


@pytest.mark.asyncio
async def test_async_streamed_refusal_is_available_as_content_and_metadata() -> None:
    event = {
        "id": "async-refusal-stream",
        "model": "k-exaone_v2.5",
        "choices": [
            {
                "index": 0,
                "delta": {"refusal": "This request is refused."},
                "finish_reason": "stop",
            }
        ],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)))
    try:
        chunks = [
            chunk async for chunk in ChatEXAONE(api_key="synthetic-key", http_async_client=client).astream("hello")
        ]
    finally:
        await client.aclose()

    final = chunks[0]
    for chunk in chunks[1:]:
        final = final + chunk
    assert final.content == "This request is refused."
    assert final.additional_kwargs["refusal"] == "This request is refused."


def test_stream_rejects_multiple_choices_instead_of_merging_them() -> None:
    event = {
        "id": "multi-choice-stream",
        "model": "k-exaone_v2.5",
        "choices": [
            {"index": 0, "delta": {"content": "first"}},
            {"index": 1, "delta": {"content": "second"}},
        ],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        model = ChatEXAONE(api_key="synthetic-key", http_client=client, model_kwargs={"n": 2})
        with pytest.raises(ValueError, match="multiple streamed choices"):
            list(model.stream("hello"))
    finally:
        client.close()

    assert requests[0]["n"] == 2


@pytest.mark.asyncio
async def test_async_stream_rejects_multiple_choices_instead_of_merging_them() -> None:
    event = {
        "id": "async-multi-choice-stream",
        "model": "k-exaone_v2.5",
        "choices": [
            {"index": 0, "delta": {"content": "first"}},
            {"index": 1, "delta": {"content": "second"}},
        ],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)))
    try:
        model = ChatEXAONE(api_key="synthetic-key", http_async_client=client, model_kwargs={"n": 2})
        with pytest.raises(ValueError, match="multiple streamed choices"):
            _ = [chunk async for chunk in model.astream("hello")]
    finally:
        await client.aclose()


@pytest.mark.parametrize("event", [{}, {"choices": None}])
def test_stream_rejects_events_without_a_choices_array(event: dict[str, Any]) -> None:
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)))
    try:
        with pytest.raises(ValueError, match="choices must be a list"):
            list(ChatEXAONE(api_key="synthetic-key", http_client=client).stream("hello"))
    finally:
        client.close()


@pytest.mark.parametrize("content", [0, False])
def test_stream_rejects_falsy_non_text_content(content: Any) -> None:
    event = {
        "id": "invalid-content-stream",
        "model": "k-exaone_v2.5",
        "choices": [{"index": 0, "delta": {"content": content}}],
    }
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)))
    try:
        with pytest.raises(ValueError, match="stream delta content"):
            list(ChatEXAONE(api_key="synthetic-key", http_client=client).stream("hello"))
    finally:
        client.close()


def test_api_error_keeps_rate_headers_and_does_not_expose_key() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                429,
                headers={"retry-after-ms": "1000", "x-second-rate-limit-remaining": "1"},
                json={"error": {"message": "service busy synthetic-key", "type": "rate_limit_error", "code": "busy"}},
            )
        )
    )
    try:
        with pytest.raises(ExaoneAPIError) as error:
            ChatEXAONE(api_key="synthetic-key", http_client=client).invoke("hello")
    finally:
        client.close()
    assert error.value.status_code == 429
    assert error.value.headers["retry-after-ms"] == "1000"
    assert error.value.headers["x-second-rate-limit-remaining"] == "1"
    assert "synthetic-key" not in str(error.value)


def test_api_error_redacts_key_from_retained_error_fields() -> None:
    api_key = "synthetic-error-key"
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                422,
                json={
                    "error": {
                        "message": "Invalid request.",
                        "type": f"type-{api_key}",
                        "code": f"code-{api_key}",
                        "param": {"field": f"param-{api_key}"},
                    }
                },
            )
        )
    )
    try:
        with pytest.raises(ExaoneAPIError) as error:
            ChatEXAONE(api_key=api_key, http_client=client).invoke("hello")
    finally:
        client.close()

    assert error.value.error_type == "type-[REDACTED]"
    assert error.value.code == "code-[REDACTED]"
    assert error.value.param == {"field": "param-[REDACTED]"}


def test_api_errors_redact_percent_encoded_keys_from_messages_and_headers() -> None:
    api_key = "synthetic+reserved/key with spaces"
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                422,
                headers={"x-request-id": quote_plus(api_key, safe="")},
                json={
                    "error": {
                        "message": f"Rejected {quote(api_key, safe='').replace('%2F', '%2f')}",
                        "param": quote(api_key, safe="").replace("%2F", "%3F"),
                    }
                },
            )
        )
    )
    try:
        with pytest.raises(ExaoneAPIError) as error:
            ChatEXAONE(api_key=api_key, http_client=client).invoke("hello")
    finally:
        client.close()

    assert str(error.value) == "Rejected [REDACTED]"
    assert error.value.headers["x-request-id"] == "[REDACTED]"
    assert error.value.param == quote(api_key, safe="").replace("%2F", "%3F")


@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_malformed_success_payloads_do_not_retain_response_text(operation: str) -> None:
    api_key = "synthetic+response/key"
    encoded_key = quote_plus(api_key, safe="")
    malformed_json = f'{{"message":"{encoded_key}"'
    body = f"data: {malformed_json}\n\n" if operation in {"stream", "astream"} else malformed_json
    sync_client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body.encode())))
    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body.encode()))
    )
    model = ChatEXAONE(api_key=api_key, http_client=sync_client, http_async_client=async_client)
    try:
        with pytest.raises(ValueError) as caught:
            if operation == "invoke":
                model.invoke("hello")
            elif operation == "ainvoke":
                await model.ainvoke("hello")
            elif operation == "stream":
                list(model.stream("hello"))
            else:
                _ = [chunk async for chunk in model.astream("hello")]
    finally:
        sync_client.close()
        await async_client.aclose()

    error = caught.value
    assert str(error) == (
        "EXAONE returned malformed JSON in the event stream."
        if operation in {"stream", "astream"}
        else "EXAONE returned malformed JSON in the response body."
    )
    assert error.__cause__ is None
    assert error.__context__ is None
    assert api_key not in repr(error)
    assert encoded_key not in repr(error)


@pytest.mark.parametrize("operation", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_transport_errors_do_not_retain_authenticated_request_or_key(operation: str) -> None:
    api_key = "synthetic-transport-key"
    requests: list[httpx.Request] = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError(f"synthetic connection failure for {api_key}", request=request)

    sync_client = httpx.Client(transport=httpx.MockTransport(fail))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    model = ChatEXAONE(api_key=api_key, http_client=sync_client, http_async_client=async_client)
    try:
        if operation == "invoke":
            with pytest.raises(httpx.ConnectError) as caught:
                model.invoke("hello")
        elif operation == "ainvoke":
            with pytest.raises(httpx.ConnectError) as caught:
                await model.ainvoke("hello")
        elif operation == "stream":
            with pytest.raises(httpx.ConnectError) as caught:
                list(model.stream("hello"))
        else:
            with pytest.raises(httpx.ConnectError) as caught:
                _ = [chunk async for chunk in model.astream("hello")]
    finally:
        sync_client.close()
        await async_client.aclose()

    error = caught.value
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == f"Bearer {api_key}"
    assert str(error) == "synthetic connection failure for [REDACTED]"
    assert error.request.headers.get("authorization") is None
    assert error.request.content == b""
    assert api_key not in str(error.request.url)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert api_key not in repr(error)
    assert api_key not in repr(vars(error))


def test_http_status_transport_error_keeps_type_without_sensitive_request_or_response() -> None:
    api_key = "synthetic+status/key"
    requests: list[httpx.Request] = []

    def fail(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        request_with_secret_query = httpx.Request(
            request.method,
            request.url.copy_with(query=f"token={api_key}".encode()),
            headers=request.headers,
        )
        response = httpx.Response(
            502,
            headers={"x-sensitive": api_key},
            content=f"synthetic response body {api_key}".encode(),
            request=request_with_secret_query,
        )
        raise httpx.HTTPStatusError(
            f"synthetic status failure for {api_key}", request=request_with_secret_query, response=response
        )

    client = httpx.Client(transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            ChatEXAONE(api_key=api_key, http_client=client).invoke("hello")
    finally:
        client.close()

    error = caught.value
    assert len(requests) == 1
    assert str(error) == "synthetic status failure for [REDACTED]"
    assert error.request.headers.get("authorization") is None
    assert api_key not in str(error.request.url)
    assert "token=" not in str(error.request.url)
    assert error.response.status_code == 502
    assert error.response.content == b""
    assert error.response.headers.get("x-sensitive") is None
    assert error.response.request is not None
    assert error.response.request.headers.get("authorization") is None
    assert error.__cause__ is None
    assert error.__context__ is None
    assert api_key not in repr(vars(error))


def test_stream_api_error_reads_body_and_preserves_exaone_error_fields() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                422,
                headers={"retry-after-ms": "125"},
                json={"result_code": "INVALID_QUERY", "description": "Query cannot be empty."},
            )
        )
    )
    try:
        with pytest.raises(ExaoneAPIError) as error:
            list(ChatEXAONE(api_key="synthetic-key", http_client=client).stream("hello"))
    finally:
        client.close()
    assert error.value.status_code == 422
    assert error.value.code == "INVALID_QUERY"
    assert error.value.headers["retry-after-ms"] == "125"
    assert str(error.value) == "Query cannot be empty."


@pytest.mark.parametrize("operation", ["invoke", "ainvoke"])
@pytest.mark.asyncio
async def test_api_error_preserves_numeric_top_level_result_code(operation: str) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": {"message": "The request could not be processed."}, "result_code": 10042},
        )

    sync_client = httpx.Client(transport=httpx.MockTransport(respond))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatEXAONE(api_key="synthetic-key", http_client=sync_client, http_async_client=async_client)
    try:
        with pytest.raises(ExaoneAPIError) as error:
            if operation == "invoke":
                model.invoke("hello")
            else:
                await model.ainvoke("hello")
    finally:
        sync_client.close()
        await async_client.aclose()

    assert error.value.code == 10042
    assert type(error.value.code) is int


def test_retry_honors_longest_retry_after_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0
    delays: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0.05", "retry-after-ms": "100"},
                json={"error": {"message": "Try again."}},
            )
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr("langchain_exaone.chat_models.time.sleep", delays.append)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        result = ChatEXAONE(api_key="synthetic-key", http_client=client, max_retries=1).invoke("hello")
    finally:
        client.close()
    assert result.content == "4"
    assert requests == 2
    assert delays == [0.1]


@pytest.mark.asyncio
async def test_async_retry_honors_retry_after_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0
    delays: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(429, headers={"retry-after-ms": "125"}, json={"error": {"message": "Try again."}})
        return httpx.Response(200, json=_completion())

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("langchain_exaone.chat_models.asyncio.sleep", record_delay)
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        result = await ChatEXAONE(api_key="synthetic-key", http_async_client=client, max_retries=1).ainvoke("hello")
    finally:
        await client.aclose()

    assert result.content == "4"
    assert requests == 2
    assert delays == [0.125]


@pytest.mark.parametrize("operation", ["stream", "astream"])
@pytest.mark.asyncio
async def test_stream_retries_status_before_exposing_output(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0
    delays: list[float] = []
    event = {"id": "retry-stream", "model": "k-exaone_v2.5", "choices": [{"index": 0, "delta": {"content": "ready"}}]}
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(429, headers={"retry-after-ms": "0"}, json={"error": {"message": "Try again."}})
        return httpx.Response(200, content=body)

    def record_sync_delay(delay: float) -> None:
        delays.append(delay)

    async def record_async_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("langchain_exaone.chat_models.time.sleep", record_sync_delay)
    monkeypatch.setattr("langchain_exaone.chat_models.asyncio.sleep", record_async_delay)
    sync_client = httpx.Client(transport=httpx.MockTransport(respond))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatEXAONE(api_key="synthetic-key", http_client=sync_client, http_async_client=async_client, max_retries=1)
    try:
        if operation == "stream":
            chunks = list(model.stream("hello"))
        else:
            chunks = [chunk async for chunk in model.astream("hello")]
    finally:
        sync_client.close()
        await async_client.aclose()

    assert "".join(chunk.content for chunk in chunks) == "ready"
    assert requests == 2
    assert delays == [0.0]


@pytest.mark.parametrize("operation", ["stream", "astream"])
@pytest.mark.asyncio
async def test_stream_does_not_replay_a_tool_call_after_partial_output(operation: str) -> None:
    calls = 0
    event = {
        "id": "partial-tool-call",
        "model": "k-exaone_v2.5",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":'},
                        }
                    ]
                },
            }
        ],
    }
    frame = f"data: {json.dumps(event)}\n\n".encode()

    class FailingSyncStream(httpx.SyncByteStream):
        def __init__(self, request: httpx.Request) -> None:
            self.request = request

        def __iter__(self) -> Any:
            yield frame
            raise httpx.ReadError("synthetic mid-stream failure", request=self.request)

        def close(self) -> None:
            pass

    class FailingAsyncStream(httpx.AsyncByteStream):
        def __init__(self, request: httpx.Request) -> None:
            self.request = request

        async def __aiter__(self) -> Any:
            yield frame
            raise httpx.ReadError("synthetic mid-stream failure", request=self.request)

        async def aclose(self) -> None:
            pass

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if operation == "stream":
            return httpx.Response(200, stream=FailingSyncStream(request))
        return httpx.Response(200, stream=FailingAsyncStream(request))

    sync_client = httpx.Client(transport=httpx.MockTransport(respond))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatEXAONE(api_key="synthetic-key", http_client=sync_client, http_async_client=async_client)
    try:
        if operation == "stream":
            chunks = model.stream("weather?")
            first = next(chunks)
            with pytest.raises(httpx.ReadError, match="mid-stream failure"):
                next(chunks)
        else:
            chunks = model.astream("weather?")
            first = await anext(chunks)
            with pytest.raises(httpx.ReadError, match="mid-stream failure"):
                await anext(chunks)
    finally:
        sync_client.close()
        await async_client.aclose()

    assert first.tool_call_chunks[0]["args"] == '{"city":'
    assert calls == 1


def test_request_timeout_retries_before_response(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0
    delays: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(408, json={"error": {"message": "Request timed out."}})
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr("langchain_exaone.chat_models.time.sleep", delays.append)
    monkeypatch.setattr("langchain_exaone.chat_models.random.uniform", lambda low, high: low)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        result = ChatEXAONE(api_key="synthetic-key", http_client=client, max_retries=1).invoke("hello")
    finally:
        client.close()

    assert result.content == "4"
    assert requests == 2
    assert delays == [1.0]


def test_retry_without_hints_waits_at_least_one_second(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0
    delays: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(429, json={"error": {"message": "Try again."}})
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr("langchain_exaone.chat_models.time.sleep", delays.append)
    monkeypatch.setattr("langchain_exaone.chat_models.random.uniform", lambda low, high: low)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        result = ChatEXAONE(api_key="synthetic-key", http_client=client, max_retries=1).invoke("hello")
    finally:
        client.close()
    assert result.content == "4"
    assert requests == 2
    assert delays == [1.0]


def test_retry_without_hints_is_skipped_if_delay_cap_is_below_one_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0
    delays: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(429, json={"error": {"message": "Try again."}})

    monkeypatch.setattr("langchain_exaone.chat_models.time.sleep", delays.append)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(ExaoneAPIError):
            ChatEXAONE(api_key="synthetic-key", http_client=client, max_retries=1, max_retry_delay=0.99).invoke("hello")
    finally:
        client.close()
    assert requests == 1
    assert delays == []


def test_retry_does_not_exceed_retry_after_delay_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = 0
    delays: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            429,
            headers={"retry-after-ms": "1000"},
            json={"error": {"message": "Try again later."}},
        )

    monkeypatch.setattr("langchain_exaone.chat_models.time.sleep", delays.append)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(ExaoneAPIError):
            ChatEXAONE(api_key="synthetic-key", http_client=client, max_retries=1, max_retry_delay=0.5).invoke("hello")
    finally:
        client.close()
    assert requests == 1
    assert delays == []


@pytest.mark.asyncio
async def test_async_invoke_and_stream_use_async_transport() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["stream"]:
            event = {
                "id": "async-stream",
                "model": "k-exaone_v2.5",
                "choices": [{"index": 0, "delta": {"content": "async"}, "finish_reason": "stop"}],
            }
            content = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, content=content.encode())
        return httpx.Response(200, json=_completion(content="async", reasoning="async reason"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatEXAONE(api_key="synthetic-key", http_async_client=client)
    try:
        response = await model.ainvoke("hello")
        chunks = [chunk async for chunk in model.astream("hello")]
    finally:
        await client.aclose()
    assert response.content == "async"
    assert response.additional_kwargs["reasoning_content"] == "async reason"
    assert any(chunk.text == "async" for chunk in chunks)


@pytest.mark.asyncio
async def test_async_stream_api_error_reads_response_body() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                422,
                json={"result_code": "INVALID_QUERY", "description": "Query cannot be empty."},
            )
        )
    )
    try:
        with pytest.raises(ExaoneAPIError) as error:
            async for _ in ChatEXAONE(api_key="synthetic-key", http_async_client=client).astream("hello"):
                pass
    finally:
        await client.aclose()
    assert error.value.status_code == 422
    assert error.value.code == "INVALID_QUERY"


def test_non_text_message_blocks_are_rejected_before_request() -> None:
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_completion())

    client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(TypeError, match="does not support message block type"):
            ChatEXAONE(api_key="synthetic-key", http_client=client).invoke(
                [HumanMessage(content=[{"type": "image", "url": "https://example.test/image.png"}])]
            )
    finally:
        client.close()
    assert calls == 0
