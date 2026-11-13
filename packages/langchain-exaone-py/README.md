# `langchain-exaone`

Community-maintained LangChain chat model integration for the EXAONE API.

This package connects LangChain applications to the K-EXAONE 2.5 Chat Completions API. It implements LangChain's `BaseChatModel` interface and supports synchronous and asynchronous calls, streaming, tool calls, structured-output binding, usage metadata, and EXAONE reasoning history.

This project is not affiliated with or endorsed by LG AI Research or LangChain.

## Install

Install the package from PyPI:

```bash
pip install -U langchain-exaone
```

To install from a checkout for development:

```bash
pip install ./packages/langchain-exaone-py
```

Set `EXAONE_API_KEY` in the environment, then create the model:

```python
from langchain_exaone import ChatEXAONE

model = ChatEXAONE()
message = model.invoke("안녕하세요. K-EXAONE을 소개해 주세요.")
print(message.text)
```

The package does not load `.env` files. Use your application's secret-management setup to place the API key in the process environment, or pass `api_key` directly.

## Serialization

`ChatEXAONE` does not opt into LangChain object serialization. LangChain's deserializer only allows registered integration classes by default, and this standalone package is not registered in `langchain-core`. The provider does not modify LangChain's global serialization registry. Create the model from application configuration when starting your process; its API key is held in a `SecretStr` and omitted from the model representation.

## Reasoning history

EXAONE returns reasoning separately from final content. The provider exposes it on `AIMessage.additional_kwargs["reasoning_content"]` and includes it when an assistant message is sent back in later turns. Keep the original `AIMessage` in conversation history so reasoning and tool calls remain attached to their assistant turn.

```python
from langchain_core.messages import HumanMessage
from langchain_exaone import ChatEXAONE

model = ChatEXAONE(reasoning_effort="low")
history = [HumanMessage("2 + 2는 얼마야?")]
answer = model.invoke(history)
history.append(answer)
history.append(HumanMessage("방금 답을 확인해줘."))
follow_up = model.invoke(history)
```

Use `model.astream(...)` for asynchronous streaming. Streaming chunks retain reasoning in `additional_kwargs`, while text remains in the normal message content field.

LangChain's streaming interface represents one completion per run. If the API returns multiple choices, such as when `n` is greater than `1` in `model_kwargs`, `stream()` and `astream()` raise `ValueError` rather than merging separate candidates. Use `generate()` when you need multiple candidates.

## Tools and structured output

```python
from langchain_core.tools import tool
from langchain_exaone import ChatEXAONE


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}."


model = ChatEXAONE().bind_tools([get_weather], tool_choice="auto")
response = model.invoke("서울 날씨를 알려줘.")
```

`with_structured_output()` uses LangChain's standard tool-based strategy. You can pass additional Chat Completions fields with `model_kwargs` or per-call keyword arguments. Their acceptance depends on the EXAONE endpoint.

## Configuration

| Option | Default | Description |
| --- | --- | --- |
| `model` | `k-exaone_v2.5` | EXAONE model ID |
| `api_key` | `EXAONE_API_KEY` | API credential |
| `base_url` | `https://api.exaone.ai/v1` | API base URL. Remote endpoints must use HTTPS; HTTP is allowed for loopback test servers. |
| `temperature` | `1.0` | Recommended EXAONE sampling value |
| `top_p` | `0.95` | Recommended EXAONE sampling value |
| `top_k` | `-1` | Disable top-k filtering, as in the API default |
| `presence_penalty` | `0.0` | Recommended EXAONE sampling value |
| `max_tokens` | API default | Maximum generated tokens, including reasoning |
| `reasoning_effort` | API default | `low`, `medium`, `high`, or `max` |
| `preserve_thinking` | `true` | Keep earlier assistant reasoning in multi-turn agent conversations |
| `chat_template_kwargs` | `{}` | Additional chat-template options; `preserve_thinking` is set from its dedicated option |
| `max_retries` | `2` | Retries for transient API errors before output is returned; set to `0` to disable |
| `timeout` | `None` | No provider-imposed HTTP timeout; pass a positive number to set one |

The API always enables reasoning. The provider preserves the returned `reasoning_content` on assistant messages and sends it with later turns; `preserve_thinking=True` also enables EXAONE's recommended `chat_template_kwargs.preserve_thinking` behavior. Keep returned assistant messages intact when continuing a conversation.

`max_tokens` is capped at the documented 524,288-token context limit. The provider cannot count an arbitrary prompt locally, so the API still validates the combined input and output token budget.

The provider retries HTTP 408, 429, 500, 502, 503, and 504 errors up to two times by default; set `max_retries=0` to disable retries. It honors the longer valid `Retry-After` or `Retry-After-Ms` delay. If neither header is present, the first retry waits at least one second to respect the API's per-second rate limit; a `max_retry_delay` below one second disables that retry. It does not retry after streaming output has started.

The EXAONE API is text-only. Image, audio, and other non-text input blocks are rejected explicitly.

See the [K-EXAONE 2.5 API reference](https://docs.exaone.ai/ko/docs/k-exaone-2.5) for the model's request parameters and limits.

## Examples

See [`examples/basic.py`](examples/basic.py) for a runnable example. The package does not make API requests during import or construction.

## Development

```bash
uv sync --group test --group lint
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Unit tests use synthetic responses and do not require API credentials.
