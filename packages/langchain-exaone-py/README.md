# `langchain-exaone`

This community-maintained package connects LangChain chat models to the K-EXAONE 2.5 Chat Completions API. It is maintained independently of LG AI Research and LangChain, and is not affiliated with or endorsed by either organization.

It implements LangChain's `BaseChatModel` interface and supports synchronous and asynchronous calls, streaming, tool calls, structured-output binding, usage metadata, and EXAONE reasoning history.

## API key

Open the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview), select **Get an API Key**, and sign in or create an EXAONE ID to follow the platform's key-issuance instructions. Each user supplies their own key. The [illustrated setup guide](https://github.com/ykoh42/exaone-integrations/blob/main/guides/getting-started.md#get-an-exaone-api-key) shows the entry screens and includes Windows PowerShell instructions.

In the terminal where you will run your program, set the issued key:

```bash
export EXAONE_API_KEY='your-api-key'
```

## Install

The package requires Python 3.10 or newer and `langchain-core` 1.3.3 or newer within the 1.x line. Install it from PyPI:

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

LangChain provider catalogs use vendor names, while chat-model listings identify classes. `ChatEXAONE` follows the common class-name pattern for a `BaseChatModel`; `Chat` is not part of the package or model ID.

The package does not load `.env` files. Use your application's secret-management setup to place the API key in the process environment, or pass `api_key` directly.

## Why use this integration?

A standard OpenAI chat model configured for EXAONE's endpoint can handle basic chat. `ChatEXAONE` also carries `reasoning_content` on assistant messages and sends it on later turns. It defaults to `preserve_thinking=True` and exposes known model facts through LangChain's standard profile. Usage and response metadata appear on LangChain messages, and raised API errors include diagnostic fields.

Retries are bounded, follow server delay hints, and stop before replaying a stream that has started. Structured-output binding and tool execution follow LangChain's contracts. Other adapters may provide similar behavior through configuration or extensions; this package keeps the EXAONE mapping in one tested integration. See the [project comparison](https://github.com/ykoh42/exaone-integrations#why-use-a-dedicated-provider) for scope and trade-offs.

## Serialization

LangChain's deserializer allows only registered integration classes by default. Since this package is not registered in `langchain-core`, `ChatEXAONE` does not opt into object serialization or modify LangChain's global registry. Create the model from application configuration when the process starts. Its API key uses `SecretStr` and is omitted from the model representation.

## Model profile

`ChatEXAONE().profile` provides LangChain's standard model profile for `k-exaone_v2.5`: a 524,288-token context budget, text input and output, reasoning, tool calling and choice. Image, audio, video, and PDF inputs and non-text outputs are unsupported. The combined input and output, including reasoning, must fit within the context budget; no independent `max_output_tokens` profile value is inferred.

The profile reports reasoning levels (`low`, `medium`, `high`, `max`), the API default (`max`), streamed tool-call support, and temperature support. The endpoint reference currently lists `low`, `high`, and `max`; the [Agent guide](https://docs.exaone.ai/ko/docs/agent-start) also uses `medium`.

Older `langchain-core` versions can warn that some newer profile fields are unrecognized. The profile still retains those fields; upgrading `langchain-core` removes the warning when the newer fields are supported.

LangChain and Deep Agents can use this profile for context management and structured-output strategy selection. An unknown model ID has no inferred profile. Pass `profile={...}` to replace the defaults using LangChain's public constructor option; `profile={}` leaves capabilities unspecified. Profiles are application metadata and are not sent as API parameters.

The default profile leaves `structured_output` unspecified because native JSON Schema output combined with tool calls in one request has not been established as compatible with LangChain's provider strategy. Agents use LangChain's tool-based strategy by default. To request native output without tools, select `ProviderStrategy` explicitly. This choice does not change the provider's request parameters or structured-output methods.

See [LangChain model profiles](https://docs.langchain.com/oss/python/langchain/models#model-profiles) and the [EXAONE API specification](https://docs.exaone.ai/ko/docs/k-exaone-2.5).

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

Each candidate in a non-streaming response carries the same request-level usage. `generate()` and `agenerate()` expose totals once per prompt in `llm_output["token_usage"]`, including reasoning tokens when present.

LangChain's `output_version="v1"` standard content blocks can also be retained in conversation history, including reasoning and tool calls.

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

When a returned EXAONE message is reused in history, edits to its LangChain `tool_calls` determine the next request. Provider-specific fields from matching original calls are retained.

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

The API always enables reasoning. The provider stores returned `reasoning_content` on assistant messages and sends it with later turns. By default, `preserve_thinking=True` also sets EXAONE's recommended `chat_template_kwargs.preserve_thinking` option. Keep assistant messages intact when continuing a conversation.

`max_tokens` is capped at the documented 524,288-token context limit. The provider cannot count an arbitrary prompt locally, so the API still validates the combined input and output token budget.

The provider retries HTTP 408, 429, 500, 502, 503, and 504 errors up to two times by default; set `max_retries=0` to disable retries. It honors the longer valid `Retry-After` or `Retry-After-Ms` delay. If neither header is present, the first retry waits at least one second to respect the API's per-second rate limit; a `max_retry_delay` below one second disables that retry. It does not retry after streaming output has started.

The EXAONE API is text-only. Image, audio, and other non-text input blocks are rejected explicitly.

See the [K-EXAONE 2.5 API reference](https://docs.exaone.ai/ko/docs/k-exaone-2.5) for the model's request parameters and limits.

## Examples

See the [examples](examples/) for LangChain agents, streaming chat, structured output, LangGraph tool loops, and Deep Agents. The package does not make API requests during import or construction.

## Development

```bash
uv sync --group test --group lint
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Unit tests use synthetic responses and do not require API credentials.
