# strands-exaone

A Python model provider connecting [Strands Agents](https://strandsagents.com/) to the EXAONE Chat Completions API. It preserves EXAONE reasoning alongside assistant messages, including across tool calls, and translates streamed text and tool calls into Strands events.

This package targets `strands-agents` 1.64.x and Python 3.10 or newer. The package is not yet published; from this repository, install it with:

```sh
pip install -e ./python/strands-exaone
export EXAONE_API_KEY='your-api-key'
```

The environment variable must be set by your shell or application; the package does not load `.env` files. You can alternatively pass `api_key=` to `ExaoneModel`.

## Agent and tools

```python
from strands import Agent, tool
from strands_exaone import ExaoneModel


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"The weather in {city} is sunny."


model = ExaoneModel(
    model_id="k-exaone_v2.5",
    preserve_thinking=True,
    enable_thinking=True,
    max_tokens=2048,
)
agent = Agent(model=model, tools=[get_weather])
agent("What is the weather in Seoul?")
```

Strands retains the returned reasoning blocks in the conversation. When sending that history back to EXAONE, the provider maps them to `reasoning_content` and keeps tool requests and results linked. `preserve_thinking` and `enable_thinking` are distinct EXAONE `chat_template_kwargs`; set them explicitly according to the behavior needed. Refer to the EXAONE documentation for model-specific support and defaults.

## Structured output

```python
from pydantic import BaseModel
from strands import Agent
from strands_exaone import ExaoneModel


class Answer(BaseModel):
    city: str
    temperature_c: int


agent = Agent(model=ExaoneModel())
answer = agent.structured_output(Answer, "Return Seoul and a temperature of 20 degrees Celsius.")
print(answer.city, answer.temperature_c)
```

The provider sends a JSON Schema and validates the response against the Pydantic model. Structured output uses a non-streaming request and requires a normally completed response. For JSON object mode with ordinary agent calls, configure `ExaoneModel(response_format={"type": "json_object"})`. A structured-output call supplies its own schema and takes precedence over that setting.

## Configuration

The constructor supports `model_id`, `base_url`, `stream`, `max_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty`, `reasoning_effort`, `preserve_thinking`, `enable_thinking`, `include_usage`, `response_format`, `timeout`, and `context_window_limit`. Defaults are left to the service where possible. `stream=False` requests a non-streaming completion while still presenting Strands events to the agent. `max_tokens=0` and boolean `False` are passed through without being discarded.

Use `params` for additional API request fields, such as `params={"parallel_tool_calls": False}` or additional `chat_template_kwargs`. Explicit constructor settings take precedence over matching values in `params`. Fields managed by Strands, including `messages`, `model`, `tools`, `tool_choice`, `stream`, `stream_options`, and `response_format`, cannot be supplied in `params`. Strands consumes one completion, so `n` must be 1. EXAONE model and endpoint capabilities can differ; consult the service documentation before enabling an option.

Only text content, reasoning, tool calls, and text/JSON tool results are supported. Unsupported content types raise an error instead of being silently removed. The provider does not add automatic HTTP retries.

## Per-request metadata and errors

```python
from strands_exaone import ExaoneModel


def on_metadata(metadata: dict) -> None:
    print(metadata["request_id"], metadata["usage"])


model = ExaoneModel(include_usage=True, metadata_callback=on_metadata)
```

On successful completion the callback runs once per request with `status_code`, `request_id`, selected response `headers`, and raw `usage` when provided. This includes EXAONE-specific reasoning token usage when returned by the service. Concurrent requests each produce their own callback. The callback can be synchronous or asynchronous; callback exceptions propagate to the caller. It is not called on HTTP errors or interrupted streams. The API key is redacted from callback metadata and exposed errors. Strands' ordinary usage event contains the standard input, output, and total token counts.

`ExaoneThrottledError` is also a Strands `ModelThrottledException`; its `headers` preserve supported rate-limit and retry information. Other service errors expose `status_code`, `code`, `param`, `request_id`, `headers`, and the sanitized response body on `ExaoneError`. Context-window errors recognized from the response are also Strands `ContextWindowOverflowException` instances. Network errors from HTTPX may propagate separately. The provider does not retry 429 responses.

Advanced users may pass an `httpx.AsyncClient` as `http_client`, for example to configure a proxy or default request headers. The caller owns and closes that client; internally created clients and response streams are closed by the provider. Strands cancellation signals abort in-flight reads and close the response.
