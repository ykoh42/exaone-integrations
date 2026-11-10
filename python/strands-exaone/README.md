# strands-exaone for Python

Use the EXAONE Chat Completions API as a model provider for [Strands Agents](https://strandsagents.com/). This community-maintained package is not affiliated with LG AI Research or the Strands Agents project.

## Quick start

The package requires Python 3.10 or newer and `strands-agents` 1.64.x. It has not been published to PyPI. From the root of a checkout, install it in your virtual environment:

```bash
python -m pip install -e ./python/strands-exaone
export EXAONE_API_KEY='your-api-key'
```

```python
from strands import Agent
from strands_exaone import ExaoneModel

model = ExaoneModel(preserve_thinking=True)
agent = Agent(model=model)
response = agent("Explain what an AI agent is in one sentence.")
print(response)
```

The provider reads `EXAONE_API_KEY` when the model is constructed. You can instead pass `api_key=...`. The package does not load `.env` files automatically.

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `model_id` | EXAONE model ID. | `k-exaone_v2.5` |
| `base_url` | API base URL, including `/v1`. | `https://api.exaone.ai/v1` |
| `stream` | Request a streaming response; `False` adapts a non-streaming response to Strands events. | `True` |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty` | Generation parameters. | Service defaults when omitted |
| `reasoning_effort` | EXAONE reasoning setting. | Service default |
| `preserve_thinking`, `enable_thinking` | EXAONE `chat_template_kwargs` for previous reasoning and reasoning generation. | Service defaults |
| `include_usage` | Set `stream_options.include_usage` on streaming requests. | Omitted |
| `response_format` | Response format for ordinary agent calls. | Omitted |
| `timeout` | HTTPX timeout in seconds for each attempt. | `120` |
| `context_window_limit` | Context window reported to Strands, in tokens. | `524288` |
| `max_retries`, `base_retry_delay`, `max_retry_delay` | Optional retry policy. | `0` retries; `0.5` and `30` seconds |
| `metadata_callback` | Callback for successful request metadata. | None |
| `http_client` | Caller-owned `httpx.AsyncClient` for custom transport or headers. | Provider-managed client |
| `params` | Additional Chat Completions parameters. | None |

Explicit options take precedence over matching `params` values. The provider manages `model`, `messages`, `stream`, `stream_options`, `tools`, `tool_choice`, and `response_format`; do not set them in `params`. Strands represents one completion, so `n` must be 1. Supported generation settings can vary by service version; see the [EXAONE API documentation](https://docs.exaone.ai/en/docs/k-exaone-2.5).

## Reasoning and tools

Reasoning is stored separately from the final answer and sent back as `reasoning_content` when it is present in the Strands conversation. Set `preserve_thinking=True` to request preservation of previous reasoning. `enable_thinking` is a separate setting; its observed behavior has differed from the documentation, so verify it for your use case.

In one live request with `enable_thinking=False`, the API returned `reasoning_tokens=0` and no separate `reasoning_content`, but the response `content` contained a `<think>` or `</think>` tag. A separate request with `enable_thinking=True` returned both reasoning tokens and separate reasoning content. These individual observations do not establish a general rule: `False` does not guarantee that reasoning will be absent from the visible response.

```python
from strands import Agent, tool
from strands_exaone import ExaoneModel


@tool
def get_number() -> int:
    """Return the example number."""
    return 7


agent = Agent(model=ExaoneModel(preserve_thinking=True), tools=[get_number])
agent("Use the tool to get the number, then add 3.")
```

The provider assembles streamed tool-call arguments before exposing calls to Strands. It accepts text, reasoning, tool calls, and text/JSON tool results; unsupported content types are rejected.

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

The provider requests JSON Schema output without streaming and validates the result against the Pydantic model. For ordinary model calls, `response_format={"type": "json_object"}` requests JSON object mode. A structured-output call supplies its own schema.

EXAONE returned `finish_reason=abort` on some observed structured-output requests; the cause is not established. The provider reports an error instead of treating an aborted result as valid. Schema support may vary with the service and schema.

## Retries

Retries are disabled by default. Set `max_retries=2` to allow two additional attempts for connection failures before a response, or HTTP 429, 500, 502, 503, and 504. The provider never retries after a successful response starts or replays a partially delivered stream.

The provider honors `retry-after` in seconds or HTTP-date form and `retry-after-ms`, using the longer valid delay when both are present. If the server's delay exceeds `max_retry_delay`, it returns the original error rather than retrying early. Without a usable header, it uses capped exponential backoff with jitter. Cancellation interrupts requests and retry waits. `timeout` applies per HTTPX attempt, not to the overall operation including retries and waits. Strands or application-level retries are separate.

## Metadata and errors

```python
from strands_exaone import ExaoneModel


def on_metadata(metadata: dict) -> None:
    print(metadata["request_id"], metadata["usage"])


model = ExaoneModel(include_usage=True, metadata_callback=on_metadata)
```

On successful completion, the callback receives `status_code`, `request_id`, selected response `headers`, `usage`, and `response`. Raw usage can include `reasoning_tokens`. Available response metadata includes top-level IDs, model, service and weight metadata; choice-level `finish_reason`, `matched_stop`, and `logprobs`; and selected message fields. Streaming logprob and annotation arrays are accumulated in order; refusal fragments are joined, while audio and legacy function-call fragments remain in `message_deltas`. Fields are present only when supplied by the service. The callback may be synchronous or asynchronous and is not called for failed or interrupted streams.

**Metadata can contain user or generated content**, including logprob tokens, refusal text, audio, and function arguments. It is not size-limited and can grow with the response. API keys are redacted, but metadata should only be logged or shared under an appropriate data policy. The provider does not copy ordinary assistant content, reasoning, or tool calls into response metadata.

`ExaoneThrottledError` is a Strands `ModelThrottledException` and preserves request ID, rate-limit headers, and a sanitized error body. Other API errors expose details through `ExaoneError`; recognized context-window errors also implement Strands' `ContextWindowOverflowException`. An injected `http_client` remains caller-owned.

## License

[MIT](LICENSE)
