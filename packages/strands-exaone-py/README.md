# strands-exaone for Python

Use the EXAONE Chat Completions API as a model provider for [Strands Agents](https://strandsagents.com/). This community-maintained package is not affiliated with LG AI Research or the Strands Agents project.

Users are responsible for reviewing and complying with the terms that apply to their EXAONE API use.

## Quick start

The Python distribution requires Python 3.10 or newer and `strands-agents` 1.64.x. Install it from PyPI:

```bash
python -m pip install -U strands-exaone
export EXAONE_API_KEY='your-api-key'
```

```python
from strands import Agent
from strands_exaone import ExaoneModel, ExaoneRetryStrategy

model = ExaoneModel(preserve_thinking=True)
agent = Agent(model=model, retry_strategy=ExaoneRetryStrategy(), callback_handler=None)
response = agent("Explain what an AI agent is in one sentence.")
print(response)
```

The provider reads `EXAONE_API_KEY` when the model is constructed. You can instead pass `api_key=...`. The package does not load `.env` files automatically.

The configured endpoint and any custom HTTP client receive your API key and conversation. Use a trusted endpoint and transport, and use HTTPS except for controlled local testing.

The callback handler determines whether intermediate reasoning and response text are printed. This example disables the default callback and prints only the final response; reasoning remains in conversation history for later turns.

For SDK tool use, streaming, structured output, Harness, and CLI walkthroughs, see the [examples](examples/).

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `model_id` | EXAONE model ID. | `k-exaone_v2.5` |
| `base_url` | API base URL, including `/v1`. Remote endpoints must use HTTPS; HTTP is allowed for loopback test servers. | `https://api.exaone.ai/v1` |
| `stream` | Request a streaming response; `False` adapts a non-streaming response to Strands events. | `True` |
| `temperature`, `top_p` | Sampling parameters. | `1.0`, `0.95` |
| `max_tokens`, `top_k`, `presence_penalty` | Generation parameters. | Service defaults when omitted |
| `reasoning_effort` | EXAONE reasoning setting. | Service default |
| `preserve_thinking` | Ask EXAONE to preserve previous reasoning in model input. | `True` |
| `enable_thinking` | EXAONE option for reasoning generation. | Service default |
| `include_usage` | Set `stream_options.include_usage` on streaming requests. | `True` |
| `response_format` | Response format for ordinary agent calls. | Omitted |
| `timeout` | Deadline for sending the request, retry waits, reading the response, asynchronous metadata callbacks, and time the caller pauses between yielded stream events. A custom transport that stalls during cleanup can finish after the deadline. | No provider-imposed timeout |
| `context_window_limit` | Context window reported to Strands. Set a positive token count. | `524288` |
| `max_retries`, `base_retry_delay`, `max_retry_delay` | Optional retry policy. | `0` retries; `1` and `30` seconds |
| `metadata_callback` | Callback for successful request metadata. | None |
| `http_client` | Caller-owned `httpx.AsyncClient` for custom transport or headers. | Provider-managed client |
| `params` | Additional Chat Completions parameters. | None |

Explicit options take precedence over matching `params` values, which take precedence over provider defaults. The provider manages `model`, `messages`, `stream`, `tools`, and `tool_choice`; do not set them in `params`. For streaming requests, `params["stream_options"]` can set `include_usage` and other service-supported options; the named `include_usage` option takes precedence. `stream_options` is omitted from non-streaming requests. `params["response_format"]` can set a format for ordinary model calls; the provider's `response_format` option takes precedence. Strands represents one completion, so `n` must be 1. Supported generation settings can vary by service version; see the [EXAONE API documentation](https://docs.exaone.ai/en/docs/k-exaone-2.5).

`model_info.limits.context_window`, when set, takes precedence over `context_window_limit`, following the Strands SDK's model info merge contract.

First-class generation options are validated before storage or a request. Values supplied through `params` remain an extension point and are passed through unless the provider owns the field; EXAONE may reject unsupported values.

`update_config(...)` applies to requests that have not started. Each in-flight request keeps the configuration it started with, including its retry policy.

## Reasoning and tools

Reasoning is stored separately from the final answer and sent back as reasoning_content when the Strands conversation includes it. The provider requests EXAONE's agent-use settings by default: temperature=1.0, top_p=0.95, and preserve_thinking=True. enable_thinking is a separate request option; check the returned fields and content when your application needs specific reasoning-handling behavior.

Strands can summarize older conversation history during automatic context management. The provider forwards the reasoning that remains in the history it receives; it cannot restore content that was summarized or removed. Token counts used for proactive context management are estimates, not exact EXAONE token counts.

Long-running agents should monitor memory and trace growth; behavior depends on the SDK version and workload.

The provider requests streaming usage by default (include_usage=True) so Strands can use token counts for context management. EXAONE may omit usage, and downstream context management is controlled by the framework.

The returned response fields are authoritative. Do not assume the request setting alone guarantees that reasoning is absent from visible content.

When EXAONE returns a refusal without non-whitespace ordinary response content, the provider exposes that refusal as a Strands text block and preserves any whitespace content. Strands has no separate refusal event; the refusal remains available in response metadata as well. System prompt blocks follow the Strands OpenAI adapter contract: text blocks are sent as separate messages and blocks without text, such as cache points, are omitted.

For direct `structured_output()` calls, a refusal without JSON raises `ExaoneError`; its sanitized `body` includes the refusal text. Strands tool execution uses the Chat Completions `tool_calls` field. Legacy `function_call` response fields are retained in metadata but do not trigger a Strands tool.

```python
from strands import Agent, tool
from strands_exaone import ExaoneModel, ExaoneRetryStrategy


@tool
def get_number() -> int:
    """Return the example number."""
    return 7


agent = Agent(
    model=ExaoneModel(preserve_thinking=True),
    tools=[get_number],
    retry_strategy=ExaoneRetryStrategy(),
    callback_handler=None,
)
print(agent("Use the tool to get the number, then add 3."))
```

The provider assembles streamed tool-call arguments before exposing calls to Strands. If a completed tool call contains a JSON object with only closing braces or brackets missing, or a trailing comma, the provider completes that syntax without changing argument values. It rejects unterminated strings, missing values, and other malformed arguments. The Harness still validates arguments against each tool's schema. The provider accepts text, reasoning, tool calls, and text/JSON tool results. Unsupported content types and reasoning signatures present in the history received by the provider are rejected.

Because Chat Completions tool messages have no separate status field, tool results marked as errors are sent with an `[ERROR] ` prefix. Successful results are sent without a prefix.

When Strands supplies multiple system text blocks, the provider sends them as separate system messages.

A user message containing both text and tool results is split into tool messages followed by a user text message, matching Strands' OpenAI Chat Completions adapter so tool responses remain adjacent to their assistant tool calls.

When using Strands Harness, configure its `read` tool to return a text description for image and document files:

```python
from strands_harness import create_harness
from strands_exaone import ExaoneRetryStrategy

agent = create_harness(
    model=ExaoneModel(),
    retry_strategy=ExaoneRetryStrategy(),
    builtin_tools={"*": False, "read": {"media": False}},
    builtin_plugins=[],
    session=False,
    memory=False,
    skills=False,
    caching=False,
)
```

This setting avoids sending unsupported media blocks; it does not give the model image or document understanding. The `read` tool can access files permitted by the agent's sandbox, and its results are sent to the model endpoint. Use an isolated sandbox or an approval mechanism to restrict access to files you intend to share.

Tools that use their own network transport may have TLS trust settings separate from the EXAONE provider's client.

After an interrupted tool call, inspect external state before repeating a task that could have side effects.

Use separate session IDs for concurrently active agents or serialize access to shared file sessions.

Cancellation may not stop work already in progress. Use suitable tool timeouts and check external effects before retrying.

For side-effecting tools, ensure that cancelling an invocation also invalidates any pending approval.

Configure callbacks for child agents as well as the parent when generated reasoning or tool output must not be printed.

Text is forwarded without trimming, including leading and trailing whitespace. Strands may remove a text block containing only whitespace during a tool turn, so an exact round trip is not guaranteed for such blocks.

## Structured output

```python
from pydantic import BaseModel
from strands import Agent
from strands_exaone import ExaoneModel


class Answer(BaseModel):
    city: str
    temperature_c: int


agent = Agent(model=ExaoneModel(), callback_handler=None)
result = agent("Return Seoul and a temperature of 20 degrees Celsius.", structured_output_model=Answer)
answer = result.structured_output
print(answer.city, answer.temperature_c)
```

In Strands SDK 1.64, `structured_output_model` uses the SDK's structured-output tool and may require additional model calls. A configured `response_format` is also sent on those tool-based calls; EXAONE's support for combining them has not been verified. The provider's direct async `model.structured_output(...)` method instead requests its own JSON Schema without streaming and validates the result against the Pydantic model. For ordinary model calls, `response_format={"type": "json_object"}` requests JSON object mode.

When using recursive schemas, verify that the selected framework path preserves schema references and that EXAONE supports the schema.

Structured-output calls reject aborted responses rather than treating them as valid schema results.

## Retries

`ExaoneModel`'s own HTTP retry loop is disabled by default. For Agent and Harness use, configure `ExaoneRetryStrategy` as the Strands retry strategy; it replaces the SDK default and honors EXAONE's `Retry-After` and `Retry-After-Ms` headers. It defaults to three total attempts, a one-second initial backoff, and a 30-second delay cap. If a server hint exceeds the cap, it surfaces the error instead of retrying early. Without a usable hint, it uses bounded exponential backoff with jitter.

Strands owns Agent-level retry selection, so a model provider cannot silently replace the SDK's strategy. `ExaoneRetryStrategy` is opt-in: pass an instance to `Agent` or `create_harness` to replace the generic policy with EXAONE-aware waits. The quick start and Harness example show this explicit configuration.

Strands' default retry strategy does not inspect EXAONE headers. Keeping it alongside the provider's own retry loop can multiply attempts, so leave `max_retries` at its default of zero when using `ExaoneRetryStrategy`. The strategy retries EXAONE transient HTTP errors, safe connection failures, and Strands throttling failures before the provider exposes output. It does not retry tool arguments that remain invalid after syntax-only completion or other model response errors.

`ExaoneModel`'s optional `max_retries` setting remains available for direct model calls that do not use an Agent retry strategy. It retries connection failures before a response and retryable HTTP responses. The provider never replays a partially delivered stream. `timeout` bounds request work, provider retry waits, response reads, asynchronous metadata callbacks, and time the consumer pauses between yielded stream events. Agent-level `ExaoneRetryStrategy` waits are outside the model timeout. It cannot interrupt a synchronous callback while that function is running; an uncooperative custom transport can also delay response cleanup beyond the deadline. A caller-owned `http_client` retains its own transport timeouts.

When stopping iteration early, explicitly close the async generator so its response can be released promptly. This applies to both `model.stream(...)` and `agent.stream_async(...)`:

```python
from contextlib import aclosing

from strands_exaone import ExaoneModel


async def first_event():
    model = ExaoneModel()
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    async with aclosing(model.stream(messages)) as stream:
        return await anext(stream)
```

Alternatively, signal cancellation and wait for the stream to finish closing.

Use `ExaoneRetryStrategy` to replace the SDK's generic throttling backoff with EXAONE-aware handling. This keeps the Harness retry boundary intact while preventing retries from ignoring the server's delay hint.

Model routers can behave differently from provider retries when they switch after streaming begins. Test the framework's fallback behavior when partial output matters to your application.

For direct model calls, this configuration disables the Agent retry strategy and permits one provider retry with a maximum delay of 10 seconds:

```python
from strands_harness import create_harness
from strands_exaone import ExaoneModel

agent = create_harness(
    model=ExaoneModel(max_retries=1, max_retry_delay=10),
    retry_strategy=None,
)
```

This option is intended for direct model calls or when choosing provider-level retries explicitly. For Harness use, `ExaoneRetryStrategy` is the recommended way to honor each EXAONE response's retry headers.

The provider preserves invalid model output as an error rather than repairing or silently dropping it. Malformed tool arguments are rejected before Strands can execute the tool. Structured-output validation errors should be handled through the Strands structured-output contract; automatic prompt rewriting or repeating a model-generated mistake is not enabled.

## Metadata and errors

```python
from strands_exaone import ExaoneModel


def on_metadata(metadata: dict) -> None:
    print(metadata["request_id"], metadata["usage"])


model = ExaoneModel(include_usage=True, metadata_callback=on_metadata)
```

After a successful response, the callback receives `status_code`, `request_id`, selected response `headers`, `usage`, and `response`. Raw usage can include `reasoning_tokens`. Available response metadata includes top-level IDs, model, service and weight metadata, `result_code`, and `description`; choice-level `finish_reason`, `matched_stop`, and `logprobs`; and selected message fields. Streaming logprob and annotation arrays are accumulated in order; refusal fragments are joined, while audio and legacy function-call fragments remain in `message_deltas`. Fields are present only when supplied by the service. The callback is not called if the response fails or is interrupted before completion.

Asynchronous callbacks are awaited and receive task cancellation when the invocation's cancellation signal is set. When a signal is supplied, an asynchronous callback runs in a separate task; changes it makes to `ContextVar` values do not propagate back to the caller. A callback that suppresses cancellation may delay termination; a synchronous callback cannot be interrupted while it is running. Callback errors propagate to the caller. A callback's Strands throttling or context-overflow exception is wrapped in `ExaoneError`, with the original as its cause, so it is not mistaken for a model error after the response.

Strands usage metrics are emitted only when all three token counts are non-negative safe integers; raw usage remains available to the metadata callback even when those metrics cannot be emitted.

`metrics.latencyMs` is captured before the metadata callback runs. A streaming call still includes time the consumer spends paused between events, because the provider stream follows the caller's async iteration pace.

**Metadata can contain user or generated content**, including logprob tokens, refusal text, audio, and function arguments. It is not size-limited and can grow with the response. API keys are redacted, but metadata should only be logged or shared under an appropriate data policy. The provider does not copy ordinary assistant content, reasoning, or tool calls into response metadata.

`ExaoneError` exposes `status_code`, `request_id`, selected `headers`, sanitized `body`, and normalized `error_type`, `code`, and `param` fields. `error_type` and `param` come from the nested `error` object. `code` comes from `error.code` when that key is present, including when its value is `null`; otherwise it comes from the documented top-level `result_code`. Error messages use the sanitized `error.message`, then top-level `description`; HTTP failures fall back to `HTTP {status}`, while an error envelope in a successful response uses a generic provider message. `ExaoneThrottledError` subclasses Strands' `ModelThrottledException`, and `ExaoneContextWindowError` subclasses its `ContextWindowOverflowException`. Neither is a subclass of `ExaoneError`; each provides an `ExaoneError` in `.details` with the same diagnostics. Other API errors use `ExaoneError`. An injected `http_client` remains caller-owned.

HTTP 400 or 413 is classified as a context-window error only when `error.code` is exactly `context_length_exceeded`. Other errors, including messages that merely mention the context window, remain `ExaoneError` and do not trigger Strands' overflow recovery. Whether EXAONE returns this specific code for an actual overflow has not been verified.

Standard HTTPX transport and stream errors retain their exception type, with the API key redacted from the message. Custom HTTPX subclasses are forwarded as their nearest standard HTTPX base type. An `HTTPStatusError` raised by a custom client is forwarded as `HTTPError` with its `status_code` preserved, but without its original response. Forwarded HTTPX errors do not include the original request or chained transport errors.

Other exceptions raised by a custom HTTPX transport are passed through unchanged. Their messages and chained exceptions may contain sensitive data; configure custom transports and logging accordingly.

## License

[MIT](https://github.com/ykoh42/exaone-integrations/blob/main/packages/strands-exaone-py/LICENSE)
