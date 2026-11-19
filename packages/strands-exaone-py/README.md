# strands-exaone for Python

Use the EXAONE Chat Completions API as a model provider for [Strands Agents](https://strandsagents.com/). This community-maintained package is not affiliated with LG AI Research or the Strands Agents project.

## API key

Get a key from the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview) and set it in your environment. See the [setup guide](https://github.com/ykoh42/exaone-integrations/blob/main/guides/getting-started.md#set-your-api-key) for Windows PowerShell.

```bash
export EXAONE_API_KEY='your-api-key'
```

## Quick start

Use Python 3.10 or newer. Install the provider and Strands Harness.

```bash
python -m pip install strands-exaone 'strands-harness~=0.4.0'
```

```python
from strands_exaone import ExaoneModel, ExaoneRetryStrategy
from strands_harness import create_harness

agent = create_harness(
    model=ExaoneModel(),
    retry_strategy=ExaoneRetryStrategy(),
    builtin_tools={"*": False},
    builtin_plugins=[],
    session=False,
    memory=False,
    skills=False,
    caching=False,
    callback_handler=None,
)
result = agent("Explain what an AI agent is in one sentence.")
for block in result.message["content"]:
    if "text" in block:
        print(block["text"])
```

See the [Harness examples](examples/harness/) for tools and conversation history. For the model provider alone, install `strands-exaone` and see the [Strands SDK examples](examples/sdk/). The provider supports `strands-agents` 1.63.x through 1.65.x.

Load any application `.env` file before constructing the model, or pass `api_key` explicitly.

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `model_id` | EXAONE model ID. | `k-exaone_v2.5` |
| `base_url` | HTTP or HTTPS API base URL, including `/v1`. | `https://api.exaone.ai/v1` |
| `stream` | Request a streaming response; `False` adapts a non-streaming response to Strands events. | `True` |
| `temperature`, `top_p` | Sampling parameters. | `1.0`, `0.95` |
| `max_tokens`, `top_k`, `presence_penalty` | Generation parameters. | Service defaults when omitted |
| `reasoning_effort` | EXAONE reasoning setting. | Service default |
| `preserve_thinking` | Ask EXAONE to preserve previous reasoning in model input. | `True` |
| `enable_thinking` | Reasoning-generation option for compatible custom models. K-EXAONE 2.5 always reasons. | Service default |
| `include_usage` | Set `stream_options.include_usage` on streaming requests. | `True` |
| `response_format` | Response format for ordinary agent calls. | Omitted |
| `timeout` | Deadline for sending the request, retry waits, reading the response, asynchronous metadata callbacks, and time the caller pauses between yielded stream events. A custom transport that stalls during cleanup can finish after the deadline. | No provider-imposed timeout |
| `context_window_limit` | Override the context window reported to Strands. Set a positive token count. | `524288` for `k-exaone_v2.5`; otherwise resolved by the SDK |
| `max_retries`, `base_retry_delay`, `max_retry_delay` | Optional retry policy. | `0` retries; `1` and `30` seconds |
| `metadata_callback` | Callback for successful request metadata. | None |
| `http_client` | Caller-owned `httpx.AsyncClient` for custom transport or headers. | Provider-managed client |
| `params` | Additional Chat Completions parameters. | None |

Explicit options take precedence over matching `params` values, which take precedence over provider defaults. The provider manages `model`, `messages`, `stream`, `tools`, and `tool_choice`; do not set them in `params`. For streaming requests, `params["stream_options"]` can set `include_usage` and other service-supported options; the named `include_usage` option takes precedence. `stream_options` is omitted from non-streaming requests. `params["response_format"]` can set a format for ordinary model calls; the provider's `response_format` option takes precedence. Strands represents one completion, so `n` must be 1. Supported generation settings can vary by service version; see the [EXAONE API documentation](https://docs.exaone.ai/en/docs/k-exaone-2.5).

`model_info.limits.context_window`, when set, takes precedence over `context_window_limit`, following the Strands SDK's model info merge contract.

First-class generation options are validated before storage or a request. Values supplied through `params` remain an extension point and are passed through unless the provider owns the field; EXAONE may reject unsupported values.

`update_config(...)` applies to requests that have not started. Each in-flight request keeps the configuration it started with, including its retry policy.

## Self-hosted K-EXAONE 2.0

For an OpenAI-compatible local server, pass the model name exposed by its `/v1/models` endpoint:

```python
from strands_exaone import ExaoneModel

model = ExaoneModel(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",  # Use the server's configured credential, if any.
    model_id="K-EXAONE-2.0-750B-A37B",  # Use the server's actual model name.
)
```

The provider accepts both `reasoning_content` and `reasoning` in compatible server responses and sends preserved reasoning back as `reasoning_content`. Identical values are emitted once; conflicting nonempty values are rejected. The model card lists a 262,144-token context window, but a deployment can configure a lower limit. Use the limit exposed by your deployment. Server and model compatibility depend on the versions and serving options you use.

For serving instructions and parser settings, see the [model card](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B). Its examples use forked servers; [vLLM 0.27.0](https://github.com/vllm-project/vllm/releases/tag/v0.27.0) also includes K-EXAONE 2.0 model support. Verify the specific server version and configuration you deploy.

Some servers accept `reasoning_effort` and may use it to select thinking on or off. Accepting the field does not establish that the model supports distinct reasoning-intensity levels; check the deployed server and model documentation.

The configured API key is sent to the selected endpoint; if omitted, it comes from `EXAONE_API_KEY`. For a server that does not require authentication, pass a placeholder such as `api_key="EMPTY"` rather than the hosted API key. HTTP does not encrypt credentials or conversation data; use HTTPS or a trusted transport. Protocol fixtures have been tested, but K-EXAONE 2.0 model inference has not been verified by this project.

## Self-hosted EXAONE 4.5 image input

For an already-running compatible endpoint, set `base_url`, `api_key` and `model_id` as above, using the server's actual EXAONE 4.5 model name. Supply image bytes through a standard Strands user content block:

```python
from pathlib import Path
from strands import Agent
from strands_exaone import ExaoneModel

model = ExaoneModel(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", model_id="EXAONE-4.5-33B")
agent = Agent(model=model)
response = agent(
    [
        {"text": "Describe this image."},
        {"image": {"format": "png", "source": {"bytes": Path("image.png").read_bytes()}}},
    ]
)
```

The provider sends text and image blocks in their original order. Strands Python image locations such as S3 are not converted to URLs; supply bytes. Image tool results remain unsupported. The default hosted K-EXAONE 2.5 model is text-only.

The [EXAONE 4.5 model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) shows vLLM and an SGLang fork with `qwen3` reasoning and `hermes` tool parsers. Select and configure these on the server, not in the provider. Image requests have been tested with synthetic protocol fixtures; actual EXAONE 4.5 model inference has not been verified.

See the [compatibility matrix](https://github.com/ykoh42/exaone-integrations/blob/main/COMPATIBILITY.md#known-server-side-observations) for server versions, parser settings, and observed limitations.

## Model information

`ExaoneModel.model_info` exposes the known `k-exaone_v2.5` capabilities through Strands' standard `ModelInfo`: stateless Chat Completions, forced tool choice, native JSON Schema output, and a 524,288-token context window. An independent maximum output size and a prompt-caching mechanism are left unknown.

The SDK owns how these facts are used. Explicit `model_info` fields override the provider's defaults; unspecified fields inherit. `model_info.limits.context_window` takes precedence over `context_window_limit`. Provider defaults apply only to the known model ID at the hosted endpoint. For other endpoints or model IDs, the SDK's lookup or fallback applies unless you configure its actual limits and capabilities. Explicit limits remain in effect when changing models.

The context budget covers input plus output, including reasoning. See the [EXAONE API specification](https://docs.exaone.ai/ko/docs/k-exaone-2.5) and [Strands model-info contract](https://github.com/strands-agents/harness-sdk/blob/main/shared/README.md).

## Reasoning and tools

The provider stores reasoning separately from the final answer and sends it back as reasoning_content when it appears in the Strands conversation. By default, requests use EXAONE's recommended agent settings: temperature=1.0, top_p=0.95, and preserve_thinking=True. K-EXAONE 2.5 always reasons; `enable_thinking` can be forwarded for compatible custom models.

When consuming `model.stream()` directly, execute tools only after `messageStop` reports `tool_use`: a `max_tokens` stop can include incomplete fragments, while missing tool IDs or names and duplicate IDs cause a provider error. When the Strands SDK processes malformed tool arguments, it may log a warning containing part of those arguments, which can include sensitive data.

Strands can summarize older conversation history during automatic context management. The provider forwards the reasoning that remains in the history it receives; it cannot restore content that was summarized or removed. Token counts used for proactive context management are estimates, not exact EXAONE token counts.

The provider requests streaming usage by default (include_usage=True) so Strands can use token counts for context management. EXAONE may omit usage, and downstream context management is controlled by the framework.

Check both the returned reasoning fields and visible content. The request setting alone does not guarantee that reasoning is absent from the content.

When EXAONE returns a refusal without non-whitespace ordinary response content, the provider exposes that refusal as a Strands text block and preserves any whitespace content. Strands has no separate refusal event; the refusal remains available in response metadata as well. System prompt blocks follow the Strands OpenAI adapter contract: text blocks are sent as separate messages and blocks without text, such as cache points, are omitted.

For direct `structured_output()` calls, a refusal without JSON raises `ExaoneError`; its sanitized `body` includes the refusal text. Strands tool execution uses the Chat Completions `tool_calls` field. Legacy `function_call` response fields are retained in metadata but do not trigger a Strands tool.

The [Harness tool example](examples/harness/#custom-tools-and-conversation-history) runs a custom tool and a follow-up turn.

The provider assembles streamed tool-call arguments and validates each completed call before exposing it to Strands. Hosted and self-hosted endpoints follow the same rules. Nonblank arguments must be a valid JSON object, and blank arguments follow Strands' rules. The Harness validates the object against the tool schema. The provider accepts text, reasoning, tool calls, and text or JSON tool results. Unsupported content types and reasoning signatures in the received history are rejected.

Because Chat Completions tool messages have no separate status field, ordinary tool results marked as errors are sent with an `[ERROR] ` prefix. Successful results are sent without a prefix.

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

This setting keeps unsupported media blocks out of requests. It does not give the model image or document understanding. The `read` tool can access files allowed by the agent's sandbox, and it sends the results to the model endpoint. Use an isolated sandbox or an approval mechanism to limit access to files you intend to share.

Cancellation may leave work running; for side-effecting tools, also invalidate pending approvals and check external state before retrying.

The provider forwards text without trimming leading or trailing whitespace. During a tool turn, Strands may remove a block that contains only whitespace, so those blocks may not round-trip exactly.

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

Handle validation failures through Strands' structured-output contract; the provider does not automatically rewrite prompts or retry model-generated output.

## Tool-call recovery

Use this optional middleware when malformed JSON tool arguments interrupt agent runs. The provider works without this middleware. The [Harness tool example](examples/harness/#optional-tool-call-recovery) enables recovery with `--recover-tool-calls`. To enable it on the Harness agent from the quick start, register it before invoking the agent:

```python
from strands.middleware import InvokeModelStage
from strands_exaone.recovery import recover_tool_calls

agent.add_middleware(InvokeModelStage, recover_tool_calls)
```

Text and reasoning stream as they arrive. Tool events are released after the arguments have been checked. Recovery can retry one malformed call before any output is emitted. To wait for validation before streaming any part of the response, register `functools.partial(recover_tool_calls, buffer_response=True)`.

With the middleware enabled, a refusal blocks tool execution even when the response also includes ordinary text.
Recovery uses the actual tools and tool choice sent to EXAONE. A correction is skipped if another middleware removes that tool or overrides the correction's tool choice. A correction stopped by the output limit follows the SDK's normal max-token handling.

Recovery sends the original arguments and parser diagnostic in a matching error tool result. The preceding empty argument object is a transport placeholder and is never executed. The system prompt and user messages are retained. Truncated, mixed, incomplete and repeated failures, and ambiguous streamed call IDs, are not recovered. Recovery can add latency and model charges. Tracing may record the diagnostic and original arguments. When both attempts report usage, the SDK usage includes both. Failed-attempt usage is also supplied to `metadata_callback` under `recovery.failed_attempts` and logged at INFO. If recovery itself fails, that callback is not invoked.

The tool result contains `raw_arguments` and `parse_error`. Diagnostic line and column numbers refer to the original argument text.

Only the correction request selects the failed tool by name. If the endpoint explicitly rejects named tool choice before emitting a response, the middleware tries `tool_choice="required"` once. Recovery succeeds only with a completed reply containing exactly one call to the original tool with a valid JSON object as its arguments. The SDK still validates the tool schema before execution.

The middleware applies to direct `ExaoneModel` instances; other model implementations and subclasses pass through unchanged.

Empty or whitespace-only tool input is not automatically retried; the SDK handles it according to its existing rules, which can allow a tool's default arguments to run.

## Retries

`ExaoneModel`'s own HTTP retry loop is disabled by default. For Agent and Harness use, configure `ExaoneRetryStrategy` as the Strands retry strategy; it replaces the SDK default and honors EXAONE's `Retry-After` and `Retry-After-Ms` headers. It defaults to three total attempts, a one-second initial backoff, and a 30-second delay cap. If a server hint exceeds the cap, it surfaces the error instead of retrying early. Without a usable hint, it uses bounded exponential backoff with jitter.

Strands owns Agent-level retry selection, so a model provider cannot silently replace the SDK's strategy. `ExaoneRetryStrategy` is opt-in: pass an instance to `Agent` or `create_harness` to replace the generic policy with EXAONE-aware waits. The quick start and Harness example show this explicit configuration.

Strands' default retry strategy does not inspect EXAONE headers. Keeping it alongside the provider's own retry loop can multiply attempts, so leave `max_retries` at its default of zero when using `ExaoneRetryStrategy`. The strategy retries EXAONE transient HTTP errors, safe connection failures, and Strands throttling failures before the provider exposes output. It does not retry tool arguments that remain invalid after syntax-only completion or other model response errors.

To extend or narrow retryable errors, subclass `ExaoneRetryStrategy` and override Strands' `is_retryable()` method. Call `super().is_retryable(error)` to retain the EXAONE defaults. The strategy does not retry a provider failure after a stream event has been exposed, even if an override marks the error as retryable.

Unlike the SDK's default Python retry strategy, `ExaoneRetryStrategy` does not emit `event_loop_throttled_delay`. The current SDK exposes no public extension point for emitting this legacy event from a custom hook; supporting it while honoring server delay hints requires an upstream SDK extension.

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

Model routers can behave differently from provider retries when they switch after streaming begins. Test the framework's fallback behavior when partial output matters to your application.

To use provider-level retries with a Harness agent, disable the Agent retry strategy. This configuration permits one provider retry with a maximum delay of 10 seconds:

```python
from strands_harness import create_harness
from strands_exaone import ExaoneModel

agent = create_harness(
    model=ExaoneModel(max_retries=1, max_retry_delay=10),
    retry_strategy=None,
)
```

This option is intended for direct model calls or when choosing provider-level retries explicitly. For Harness use, `ExaoneRetryStrategy` is the recommended way to honor each EXAONE response's retry headers.

## Metadata and errors

```python
from strands_exaone import ExaoneModel


def on_metadata(metadata: dict) -> None:
    print(metadata["request_id"], metadata["usage"])


model = ExaoneModel(include_usage=True, metadata_callback=on_metadata)
```

After a successful response, the callback receives `status_code`, `request_id`, selected response `headers`, `usage`, and `response`. Raw usage can include `reasoning_tokens`. Available response metadata includes top-level IDs, model, service and weight metadata, `result_code`, and `description`; choice-level `finish_reason`, `matched_stop`, and `logprobs`; and selected message fields. Streaming logprob and annotation arrays are accumulated in order; refusal fragments are joined, while audio and legacy function-call fragments remain in `message_deltas`. Fields are present only when supplied by the service. The callback is not called if the response fails or is interrupted before completion.

When the endpoint reports cached or cache-write input tokens, the Strands usage event also exposes them as `cacheReadInputTokens` and `cacheWriteInputTokens`. These are input-token details; they are not added again to the input total.

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
