# strands-exaone for TypeScript

Use the EXAONE Chat Completions API as a model provider for [Strands Agents](https://strandsagents.com/). This community-maintained package is not affiliated with LG AI Research or the Strands Agents project.

See the [changelog](CHANGELOG.md) for release notes.

## API key

Get a key from the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview) and set it in your environment. See the [setup guide](https://github.com/ykoh42/exaone-integrations/blob/main/guides/getting-started.md#set-your-api-key) for Windows PowerShell.

```bash
export EXAONE_API_KEY='your-api-key'
```

## Quick start

The package is published on npm as `strands-exaone`. It requires Node.js 22 or newer and uses `@strands-agents/sdk` 1.24.0 or newer within the 1.x line and Zod 4.5.4 or newer within major version 4 as peer dependencies. Install it in your project with:

```bash
npm install strands-exaone
```

```typescript
import { Agent } from '@strands-agents/sdk'
import { ExaoneModel, ExaoneRetryStrategy } from 'strands-exaone'

const model = new ExaoneModel({ preserveThinking: true })
const agent = new Agent({ model, retryStrategy: new ExaoneRetryStrategy(), printer: false })
const response = await agent.invoke('Explain what an AI agent is in one sentence.')
const answer = response.lastMessage.content
  .filter((block) => block.type === 'textBlock')
  .map((block) => block.text)
  .join('')
console.log(answer)
```

The provider reads `EXAONE_API_KEY` when a request is made. You can instead pass `apiKey` to the constructor. The package does not load `.env` files automatically.

The configured endpoint and any custom fetch implementation receive your API key and conversation. Use a trusted endpoint and transport, and use HTTPS except for controlled local testing.

The example disables the SDK's default printer and prints only text blocks. In SDK 1.24, the default printer and `String(response)` can include reasoning.

For SDK tool use, streaming, structured output, Harness, and CLI walkthroughs, see the [examples](examples/).

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `modelId` | EXAONE model ID. | `k-exaone_v2.5` |
| `baseURL` | HTTP or HTTPS API base URL, including `/v1`. | `https://api.exaone.ai/v1` |
| `stream` | Request a streaming response; `false` adapts a non-streaming response to Strands events. | `true` |
| `maxTokens`, `topK`, `presencePenalty` | Generation parameters. | Service defaults when omitted |
| `temperature` | Sampling temperature. | `1.0` |
| `topP` | Nucleus sampling threshold. | `0.95` |
| `reasoningEffort` | EXAONE reasoning setting. | Service default |
| `preserveThinking` | Request preservation of earlier assistant reasoning in the model input. | `true` |
| `enableThinking` | Reasoning-generation option for compatible custom models. K-EXAONE 2.5 always reasons. | Service default |
| `includeUsage` | Set `stream_options.include_usage` on streaming requests. | `true` |
| `timeoutMs` | Deadline for attempts, retry waits, response reads, and stream consumption. A custom fetch implementation must honor the supplied abort signal. | No provider-imposed timeout |
| `contextWindowLimit` | Override the context window reported to Strands. Set a positive token count. | `524288` for `k-exaone_v2.5`; otherwise resolved by the SDK |
| `maxRetries`, `baseRetryDelayMs`, `maxRetryDelayMs` | Optional retry policy. | `0` retries; `1000` and `30000` milliseconds |
| `onMetadata` | Callback for successful request metadata. | None |
| `fetch` | Custom fetch implementation. | Global `fetch` |
| `params` | Additional Chat Completions parameters. | None |

By default, each request uses EXAONE's recommended agent settings: `temperature=1.0`, `top_p=0.95`, and `preserve_thinking=true`. `getConfig()` returns the configured settings. Explicit named options override matching `params` values, which override the defaults. The provider manages `model`, `messages`, `stream`, `tools`, and `tool_choice`; leave these fields out of `params`. For streaming requests, `params.stream_options` can set `include_usage` and other service-supported options. The named `includeUsage` option takes precedence, and `stream_options` is omitted from non-streaming requests. Strands represents one completion, so `n` must be 1. Supported generation settings can vary by service version; see the [EXAONE API documentation](https://docs.exaone.ai/en/docs/k-exaone-2.5).

The provider validates first-class generation options before storing them or sending a request. Values supplied through `params` remain an extension point and are passed through unless the provider owns the field; EXAONE may reject unsupported values.

The provider copies configuration arrays, plain objects, and `Date` values when it constructs, updates, or returns configuration. It keeps custom class instances by reference to preserve their serialization behavior. A stream uses the configuration it had at startup and serializes its request once, so retries send the same body. `updateConfig()` applies to later streams.

In Strands SDK 1.24, `modelInfo.limits.contextWindow`, when set, takes precedence over `contextWindowLimit`.

If EXAONE returns a refusal without ordinary non-whitespace response text, the provider exposes the refusal as a Strands text block and preserves any whitespace content. Strands has no separate refusal event, so the refusal is also available in response metadata. System prompt blocks follow the Strands OpenAI adapter contract: text blocks are joined into one message, while cache points and guard content are omitted with warnings.

Strands tool execution uses the Chat Completions `tool_calls` field. Legacy `function_call` response fields are retained in metadata but do not trigger a Strands tool.

## Self-hosted K-EXAONE 2.0

For an OpenAI-compatible local server, pass the model name exposed by its `/v1/models` endpoint:

```typescript
import { ExaoneModel } from 'strands-exaone'

const model = new ExaoneModel({
  baseURL: 'http://127.0.0.1:8000/v1',
  apiKey: 'EMPTY', // Use the server's configured credential, if any.
  modelId: 'K-EXAONE-2.0-750B-A37B', // Use the server's actual model name.
})
```

The provider accepts both `reasoning_content` and `reasoning` in compatible server responses and sends preserved reasoning back as `reasoning_content`. Identical values are emitted once; conflicting nonempty values are rejected. The model card lists a 262,144-token context window, but a deployment can configure a lower limit. Use the limit exposed by your deployment. Server and model compatibility depend on the versions and serving options you use.

For serving instructions and parser settings, see the [model card](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B). Its examples use forked servers; [vLLM 0.27.0](https://github.com/vllm-project/vllm/releases/tag/v0.27.0) also includes K-EXAONE 2.0 model support. Verify the specific server version and configuration you deploy.

Some servers accept `reasoning_effort` and may use it to select thinking on or off. Accepting the field does not establish that the model supports distinct reasoning-intensity levels; check the deployed server and model documentation.

The configured API key is sent to the selected endpoint; if omitted, it comes from `EXAONE_API_KEY`. For a server that does not require authentication, pass a placeholder such as `apiKey: 'EMPTY'` rather than the hosted API key. HTTP does not encrypt credentials or conversation data; use HTTPS or a trusted transport. Protocol fixtures have been tested, but K-EXAONE 2.0 model inference has not been verified by this project.

## Self-hosted EXAONE 4.5 image input

For an already-running compatible endpoint, set `baseURL`, `apiKey` and `modelId` as above, using the server's actual EXAONE 4.5 model name. Supply standard Strands image blocks with bytes or a URL:

```typescript
import { readFile } from 'node:fs/promises'
import { Agent, ImageBlock, TextBlock } from '@strands-agents/sdk'
import { ExaoneModel } from 'strands-exaone'

const model = new ExaoneModel({ baseURL: 'http://127.0.0.1:8000/v1', apiKey: 'EMPTY', modelId: 'EXAONE-4.5-33B' })
const agent = new Agent({ model })
const response = await agent.invoke([
  new TextBlock('Describe this image.'),
  new ImageBlock({ format: 'png', source: { bytes: await readFile('image.png') } }),
])
```

The provider sends text and image blocks in their original order. Strands S3 image locations are not converted to URLs; supply bytes or a URL. Image tool results remain unsupported. The default hosted K-EXAONE 2.5 model is text-only.

The [EXAONE 4.5 model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) shows vLLM and an SGLang fork with `qwen3` reasoning and `hermes` tool parsers. Select and configure these on the server, not in the provider. Image requests have been tested with synthetic protocol fixtures; actual EXAONE 4.5 model inference has not been verified.

See the [compatibility matrix](https://github.com/ykoh42/exaone-integrations/blob/main/COMPATIBILITY.md#known-server-side-observations) for server versions, parser settings, and observed limitations.

## Model information

`ExaoneModel.modelInfo` exposes the known `k-exaone_v2.5` capabilities through Strands' standard `ModelInfo`: stateless Chat Completions, forced tool choice, native JSON Schema output, and a 524,288-token context window. An independent maximum output size and a prompt-caching mechanism are left unknown.

The SDK owns how these facts are used. Explicit `modelInfo` fields override the provider's defaults; unspecified fields inherit. `modelInfo.limits.contextWindow` takes precedence over `contextWindowLimit`. Provider defaults apply only to the known model ID at the hosted endpoint. For other endpoints or model IDs, the SDK's lookup or fallback applies unless you configure its actual limits and capabilities. Explicit limits remain in effect when changing models.

The context budget covers input plus output, including reasoning. See the [EXAONE API specification](https://docs.exaone.ai/ko/docs/k-exaone-2.5) and [Strands model-info contract](https://github.com/strands-agents/harness-sdk/blob/main/shared/README.md).

## Reasoning and tools

The provider stores reasoning separately from the final answer and sends it back as reasoning_content when it appears in the Strands conversation. preserveThinking defaults to true; set it to false to stop requesting preservation of earlier reasoning. K-EXAONE 2.5 always reasons; `enableThinking` can be forwarded for compatible custom models.

When an output-limit response maps to the SDK `maxTokens` stop, the SDK Agent raises `MaxTokensError` instead of executing its tool calls. Calls with missing names or missing or duplicate IDs are rejected before that stop. Low-level stream consumers should wait for the stop reason before executing tools: earlier tool fragments may be incomplete. The SDK logs tool-input JSON parse errors, which can include fragments of malformed input; configure logging accordingly.

Strands can summarize older conversation history during automatic context management. The provider forwards the reasoning that remains in the history it receives; it cannot restore content that was summarized or removed. Token counts used for proactive context management are estimates, not exact EXAONE token counts.

Check both the returned reasoning fields and visible content. The request setting alone does not guarantee that reasoning is absent from the content.

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel, ExaoneRetryStrategy } from 'strands-exaone'

const getNumber = tool({
  name: 'get_number',
  description: 'Return the example number.',
  inputSchema: z.object({}),
  callback: () => '7',
})

const agent = new Agent({
  model: new ExaoneModel({ preserveThinking: true }),
  tools: [getNumber],
  retryStrategy: new ExaoneRetryStrategy(),
})
await agent.invoke('Use the tool to get the number, then add 3.')
```

The provider assembles streamed tool-call arguments and validates each completed call before exposing it to Strands. Hosted and self-hosted endpoints follow the same rules. Nonblank arguments must be a valid JSON object, and blank arguments follow Strands' rules. The Harness validates the object against the tool schema. The provider accepts text, reasoning, tool calls, and text or JSON tool results. Unsupported content types and reasoning signatures in the received history are rejected. Strands SDK 1.25 may restore unrecognized session data as `UnknownBlock`. These blocks stay in the SDK session and are omitted from EXAONE requests.

Because Chat Completions tool messages have no separate status field, tool results marked as errors are sent with an `[ERROR] ` prefix. Successful results are sent without a prefix.

Multiple system text blocks are concatenated without separators into one system message, matching Strands' TypeScript OpenAI Chat Completions adapter. Include separators in the blocks when needed.

A user message containing both text and tool results is split into tool messages followed by a user text message, matching Strands' OpenAI Chat Completions adapter so tool responses remain adjacent to their assistant tool calls.

JavaScript parses JSON numbers as `Number`, so integers outside the safe integer range can lose precision in tool arguments and history. Use a string schema for large identifiers or exact decimal values.

The provider preserves whitespace in text; Strands SDK aggregation omits text blocks that consist entirely of whitespace.

When using Strands Harness, configure the built-in `read` tool to return text descriptions for images and binary documents:

```typescript
import { createHarness } from '@strands-agents/harness'
import { ExaoneModel, ExaoneRetryStrategy } from 'strands-exaone'

const agent = await createHarness({
  model: new ExaoneModel(),
  retryStrategy: new ExaoneRetryStrategy(),
  builtinTools: { '*': false, read: { media: false } },
  builtinPlugins: [],
  session: false,
  memory: false,
  skills: false,
  caching: false,
})
```

These descriptions identify the file but do not give the model image or document understanding. The `read` tool can access files allowed by the agent's sandbox, and it sends the results to the model endpoint. Use an isolated sandbox or an approval mechanism to limit access to files you intend to share.

Cancellation may leave work running; check external state before retrying side-effecting tools.

In SDK 1.24.0, a pending `HumanInTheLoop` callback can approve and execute a tool after the invocation's cancellation signal is set. Coordinate approval with cancellation in your application. Cancellation alone does not invalidate an approval request.

## Structured output

```typescript
import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel } from 'strands-exaone'

const agent = new Agent({
  model: new ExaoneModel(),
  structuredOutputSchema: z.object({ city: z.string(), temperature_c: z.number() }),
})
const result = await agent.invoke('Return Seoul and a temperature of 20 degrees Celsius.')
console.log(result.structuredOutput)
```

The provider maps structured-output requests to JSON Schema or a forced tool call when Strands requests one. An explicit params.response_format takes precedence over the SDK's JSON Schema and can also be sent with a structured-output tool; EXAONE support for that combination has not been verified. The SDK validates structuredOutput for an agent invocation; callers using model.stream() directly must validate returned text. The provider treats abort as an error rather than a valid result; schema support may vary with the service and schema.

Handle validation failures through Strands' structured-output contract; the provider does not automatically rewrite prompts or retry model-generated output.

## Retries

`ExaoneModel`'s own HTTP retry loop is disabled by default. For Agent and Harness use, configure `ExaoneRetryStrategy` as the Strands retry strategy; it replaces the SDK default and honors EXAONE's `Retry-After` and `Retry-After-Ms` headers. It defaults to three total attempts, a one-second initial backoff, and a 30-second delay cap. If a server hint exceeds the cap, it surfaces the error instead of retrying early. Without a usable hint, it uses bounded exponential backoff with jitter.

Strands owns Agent-level retry selection, so a model provider cannot silently replace the SDK's strategy. `ExaoneRetryStrategy` is opt-in: pass an instance to `Agent` or `createHarness` to replace the generic policy with EXAONE-aware waits. The quick start and Harness example show this explicit configuration.

Strands' default retry strategy does not inspect EXAONE headers. Keeping it alongside the provider's own retry loop can multiply attempts, so leave `maxRetries` at its default of zero when using `ExaoneRetryStrategy`. The strategy retries EXAONE transient HTTP errors and Strands network errors before the provider exposes output; it does not retry invalid model-generated tool arguments or other model response errors.

`ExaoneModel`'s optional `maxRetries` setting remains available for direct model calls that do not use an Agent retry strategy. It retries retryable HTTP responses before emitting model events. Fetch failures are not retried because the fetch API does not reliably say whether the request was received. The provider never replays a partially delivered stream.

To use provider-level retries with a Harness agent, disable the Agent retry strategy. This configuration allows one provider retry with a maximum wait of 10 seconds:

```typescript
import { createHarness } from '@strands-agents/harness'
import { ExaoneModel } from 'strands-exaone'

const agent = await createHarness({
  model: new ExaoneModel({ maxRetries: 1, maxRetryDelayMs: 10_000 }),
  retryStrategy: null,
})
```

This disables the SDK's model retry strategy for that Agent. Cancellation interrupts provider retry waits; `timeoutMs` bounds provider attempts, provider retry waits, response reads, and stream consumption. Agent-level `ExaoneRetryStrategy` waits are outside the model timeout and follow Strands retry-strategy semantics. A custom `fetch` implementation must honor the supplied abort signal.

When stopping an agent stream early, use SDK cancellation APIs and let the model stream close; custom fetch implementations should honor the supplied abort signal.

## Tool-call recovery

To opt in to recovery from one completed tool call with malformed JSON arguments, register the model-stage middleware:

```typescript
import { Agent, InvokeModelStage } from '@strands-agents/sdk'
import { ExaoneModel, toolCallRecovery } from 'strands-exaone'

const agent = new Agent({ model: new ExaoneModel(), tools: [] })
agent.addMiddleware(InvokeModelStage, toolCallRecovery())
```

Text and reasoning stream as they arrive. Tool events are released after the arguments have been checked. Recovery can retry one malformed call before any output is emitted. To wait for validation before streaming any part of the response, register `toolCallRecovery({ bufferResponse: true })`.

With the middleware enabled, a refusal blocks tool execution even when the response also includes ordinary text.
Recovery uses the actual tools and tool choice sent to EXAONE. A correction is skipped if another middleware removes that tool or overrides the correction's tool choice. A correction stopped by the output limit follows the SDK's normal max-token handling.

Recovery sends the original argument text and JSON diagnostic as a matching error tool result. For transport compatibility the failed assistant call has `{}` arguments in that retry, explicitly described as a placeholder in the tool result. The placeholder is never executed. The original system prompt and user messages are retained. Mixed calls, incomplete calls, length stops, refusals and ambiguous streamed call IDs are not automatically recovered. If recovery is unsuccessful, the error is propagated.

The tool result contains `raw_arguments` and `parse_error`. Diagnostic line and column numbers refer to the original argument text.

Only the correction request selects the failed tool by name. If the endpoint explicitly rejects named tool choice before emitting a response, the middleware tries `tool_choice="required"` once. Recovery succeeds only with a completed reply containing exactly one call to the original tool with a valid JSON object as its arguments. The SDK validates Zod-backed input schemas before execution. Plain JSON Schema describes the expected inputs, and the tool implementation defines its runtime validation.

The middleware applies to direct `ExaoneModel` instances; other model implementations and subclasses pass through unchanged.

Empty or whitespace-only tool input is not automatically retried; the SDK handles it according to its existing rules, which can allow a tool's default arguments to run.

On successful recovery, `onMetadata` receives `recovery.failedAttempts` with the failed response's usage if the server supplied it. When both attempts report valid token counts, the SDK model usage includes both attempts; `onMetadata.usage` remains the raw usage from the successful response. Missing usage cannot be inferred. If the retry fails before a successful response, the earlier attempt's usage is not currently reported by `onMetadata`.

## Metadata and errors

```typescript
import { ExaoneModel } from 'strands-exaone'

const model = new ExaoneModel({
  includeUsage: true,
  onMetadata: (metadata) => console.log(metadata.requestId, metadata.usage),
})
```

After a successful response, `onMetadata` receives the HTTP status, request ID, selected rate-limit headers, raw `usage`, and `response`. Raw usage can include `reasoning_tokens`. Available response metadata includes top-level IDs, model, service and weight metadata, `result_code`, and `description`; choice-level `finish_reason`, `matched_stop`, and `logprobs`; and selected message fields. Streaming logprob and annotation arrays are accumulated in order; refusal fragments are joined, while audio and legacy function-call fragments remain in `message_deltas`. Fields are present only when supplied by the service. The same metadata is also available in the Strands model metadata event under `trace.exaone`. The callback is not called if the response fails or is interrupted before completion.

When the endpoint reports cached or cache-write input tokens, the Strands usage event also exposes them as `cacheReadInputTokens` and `cacheWriteInputTokens`. These are input-token details; they are not added again to the input total.

Copy metric values that must remain stable across later invocations; framework metrics may be live snapshots.

The callback may be synchronous or asynchronous; the provider waits for it to finish. Cancellation or timeout stops waiting but cannot forcibly stop work the callback has already started. Callback failures are reported as `ModelError` with the original error as `cause`, so the SDK's default retry strategy does not retry a completed response because of a callback error.

**Metadata can contain user or generated content**, including logprob tokens, refusal text, audio, and function arguments. It is not size-limited and can grow with the response. API keys are redacted, but metadata and traces should only be logged or shared under an appropriate data policy. The provider does not copy ordinary assistant content, reasoning, or tool calls into response metadata.

HTTP errors expose `status`, `requestId`, selected rate-limit headers, sanitized `body`, and normalized `errorType`, `code`, and `param` fields on `ExaoneAPIError`. `errorType` and `param` come from the nested `error` object. `code` comes from `error.code` when that key is present, including when its value is `null`; otherwise it comes from the documented top-level `result_code`. Error messages use the sanitized `error.message`, then top-level `description`; HTTP failures fall back to `HTTP <status>`, while an error envelope in a successful response uses a generic provider message. HTTP 429 is surfaced as the Strands `ModelThrottledError`, with the `ExaoneAPIError` as its cause. An error in a successful HTTP response body or event stream raises `ExaoneResponseError`, whose `body`, `errorType`, `code`, and `param` contain sanitized response details.

If an HTTP error body cannot be read, `body` is `undefined`; the status, request ID, and available rate-limit headers remain accessible. Its cause is a generic read error that does not retain the transport's original message or request data.

For a failed `fetch` request, the provider redacts the API key from the transport error message and does not retain its original cause or request object. Other text in that message, including any prompt or request content a custom transport includes, is retained.

For HTTP 400 or 413, an exact `error.code` of `context_length_exceeded` is surfaced as the Strands `ContextWindowOverflowError`, with the `ExaoneAPIError` as its cause. Strands may compact history and retry in response. EXAONE's use of this OpenAI-compatible code has not been verified; ambiguous 400 and 413 responses remain `ExaoneAPIError` and do not trigger that overflow recovery.

## License

[MIT](LICENSE)
