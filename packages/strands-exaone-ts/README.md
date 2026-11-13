# strands-exaone for TypeScript

Use the EXAONE Chat Completions API as a model provider for [Strands Agents](https://strandsagents.com/). This community-maintained package is not affiliated with LG AI Research or the Strands Agents project.

Users are responsible for reviewing and complying with the terms that apply to their EXAONE API use.

## Quick start

The package is published on npm as `strands-exaone`. It requires Node.js 22 or newer and uses `@strands-agents/sdk` and Zod 4.6.6 or newer within major version 4 as peer dependencies. Install it in your project with:

```bash
npm install strands-exaone
export EXAONE_API_KEY='your-api-key'
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

For file reading and optional web search with Strands Harness, see the [Harness example](examples/harness/).

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `modelId` | EXAONE model ID. | `k-exaone_v2.5` |
| `baseURL` | API base URL, including `/v1`. Remote endpoints must use HTTPS; HTTP is allowed for loopback test servers. | `https://api.exaone.ai/v1` |
| `stream` | Request a streaming response; `false` adapts a non-streaming response to Strands events. | `true` |
| `maxTokens`, `topK`, `presencePenalty` | Generation parameters. | Service defaults when omitted |
| `temperature` | Sampling temperature. | `1.0` |
| `topP` | Nucleus sampling threshold. | `0.95` |
| `reasoningEffort` | EXAONE reasoning setting. | Service default |
| `preserveThinking` | Preserve earlier assistant reasoning in the model input. | `true` |
| `enableThinking` | EXAONE reasoning-generation option. | Service default |
| `includeUsage` | Set `stream_options.include_usage` on streaming requests. | `true` |
| `timeoutMs` | Deadline for attempts, retry waits, response reads, and stream consumption. A custom fetch implementation must honor the supplied abort signal. | No provider-imposed timeout |
| `contextWindowLimit` | Context window reported to Strands. Set a positive token count. | `524288` |
| `maxRetries`, `baseRetryDelayMs`, `maxRetryDelayMs` | Optional retry policy. | `0` retries; `1000` and `30000` milliseconds |
| `onMetadata` | Callback for successful request metadata. | None |
| `fetch` | Custom fetch implementation. | Global `fetch` |
| `params` | Additional Chat Completions parameters. | None |

The provider sends EXAONE's recommended agent settings by default: `temperature=1.0`, `top_p=0.95`, and `preserve_thinking=true`. These defaults are applied when building each request; `getConfig()` returns the configured settings. Explicit named options take precedence over matching `params` values; `params` values take precedence over these defaults. The provider manages `model`, `messages`, `stream`, `tools`, and `tool_choice`; do not set them in `params`. For streaming requests, `params.stream_options` can set `include_usage` and other service-supported options; the named `includeUsage` option takes precedence. `stream_options` is omitted from non-streaming requests. Strands represents one completion, so `n` must be 1. Supported generation settings can vary by service version; see the [EXAONE API documentation](https://docs.exaone.ai/en/docs/k-exaone-2.5).

The provider validates first-class generation options before storing them or sending a request. Values supplied through `params` remain an extension point and are passed through unless the provider owns the field; EXAONE may reject unsupported values.

Configuration arrays, plain objects, and `Date` values are copied on construction, update, and retrieval. Custom class instances are retained by reference so their serialization behavior is preserved. Each stream uses the configuration present when it starts and serializes its request once, so retries use the same body; `updateConfig()` affects later streams.

In Strands SDK 1.24, `modelInfo.limits.contextWindow`, when set, takes precedence over `contextWindowLimit`.

When EXAONE returns a refusal without non-whitespace ordinary response content, the provider exposes that refusal as a Strands text block and preserves any whitespace content. Strands has no separate refusal event; the refusal remains available in response metadata as well. System prompt blocks follow the Strands OpenAI adapter contract: text blocks are joined into one message, while cache points and guard content are omitted with warnings.

Strands tool execution uses the Chat Completions `tool_calls` field. Legacy `function_call` response fields are retained in metadata but do not trigger a Strands tool.

## Reasoning and tools

Reasoning is stored separately from the final answer and sent back as reasoning_content when the Strands conversation includes it. preserveThinking defaults to true to request preservation of earlier reasoning; set it to false to disable that request. enableThinking is a separate request option; check the returned fields and content when your application needs specific reasoning-handling behavior.

Strands can summarize older conversation history during automatic context management. The provider forwards the reasoning that remains in the history it receives; it cannot restore content that was summarized or removed. Token counts used for proactive context management are estimates, not exact EXAONE token counts.

The returned response fields are authoritative. Do not assume the request setting alone guarantees that reasoning is absent from visible content.

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

The provider assembles streamed tool-call arguments before exposing calls to Strands. Blank tool-call arguments are treated as an empty object so zero-argument tools can run. If a completed tool call contains a JSON object with only closing braces or brackets missing, or a trailing comma, the provider completes that syntax without changing argument values. Unterminated strings, missing values, and other malformed arguments remain errors; the Harness validates arguments against each tool's schema. It accepts text, reasoning, tool calls, and text/JSON tool results. Unsupported content types and reasoning signatures present in the history received by the provider are rejected.

Because Chat Completions tool messages have no separate status field, tool results marked as errors are sent with an `[ERROR] ` prefix. Successful results are sent without a prefix.

Multiple system text blocks are concatenated without separators into one system message, matching Strands' TypeScript OpenAI Chat Completions adapter. Include separators in the blocks when needed.

A user message containing both text and tool results is split into tool messages followed by a user text message, matching Strands' OpenAI Chat Completions adapter so tool responses remain adjacent to their assistant tool calls.

JavaScript parses JSON numbers as `Number`, so integers outside the safe integer range can lose precision in tool arguments and history. Use a string schema for large identifiers or exact decimal values.

The provider preserves whitespace in text; Strands SDK aggregation omits text blocks that consist entirely of whitespace.

The provider accepts text and JSON content. When using Strands Harness, configure the built-in `read` tool to return text descriptions for images and binary documents:

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

These descriptions identify the file but do not provide image or document understanding. The `read` tool can access files permitted by the agent's sandbox, and its results are sent to the model endpoint. Use an isolated sandbox or an approval mechanism to restrict access to files you intend to share.

After an interrupted tool call, inspect external state before repeating a task that could have side effects.

Use separate session IDs for concurrently active agents or serialize access to shared file sessions.

Cancellation may not stop work already in progress. Use suitable tool timeouts and check external effects before retrying.

In SDK 1.24.0, a pending `HumanInTheLoop` approval callback can still approve and execute a tool after the invocation's cancellation signal is set. Coordinate approval and cancellation in your application; cancellation alone does not invalidate a pending approval.

Handle tool-hook errors explicitly, and make side-effecting tools idempotent when repeated actions would be harmful.

Built-in tools that summarize or delegate work may return child-agent results to the parent. Configure them carefully when generated reasoning is sensitive.

Disconnect caller-owned MCP clients when they are no longer needed.

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

## Retries

`ExaoneModel`'s own HTTP retry loop is disabled by default. For Agent and Harness use, configure `ExaoneRetryStrategy` as the Strands retry strategy; it replaces the SDK default and honors EXAONE's `Retry-After` and `Retry-After-Ms` headers. It defaults to three total attempts, a one-second initial backoff, and a 30-second delay cap. If a server hint exceeds the cap, it surfaces the error instead of retrying early. Without a usable hint, it uses bounded exponential backoff with jitter.

Strands owns Agent-level retry selection, so a model provider cannot silently replace the SDK's strategy. `ExaoneRetryStrategy` is opt-in: pass an instance to `Agent` or `createHarness` to replace the generic policy with EXAONE-aware waits. The quick start and Harness example show this explicit configuration.

Strands' default retry strategy does not inspect EXAONE headers. Keeping it alongside the provider's own retry loop can multiply attempts, so leave `maxRetries` at its default of zero when using `ExaoneRetryStrategy`. The strategy retries EXAONE transient HTTP errors and Strands network errors before the provider exposes output; it does not retry invalid model-generated tool arguments or other model response errors.

`ExaoneModel`'s optional `maxRetries` setting remains available for direct model calls that do not use an Agent retry strategy. It retries retryable HTTP responses before emitting model events. Fetch failures are not retried because the fetch API does not reliably say whether the request was received. The provider never replays a partially delivered stream.

For direct model calls, this configuration allows one provider retry with a maximum wait of 10 seconds:

```typescript
import { createHarness } from '@strands-agents/harness'
import { ExaoneModel } from 'strands-exaone'

const agent = await createHarness({
  model: new ExaoneModel({ maxRetries: 1, maxRetryDelayMs: 10_000 }),
  retryStrategy: null,
})
```

This disables the SDK's model retry strategy for that Agent. Cancellation interrupts provider retry waits; `timeoutMs` bounds provider attempts, provider retry waits, response reads, and stream consumption. Agent-level `ExaoneRetryStrategy` waits are outside the model timeout and follow Strands retry-strategy semantics. A custom `fetch` implementation must honor the supplied abort signal.

The provider preserves invalid model output as an error rather than rewriting argument values or silently dropping it. It completes only unambiguous tool-argument delimiters before Strands sees the call; remaining malformed arguments are rejected before a tool can execute. Structured-output validation errors should be handled through the Strands structured-output contract; automatic prompt rewriting or repeating a model-generated mistake is not enabled.

When stopping an agent stream early, use SDK cancellation APIs and let the model stream close; custom fetch implementations should honor the supplied abort signal.

## Metadata and errors

```typescript
import { ExaoneModel } from 'strands-exaone'

const model = new ExaoneModel({
  includeUsage: true,
  onMetadata: (metadata) => console.log(metadata.requestId, metadata.usage),
})
```

After a successful response, `onMetadata` receives the HTTP status, request ID, selected rate-limit headers, raw `usage`, and `response`. Raw usage can include `reasoning_tokens`. Available response metadata includes top-level IDs, model, service and weight metadata, `result_code`, and `description`; choice-level `finish_reason`, `matched_stop`, and `logprobs`; and selected message fields. Streaming logprob and annotation arrays are accumulated in order; refusal fragments are joined, while audio and legacy function-call fragments remain in `message_deltas`. Fields are present only when supplied by the service. The same metadata is also available in the Strands model metadata event under `trace.exaone`. The callback is not called if the response fails or is interrupted before completion.

Copy metric values that must remain stable across later invocations; framework metrics may be live snapshots.

The callback may be synchronous or asynchronous; the provider waits for it to finish. Cancellation or timeout stops waiting but cannot forcibly stop work the callback has already started. Callback failures are reported as `ModelError` with the original error as `cause`, so the SDK's default retry strategy does not retry a completed response because of a callback error.

**Metadata can contain user or generated content**, including logprob tokens, refusal text, audio, and function arguments. It is not size-limited and can grow with the response. API keys are redacted, but metadata and traces should only be logged or shared under an appropriate data policy. The provider does not copy ordinary assistant content, reasoning, or tool calls into response metadata.

HTTP errors expose `status`, `requestId`, selected rate-limit headers, sanitized `body`, and normalized `errorType`, `code`, and `param` fields on `ExaoneAPIError`. `errorType` and `param` come from the nested `error` object. `code` comes from `error.code` when that key is present, including when its value is `null`; otherwise it comes from the documented top-level `result_code`. Error messages use the sanitized `error.message`, then top-level `description`; HTTP failures fall back to `HTTP <status>`, while an error envelope in a successful response uses a generic provider message. HTTP 429 is surfaced as the Strands `ModelThrottledError`, with the `ExaoneAPIError` as its cause. An error in a successful HTTP response body or event stream raises `ExaoneResponseError`, whose `body`, `errorType`, `code`, and `param` contain sanitized response details.

If an HTTP error body cannot be read, `body` is `undefined`; the status, request ID, and available rate-limit headers remain accessible. Its cause is a generic read error that does not retain the transport's original message or request data.

For a failed `fetch` request, the provider redacts the API key from the transport error message and does not retain its original cause or request object. Other text in that message, including any prompt or request content a custom transport includes, is retained.

For HTTP 400 or 413, an exact `error.code` of `context_length_exceeded` is surfaced as the Strands `ContextWindowOverflowError`, with the `ExaoneAPIError` as its cause. Strands may compact history and retry in response. EXAONE's use of this OpenAI-compatible code has not been verified; ambiguous 400 and 413 responses remain `ExaoneAPIError` and do not trigger that overflow recovery.

## License

[MIT](LICENSE)
