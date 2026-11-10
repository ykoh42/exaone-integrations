# strands-exaone for TypeScript

Use the EXAONE Chat Completions API as a model provider for [Strands Agents](https://strandsagents.com/). This community-maintained package is not affiliated with LG AI Research or the Strands Agents project.

## Quick start

The package requires Node.js 22 or newer and uses `@strands-agents/sdk` and `zod` as peer dependencies. It has not been published to npm. Use the Node.js version in `.node-version` for the checkout's development tools. From the checkout root, install the workspace and build the package:

```bash
npm ci
npm run build --workspace=strands-exaone
export EXAONE_API_KEY='your-api-key'
```

```typescript
import { Agent } from '@strands-agents/sdk'
import { ExaoneModel } from 'strands-exaone'

const model = new ExaoneModel({ preserveThinking: true })
const agent = new Agent({ model })
const response = await agent.invoke('Explain what an AI agent is in one sentence.')
console.log(String(response))
```

The provider reads `EXAONE_API_KEY` when a request is made. You can instead pass `apiKey` to the constructor. The package does not load `.env` files automatically.

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `modelId` | EXAONE model ID. | `k-exaone_v2.5` |
| `baseURL` | API base URL, including `/v1`. | `https://api.exaone.ai/v1` |
| `stream` | Request a streaming response; `false` adapts a non-streaming response to Strands events. | `true` |
| `maxTokens`, `temperature`, `topP`, `topK`, `presencePenalty` | Generation parameters. | Service defaults when omitted |
| `reasoningEffort` | EXAONE reasoning setting. | Service default |
| `preserveThinking`, `enableThinking` | EXAONE `chat_template_kwargs` for previous reasoning and reasoning generation. | Service defaults |
| `includeUsage` | Set `stream_options.include_usage` on streaming requests. | Omitted |
| `timeoutMs` | Total timeout in milliseconds, including attempts, waits, and stream consumption. | No provider-imposed timeout |
| `contextWindowLimit` | Context window reported to Strands, in tokens. | `524288` |
| `maxRetries`, `baseRetryDelayMs`, `maxRetryDelayMs` | Optional retry policy. | `0` retries; `500` and `30000` milliseconds |
| `onMetadata` | Callback for successful request metadata. | None |
| `fetch` | Custom fetch implementation. | Global `fetch` |
| `params` | Additional Chat Completions parameters. | None |

Explicit options take precedence over matching `params` values. The provider manages `model`, `messages`, `stream`, `stream_options`, `tools`, and `tool_choice`; do not set them in `params`. Strands represents one completion, so `n` must be 1. Supported generation settings can vary by service version; see the [EXAONE API documentation](https://docs.exaone.ai/en/docs/k-exaone-2.5).

## Reasoning and tools

Reasoning is stored separately from the final answer and sent back as `reasoning_content` when it is present in the Strands conversation. Set `preserveThinking: true` to request preservation of previous reasoning. `enableThinking` is a separate setting; its observed behavior has differed from the documentation, so verify it for your use case.

In one live request with `enableThinking: false`, the API returned `reasoning_tokens=0` and no separate `reasoning_content`, but the response `content` contained a `<think>` or `</think>` tag. A separate request with `enableThinking: true` returned both reasoning tokens and separate reasoning content. These individual observations do not establish a general rule: `false` does not guarantee that reasoning will be absent from the visible response.

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel } from 'strands-exaone'

const getNumber = tool({
  name: 'get_number',
  description: 'Return the example number.',
  inputSchema: z.object({}),
  callback: () => '7',
})

const agent = new Agent({ model: new ExaoneModel({ preserveThinking: true }), tools: [getNumber] })
await agent.invoke('Use the tool to get the number, then add 3.')
```

The provider assembles streamed tool-call arguments before exposing calls to Strands. It accepts text, reasoning, tool calls, and text/JSON tool results; unsupported content types and reasoning signatures are rejected.

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

The provider maps structured-output requests to JSON Schema or a forced tool call when Strands requests one. EXAONE returned `finish_reason=abort` on some observed structured-output requests; the cause is not established. The provider reports an error instead of treating an aborted result as valid. Schema support may vary with the service and schema.

## Retries

Retries are disabled by default. Set `maxRetries: 2` to allow two additional attempts for HTTP 429, 500, 502, 503, and 504. Fetch failures are not retried: the fetch API does not reliably say whether a failed request was already sent. The provider never retries after a successful response starts or replays a partially delivered stream.

The provider honors `retry-after` in seconds or HTTP-date form and `retry-after-ms`, using the longer valid delay when both are present. If the server's delay exceeds `maxRetryDelayMs`, it returns the original error rather than retrying early. Without a usable header, it uses capped exponential backoff with jitter. Cancellation interrupts requests and retry waits. `timeoutMs` covers the entire operation, including retries, waits, and stream consumption. Strands or application-level retries are separate.

## Metadata and errors

```typescript
import { ExaoneModel } from 'strands-exaone'

const model = new ExaoneModel({
  includeUsage: true,
  onMetadata: (metadata) => console.log(metadata.requestId, metadata.usage),
})
```

On successful completion, `onMetadata` receives the request ID, selected rate-limit headers, raw `usage`, and `response`. Raw usage can include `reasoning_tokens`. Available response metadata includes top-level IDs, model, service and weight metadata; choice-level `finish_reason`, `matched_stop`, and `logprobs`; and selected message fields. Streaming logprob and annotation arrays are accumulated in order; refusal fragments are joined, while audio and legacy function-call fragments remain in `message_deltas`. Fields are present only when supplied by the service. The same metadata is also available in the Strands model metadata event under `trace.exaone`. Interrupted streams do not produce completion metadata.

**Metadata can contain user or generated content**, including logprob tokens, refusal text, audio, and function arguments. It is not size-limited and can grow with the response. API keys are redacted, but metadata and traces should only be logged or shared under an appropriate data policy. The provider does not copy ordinary assistant content, reasoning, or tool calls into response metadata.

HTTP errors expose the status, request ID, selected rate-limit headers, and sanitized body on `ExaoneAPIError`. HTTP 429 is surfaced as the Strands `ModelThrottledError`, with the `ExaoneAPIError` as its cause.

## License

[MIT](LICENSE)
