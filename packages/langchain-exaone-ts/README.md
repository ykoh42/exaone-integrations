# langchain-exaone

Use the EXAONE Chat Completions API with LangChain.js. This community-maintained package is independent of LG AI Research and LangChain, and is not affiliated with or endorsed by either organization.

See the [changelog](CHANGELOG.md) for release notes.

## API key

Open the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview), select **Get an API Key**, and sign in or create an EXAONE ID to follow the platform's key-issuance instructions. Each user supplies their own key. The [illustrated setup guide](https://github.com/ykoh42/exaone-integrations/blob/main/guides/getting-started.md#get-an-exaone-api-key) shows the entry screens and includes Windows PowerShell instructions.

In the terminal where you will run your program, set the issued key:

```bash
export EXAONE_API_KEY='your-api-key'
```

## Install

```bash
npm install langchain-exaone @langchain/core
```

Requires Node.js 22 or newer and `@langchain/core` 1.3.0 or newer within the 1.x line. Set `EXAONE_API_KEY` in the process environment, or pass `apiKey` when constructing the model. The package does not load `.env` files automatically.

## Quick start

```typescript
import { ChatExaone } from 'langchain-exaone'

const model = new ChatExaone()
const response = await model.invoke('Explain what an AI agent is in one sentence.')

console.log(response.content)
```

The package uses `k-exaone_v2.5` and `https://api.exaone.ai/v1` by default. See the [examples](examples/) for LangChain agents, streaming chat, structured output, LangGraph tool loops, and Deep Agents.

LangChain provider catalogs use vendor names, while chat-model listings identify classes. `ChatExaone` follows the usual class-name pattern for a `BaseChatModel`. The `Chat` prefix is part of the class name, not the package name or model ID.

## Why use this integration?

A standard OpenAI chat model configured for EXAONE's endpoint can handle basic chat. `ChatExaone` also carries `reasoning_content` on assistant messages and sends it on later turns. It defaults to `preserveThinking: true` and exposes known model facts through LangChain's standard profile. Usage and response metadata appear on LangChain messages, and raised API errors include diagnostic fields.

Retries are bounded, follow server delay hints, and stop before replaying a stream that has started. Structured-output binding and tool execution follow LangChain's contracts. Other adapters may provide similar behavior through configuration or extensions; this package keeps the EXAONE mapping in one tested integration. See the [project comparison](https://github.com/ykoh42/exaone-integrations#why-use-a-dedicated-provider) for scope and trade-offs.

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `model` | EXAONE model ID. | `k-exaone_v2.5` |
| `baseUrl` | Chat Completions API base URL, including `/v1`. Remote endpoints must use HTTPS; HTTP is allowed for loopback test servers. | `https://api.exaone.ai/v1` |
| `temperature`, `topP`, `topK` | Sampling settings. | Service defaults |
| `maxTokens` | Maximum generated tokens. | Service default |
| `frequencyPenalty`, `presencePenalty`, `seed` | Generation settings. | Service defaults |
| `reasoningEffort` | `low`, `medium`, `high`, or `max`. | Service default |
| `preserveThinking` | Preserve earlier reasoning in multi-turn conversations. | `true` |
| `enableThinking` | Set `chat_template_kwargs.enable_thinking` for compatible custom models. K-EXAONE 2.5 always reasons. | Service default |
| `timeout` | Request timeout in milliseconds. | No provider-imposed timeout |
| `maxRetries` | Retries for 408, 429, 500, 502, 503, and 504 responses. | `2` |
| `baseRetryDelayMs`, `maxRetryDelayMs` | Retry backoff bounds. | `1000`, `60000` |
| `streamUsage` | Request usage in the final stream event. | `true` |
| `modelKwargs` | Additional Chat Completions request fields. | None |
| `fetch` | Custom fetch implementation. | Global `fetch` |

Explicit call options use LangChain's snake_case request names, such as `max_tokens`, `top_p`, `reasoning_effort`, and `chat_template_kwargs`. `modelKwargs` is passed through for endpoint-specific fields. The provider owns `model`, `messages`, and `stream`; do not set them in `modelKwargs`.

The provider does not impose a short generation timeout. Set `timeout` only when the application has an appropriate deadline for its prompts and reasoning budget. Call-level cancellation uses LangChain's `signal` option.

The default retry policy retries only before it has received a response. It respects valid `Retry-After` and `Retry-After-Ms` values and does not retry when the requested wait exceeds `maxRetryDelayMs`. A hintless 429 waits at least one second; if the configured retry cap is below one second, that 429 is returned without retry. The provider never replays an already-started stream.

`streamEvents()` exposes streamed `reasoning_content` through LangChain's typed reasoning stream and preserves text as a separate text stream.

EXAONE API defaults and limits can change; see the [EXAONE K-EXAONE 2.5 API reference](https://docs.exaone.ai/ko/docs/k-exaone-2.5) and [rate limits](https://docs.exaone.ai/ko/docs/rate-limits).

## Model profile

`new ChatExaone().profile` provides LangChain's standard model profile for `k-exaone_v2.5`: a 524,288-token context budget, reasoning, tool calling and choice. Image, audio, video, and PDF inputs and non-text outputs are unsupported. The combined input and output, including reasoning, must fit within the context budget; no independent `maxOutputTokens` profile value is inferred.

LangChain and Deep Agents can use this profile for context management and structured-output strategy selection. An unknown model ID returns an empty profile. Pass `profile: {...}` to replace the defaults; `profile: {}` leaves capabilities unspecified. Profiles are copied on construction and access, and are not sent as API parameters.

The default profile leaves `structuredOutput` unspecified because native JSON Schema output combined with tool calls in one request has not been established as compatible with LangChain's provider strategy. Agents use LangChain's tool-based strategy by default. To request native output without tools, select `providerStrategy` explicitly. This choice does not change the provider's request parameters or structured-output methods.

See [LangChain model profiles](https://docs.langchain.com/oss/javascript/langchain/models#model-profiles) and the [EXAONE API specification](https://docs.exaone.ai/ko/docs/k-exaone-2.5).

## Reasoning and conversation history

When EXAONE returns `reasoning_content`, the integration stores it in `AIMessage.additional_kwargs.reasoning_content` and sends it back on later requests that include the message. Requests set `chat_template_kwargs.preserve_thinking=true` by default. Set `preserveThinking: false` to turn that option off. Keep the returned AI message in conversation history:

```typescript
import { HumanMessage } from '@langchain/core/messages'
import { ChatExaone } from 'langchain-exaone'

const model = new ChatExaone()
const first = await model.invoke('Compare two approaches and recommend one.')
const followUp = await model.invoke([
  new HumanMessage('Compare two approaches and recommend one.'),
  first,
  new HumanMessage('Now give me the recommendation in one sentence.'),
])

console.log(followUp.content)
```

The provider forwards reasoning found in the history it receives. It cannot restore reasoning that application memory or a chain has removed or summarized. Reasoning remains in `additional_kwargs.reasoning_content`; it is not merged into the answer text.

LangChain's `outputVersion: "v1"` exposes reasoning and tool calls as standard content blocks. Keep the returned message, or a completed aggregation of stream chunks, to include those blocks in a later conversation.

Like LangChain's OpenAI integration, a normal `invoke()` returns an `AIMessage`; streaming yields `AIMessageChunk` objects. The current `BaseChatModel` TypeScript signature types `invoke()` as returning a chunk, so use chunk-specific methods such as `concat()` on streamed chunks only.

## Tools

`bindTools()` accepts LangChain tools and OpenAI-shaped function tool definitions. The response exposes parsed `tool_calls` and the raw response metadata. The [LangChain agent example](examples/langchain/agent.ts) runs tools and returns their results through LangChain's agent loop.

When a returned EXAONE message is reused in history, edits to its LangChain `tool_calls` determine the next request. Provider-specific fields from matching original calls are retained.

`withStructuredOutput()` uses the `functionCalling` method: it requests its named tool and uses LangChain's parser. Zod schemas are validated, and `{ includeRaw: true }` includes the original message when parsing fails. The `jsonMode` and `jsonSchema` methods are not supported by this wrapper. Invalid tool arguments remain available in `invalid_tool_calls`, and the original calls are retained in response metadata for conversation history.

The integration also preserves legacy `function_call` fields in assistant history. It does not execute tools by itself.

## API compatibility

This package sends requests directly to the EXAONE Chat Completions endpoint. It supports text messages, assistant reasoning history, function tools, sampling settings, JSON response-format passthrough, streaming, usage, response metadata, and HTTP errors. The service decides which forwarded tool definitions and optional generation fields it accepts.

The integration accepts text and reasoning input blocks. It rejects image, audio, video, and other multimodal content. Non-streaming `generate()` returns every candidate when `modelKwargs.n` requests multiple choices; `invoke()` returns the first candidate. Each candidate carries the same request-level usage, so use `generate()`'s `llmOutput` for totals rather than adding candidate usage together. Streaming raises an error if the endpoint returns multiple choices, since LangChain's stream represents one message. When returned, `logprobs` and other response-specific fields are kept in metadata; the integration does not expose token-level log probabilities as a separate abstraction.

## Development

From this package directory, install and verify this workspace independently:

```bash
npm ci --ignore-scripts --workspaces=false
npm run check --workspaces=false
```

`check` runs lint, TypeScript type checking, package build, and synthetic fixture tests. The tests do not call the live EXAONE API.
