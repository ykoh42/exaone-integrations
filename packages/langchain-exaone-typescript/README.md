# langchain-exaone

Use the EXAONE Chat Completions API with LangChain.js. This community-maintained integration is not affiliated with LG AI Research or the LangChain project.

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

The package uses `k-exaone_v2.5` and `https://api.exaone.ai/v1` by default. A minimal example with preserved reasoning is available in [`examples/basic.ts`](examples/basic.ts).

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
| `enableThinking` | Set `chat_template_kwargs.enable_thinking`. | Service default |
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

## Reasoning and conversation history

When EXAONE returns `reasoning_content`, the integration stores it in `AIMessage.additional_kwargs.reasoning_content`. It sends that value back as `reasoning_content` if the message is included in a later request. Requests also set `chat_template_kwargs.preserve_thinking=true` by default; set `preserveThinking: false` to disable that option. Keep the returned AI message in conversation history to preserve it:

```typescript
import { HumanMessage } from '@langchain/core/messages'
import { ChatExaone } from 'langchain-exaone'

const model = new ChatExaone({ enableThinking: true })
const first = await model.invoke('Compare two approaches and recommend one.')
const followUp = await model.invoke([
  new HumanMessage('Compare two approaches and recommend one.'),
  first,
  new HumanMessage('Now give me the recommendation in one sentence.'),
])

console.log(followUp.content)
```

The provider forwards reasoning present in the message history it receives. It cannot recover reasoning removed or summarized by application memory or a chain. Reasoning stays in the provider-specific `additional_kwargs.reasoning_content` field and is not merged into the answer text.

## Tools

`bindTools()` accepts LangChain tools and OpenAI-shaped function tool definitions. The response exposes parsed `tool_calls` and the raw response metadata. The [tool example](examples/tool-call.ts) shows how to inspect a model-requested tool call; your application or agent executor is responsible for running tools and returning `ToolMessage` results.

The integration also preserves legacy `function_call` fields in assistant history. It does not execute tools by itself.

## API compatibility

This package sends requests directly to the EXAONE Chat Completions endpoint. It supports text messages, assistant reasoning history, function tools, sampling settings, JSON response-format passthrough, streaming, usage, response metadata, and HTTP errors. Tool definitions and optional generation fields are forwarded to the service; the service remains authoritative about which combinations it accepts.

The integration rejects input content blocks other than text and reasoning. Image, audio, video, and other multimodal requests are not supported. LangChain chat models represent one completion per run, so `n` must be `1`. `logprobs` and other response-specific fields are retained in metadata where returned; the integration does not provide a token-level logprob abstraction.

## Development

From this package directory, install and verify this workspace independently:

```bash
npm ci --ignore-scripts --workspaces=false
npm run check --workspaces=false
```

`check` runs lint, TypeScript type checking, package build, and synthetic fixture tests. The tests do not call the live EXAONE API.
