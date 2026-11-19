# langchain-exaone

Use the EXAONE Chat Completions API with LangChain.js. This community-maintained package is independent of LG AI Research and LangChain, and is not affiliated with or endorsed by either organization.

See the [changelog](CHANGELOG.md) for release notes.

## API key

Get a key from the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview) and set it in your environment. See the [setup guide](https://github.com/ykoh42/exaone-integrations/blob/main/guides/getting-started.md#set-your-api-key) for Windows PowerShell.

```bash
export EXAONE_API_KEY='your-api-key'
```

## Quick start

Use Node.js 22 or newer for the Deep Agents example.

```bash
npm install langchain-exaone 'deepagents@~1.17.0' langchain @langchain/core
```

```typescript
import { createDeepAgent } from 'deepagents'
import { ChatExaone } from 'langchain-exaone'

const agent = await createDeepAgent({ model: new ChatExaone() })
const result = await agent.invoke({
  messages: [{ role: 'user', content: 'Explain what an AI agent is in one sentence.' }],
})
console.log(result.messages.at(-1)?.text)
```

See the [Deep Agents example](examples/deepagents/) for tool use and delegation. For the model provider alone, install `langchain-exaone @langchain/core` and see the [LangChain examples](examples/langchain/). The provider supports Node.js 20.3 and newer and `@langchain/core` 1.2 and newer within 1.x.

Set `EXAONE_API_KEY` in the process environment or pass `apiKey` to `ChatExaone`. Load any application `.env` file before constructing the model.

## Configuration

| Option | Description | Default |
| --- | --- | --- |
| `model` | EXAONE model ID. | `k-exaone_v2.5` |
| `baseUrl` | HTTP or HTTPS Chat Completions API base URL, including `/v1`. | `https://api.exaone.ai/v1` |
| `temperature`, `topP`, `topK` | Sampling settings. | Service defaults |
| `maxTokens` | Maximum generated tokens. | Service default |
| `frequencyPenalty`, `presencePenalty`, `seed` | Generation settings. | Service defaults |
| `reasoningEffort` | Reasoning setting accepted by the selected endpoint. | Service default |
| `preserveThinking` | Request preservation of earlier assistant reasoning in the model input. | `true` |
| `enableThinking` | Set `chat_template_kwargs.enable_thinking` for compatible custom models. K-EXAONE 2.5 always reasons. | Service default |
| `timeout` | Request timeout in milliseconds. | No provider-imposed timeout |
| `maxRetries` | Retries for 408, 429, 500, 502, 503, and 504 responses. | `2` |
| `baseRetryDelayMs`, `maxRetryDelayMs` | Retry backoff bounds. | `1000`, `60000` |
| `streamUsage` | Request usage in the final stream event. | `true` |
| `modelKwargs` | Additional Chat Completions request fields. | None |
| `fetch` | Custom fetch implementation. | Global `fetch` |

Explicit call options use LangChain's snake_case request names, such as `max_tokens`, `top_p`, `reasoning_effort`, and `chat_template_kwargs`. `modelKwargs` is passed through for endpoint-specific fields. The provider owns `model`, `messages`, and `stream`; do not set them in `modelKwargs`.

For an already-running K-EXAONE 2.0 endpoint, set `baseUrl` and `model` to the server's URL and served model name. If supported by the server, `enableThinking: false` requests non-reasoning mode. The provider accepts `reasoning` and `reasoning_content` response fields and preserves them as `reasoning_content` for later turns. Identical values are emitted once; conflicting nonempty values are rejected.

See the [model-card serving configuration](https://github.com/ykoh42/exaone-integrations/blob/main/COMPATIBILITY.md#server-versions-and-parser-configurations).

Some servers accept `reasoning_effort` and may use it to select thinking on or off. Accepting the field does not establish that the model supports distinct reasoning-intensity levels; check the deployed server and model documentation.

```typescript
const model = new ChatExaone({
  baseUrl: 'http://127.0.0.1:8000/v1',
  model: 'K-EXAONE-2.0-750B-A37B', // Use the served name.
  apiKey: 'EMPTY', // Use the server's configured credential, if any.
})
```

The configured API key is sent to the selected endpoint; if omitted, it comes from `EXAONE_API_KEY`. For a server that does not require authentication, pass a placeholder such as `apiKey: 'EMPTY'` rather than the hosted API key. HTTP does not encrypt credentials or conversation data; use HTTPS or a trusted transport.

The provider does not impose a short generation timeout. Set `timeout` only when the application has an appropriate deadline for its prompts and reasoning budget. Call-level cancellation uses LangChain's `signal` option.

The default retry policy retries only before it has received a response. It respects valid `Retry-After` and `Retry-After-Ms` values and does not retry when the requested wait exceeds `maxRetryDelayMs`. A hintless 429 waits at least one second; if the configured retry cap is below one second, that 429 is returned without retry. The provider never replays an already-started stream.

`streamEvents()` exposes streamed `reasoning_content` through LangChain's typed reasoning stream and preserves text as a separate text stream.

EXAONE API defaults and limits can change; see the [EXAONE K-EXAONE 2.5 API reference](https://docs.exaone.ai/ko/docs/k-exaone-2.5) and [rate limits](https://docs.exaone.ai/ko/docs/rate-limits).

## Model profile

`new ChatExaone().profile` provides LangChain's standard model profile for `k-exaone_v2.5`: a 524,288-token context budget, reasoning, tool calling and choice. Image, audio, video, and PDF inputs and non-text outputs are unsupported. The combined input and output, including reasoning, must fit within the context budget; no independent `maxOutputTokens` profile value is inferred.

LangChain and Deep Agents can use this profile for context management and structured-output strategy selection. A custom endpoint or unknown model ID returns an empty profile. Pass `profile: {...}` to replace the defaults; `profile: {}` leaves capabilities unspecified. Profiles are copied on construction and access, and are not sent as API parameters.

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

For v1 messages, reasoning blocks in `content` are authoritative: removing them prevents that reasoning from being sent again. For older message formats, the provider also reads `additional_kwargs.reasoning_content`.

Refusal text is available in `additional_kwargs.refusal` and as a `refusal` content block in v1 messages. When reusing an assistant message, the provider sends its answer text or falls back to the refusal text if the answer is empty.

Like LangChain's OpenAI integration, a normal `invoke()` returns an `AIMessage`; streaming yields `AIMessageChunk` objects. The current `BaseChatModel` TypeScript signature types `invoke()` as returning a chunk, so use chunk-specific methods such as `concat()` on streamed chunks only.

## Tools

`bindTools()` accepts LangChain tools and OpenAI-shaped function tool definitions. The response exposes parsed `tool_calls` and the raw response metadata. The [LangChain agent example](examples/langchain/agent.ts) runs tools and returns their results through LangChain's agent loop.

Pass `{ strict: true }` or `{ strict: false }` to `bindTools()` to set `strict` on every function tool. Omit it to preserve each tool's own setting. Support and enforcement depend on the endpoint.

When a returned EXAONE message is reused in history, edits to its LangChain `tool_calls` determine the next request. Provider-specific fields from matching original calls are retained.

`withStructuredOutput()` defaults to `functionCalling`: it requests its named tool and uses LangChain's parser. Set `{ method: "jsonSchema" }` for native JSON Schema output or `{ method: "jsonMode" }` for a JSON object. With JSON mode, ask for JSON in the prompt. Zod schemas are validated; a plain JSON Schema object is parsed as JSON without additional schema validation, following LangChain's parser. `{ includeRaw: true }` retains the original message when parsing fails. Invalid non-streamed tool arguments remain available in `invalid_tool_calls`, and the original calls are retained in response metadata for conversation history.

Malformed non-streamed JSON includes an error and, when the runtime reports one, a UTF-16 offset. This does not automatically send the error back to the model or retry the call; that behavior depends on the agent.

### Optional malformed-tool recovery

Use this optional middleware when malformed JSON tool arguments interrupt agent runs. The provider works without this middleware. Try `--recover-tool-calls` in the [Deep Agents example](examples/deepagents/#optional-tool-call-recovery).

```typescript
import { createDeepAgent } from 'deepagents'
import { createMiddleware } from 'langchain'
import { ChatExaone, toolCallRecovery } from 'langchain-exaone'

const agent = await createDeepAgent({
  model: new ChatExaone(),
  tools: [], // Add your LangChain tools here.
  middleware: [createMiddleware(toolCallRecovery())],
})
```

Text and reasoning stream as they arrive. Tool arguments are checked before they are sent to the agent. If a single completed call contains broken JSON, the provider requests a correction and emits only the validated arguments under the original tool call ID.

Recovery sends the original arguments and parser diagnostic in a matched tool-error message. The preceding assistant tool-call arguments use an explicitly identified `{}` placeholder. The tool runs only after the model returns a valid call. The system prompt, user messages and tool definitions are retained. An incomplete, filtered, or refused response cannot run a tool. Responses with multiple choices or tool calls are rejected if any call has malformed JSON. Server acceptance of this recovery exchange and successful correction depend on the endpoint and model.

If LangChain's response cache is configured, the middleware uses a separate cache key for validated responses. The key is local to LangChain and is omitted from API requests.

Streaming recovery uses the effective EXAONE request after SDK middleware has applied tool selection. Streamed usage includes both attempts when the server reports both. With non-streaming model calls, the correction also passes through SDK middleware. Register recovery after routing middleware that switches providers or makes its own model calls, so the correction applies to the selected EXAONE model call.

The tool result contains `raw_arguments` and `parse_error`. Diagnostic line and column numbers refer to the original argument text.

Only the correction request selects the failed tool by name. If the endpoint explicitly rejects named tool choice before emitting a response, the provider tries `tool_choice="required"` once. Recovery succeeds only with a completed reply containing exactly one call to the original tool with a valid JSON object as its arguments. The SDK still validates the tool schema before execution.

Empty or whitespace-only arguments follow LangChain's existing behavior and do not trigger recovery. LangChain may treat an empty streamed input as `{}` and run a zero-argument or defaulted tool; an empty input alone does not show whether the server omitted arguments. The middleware still rejects tool calls ending due to length.

The recovered assistant message includes `response_metadata.exaone_tool_call_recovery`. For streams, `usage_metadata` combines the reported counts from both requests and the recovery metadata records `failed_usage` and `correction_usage` separately. `response_metadata.usage` belongs to the original response. For non-streaming calls, LangChain can report the failed attempt through error callbacks and tracing, and the returned message's usage describes the successful request. Tool-result feedback and tracing may contain model-generated arguments.

LangChain can parse incomplete streamed JSON as valid tool arguments. If you dispatch tools yourself, reject `length` finishes and check the assembled `tool_call_chunks[].args` with `JSON.parse`.

Use `tool_calls` for hosted EXAONE 2.5. The provider can also forward legacy `function_call` and `FunctionMessage` history to compatible endpoints.

## API compatibility

The provider accepts function tools. The endpoint controls support for forwarded tool options and generation settings.

The integration accepts text and reasoning blocks and, for compatible self-hosted endpoints, user image blocks. Audio, video, image tool messages, and other unsupported content are rejected. Non-streaming `generate()` returns every candidate when `modelKwargs.n` requests multiple choices; `invoke()` returns the first candidate. Each candidate carries the same request-level usage, so use `generate()`'s `llmOutput` for totals rather than adding candidate usage together. Streaming raises an error if the endpoint returns multiple choices, since LangChain's stream represents one message. When returned, `logprobs` and other response-specific fields are kept in metadata; the integration does not expose token-level log probabilities as a separate abstraction.

For an already-running EXAONE 4.5 endpoint, set `baseUrl`, `model`, and `apiKey` to its connection details and use LangChain's image blocks:

```typescript
import { HumanMessage } from '@langchain/core/messages'
import { ChatExaone } from 'langchain-exaone'

const model = new ChatExaone({ baseUrl: 'http://127.0.0.1:8000/v1', apiKey: 'EMPTY', model: 'EXAONE-4.5-33B' })
const message = new HumanMessage({ content: [
  { type: 'text', text: 'Describe this image.' },
  { type: 'image', url: 'https://example.com/image.png' },
] })
const response = await model.invoke([message])
```

User images support URLs and base64 data with an image MIME type; the provider sends text and image blocks in their original order. File IDs are unsupported. The default hosted K-EXAONE 2.5 model is text-only.

See the [compatibility matrix](https://github.com/ykoh42/exaone-integrations/blob/main/COMPATIBILITY.md#known-server-side-observations) for server options and known limitations.

## Development

From this package directory, install and verify this workspace independently:

```bash
npm ci --ignore-scripts --workspaces=false
npm run check --workspaces=false
```

`check` runs lint, TypeScript type checking, package build, and synthetic fixture tests. The tests do not call the live EXAONE API.
