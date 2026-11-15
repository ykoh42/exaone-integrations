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

Requires Node.js 20.3.0 or newer and `@langchain/core` 1.2.0 or newer within the 1.x line. Set `EXAONE_API_KEY` in the process environment, or pass `apiKey` when constructing the model. The package does not load `.env` files automatically.

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
| `baseUrl` | HTTP or HTTPS Chat Completions API base URL, including `/v1`. | `https://api.exaone.ai/v1` |
| `temperature`, `topP`, `topK` | Sampling settings. | Service defaults |
| `maxTokens` | Maximum generated tokens. | Service default |
| `frequencyPenalty`, `presencePenalty`, `seed` | Generation settings. | Service defaults |
| `reasoningEffort` | Reasoning setting accepted by the selected endpoint. | Service default |
| `preserveThinking` | Preserve earlier reasoning in multi-turn conversations. | `true` |
| `enableThinking` | Set `chat_template_kwargs.enable_thinking` for compatible custom models. K-EXAONE 2.5 always reasons. | Service default |
| `timeout` | Request timeout in milliseconds. | No provider-imposed timeout |
| `maxRetries` | Retries for 408, 429, 500, 502, 503, and 504 responses. | `2` |
| `baseRetryDelayMs`, `maxRetryDelayMs` | Retry backoff bounds. | `1000`, `60000` |
| `streamUsage` | Request usage in the final stream event. | `true` |
| `modelKwargs` | Additional Chat Completions request fields. | None |
| `fetch` | Custom fetch implementation. | Global `fetch` |

Explicit call options use LangChain's snake_case request names, such as `max_tokens`, `top_p`, `reasoning_effort`, and `chat_template_kwargs`. `modelKwargs` is passed through for endpoint-specific fields. The provider owns `model`, `messages`, and `stream`; do not set them in `modelKwargs`.

For an already-running K-EXAONE 2.0 endpoint, set `baseUrl` and `model` to the server's URL and served model name. If supported by the server, `enableThinking: false` requests non-reasoning mode. The provider accepts `reasoning` and `reasoning_content` response fields and preserves them as `reasoning_content` for later turns. Identical values are emitted once; conflicting nonempty values are rejected. Protocol fixtures have been tested; actual K-EXAONE 2.0 inference has not been verified by this project.

For serving instructions and parser settings, see the [model card](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B). Its examples use forked servers; [vLLM 0.27.0](https://github.com/vllm-project/vllm/releases/tag/v0.27.0) also includes K-EXAONE 2.0 model support. Verify the specific server version and configuration you deploy.

The SGLang `qwen3_coder` parser in the inspected [model-card fork](https://github.com/lkm2835/sglang/blob/918653d22b2f1ae853b81fc0f5539b7c9311487e/python/sglang/srt/function_call/qwen3_coder_detector.py#L104) and [upstream revision](https://github.com/sgl-project/sglang/blob/d9ec74eb72b856419c6eb44733e00193d28ed840/python/sglang/srt/function_call/qwen3_coder_detector.py#L112) converts the text `null` to JSON null even for a string parameter. This can cause tool validation to fail; check your server's parser version.

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

Refusal text is kept separately in `additional_kwargs.refusal`, and v1 messages also expose a `refusal` content block. It is not merged into answer text. When reusing an assistant message that has no answer text, the provider sends its refusal as text in the next request: a refusal content block in message history was rejected by the hosted 2.5 endpoint in a compatibility check. If answer text is present, that text is sent instead.

Like LangChain's OpenAI integration, a normal `invoke()` returns an `AIMessage`; streaming yields `AIMessageChunk` objects. The current `BaseChatModel` TypeScript signature types `invoke()` as returning a chunk, so use chunk-specific methods such as `concat()` on streamed chunks only.

## Tools

`bindTools()` accepts LangChain tools and OpenAI-shaped function tool definitions. The response exposes parsed `tool_calls` and the raw response metadata. The [LangChain agent example](examples/langchain/agent.ts) runs tools and returns their results through LangChain's agent loop.

When a returned EXAONE message is reused in history, edits to its LangChain `tool_calls` determine the next request. Provider-specific fields from matching original calls are retained.

`withStructuredOutput()` defaults to `functionCalling`: it requests its named tool and uses LangChain's parser. Set `{ method: "jsonSchema" }` for native JSON Schema output or `{ method: "jsonMode" }` for a JSON object. With JSON mode, ask for JSON in the prompt. Zod schemas are validated; a plain JSON Schema object is parsed as JSON without additional schema validation, following LangChain's parser. `{ includeRaw: true }` retains the original message when parsing fails. Invalid tool arguments remain available in `invalid_tool_calls`, and the original calls are retained in response metadata for conversation history.

The integration also preserves legacy `function_call` fields and LangChain `FunctionMessage` results in conversation history for compatible endpoints. A synthetic history containing both fields and a `function` role returned HTTP 400 from the default EXAONE endpoint; use the tool-calling protocol for EXAONE. The integration does not execute tools by itself.

## API compatibility

This package sends requests directly to the EXAONE Chat Completions endpoint. It supports text messages, assistant reasoning history, function tools, sampling settings, JSON response-format passthrough, streaming, usage, response metadata, and HTTP errors. Other response tool-call types cannot be represented as function tools and are rejected. The service decides which forwarded tool definitions and optional generation fields it accepts.

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

User images support URLs and base64 data with an image MIME type; the provider sends text and image blocks in their original order. File IDs are unsupported. The default hosted K-EXAONE 2.5 model is text-only. The [EXAONE 4.5 model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) shows vLLM and an SGLang fork with `qwen3` reasoning and `hermes` tool parsers. Configure these on the server. Image requests have been tested with synthetic protocol fixtures; actual EXAONE 4.5 inference has not been verified.

### Server-specific observations

The provider preserves text/image order in the **request sent to the endpoint**. The provider does not pre-check whether the endpoint supports images; the server may reject them. What reaches the model depends on the server's message normalization, chat template, and configuration. The observations below come from synthetic inputs and isolated server code; no running inference server or EXAONE 4.5 model was tested.

- **vLLM:** With the official 4.5 template and vLLM 0.27.0's automatic content-format detection, an isolated message-normalization test converted `text → image → text` to `image → text → text` with the default `interleave_mm_strings=False`. Setting [`--interleave-mm-strings`](https://docs.vllm.ai/en/v0.27.0/cli/serve/#interleave-mm-strings) preserved the order in that test. Verify the setting and behavior on your server; this is not a real-image inference result.
- **SGLang image/tool history:** In an isolated test using the model-card fork's message normalization and the 4.5 template at revision `570aa4b15a4f45ba1133072b45f50198f6e3b4fd`, an image-containing first user message followed by an assistant tool call and tool result caused earlier assistant reasoning to be omitted from the rendered prompt, even though the request contained it. A preceding string-valued user message changed the result. Other templates or server settings may behave differently.
- **SGLang streaming tools:** With automatic tool choice, an isolated execution of the [model-card fork at `6757c9f`](https://github.com/lkm2835/sglang/tree/6757c9f904cdb8ae9028a394a2108d079b9e088c)'s streaming handler and `hermes` parser emitted a tool name without its arguments when an entire synthetic tool call and the stop arrived in the same final event. The parser-level condition also reproduced on SGLang 0.5.23; this does not verify that release's EXAONE 4.5 model support. Depending on the framework and tool schema, missing arguments can cause validation to fail or a tool to run with default values. See [SGLang issue #31915](https://github.com/sgl-project/sglang/issues/31915) and [proposed fix #32167](https://github.com/sgl-project/sglang/pull/32167). Actual server chunking and inference were not tested.

These server behaviors are not repaired by the provider. Check the exact server version, template, and settings you deploy.

## Development

From this package directory, install and verify this workspace independently:

```bash
npm ci --ignore-scripts --workspaces=false
npm run check --workspaces=false
```

`check` runs lint, TypeScript type checking, package build, and synthetic fixture tests. The tests do not call the live EXAONE API.
