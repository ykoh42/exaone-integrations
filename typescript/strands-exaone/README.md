# strands-exaone

A Strands Agents model provider for the EXAONE Chat Completions API. It supports K-EXAONE reasoning, tools, and both streaming and non-streaming responses. Node.js 22 or newer is required.

This package is being prepared for release and has not yet been published. Once published, install it with its peer dependencies:

```sh
npm install strands-exaone @strands-agents/sdk zod
export EXAONE_API_KEY='your-api-key'
```

```ts
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'
import { ExaoneModel } from 'strands-exaone'

const lookupNumber = tool({
  name: 'lookup_number',
  description: 'Look up the current example number.',
  inputSchema: z.object({}),
  callback: () => '7',
})

const model = new ExaoneModel({
  reasoningEffort: 'low',
  preserveThinking: true,
  maxTokens: 512,
})
const agent = new Agent({ model, tools: [lookupNumber] })
console.log(String(await agent.invoke('Look up the number and add 3.')))
```

The provider reads `EXAONE_API_KEY` at request time. Alternatively, pass `apiKey` in the constructor. It does not load `.env` files automatically.

## Configuration

`new ExaoneModel(options)` accepts the Strands base model options such as `modelId`, `contextWindowLimit`, `maxTokens`, `temperature` and `topP`, as well as:

- `baseURL`: API base URL including `/v1`; defaults to `https://api.exaone.ai/v1`.
- `stream`: default `true`; `false` adapts a regular JSON response to Strands events.
- `includeUsage`: sets `stream_options.include_usage` when streaming. If omitted, the server chooses its default.
- `topK`, `presencePenalty`, `reasoningEffort` (`low`, `medium`, `high`, or `max`), `preserveThinking`, and `enableThinking`: EXAONE request options. `enableThinking` behavior has been observed in live API tests but differs from the published specification; API behavior may change.
- `timeoutMs`: request timeout in milliseconds. The default has no provider-imposed timeout. Strands cancellation signals are supported.
- `params`: additional request fields. Explicit named configuration options take precedence. The provider controls `model`, `messages`, `stream`, `stream_options`, `tools`, and `tool_choice`; they cannot be overridden using `params`. Only one completion choice is supported.
- `fetch`: custom fetch implementation.
- `onMetadata`: receives the API request ID, rate-limit headers, and raw usage when available. This includes `reasoning_tokens` when the API returns it. Strands model metadata also exposes this information under `trace.exaone`.

Assistant reasoning is preserved when prior Strands messages contain reasoning blocks, enabling `preserveThinking` across tool turns. Tool calls are assembled and validated before being exposed to Strands, including when multiple calls arrive interleaved. Text conversations and text tool results are supported; unsupported media or reasoning signatures are rejected rather than discarded. Strands structured-output requests are mapped to JSON Schema, or to a forced tool call when Strands requests one.

HTTP errors expose `status`, `body`, `requestId`, and selected rate-limit `headers` on `ExaoneAPIError`. HTTP 429 is exposed as the Strands `ModelThrottledError`, with the `ExaoneAPIError` available as its `cause`. The API key is redacted from the exposed HTTP error body. The provider itself does not automatically retry requests; a Strands agent may apply its own retry behavior.
