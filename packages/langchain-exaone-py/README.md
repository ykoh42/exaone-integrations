# `langchain-exaone`

This community-maintained package connects LangChain chat models to the K-EXAONE 2.5 Chat Completions API. It is maintained independently of LG AI Research and LangChain, and is not affiliated with or endorsed by either organization.

It implements LangChain's `BaseChatModel` interface and supports synchronous and asynchronous calls, streaming, tool calls, structured-output binding, usage metadata, and EXAONE reasoning history.

## API key

Open the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview), select **Get an API Key**, and sign in or create an EXAONE ID to follow the platform's key-issuance instructions. Each user supplies their own key. The [illustrated setup guide](https://github.com/ykoh42/exaone-integrations/blob/main/guides/getting-started.md#get-an-exaone-api-key) shows the entry screens and includes Windows PowerShell instructions.

In the terminal where you will run your program, set the issued key:

```bash
export EXAONE_API_KEY='your-api-key'
```

## Install

The package requires Python 3.10 or newer and `langchain-core` 1.3.3 or newer within the 1.x line. Install it from PyPI:

```bash
pip install -U langchain-exaone
```

To install from a checkout for development:

```bash
pip install ./packages/langchain-exaone-py
```

Set `EXAONE_API_KEY` in the environment, then create the model:

```python
from langchain_exaone import ChatEXAONE

model = ChatEXAONE()
message = model.invoke("안녕하세요. K-EXAONE을 소개해 주세요.")
print(message.text)
```

The package does not load `.env` files. Use your application's secret-management setup to place the API key in the process environment, or pass `api_key` directly.

## Why use this integration?

The integration preserves EXAONE reasoning across LangChain turns, exposes model profiles and response metadata, and supports tools and structured output. Retries honor server delay hints before streaming output begins.

## Serialization

LangChain's deserializer allows only registered integration classes by default. Since this package is not registered in `langchain-core`, `ChatEXAONE` does not opt into object serialization or modify LangChain's global registry. Create the model from application configuration when the process starts. Its API key uses `SecretStr` and is omitted from the model representation.

## Model profile

`ChatEXAONE().profile` provides LangChain's standard model profile for `k-exaone_v2.5`: a 524,288-token context budget, text input and output, reasoning, tool calling and choice. Image, audio, video, and PDF inputs and non-text outputs are unsupported. The combined input and output, including reasoning, must fit within the context budget; no independent `max_output_tokens` profile value is inferred.

Where supported by the installed LangChain version, the profile reports reasoning levels (`low`, `medium`, `high`, `max`), the API default (`max`), streamed tool-call support, and temperature support. The endpoint reference currently lists `low`, `high`, and `max`; the [Agent guide](https://docs.exaone.ai/ko/docs/agent-start) also uses `medium`.

The default profile includes only fields recognized by the installed `langchain-core` version. Upgrading LangChain makes additional profile fields available without changing the model's API behavior.

LangChain and Deep Agents can use this profile for context management and structured-output strategy selection. A custom endpoint or unknown model ID has no inferred profile. Pass `profile={...}` to replace the defaults using LangChain's public constructor option; `profile={}` leaves capabilities unspecified. Profiles are application metadata and are not sent as API parameters.

The default profile leaves `structured_output` unspecified because native JSON Schema output combined with tool calls in one request has not been established as compatible with LangChain's provider strategy. Agents use LangChain's tool-based strategy by default. To request native output without tools, select `ProviderStrategy` explicitly. This choice does not change the provider's request parameters or structured-output methods.

See [LangChain model profiles](https://docs.langchain.com/oss/python/langchain/models#model-profiles) and the [EXAONE API specification](https://docs.exaone.ai/ko/docs/k-exaone-2.5).

## Reasoning history

The hosted K-EXAONE 2.5 API always reasons and returns reasoning separately from final content. The provider exposes it on `AIMessage.additional_kwargs["reasoning_content"]` and includes it when an assistant message is sent back in later turns. Keep the original `AIMessage` in conversation history so reasoning and tool calls remain attached to their assistant turn.

```python
from langchain_core.messages import HumanMessage
from langchain_exaone import ChatEXAONE

model = ChatEXAONE(reasoning_effort="low")
history = [HumanMessage("2 + 2는 얼마야?")]
answer = model.invoke(history)
history.append(answer)
history.append(HumanMessage("방금 답을 확인해줘."))
follow_up = model.invoke(history)
```

Use `model.astream(...)` for asynchronous streaming. Streaming chunks retain reasoning in `additional_kwargs`, while text remains in the normal message content field.

LangChain's streaming interface represents one completion per run. If the API returns multiple choices, such as when `n` is greater than `1` in `model_kwargs`, `stream()` and `astream()` raise `ValueError` rather than merging separate candidates. Use `generate()` when you need multiple candidates.

For a non-streaming response, malformed tool arguments are retained in `invalid_tool_calls` with a JSON error and location when available. This does not automatically send the error back to the model or retry the call; that behavior depends on the agent.

In an experimental EXAONE 4.5 run with vLLM 0.33.0+cpu, vLLM Metal 0.33.0 (`qwen3`/`hermes`), and LangChain Core 1.8.1, incomplete streamed tool-call JSON became a partial object that passed the tool schema, and a LangChain 1.5.0 agent invoked a synthetic tool. The same behavior occurred with `langchain-openai` 1.8.0 `ChatOpenAI`. With both integrations, the agent also executed synthetic tools after `finish_reason=length` when parsed arguments passed the schema, including a case with incomplete streamed JSON. If you control tool dispatch, treat `finish_reason=length` as incomplete and stop before execution; for streamed calls, also validate the assembled raw `tool_call_chunks` arguments as complete JSON. JSON or schema validity alone cannot establish that truncated values are complete.

Each candidate in a non-streaming response carries the same request-level usage. `generate()` and `agenerate()` expose totals once per prompt in `llm_output["token_usage"]`, including reasoning tokens when present.

When the endpoint reports cache reads or writes, `usage_metadata.input_token_details` exposes them as `cache_read` and `cache_creation`. Reasoning counts are exposed under `output_token_details` when reported.

LangChain's `output_version="v1"` standard content blocks can also be retained in conversation history, including reasoning and tool calls.

For v1 messages, reasoning blocks in `content` are authoritative: removing them prevents that reasoning from being sent again. For older message formats, the provider also reads `additional_kwargs["reasoning_content"]`.

Refusal text is kept separately in `additional_kwargs["refusal"]`, and v1 messages also expose a `refusal` content block. It is not merged into answer text. When reusing an assistant message that has no answer text, the provider sends its refusal as text in the next request: a refusal content block in message history was rejected by the hosted 2.5 endpoint in a compatibility check. If answer text is present, that text is sent instead.

## Tools and structured output

```python
from langchain_core.tools import tool
from langchain_exaone import ChatEXAONE


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}."


model = ChatEXAONE().bind_tools([get_weather], tool_choice="auto")
response = model.invoke("서울 날씨를 알려줘.")
```

Pass `strict=True` to `bind_tools()` to request strict schemas for converted tools when the endpoint supports them. Already formatted OpenAI tool definitions retain their own `strict` value, following LangChain's converter.

### Optional malformed-tool recovery

For LangChain agents on Python 3.11 or newer, install `langchain-exaone[recovery]` and attach the optional middleware:

```python
from langchain.agents import create_agent
from langchain_exaone.recovery import ToolCallRecoveryMiddleware

agent = create_agent(
    model=ChatEXAONE(),
    tools=[get_weather],
    middleware=[ToolCallRecoveryMiddleware()],
)
```

If one complete tool call has invalid JSON, the middleware requests a correction with the original arguments and JSON error in a matching tool-result error. To support endpoints that reject invalid JSON in assistant history, that temporary request uses an explicitly identified `{}` placeholder for the failed call; the placeholder is not executed or added to agent history. The original system prompt, user messages and tools are retained. Schema validation remains with LangChain.

Text and reasoning stream as they arrive. Tool chunks are released after the arguments have been checked. Recovery can retry one malformed call before any output is emitted. To wait for validation before streaming any part of the response, use `ToolCallRecoveryMiddleware(buffer_response=True)`.

The middleware requires a completed response before a tool can run. Length stops, filtered responses, refusals, streams missing `[DONE]`, and non-standard JSON numbers such as `NaN` are rejected. Responses with multiple choices or tool calls are rejected if any call has malformed JSON. Malformed calls with a missing or ambiguous ID are not retried. Calls containing the API credential in the arguments, ID, or name, including a URL-encoded form, are rejected.

If LangChain's response cache is configured, the middleware uses a separate cache key for validated responses. The key is local to LangChain and is omitted from API requests.

Recovery uses the effective EXAONE request after SDK middleware has applied tool selection. It stops before sending a correction if the tool was removed or the SDK owns the tool choice, as with `ToolStrategy` in LangChain Python. Register recovery after routing middleware that switches providers or makes its own model calls, so the correction applies to the selected EXAONE model call.

The tool error includes a separate JSON record with `raw_arguments` as a string and `parse_error` as the diagnostic. When available, the location is a one-based line and column in the original argument text. It refers to the decoded argument text, before JSON escaping.

Only the correction request selects the failed tool by name. If the endpoint explicitly rejects named tool choice before emitting a response, the middleware tries `tool_choice="required"` once. Recovery succeeds only with a completed reply containing exactly one call to the original tool with a valid JSON object as its arguments. The SDK still validates the tool schema before execution.

Without the middleware, LangChain's partial JSON parser can turn incomplete streamed arguments into an executable tool call. For example, an Agent can execute a tool with `amount=12` when the stream ends with the raw arguments `{"amount":12`.

Empty or whitespace-only tool input is not automatically retried; LangChain handles it according to its existing rules, which can allow a tool's default arguments to run during streaming.

The recovered assistant message exposes `response_metadata["exaone_tool_call_recovery"]` with `attempts: 1` and, when reported by the service, `failed_usage`. The successful response's own usage does not include the failed attempt. In streaming runs, inspect the agent's final state or `updates` stream for this metadata; individual `messages` chunks have already been emitted before the middleware attaches it. The SDK can also emit an `on_llm_error` callback for the failed attempt, whose `MalformedToolArguments` exception exposes numeric `.usage`. The tool-result feedback and SDK tracing may contain the model-generated argument text; configure tracing according to your application's data handling needs.

`with_structured_output()` uses LangChain's tool-based strategy by default. Pass `method="json_schema"` for native JSON Schema output or `method="json_mode"` for a JSON object. With JSON mode, ask for JSON in the prompt. Pydantic schemas validate the result; a plain JSON Schema dictionary is parsed as JSON without additional schema validation, following LangChain's parser. `include_raw=True` retains the original message and parsing error.

You can pass additional Chat Completions fields with `model_kwargs` or per-call keyword arguments. Their acceptance depends on the EXAONE endpoint.

When a returned EXAONE message is reused in history, edits to its LangChain `tool_calls` determine the next request. Provider-specific fields from matching original calls are retained. Response tool-call types other than function calls cannot be represented by this integration and are rejected.

Legacy `function_call` fields and LangChain `FunctionMessage` results are also preserved in conversation history for compatible endpoints. A synthetic history containing both fields and a `function` role returned HTTP 400 from the default EXAONE endpoint; use the tool-calling protocol for EXAONE.

## Configuration

| Option | Default | Description |
| --- | --- | --- |
| `model` | `k-exaone_v2.5` | EXAONE model ID |
| `api_key` | `EXAONE_API_KEY` | API credential |
| `base_url` | `https://api.exaone.ai/v1` | HTTP or HTTPS API base URL. |
| `temperature` | `1.0` | Recommended EXAONE sampling value |
| `top_p` | `0.95` | Recommended EXAONE sampling value |
| `top_k` | `-1` | Disable top-k filtering, as in the API default |
| `presence_penalty` | `0.0` | Recommended EXAONE sampling value |
| `max_tokens` | API default | Maximum generated tokens, including reasoning |
| `reasoning_effort` | API default | Reasoning setting accepted by the selected endpoint |
| `preserve_thinking` | `true` | Request preservation of earlier assistant reasoning in multi-turn agent conversations |
| `chat_template_kwargs` | `{}` | Additional chat-template options; `preserve_thinking` is set from its dedicated option |
| `max_retries` | `2` | Retries for transient API errors before output is returned; set to `0` to disable |
| `timeout` | `None` | No provider-imposed HTTP timeout; pass a positive number to set one |

For an already-running K-EXAONE 2.0 endpoint, set `base_url` and `model` to the server's URL and served model name. If supported by the server, `chat_template_kwargs={"enable_thinking": False}` requests non-reasoning mode. The provider accepts `reasoning` and `reasoning_content` response fields and preserves them as `reasoning_content` for later turns. Identical values are emitted once; conflicting nonempty values are rejected. Protocol fixtures have been tested; actual K-EXAONE 2.0 inference has not been verified by this project.

For serving instructions and parser settings, see the [model card](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B). Its examples use forked servers; [vLLM 0.27.0](https://github.com/vllm-project/vllm/releases/tag/v0.27.0) also includes K-EXAONE 2.0 model support. Verify the specific server version and configuration you deploy.

The SGLang `qwen3_coder` parser in the inspected [model-card fork](https://github.com/lkm2835/sglang/blob/918653d22b2f1ae853b81fc0f5539b7c9311487e/python/sglang/srt/function_call/qwen3_coder_detector.py#L104) and [upstream revision](https://github.com/sgl-project/sglang/blob/d9ec74eb72b856419c6eb44733e00193d28ed840/python/sglang/srt/function_call/qwen3_coder_detector.py#L112) converts the text `null` to JSON null even for a string parameter. This can cause tool validation to fail; check your server's parser version.

Some servers accept `reasoning_effort` and may use it to select thinking on or off. Accepting the field does not establish that the model supports distinct reasoning-intensity levels; check the deployed server and model documentation.

```python
model = ChatEXAONE(
    base_url="http://127.0.0.1:8000/v1",
    model="K-EXAONE-2.0-750B-A37B",  # Use the served name.
    api_key="EMPTY",  # Use the server's configured credential, if any.
)
```

The configured API key is sent to the selected endpoint; if omitted, it comes from `EXAONE_API_KEY`. For a server that does not require authentication, pass a placeholder such as `api_key="EMPTY"` rather than the hosted API key. HTTP does not encrypt credentials or conversation data; use HTTPS or a trusted transport.

The default model's documented context window is 524,288 tokens. The selected endpoint validates its own input and output token budget.

The provider retries HTTP 408, 429, 500, 502, 503, and 504 errors up to two times by default; set `max_retries=0` to disable retries. It honors the longer valid `Retry-After` or `Retry-After-Ms` delay. If neither header is present, the first retry waits at least one second to respect the API's per-second rate limit; a `max_retry_delay` below one second disables that retry. It does not retry after streaming output has started.

The default hosted K-EXAONE 2.5 API is text-only. For an already-running compatible EXAONE 4.5 endpoint, set `base_url`, `model`, and `api_key` to its connection details and use LangChain image content blocks:

```python
from langchain_core.messages import HumanMessage
from langchain_exaone import ChatEXAONE

model = ChatEXAONE(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", model="EXAONE-4.5-33B")
message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image."},
        {"type": "image", "url": "https://example.com/image.png"},
    ]
)
response = model.invoke([message])
```

User images support URLs and base64 data with an image MIME type; the provider sends text and image blocks in their original order. File IDs, image tool messages, audio, and other unsupported blocks are rejected. The [EXAONE 4.5 model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) shows vLLM and an SGLang fork with `qwen3` reasoning and `hermes` tool parsers. Configure these on the server. Image requests have been tested with synthetic protocol fixtures; actual EXAONE 4.5 image inference has not been verified.

### Server-specific observations

These observations use synthetic inputs and isolated server code, not running servers or EXAONE 4.5 inference. The provider preserves image/text order in its request; server processing determines what reaches the model.

- **vLLM 0.27.0:** With the official 4.5 template and automatic content-format detection, default normalization moved an image before surrounding text. [`--interleave-mm-strings`](https://docs.vllm.ai/en/v0.27.0/cli/serve/#interleave-mm-strings) preserved the order in the isolated test.
- **SGLang image/tool history:** With the model-card fork `6757c9f` and 4.5 template `570aa4b`, a first user message containing an image followed by a tool call and result could omit earlier assistant reasoning from the rendered prompt. A prior string-valued user message changed the result.
- **SGLang streaming tools:** With automatic tool choice, the same fork could emit a tool name without arguments when the complete synthetic call and stop arrived in one final event. Missing arguments can fail validation or invoke a tool with defaults. The parser condition also occurred in SGLang 0.5.23 in isolation; this does not establish 4.5 model support. See [issue #31915](https://github.com/sgl-project/sglang/issues/31915) and [proposed fix #32167](https://github.com/sgl-project/sglang/pull/32167).

The provider does not repair these server behaviors. Check your server version, template and settings; see the [compatibility matrix](https://github.com/ykoh42/exaone-integrations/blob/main/COMPATIBILITY.md#known-server-side-observations) for details.

See the [K-EXAONE 2.5 API reference](https://docs.exaone.ai/ko/docs/k-exaone-2.5) for the model's request parameters and limits.

## Examples

See the [examples](examples/) for LangChain agents, streaming chat, structured output, LangGraph tool loops, and Deep Agents. The package does not make API requests during import or construction.

## Development

```bash
uv sync --group test --group lint
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Unit tests use synthetic responses and do not require API credentials.
