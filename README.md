# EXAONE Integrations

Community-maintained model providers for the EXAONE Chat Completions API. This project is not affiliated with LG AI Research, Strands Agents, or LangChain.

Review and follow the terms that apply to your use of the EXAONE API.

| Framework | Language | Package | Status |
| --- | --- | --- | --- |
| Strands | Python | [`strands-exaone`](packages/strands-exaone-py/) | Published on PyPI |
| Strands | TypeScript | [`strands-exaone`](packages/strands-exaone-ts/) | Published on npm |
| LangChain | Python | [`langchain-exaone`](packages/langchain-exaone-py/) | Published on PyPI |
| LangChain | TypeScript | [`langchain-exaone`](packages/langchain-exaone-ts/) | Published on npm |

Each package README covers installation, examples, configuration, supported framework features, and API limitations.

## Why use a dedicated provider?

EXAONE provides an [OpenAI-compatible API](https://docs.exaone.ai/ko/docs/k-exaone-2.5). An OpenAI provider configured with EXAONE's base URL, model ID, and API key can handle basic chat. Some adapters also support EXAONE-specific fields through configuration or extensions, depending on the adapter and version.

These packages provide a maintained, tested mapping between EXAONE and each framework:

| Area | What the integration provides |
| --- | --- |
| Reasoning history | Maps `reasoning_content` into framework messages and sends it back on later turns. Requests `preserve_thinking=true` by default. Keep returned assistant messages in history; reasoning removed by application memory cannot be restored. |
| Model information | Exposes known context limits and capabilities through Strands `ModelInfo` or LangChain `profile`, with caller overrides. Unknown facts remain unspecified. The framework owns context management and strategy selection. |
| Retry handling | Honors valid `Retry-After` and `Retry-After-Ms` hints within the configured retry policy. Strands uses the explicitly configured `ExaoneRetryStrategy`; LangChain includes bounded retries by default. A hint beyond the delay cap surfaces the error instead of retrying early. Already-started streams are never replayed by the provider. |
| Streaming and tools | Maps reasoning, text, streamed tool calls, and tool results to framework types. Frameworks execute tools and validate their schemas. Unsupported inputs and malformed responses follow the package's documented error handling. |
| Diagnostics | Adapts usage, including reasoning-token counts where available, response metadata, and API errors to the framework's interfaces. Exact fields and access methods are documented per package. |

The Strands packages can repair missing closing braces or brackets and trailing commas in tool-argument JSON objects without guessing argument values. They reject other malformed arguments before tool execution. See the Strands [Python](packages/strands-exaone-py/#reasoning-and-tools) and [TypeScript](packages/strands-exaone-ts/#reasoning-and-tools) documentation for the exact limits of this behavior.

The integrations follow each framework's public contracts and document API compatibility limits. Model decisions can still be wrong, and service errors can still occur. If your existing OpenAI adapter preserves the EXAONE fields you need and handles your application's limits and retry policy, you can continue using it.

## Get started

The [getting-started guide](guides/getting-started.md) walks through API-key setup with screenshots and includes runnable examples for all four packages. Each user gets their own key from the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview).

Install the package for your framework and language. You do not need a repository checkout:

| Framework | Python | TypeScript |
| --- | --- | --- |
| Strands | `python -m pip install strands-exaone` | `npm install strands-exaone` |
| LangChain | `python -m pip install langchain-exaone` | `npm install langchain-exaone @langchain/core` |

Set `EXAONE_API_KEY` in the terminal where you run your program. The packages do not load `.env` files automatically. See the [guide](guides/getting-started.md#set-your-api-key) for macOS/Linux/WSL and Windows PowerShell commands, and the package READMEs for configuration. For development from this checkout, see [CONTRIBUTING.md](CONTRIBUTING.md).

For a higher daily API quota, see EXAONE's [official FAQ](https://docs.exaone.ai/en/terms/faq) and the guide's [limit-increase instructions](guides/getting-started.md#increase-your-daily-limit).

## Examples

Each example has its own dependencies and execution instructions.

| Framework | Python | TypeScript |
| --- | --- | --- |
| Strands SDK | [Example](packages/strands-exaone-py/examples/sdk/) | [Example](packages/strands-exaone-ts/examples/sdk/) |
| Strands Harness | [Example](packages/strands-exaone-py/examples/harness/) | [Example](packages/strands-exaone-ts/examples/harness/) |
| Strands CLI | [Example](packages/strands-exaone-py/examples/strands-cli/) | [Example](packages/strands-exaone-ts/examples/strands-cli/) |
| LangChain | [Example](packages/langchain-exaone-py/examples/langchain/) | [Example](packages/langchain-exaone-ts/examples/langchain/) |
| LangGraph | [Example](packages/langchain-exaone-py/examples/langgraph/) | [Example](packages/langchain-exaone-ts/examples/langgraph/) |
| Deep Agents | [Example](packages/langchain-exaone-py/examples/deepagents/) | [Example](packages/langchain-exaone-ts/examples/deepagents/) |

The Strands integrations adapt one completion choice to Strands' single-message model contract. EXAONE Tabular is a separate API and is outside this project's scope.

## License

This project is licensed under the [MIT License](LICENSE).
