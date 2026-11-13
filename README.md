# EXAONE Integrations

Community-maintained model providers for the EXAONE Chat Completions API. This project is not affiliated with LG AI Research, Strands Agents, or LangChain.

Users are responsible for reviewing and complying with the terms that apply to their EXAONE API use.

| Framework | Language | Package | Status |
| --- | --- | --- | --- |
| Strands | Python | [`strands-exaone`](packages/strands-exaone-py/) | Published on PyPI |
| Strands | TypeScript | [`strands-exaone`](packages/strands-exaone-ts/) | Published on npm |
| LangChain | Python | [`langchain-exaone`](packages/langchain-exaone-py/) | Published on PyPI |
| LangChain | TypeScript | [`langchain-exaone`](packages/langchain-exaone-ts/) | Published on npm |

Each package documents its supported framework features, installation, examples, configuration, and API limitations.

## Why use a dedicated provider?

EXAONE provides an [OpenAI-compatible API](https://docs.exaone.ai/ko/docs/k-exaone-2.5). For basic chat, an OpenAI provider configured with EXAONE's base URL, model ID, and API key can be sufficient. Generic adapters can also support EXAONE-specific fields through configuration or extension points; the behavior depends on the adapter and its version.

These packages maintain the EXAONE-to-framework mapping so applications can use the following behavior without implementing it separately:

| Area | What the integration provides |
| --- | --- |
| Reasoning history | Maps `reasoning_content` into framework messages and sends it back on later turns. Requests `preserve_thinking=true` by default. Keep returned assistant messages in history; reasoning removed by application memory cannot be restored. |
| Model information | Exposes known context limits and capabilities through Strands `ModelInfo` or LangChain `profile`, with caller overrides. Unknown facts remain unspecified. The framework owns context management and strategy selection. |
| Retry handling | Honors valid `Retry-After` and `Retry-After-Ms` hints within the configured retry policy. Strands uses the explicitly configured `ExaoneRetryStrategy`; LangChain includes bounded retries by default. A hint beyond the delay cap surfaces the error instead of retrying early. Already-started streams are never replayed by the provider. |
| Streaming and tools | Maps reasoning, text, streamed tool calls, and tool results to framework types. Frameworks execute tools and validate their schemas. Unsupported inputs and malformed responses follow the package's documented error handling. |
| Diagnostics | Adapts usage, including reasoning-token counts where available, response metadata, and API errors to the framework's interfaces. Exact fields and access methods are documented per package. |

The Strands packages also apply narrowly scoped syntax completion to tool-argument JSON objects: missing closing braces or brackets and trailing commas can be corrected without guessing argument values. Other malformed arguments are rejected before tool execution. This behavior is specific to the Strands packages; see their [Python](packages/strands-exaone-py/#reasoning-and-tools) and [TypeScript](packages/strands-exaone-ts/#reasoning-and-tools) documentation for the boundaries.

The integrations follow each framework's public contracts and keep API compatibility limits explicit. They do not guarantee correct model decisions or eliminate service errors. If your existing OpenAI adapter already preserves the required EXAONE fields and handles your application's limits and retry policy, continuing to use it is a reasonable choice. The dedicated packages provide a maintained, tested implementation of that integration work.

## Get started

The Python packages are available on PyPI and the TypeScript packages are available on npm. The commands below install from this checkout for development. For TypeScript, use the Node.js version in `.node-version` and npm 11.20.0.

```bash
# Python, in your virtual environment
python -m pip install -e ./packages/strands-exaone-py

# TypeScript / Node.js
npm ci
npm run build --workspace=strands-exaone

# LangChain Python
python -m pip install ./packages/langchain-exaone-py

# LangChain.js
npm run build --workspace=langchain-exaone
```

Set `EXAONE_API_KEY` in your environment before running an agent. See the package READMEs for examples and [CONTRIBUTING.md](CONTRIBUTING.md) for development checks.

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
