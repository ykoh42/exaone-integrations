# EXAONE Integrations

Community-maintained providers for the EXAONE Chat Completions API. They preserve EXAONE reasoning history and expose model capabilities and token usage through Strands and LangChain.

## Get started

Get an API key from the [EXAONE API Platform](https://docs.exaone.ai/en/docs/overview) and [set `EXAONE_API_KEY`](guides/getting-started.md#set-your-api-key). Pick a package for your framework:

| Framework | Python | TypeScript |
| --- | --- | --- |
| Strands | [Python guide](packages/strands-exaone-py/) | [TypeScript guide](packages/strands-exaone-ts/) |
| LangChain | [Python guide](packages/langchain-exaone-py/) | [TypeScript guide](packages/langchain-exaone-ts/) |

The providers default to hosted K-EXAONE 2.5. See [model and server compatibility](COMPATIBILITY.md) for self-hosted endpoints, the [roadmap](ROADMAP.md) for planned work, and [contributing](CONTRIBUTING.md) for development.

## Examples

| Framework | Python | TypeScript |
| --- | --- | --- |
| Strands SDK | [Example](packages/strands-exaone-py/examples/sdk/) | [Example](packages/strands-exaone-ts/examples/sdk/) |
| Strands Harness | [Example](packages/strands-exaone-py/examples/harness/) | [Example](packages/strands-exaone-ts/examples/harness/) |
| Strands CLI | [Example](packages/strands-exaone-py/examples/strands-cli/) | [Example](packages/strands-exaone-ts/examples/strands-cli/) |
| LangChain | [Example](packages/langchain-exaone-py/examples/langchain/) | [Example](packages/langchain-exaone-ts/examples/langchain/) |
| LangGraph | [Example](packages/langchain-exaone-py/examples/langgraph/) | [Example](packages/langchain-exaone-ts/examples/langgraph/) |
| Deep Agents | [Example](packages/langchain-exaone-py/examples/deepagents/) | [Example](packages/langchain-exaone-ts/examples/deepagents/) |

## License

This project is licensed under the [MIT License](LICENSE).
