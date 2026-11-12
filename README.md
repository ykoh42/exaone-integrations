# EXAONE Integrations

Community-maintained model providers for the EXAONE Chat Completions API. This project is not affiliated with LG AI Research, Strands Agents, or LangChain.

Users are responsible for reviewing and complying with the terms that apply to their EXAONE API use.

| Framework | Language | Package | Status |
| --- | --- | --- | --- |
| Strands | Python | [`strands-exaone`](packages/strands-exaone-python/) | Implemented; not published |
| Strands | TypeScript | [`strands-exaone`](packages/strands-exaone-typescript/) | Implemented; not published |
| LangChain | Python | [`langchain-exaone`](packages/langchain-exaone-python/) | Implemented; not published |
| LangChain | TypeScript | [`langchain-exaone`](packages/langchain-exaone-typescript/) | Implemented; not published |

Each package documents its supported framework features, installation, examples, configuration, and API limitations.

## Get started

Clone this repository and run the following commands from its root. The packages are not yet published to PyPI or npm. For the TypeScript checkout, use the Node.js version in `.node-version` and npm 11.20.0.

```bash
# Python, in your virtual environment
python -m pip install -e ./packages/strands-exaone-python

# TypeScript / Node.js
npm ci
npm run build --workspace=strands-exaone

# LangChain Python
python -m pip install ./packages/langchain-exaone-python

# LangChain.js
npm run build --workspace=langchain-exaone
```

Set `EXAONE_API_KEY` in your environment before running an agent. See the package READMEs for examples and [CONTRIBUTING.md](CONTRIBUTING.md) for development checks.

For runnable Strands examples, see the Python [Harness](packages/strands-exaone-python/examples/harness/) and [CLI](packages/strands-exaone-python/examples/strands-cli/) examples, or the TypeScript [Harness](packages/strands-exaone-typescript/examples/harness/) and [CLI](packages/strands-exaone-typescript/examples/strands-cli/) examples. LangChain examples are in each provider's `examples/` directory.

The Strands integrations adapt one completion choice to Strands' single-message model contract. EXAONE Tabular is a separate API and is outside this project's scope.

## License

This project is licensed under the [MIT License](LICENSE).
