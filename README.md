# EXAONE Integrations

Community-maintained model providers for the EXAONE Chat Completions API. This project is not affiliated with LG AI Research, Strands Agents, or LangChain.

Users are responsible for reviewing and complying with the terms that apply to their EXAONE API use.

| Framework | Language | Package | Status |
| --- | --- | --- | --- |
| Strands | Python | [`strands-exaone`](packages/strands-exaone-py/) | Published on PyPI |
| Strands | TypeScript | [`strands-exaone`](packages/strands-exaone-ts/) | Published on npm as `0.1.0` |
| LangChain | Python | [`langchain-exaone`](packages/langchain-exaone-py/) | Published on PyPI |
| LangChain | TypeScript | [`langchain-exaone`](packages/langchain-exaone-ts/) | Published on npm as `0.1.0` |

Each package documents its supported framework features, installation, examples, configuration, and API limitations.

## Get started

The Python packages are available on PyPI and the TypeScript packages are available on npm as version `0.1.0`. The commands below install from this checkout for development. For TypeScript, use the Node.js version in `.node-version` and npm 11.20.0.

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

For runnable Strands examples, see the Python [Harness](packages/strands-exaone-py/examples/harness/) and [CLI](packages/strands-exaone-py/examples/strands-cli/) examples, or the TypeScript [Harness](packages/strands-exaone-ts/examples/harness/) and [CLI](packages/strands-exaone-ts/examples/strands-cli/) examples. LangChain examples are in each provider's `examples/` directory.

The Strands integrations adapt one completion choice to Strands' single-message model contract. EXAONE Tabular is a separate API and is outside this project's scope.

## License

This project is licensed under the [MIT License](LICENSE).
