# EXAONE Integrations

Model providers for using the EXAONE Chat Completions API with [Strands Agents](https://strandsagents.com/). This is a community-maintained project and is not affiliated with LG AI Research or the Strands Agents project.

Use of the EXAONE API is subject to the [EXAONE API Platform terms](https://docs.exaone.ai/en/terms/terms-of-service), including non-commercial use and restrictions on third-party access through products or services.

| Language | Package | Status |
| --- | --- | --- |
| Python | [`strands-exaone`](packages/strands-exaone-python/) | Implemented; not published |
| TypeScript | [`strands-exaone`](packages/strands-exaone-typescript/) | Implemented; not published |

Both providers support streamed and non-streamed responses, reasoning history, tool calls, structured output, and EXAONE response metadata. Select a language above for installation, examples, configuration, and API limitations.

## Get started

Clone this repository and run the following commands from its root. The packages are not yet published to PyPI or npm. For the TypeScript checkout, use the Node.js version in `.node-version` and npm 11.20.0.

```bash
# Python, in your virtual environment
python -m pip install -e ./packages/strands-exaone-python

# TypeScript / Node.js
npm ci
npm run build --workspace=strands-exaone
```

Set `EXAONE_API_KEY` in your environment before running an agent. See the package READMEs for examples and [CONTRIBUTING.md](CONTRIBUTING.md) for development checks.

For runnable examples, see the Python [Harness](packages/strands-exaone-python/examples/harness/) and [CLI](packages/strands-exaone-python/examples/strands-cli/) examples, or the TypeScript [Harness](packages/strands-exaone-typescript/examples/harness/) and [CLI](packages/strands-exaone-typescript/examples/strands-cli/) examples.

The providers adapt a single completion choice to Strands. EXAONE Tabular is a separate API and is outside this project's scope.

## License

This project is licensed under the [MIT License](LICENSE).
