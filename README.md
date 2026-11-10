# EXAONE Integrations

Model providers for using the EXAONE Chat Completions API with [Strands Agents](https://strandsagents.com/). This is a community-maintained project and is not affiliated with LG AI Research or the Strands Agents project.

| Language | Package | Status |
| --- | --- | --- |
| Python | [`strands-exaone`](python/strands-exaone/) | Implemented; not published |
| TypeScript | [`strands-exaone`](typescript/strands-exaone/) | Implemented; not published |

Both providers support streamed and non-streamed responses, reasoning history, tool calls, structured output, and EXAONE response metadata. Select a language above for installation, examples, configuration, and API limitations.

## Get started

Clone this repository and run the following commands from its root. The packages are not yet published to PyPI or npm.

```bash
# Python, in your virtual environment
python -m pip install -e ./python/strands-exaone

# TypeScript / Node.js
npm ci
npm run build --workspace=strands-exaone
```

Set `EXAONE_API_KEY` in your environment before running an agent. See the package READMEs for examples and [CONTRIBUTING.md](CONTRIBUTING.md) for development checks.

The providers adapt a single completion choice to Strands. EXAONE Tabular is a separate API and is outside this project's scope.

## License

This project is licensed under the [MIT License](LICENSE).
