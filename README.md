# EXAONE Integrations

Community-maintained integrations for the EXAONE API and agent frameworks. This project is not affiliated with LG AI Research.

| Package | Directory | Status |
| --- | --- | --- |
| `strands-exaone` for Python | [`python/strands-exaone`](python/strands-exaone/) | Implemented, not published |
| `strands-exaone` for TypeScript | [`typescript/strands-exaone`](typescript/strands-exaone/) | Implemented, not published |

Both packages implement Strands' model interface for K-EXAONE 2.5. They support streaming and non-streaming chat, separate reasoning content and its preservation across turns, tool calls, structured output requests, EXAONE generation options, and request-specific usage metadata. The package READMEs contain usage and configuration examples. Until release, install from a local checkout:

```sh
python -m pip install ./python/strands-exaone
npm install ./typescript/strands-exaone @strands-agents/sdk zod
```

Set `EXAONE_API_KEY` in your environment before running a provider. Do not commit API keys. See [CONTRIBUTING.md](CONTRIBUTING.md) for development and opt-in live checks.

The [K-EXAONE 2.5 checklist](docs/k-exaone-2.5-checklist.md) distinguishes implemented, live-tested, fixture-tested, and unverified behavior. API observations are recorded for [November 10](docs/live-api-observations-2026-11-10.md) and [November 11](docs/live-api-observations-2026-11-11.md); the [provider design analysis](docs/strands-provider-design.md) compares the alternatives. Both providers expose one completion choice, as required by the Strands model interface; the separate EXAONE Tabular API is outside their scope.

The package setup draws on the [Strands extension template](https://github.com/strands-agents/extension-template). The workspace and development guidelines also draw on [web-llm-adapters](https://github.com/ykoh42/web-llm-adapters).
