# EXAONE Integrations

Community-maintained integrations for using the EXAONE API with agent frameworks. This project is not affiliated with LG AI Research.

The initial work targets Strands Agents for Python and TypeScript:

| Package | Directory | Status |
| --- | --- | --- |
| `strands-exaone` (PyPI) | [`python/strands-exaone`](python/strands-exaone/) | Setup only |
| `strands-exaone` (npm) | [`typescript/strands-exaone`](typescript/strands-exaone/) | Setup only |

No model provider has been implemented or published yet. See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks.

Before implementation, consult the [K-EXAONE 2.5 specification checklist](docs/k-exaone-2.5-checklist.md).
See the [live API observations](docs/live-api-observations-2026-11-10.md) for verified behavior and documentation discrepancies.

The Strands package setup draws on the [Strands extension template](https://github.com/strands-agents/extension-template). The workspace and development guidelines also draw on [web-llm-adapters](https://github.com/ykoh42/web-llm-adapters).
