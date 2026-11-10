# Contributing

The initial packages are scaffolds; model behavior and live API support have not been implemented yet.

Use the Node.js version in `.node-version` and Python 3.10 or newer. Install dependencies and run the available checks:

```sh
npm ci
npm run check
python -m pip install -e './python/strands-exaone[dev]'
ruff check python/strands-exaone
ruff format --check python/strands-exaone
complexipy python/strands-exaone/src --max-complexity-allowed 15
python -c 'import strands_exaone'
python -m pytest python/strands-exaone/tests  # once tests exist
(cd python/strands-exaone && hatch build)
```

Tests use pytest and Vitest, following the Strands SDK. The TypeScript test command permits an empty suite while this is a scaffold; CI reports the Python suite as absent until tests are added. Add behavior-focused tests as the provider is implemented. For protocol changes, verify streaming, tool calls, and reasoning history with fixtures and, when credentials are available, the real API. Never commit API keys or model outputs containing sensitive data.

Python and TypeScript checks enforce a cognitive complexity threshold of 15 for source functions. The tools use similar metrics, but scores across languages are not necessarily identical.
