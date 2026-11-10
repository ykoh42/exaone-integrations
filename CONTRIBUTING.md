# Contributing

The initial packages are scaffolds; model behavior and live API support have not been implemented yet.

Use the Node.js version in `.node-version` and Python 3.10 or newer. Install dependencies and run the available checks:

```sh
npm ci
npm run check
python -m pip install -e './python/strands-exaone[dev]'
ruff check python/strands-exaone
ruff format --check python/strands-exaone
python -c 'import strands_exaone'
```

Add behavior-focused tests as the provider is implemented. For protocol changes, verify streaming, tool calls, and reasoning history with fixtures and, when credentials are available, the real API. Never commit API keys or model outputs containing sensitive data.
