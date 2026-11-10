# Contributing

Use the Node.js version in `.node-version` and Python 3.10 or newer. Install dependencies and run the checks:

```sh
npm ci
npm run check
python -m pip install -e './python/strands-exaone[dev]'
ruff check python/strands-exaone
ruff format --check python/strands-exaone
complexipy python/strands-exaone/src --max-complexity-allowed 15
python -m pytest python/strands-exaone/tests
(cd python/strands-exaone && hatch build)
npm pack --workspace=strands-exaone --dry-run
```

Tests use pytest and Vitest, following the Strands SDK. Python and TypeScript enforce a cognitive complexity threshold of 15 for source functions; the tools use similar metrics, but their scores are not directly comparable.

## Opt-in live tests

Live tests send synthetic prompts to EXAONE and consume API quota. Set `EXAONE_API_KEY` in the environment, install both development workspaces, and run these serially (not concurrently):

```sh
python scripts/live-smoke-python.py
npm run build --workspace=strands-exaone
node scripts/live-smoke-typescript.mjs
```

The TypeScript smoke test also compares the Strands OpenAI chat provider. For the separate Python comparison, install `strands-agents[openai]` and run `python scripts/live-compare-python.py`. The scripts rate-limit requests, record only boolean/counter observations, and do not print the API key or model output. They do not automatically load `.env` files. Never commit API keys or model outputs containing sensitive data. API service conditions and rates can change; local mock tests do not require credentials.
