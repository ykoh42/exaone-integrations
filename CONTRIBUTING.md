# Contributing

This repository contains independently installable Python and TypeScript Strands model providers. Run the commands below from the repository root.

## Development setup

Use Python 3.10 or newer, the Node.js version in `.node-version`, and npm 11.20.0. The npm version supports this repository's install-script settings. Create and activate a Python virtual environment, then install the development dependencies:

```bash
python -m pip install -e './packages/strands-exaone-python[dev]'
npm ci
```

## Checks

```bash
# Python
ruff check packages/strands-exaone-python
ruff format --check packages/strands-exaone-python
complexipy packages/strands-exaone-python/src --max-complexity-allowed 15
python -m pytest packages/strands-exaone-python/tests
(cd packages/strands-exaone-python && hatch build)

# TypeScript
npm run check
npm pack --workspace=strands-exaone --dry-run
```

The packages use pytest and Vitest. Python and TypeScript checks enforce a cognitive complexity threshold of 15; the scores from the two tools are not directly comparable. CI builds a Python wheel and installs the npm tarball in a separate project.

## Live tests

Live tests are optional. They send synthetic prompts to EXAONE and consume API quota. Set `EXAONE_API_KEY` in your environment, install the development dependencies, and run the tests one at a time:

```bash
python scripts/live-smoke-python.py
npm run build --workspace=strands-exaone
node scripts/live-smoke-typescript.mjs
```

The TypeScript smoke test includes a comparison with the Strands OpenAI chat provider. To run the Python comparison, install `strands-agents[openai]` and run `python scripts/live-compare-python.py`.

The scripts do not load `.env` automatically. Run live checks serially within the account's limits and use synthetic inputs. Never commit API keys, generated responses, or reasoning. Logprob tokens and other response metadata can also contain generated or user-provided content.
