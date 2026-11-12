# Contributing

This repository contains independently installable Python and TypeScript providers for Strands and LangChain. Run the checks below from the repository root unless a command changes directories.

## Development setup

Use Python 3.10 or newer, uv for the LangChain Python lockfile, the Node.js version in `.node-version`, and npm 11.20.0. The npm version supports this repository's install-script settings. Create and activate a Python virtual environment, then install the Strands development dependencies:

```bash
python -m pip install -e './packages/strands-exaone-python[dev]'
npm ci
(cd packages/strands-exaone-typescript && npm ci --ignore-scripts --workspaces=false)
(cd packages/langchain-exaone-python && uv sync --locked --all-groups)
(cd packages/langchain-exaone-typescript && npm ci --ignore-scripts --workspaces=false)
```

## Checks

```bash
# Python
ruff check packages/strands-exaone-python
ruff format --check packages/strands-exaone-python
complexipy packages/strands-exaone-python/src --max-complexity-allowed 15
python -m pytest packages/strands-exaone-python/tests
(cd packages/strands-exaone-python && hatch build)

# LangChain Python
(cd packages/langchain-exaone-python && uv run --locked ruff check src tests)
(cd packages/langchain-exaone-python && uv run --locked ruff format --check src tests)
(cd packages/langchain-exaone-python && uv run --locked pytest)
(cd packages/langchain-exaone-python && uv build)

# TypeScript
npm run check
npm pack --workspace=strands-exaone --dry-run
(cd packages/strands-exaone-typescript && npm run check --workspaces=false)

# LangChain TypeScript
(cd packages/langchain-exaone-typescript && npm run check --workspaces=false)
(cd packages/langchain-exaone-typescript && npm pack --pack-destination "${TMPDIR:-/tmp}")
```

All four packages have fixture-based tests. The Strands packages enforce a cognitive complexity threshold of 15 in Python and TypeScript; those scores are not directly comparable. CI tests Strands against Harness 0.1.x and 0.3.4, exercises both LangChain packages across supported runtime versions, builds Python wheels and npm tarballs, and smoke-tests the artifacts in independent environments.

For TypeScript watch mode, run `npm run test:watch`. The Harness example tests run a separate process that imports the built package, so source changes require a fresh build. Either restart watch mode after each change, or run the following in another terminal and wait for compilation to finish before pressing `a` in Vitest to rerun all tests:

```bash
npx tsc -p packages/strands-exaone-typescript/tsconfig.json --watch
```

## Live tests

Live tests are optional. They send synthetic prompts to EXAONE and consume API quota. Set `EXAONE_API_KEY` in your environment, install the development dependencies, and run the tests one at a time:

```bash
python scripts/live-smoke-python.py
npm run build --workspace=strands-exaone
node scripts/live-smoke-typescript.mjs
```

The TypeScript smoke test includes a comparison with the Strands OpenAI chat provider. To run the Python comparison, install `strands-agents[openai]` and run `python scripts/live-compare-python.py`.

The scripts do not load `.env` automatically. Run live checks serially within the account's limits and use synthetic inputs. Never commit API keys, generated responses, or reasoning. Logprob tokens and other response metadata can also contain generated or user-provided content.
