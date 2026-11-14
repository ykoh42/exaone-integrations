# Contributing

This repository contains independently installable Python and TypeScript providers for Strands and LangChain. Run the checks below from the repository root unless a command changes directories.

## Development setup

Use Python 3.10 or newer, uv for the LangChain Python lockfile, the Node.js version in `.node-version`, and npm 11.20.0. The npm version supports this repository's install-script settings. Create and activate a Python virtual environment, then install the Strands development dependencies:

```bash
python -m pip install -e './packages/strands-exaone-py[dev]'
npm ci
(cd packages/strands-exaone-ts && npm ci --ignore-scripts --workspaces=false)
(cd packages/langchain-exaone-py && uv sync --locked --all-groups)
(cd packages/langchain-exaone-ts && npm ci --ignore-scripts --workspaces=false)
```

## Checks

```bash
# Python
ruff check packages/strands-exaone-py
ruff format --check packages/strands-exaone-py
complexipy packages/strands-exaone-py/src --max-complexity-allowed 15
python -m pytest packages/strands-exaone-py/tests
(cd packages/strands-exaone-py && hatch build)

# LangChain Python
(cd packages/langchain-exaone-py && uv run --locked ruff check src tests examples)
(cd packages/langchain-exaone-py && uv run --locked ruff format --check src tests examples)
(cd packages/langchain-exaone-py && uv run --locked pytest)
(cd packages/langchain-exaone-py && uv build)

# TypeScript
npm run check
npm pack --workspace=strands-exaone --dry-run
(cd packages/strands-exaone-ts && npm run check --workspaces=false)

# LangChain TypeScript
(cd packages/langchain-exaone-ts && npm run check --workspaces=false)
(cd packages/langchain-exaone-ts && npm pack --pack-destination "${TMPDIR:-/tmp}")
```

All four packages have fixture-based tests. The Strands packages enforce a cognitive complexity threshold of 15 in Python and TypeScript; those scores are not directly comparable. CI tests the minimum and newer Strands SDK and Harness combinations, exercises both LangChain packages across supported runtime versions and dependency boundaries, builds Python wheels and npm tarballs, and tests the artifacts in independent environments. LangChain TypeScript's published runtime supports Node.js 20.3 and newer; the repository's development tools require the newer Node.js version in `.node-version`.

## Examples

Each example directory has its own setup instructions and framework dependencies. CI lints the Python examples and independently installs and type-checks the five TypeScript example projects. To check a TypeScript example locally, follow its README and run `npm run check` from that directory. Running an example uses the configured EXAONE account; type-checking does not make API requests.

For TypeScript watch mode, run `npm run test:watch`. The Harness example tests run a separate process that imports the built package, so source changes require a fresh build. Either restart watch mode after each change, or run the following in another terminal and wait for compilation to finish before pressing `a` in Vitest to rerun all tests:

```bash
npx tsc -p packages/strands-exaone-ts/tsconfig.json --watch
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

## Releases

The `Publish` workflow releases one distribution at a time. Create a tag named `<package-directory>-v<version>` from `main`, then publish a stable GitHub release from that tag. For example, the Strands Python distribution uses `strands-exaone-py-v0.1.0`. Running `workflow_dispatch` with a tag validates and builds the release without publishing it.

Before publishing, configure OIDC trusted publishers for each registry package. For npm, use the GitHub Actions publisher for `ykoh42/exaone-integrations`, workflow `publish.yml`, and environment `npm`; leave direct publishing disabled so the workflow can only stage a package. A maintainer must review and approve the staged npm package with 2FA. For PyPI, configure a trusted publisher for each project with owner `ykoh42`, repository `exaone-integrations`, and workflow `publish.yml`. Use `pypi` for `strands-exaone` and `pypi-langchain` for `langchain-exaone`; PyPI requires distinct pending publisher configurations when creating multiple projects from one repository. Protect both GitHub environments with a required reviewer before enabling their publishers; PyPI uploads immediately after the matching job is approved. See the [npm trusted publishing guide](https://docs.npmjs.com/trusted-publishers/) and [PyPI trusted publishers guide](https://docs.pypi.org/trusted-publishers/) for account setup.
