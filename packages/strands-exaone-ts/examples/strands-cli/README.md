# EXAONE with the Strands CLI

Run a TypeScript Strands harness with the EXAONE model provider and a local example tool. This community-maintained integration is not affiliated with LG AI Research or the Strands Agents project.

## Setup

The provider runs on Node.js 22 or newer. Building this checkout requires Node.js 22.13.0 or newer and npm 11.20.0; `.node-version` selects the checkout's development version. The `strands-exaone` package is not published yet, so the example installs it from this checkout. From the repository root, install the workspace dependencies and build the provider:

```bash
npm ci
npm run build --workspace=strands-exaone
cd packages/strands-exaone-ts/examples/strands-cli
npm install --install-links --no-package-lock
export EXAONE_API_KEY='your-api-key'
```

`--install-links` packs the local provider as a regular dependency, allowing the example and CLI to share the installed Strands SDK. Provide the key through your environment or a trusted file selected with `--env-file`; the provider does not load `.env` files automatically.

## Run

From this directory, start an interactive session:

```bash
npx strands --agent ./agent.ts
```

Ask the agent to use `get_example_number` and add three to its result, then ask a follow-up in the same conversation. This tool returns a fixed value and does not access external systems. The agent config disables built-in tools, memory, skills, and session persistence. The default context manager and background-task support register additional management tools, and the CLI can add its own tools. Conversation history remains available until you exit; model requests still use the EXAONE API. The interactive CLI can display model reasoning.

`preserveThinking: true` requests preservation of earlier reasoning. The provider forwards available reasoning with the corresponding assistant and tool history. Set `EXAONE_BASE_URL` only when you need to use another compatible endpoint. See the [TypeScript package README](../../README.md) for configuration and API behavior.

The harness uses automatic context management by default. It can summarize older conversation history, so `preserveThinking` can only retain reasoning that remains in that history. If you require the original history to stay intact, pass `contextManager: false` to `createHarness`; then manage the conversation length yourself so input tokens plus `maxTokens` fit within the model's context limit.
