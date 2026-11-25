# EXAONE with the Strands CLI

Run a TypeScript Strands harness with the EXAONE model provider and a local example tool. This community-maintained integration is not affiliated with LG AI Research or the Strands Agents project.

## Setup

Strands CLI 0.4 requires Node.js 22.12 or newer. For this checkout, use the Node.js version in `.node-version` and npm 11.20.0. This example installs `strands-exaone` from the local checkout so you can try source changes. The published package is also available on npm. From the repository root, install the workspace dependencies and build the provider:

```bash
npm ci
npm run build --workspace=strands-exaone
cd packages/strands-exaone-ts/examples/strands-cli
npm install --install-links --no-package-lock --workspaces=false
export EXAONE_API_KEY='your-api-key'
```

`--install-links` packs the local provider as a regular dependency, allowing the example and CLI to share the installed Strands SDK. Provide the key through your environment or a trusted file selected with `--env-file`; the provider does not load `.env` files automatically.

The CLI 0.4 loads the Harness version it depends on. This example uses its compatible Harness 0.3.x range. Run `npm run check` to type-check the example.

## Run

From this directory, start an interactive session:

```bash
npx strands --agent ./agent.ts
```

Ask the agent to use `get_example_number` and add three to its result, then ask a follow-up in the same conversation. This tool returns a fixed value and does not access external systems.

The agent disables built-in tools, memory, skills, and session persistence. The default context manager and background-task support register additional management tools, and the CLI can add its own tools. Conversation history lasts until you exit. Requests go to the EXAONE API, and the interactive CLI can display model reasoning.

`preserveThinking: true` requests preservation of earlier reasoning. The provider forwards available reasoning with the corresponding assistant and tool history. Set `EXAONE_BASE_URL` only when you need to use another compatible endpoint. See the [TypeScript package README](../../README.md) for configuration and API behavior.

The harness automatically manages context and may summarize older messages, so `preserveThinking` can only retain reasoning that remains in that history. If you require the original history to stay intact, pass `contextManager: false` to `createHarness`; then manage the conversation length yourself so input tokens plus `maxTokens` fit within the model's context limit.
