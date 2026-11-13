# EXAONE with Deep Agents

Ask a Deep Agent to plan a release brief, delegate to an EXAONE-backed reviewer, and save the brief in its virtual filesystem. The release notes are synthetic and require no search credentials.

This example uses the community-maintained `langchain-exaone` provider.

## Setup

Use the Node.js version in the repository's `.node-version` and npm 11.20.0. Build the provider from the repository root, then install the example's dependencies:

```bash
npm ci
npm run build --workspace=langchain-exaone
cd packages/langchain-exaone-ts/examples/deepagents
npm install --install-links --no-package-lock --workspaces=false
export EXAONE_API_KEY='your-api-key'
```

## Run

This command makes real EXAONE API requests and consumes your account quota.

```bash
npm start
```

Run `npm run check` in this directory to type-check the examples. The manifest installs this checkout; replace `"langchain-exaone": "file:../.."` with a published version to run without the repository.

The examples print only the answer text. They keep complete assistant messages, including supported reasoning, in the history sent to EXAONE without trimming or rebuilding responses. The framework may summarize or remove older messages during context management; the provider forwards the resulting history.

The default StateBackend keeps files in graph state and does not read or write host files. The parent and named subagent use the same EXAONE model. Deep Agents manages tools and context; the model decides when to delegate or use filesystem tools. The in-memory checkpoint lasts for this process only, and the example configures no shell-capable backend.

The examples read `EXAONE_API_KEY` from the environment without loading `.env` files. To use another trusted compatible endpoint, set `EXAONE_BASE_URL` to its base URL, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [Deep Agents guide](https://docs.langchain.com/oss/javascript/deepagents/customization) and [reference guide](https://docs.langchain.com/oss/javascript/deepagents/backends), with the EXAONE provider passed explicitly in place of the default model.
