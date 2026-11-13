# EXAONE with Deep Agents

Ask a Deep Agent to plan a release brief, delegate to an EXAONE-backed reviewer, and save the brief in its virtual filesystem. The release notes are synthetic and require no search credentials.

This example uses the community-maintained `langchain-exaone` provider.

## Setup

Use the Node.js version in the repository's `.node-version` and npm 11.20.0. From the repository root, build the provider, then install this example independently:

```bash
npm ci
npm run build --workspace=langchain-exaone
cd packages/langchain-exaone-ts/examples/deepagents
npm install --install-links --no-package-lock --workspaces=false
export EXAONE_API_KEY='your-api-key'
```

## Run

Run each command separately; each makes EXAONE API requests and consumes your account quota.

```bash
npm start
```

Run `npm run check` in this directory to type-check the examples. The manifest installs this checkout; replace `"langchain-exaone": "file:../.."` with a published version to run without the repository.

The examples print answer text only. Complete assistant messages, including supported reasoning, remain in the conversation sent to EXAONE. They do not trim or rebuild model responses. Framework context management can summarize or remove older messages; the provider forwards the history it receives.

The default StateBackend stores files in graph state, without reading or writing host files. The parent and named subagent use the same EXAONE model. Deep Agents owns tool orchestration and context management; the model chooses when to delegate or use filesystem tools. The in-memory checkpoint lasts for this process only. No shell-capable backend is configured.

The examples read `EXAONE_API_KEY` from the environment and do not load `.env` files. `EXAONE_BASE_URL` optionally selects a trusted compatible endpoint, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [Deep Agents guide](https://docs.langchain.com/oss/javascript/deepagents/customization) and [reference guide](https://docs.langchain.com/oss/javascript/deepagents/backends); the provider is supplied explicitly instead of the guide's default model.
