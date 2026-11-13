# EXAONE with LangGraph

Build an explicit model → tool → model graph. A second turn uses the first result through the same in-memory checkpoint thread.

This example uses the community-maintained `langchain-exaone` provider.

## Setup

Use the Node.js version in the repository's `.node-version` and npm 11.20.0. From the repository root, build the provider, then install this example independently:

```bash
npm ci
npm run build --workspace=langchain-exaone
cd packages/langchain-exaone-ts/examples/langgraph
npm install --install-links --no-package-lock --workspaces=false
export EXAONE_API_KEY='your-api-key'
```

## Run

Run each command separately; each makes EXAONE API requests and consumes your account quota.

```bash
npm start
```

Run `npm run check` in this directory to type-check the examples. The manifest installs this checkout; replace `"langchain-exaone": "file:../.."` with a published version to run without the repository.

The examples print answer text only. Complete assistant messages, including supported reasoning, remain in the conversation sent to EXAONE. They do not trim or rebuild model responses. Framework context management can summarize or remove older messages; the provider forwards the history it receives. Checkpoints live only in this process; use a persistent checkpointer in your application to keep threads across restarts.

The examples read `EXAONE_API_KEY` from the environment and do not load `.env` files. `EXAONE_BASE_URL` optionally selects a trusted compatible endpoint, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [LangGraph guide](https://docs.langchain.com/oss/javascript/langgraph/quickstart) and [reference guide](https://docs.langchain.com/oss/javascript/langgraph/persistence); the provider is supplied explicitly instead of the guide's default model.
