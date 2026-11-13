# EXAONE with LangGraph

Build a graph that calls the model, runs a tool, and returns the result to the model. A second turn continues the same in-memory checkpoint thread using the first result.

This example uses the community-maintained `langchain-exaone` provider.

## Setup

Use the Node.js version in the repository's `.node-version` and npm 11.20.0. Build the provider from the repository root, then install the example's dependencies:

```bash
npm ci
npm run build --workspace=langchain-exaone
cd packages/langchain-exaone-ts/examples/langgraph
npm install --install-links --no-package-lock --workspaces=false
export EXAONE_API_KEY='your-api-key'
```

## Run

This command makes real EXAONE API requests and consumes your account quota.

```bash
npm start
```

Run `npm run check` in this directory to type-check the examples. The manifest installs this checkout; replace `"langchain-exaone": "file:../.."` with a published version to run without the repository.

The examples print only the answer text. They keep complete assistant messages, including supported reasoning, in the history sent to EXAONE without trimming or rebuilding responses. The framework may summarize or remove older messages during context management; the provider forwards the resulting history. Checkpoints last for this process only. Use a persistent checkpointer to keep threads across restarts.

The examples read `EXAONE_API_KEY` from the environment without loading `.env` files. To use another trusted compatible endpoint, set `EXAONE_BASE_URL` to its base URL, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [LangGraph guide](https://docs.langchain.com/oss/javascript/langgraph/quickstart) and [reference guide](https://docs.langchain.com/oss/javascript/langgraph/persistence), with the EXAONE provider passed explicitly in place of the default model.
