# EXAONE with LangChain

Stream a chat response, run a calculator agent with conversation checkpoints, or extract a typed contact.

This example uses the community-maintained `langchain-exaone` provider.

## Setup

Use the Node.js version in the repository's `.node-version` and npm 11.20.0. Build the provider from the repository root, then install the example's dependencies:

```bash
npm ci
npm run build --workspace=langchain-exaone
cd packages/langchain-exaone-ts/examples/langchain
npm install --install-links --no-package-lock --workspaces=false
export EXAONE_API_KEY='your-api-key'
```

## Run

Run the commands one at a time. Each makes real EXAONE API requests and consumes your account quota.

```bash
npm run chat
npm run agent
npm run structured-output
```

Run `npm run check` in this directory to type-check the examples. The manifest installs this checkout; replace `"langchain-exaone": "file:../.."` with a published version to run without the repository.

The examples print only the answer text. They keep complete assistant messages, including supported reasoning, in the history sent to EXAONE without trimming or rebuilding responses. The framework may summarize or remove older messages during context management; the provider forwards the resulting history. Checkpoints last for this process only. Use a persistent checkpointer to keep threads across restarts.

The examples read `EXAONE_API_KEY` from the environment without loading `.env` files. To use another trusted compatible endpoint, set `EXAONE_BASE_URL` to its base URL, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [LangChain guide](https://docs.langchain.com/oss/javascript/langchain/agents) and [reference guide](https://docs.langchain.com/oss/javascript/langchain/structured-output), with the EXAONE provider passed explicitly in place of the default model.
