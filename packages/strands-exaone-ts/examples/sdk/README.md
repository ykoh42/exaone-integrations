# EXAONE with Strands SDK

Run a typed calculator tool, stream a two-turn conversation, or extract a schema-validated contact.

This example uses the community-maintained `strands-exaone` provider.

## Setup

Use the Node.js version in the repository's `.node-version` and npm 11.20.0. From the repository root, build the provider, then install this example independently:

```bash
npm ci
npm run build --workspace=strands-exaone
cd packages/strands-exaone-ts/examples/sdk
npm install --install-links --no-package-lock --workspaces=false
export EXAONE_API_KEY='your-api-key'
```

## Run

Run each command separately; each makes EXAONE API requests and consumes your account quota.

```bash
npm run agent
npm run streaming
npm run structured-output
```

Run `npm run check` in this directory to type-check the examples. The manifest installs this checkout; replace `"strands-exaone": "file:../.."` with a published version to run without the repository.

The examples print answer text only. Complete assistant messages, including supported reasoning, remain in the conversation sent to EXAONE. They do not trim or rebuild model responses. Framework context management can summarize or remove older messages; the provider forwards the history it receives. `ExaoneRetryStrategy` handles rate-limit hints at the Agent level. Each script owns its own agent; the streaming script reuses one agent for both turns.

The examples read `EXAONE_API_KEY` from the environment and do not load `.env` files. `EXAONE_BASE_URL` optionally selects a trusted compatible endpoint, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [Strands SDK guide](https://strandsagents.com/docs/user-guide/quickstart/) and [reference guide](https://strandsagents.com/docs/user-guide/sdk/streaming/); the provider is supplied explicitly instead of the guide's default model.
