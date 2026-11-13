# EXAONE with Strands Harness

Run a prompt with the EXAONE provider and Strands Harness. File reading is off unless you pass an explicit flag. This community-maintained integration is not affiliated with LG AI Research or the Strands Agents project.

## Setup

The provider runs on Node.js 22 or newer. For this checkout, use the Node.js version in `.node-version` and npm 11.20.0. The published `strands-exaone` package is available on npm; this example installs the local checkout so you can test source changes:

```bash
npm ci
npm run build --workspace=strands-exaone
cd packages/strands-exaone-ts/examples/harness
npm install --install-links --no-package-lock
export EXAONE_API_KEY='your-api-key'
```

From this directory, pass a prompt. To let the agent read a file you choose to share, enable file reading explicitly and use an absolute path:

```bash
npm start -- --allow-file-read 'Summarize /absolute/path/to/notes.txt'
```

The example sets `media: false` for `read`, so images and binary documents are described in text without their contents being interpreted by the model. It uses `ExaoneRetryStrategy` to honor EXAONE's `Retry-After` hints and disables the other built-in tools, memory, skills, built-in plugins, and session persistence. Harness automatic context management remains enabled and can summarize older conversation history. The example prints only the final text; reasoning stays in the agent's conversation.

When file reading is enabled, the model can request any file allowed by the Harness sandbox. Tool results are sent to the configured model endpoint. Run the example in an isolated sandbox with only the files you intend to share.

The default context manager and background-task support also register management tools for retrieving context and managing background work.

## Optional web search

Set `EXA_WEB_SEARCH=true` to enable the Harness `web_search` tool, which uses [Exa](https://exa.ai/), a separate search service. Search queries go to Exa when the agent uses the tool and are subject to its [privacy policy](https://exa.ai/privacy-policy). `EXA_API_KEY` is optional; see the Harness documentation for search configuration.

```bash
export EXA_WEB_SEARCH=true
export EXA_API_KEY='your-exa-search-key'
npm start -- 'Search for the official TypeScript documentation'
```

`EXA_API_KEY` is for Exa search and is separate from `EXAONE_API_KEY`. The example does not load `.env` files automatically. `EXAONE_BASE_URL` can be set to another compatible API endpoint. See the [package README](../../README.md) for provider configuration and limitations.

## Custom tools and conversation history

Run the support-queue example to let the Harness execute `get_ticket` and answer a follow-up in the same conversation:

```bash
npm run tools
```

This example uses synthetic ticket data. It disables built-in file and search tools, session persistence, memory, and skills. Harness automatic context management remains enabled. Both turns keep the complete assistant messages and supported reasoning in history.

See the official [Harness quickstart](https://strandsagents.com/docs/user-guide/harness/quickstart/) for the framework APIs.
