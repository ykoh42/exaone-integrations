# EXAONE with Strands Harness

Run a prompt with the EXAONE provider and Strands Harness. File reading is disabled by default and requires an explicit flag. This community-maintained integration is not affiliated with LG AI Research or the Strands Agents project.

## Setup

The provider runs on Node.js 22 or newer. Building this checkout requires Node.js 22.13.0 or newer and npm 11.20.0. The `strands-exaone` package is not published yet, so install it from this checkout:

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

The example configures `read` with `media: false`. Images and binary documents receive a text description; their contents are not understood by the model. File reading is disabled unless you pass `--allow-file-read`. The example uses `ExaoneRetryStrategy` to honor EXAONE's `Retry-After` hints, and disables the other built-in tools, memory, skills, built-in plugins, and session persistence. The Harness still uses automatic context management, which can summarize older conversation history. The example prints final text only; reasoning remains available in the agent's conversation.

When enabled, the model can request any file the Harness sandbox permits it to read, and tool results are sent to the configured model endpoint. Run the example in an isolated sandbox containing only files you intend to share.

The default context manager and background-task support also register management tools for retrieving context and managing background work.

## Optional web search

Set `EXA_WEB_SEARCH=true` to enable the Harness `web_search` tool backed by [Exa](https://exa.ai/), a separate search service. When the agent uses this tool, its search queries are sent to Exa and are subject to Exa's [privacy policy](https://exa.ai/privacy-policy). `EXA_API_KEY` is optional; see the Harness documentation for its search configuration.

```bash
export EXA_WEB_SEARCH=true
export EXA_API_KEY='your-exa-search-key'
npm start -- 'Search for the official TypeScript documentation'
```

`EXA_API_KEY` is for Exa search and is separate from `EXAONE_API_KEY`. The example does not load `.env` files automatically. `EXAONE_BASE_URL` can be set to another compatible API endpoint. See the [package README](../../README.md) for provider configuration and limitations.
