# EXAONE with Strands Harness

Run the Python model provider with Strands Harness. File reading is off unless you pass an explicit flag. This community-maintained integration is not affiliated with LG AI Research, Strands Agents, or Exa.

## Setup

Python 3.10 or newer is required. From this example directory, create an environment and install the local provider with the Harness:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'strands-harness~=0.3.4' ../..
export EXAONE_API_KEY='your-api-key'
```

The provider reads `EXAONE_API_KEY` from the environment; it does not load `.env` files automatically. By default, model requests are sent to the EXAONE API.

To use another compatible endpoint, set `EXAONE_BASE_URL` to its base URL, including `/v1`. If the variable is unset, the provider's default EXAONE endpoint is used.

## Run

To let the agent read a file you choose to share with the model, enable file reading explicitly:

```bash
.venv/bin/python agent.py --allow-file-read 'Read /absolute/path/to/a/text-file.txt and summarize it.'
```

The example sets `media=False` for `read`, so image and document files are described in text instead of being sent as unsupported media blocks. The model does not interpret their contents.

When file reading is enabled, the model can request any file allowed by the Harness sandbox. Tool results are sent to the configured model endpoint. Run the example in an isolated sandbox with only the files you intend to share.

To allow web search, pass `--web-search`:

```bash
.venv/bin/python agent.py --web-search 'Find the official Python documentation.'
```

This enables the third-party Exa search tool. **Search queries are sent to Exa**, subject to its [privacy policy](https://exa.ai/privacy-policy). Search is disabled by default. Set `EXA_API_KEY` in the environment if you have an Exa key; the Harness also supports keyless search with a lower rate limit. The model chooses whether to call the enabled tools.

The example uses `ExaoneRetryStrategy` so Strands waits for EXAONE's `Retry-After` hints instead of applying its generic throttling backoff. It disables the default printing callback, session persistence, memory, and skills. It uses the Harness default automatic context manager, which may summarize or remove older history. See the [Python package README](../../README.md) for reasoning preservation, configuration, and context-management details.

The default context manager and background-task support also register management tools for retrieving context and managing background work.

## Custom tools and conversation history

Run the support-queue example to let the Harness execute `get_ticket` and answer a follow-up in the same conversation:

```bash
.venv/bin/python tools.py
```

This example uses synthetic ticket data. It disables built-in file and search tools, session persistence, memory, and skills. Harness automatic context management remains enabled. Both turns keep the complete assistant messages and supported reasoning in history.

See the official [Harness quickstart](https://strandsagents.com/docs/user-guide/harness/quickstart/) for the framework APIs.
