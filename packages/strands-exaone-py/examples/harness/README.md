# EXAONE with Strands Harness

Use the Python model provider with Strands Harness. File reading is disabled by default and requires an explicit flag. This community-maintained integration is not affiliated with LG AI Research, Strands Agents, or Exa.

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

The example configures `read` with `media=False`: image and document files are described as text instead of being sent as unsupported media blocks. This does not give the model image or document understanding. The tool is disabled unless you pass `--allow-file-read`.

When enabled, the model can request any file the Harness sandbox permits it to read, and tool results are sent to the configured model endpoint. Run the example in an isolated sandbox containing only files you intend to share.

To allow web search, pass `--web-search`:

```bash
.venv/bin/python agent.py --web-search 'Find the official Python documentation.'
```

This enables the third-party Exa search tool. **Search queries are sent to Exa**, subject to its [privacy policy](https://exa.ai/privacy-policy). Search is disabled by default. Set `EXA_API_KEY` in the environment if you have an Exa key; the Harness also supports keyless search with a lower rate limit. The model chooses whether to call the enabled tools.

The example uses `ExaoneRetryStrategy` so Strands waits for EXAONE's `Retry-After` hints instead of applying its generic throttling backoff. It disables the default printing callback, session persistence, memory, and skills. It uses the Harness default automatic context manager, which may summarize or remove older history. See the [Python package README](../../README.md) for reasoning preservation, configuration, and context-management details.

The default context manager and background-task support also register management tools for retrieving context and managing background work.
