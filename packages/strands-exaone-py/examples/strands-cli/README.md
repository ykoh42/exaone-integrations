# EXAONE with the Strands CLI

Run a Python Strands harness with the EXAONE model provider and a local example tool. This community-maintained integration is not affiliated with LG AI Research or the Strands Agents project.

## Setup

Python 3.10 or newer and Node.js 22 or newer are required. Create an environment for the agent and install the provider and Harness from PyPI:

```bash
cd packages/strands-exaone-py/examples/strands-cli
python3 -m venv .venv
.venv/bin/python -m pip install 'strands-exaone' 'strands-harness~=0.3.4'
npm install --no-package-lock
export EXAONE_API_KEY='your-api-key'
```

To test the provider from this checkout, replace `strands-exaone` with `-e ../..` in the install command.

The CLI runs `agent.py` with this directory's `.venv` interpreter. Provide the key through your environment or a trusted file selected with `--env-file`; the provider does not load `.env` files automatically.

## Run

From this directory, start an interactive session:

```bash
npx strands --agent ./agent.py
```

Ask the agent to use `get_example_number` and add three to its result, then ask a follow-up in the same conversation. This tool returns a fixed value and does not access external systems. The agent config disables built-in tools, memory, skills, and session persistence. The default context manager and background-task support register additional management tools, and the CLI can add its own tools. Conversation history remains available until you exit; model requests still use the EXAONE API. The interactive CLI can display model reasoning.

The CLI's skill-discovery and forwarding behavior depends on its version and configuration. Verify discovery settings before sending private skill content; use the direct Harness example when you need explicit tool configuration.

`preserve_thinking=True` requests preservation of earlier reasoning. The provider forwards available reasoning with the corresponding assistant and tool history. Set `EXAONE_BASE_URL` only when you need to use another compatible endpoint. See the [Python package README](../../README.md) for configuration and API behavior.

The harness uses automatic context management by default. It can summarize older conversation history, so `preserve_thinking` can only retain reasoning that remains in that history. If you require the original history to stay intact, pass `context_manager=False` to `create_harness`; then manage the conversation length yourself so input tokens plus `max_tokens` fit within the model's context limit.
