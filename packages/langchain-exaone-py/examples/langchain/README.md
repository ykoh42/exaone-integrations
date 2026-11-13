# EXAONE with LangChain

Stream a chat response, run a calculator agent with conversation checkpoints, or extract a typed contact.

This example uses the community-maintained `langchain-exaone` provider.

## Setup

Python 3.11 or newer is required. From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export EXAONE_API_KEY='your-api-key'
```

## Run

Run the commands one at a time. Each makes real EXAONE API requests and consumes your account quota.

```bash
.venv/bin/python chat.py
.venv/bin/python agent.py
.venv/bin/python structured_output.py
```

The requirements install the provider from this checkout. To use the published package, replace `../..` with `langchain-exaone` in `requirements.txt`.

The examples print only the answer text. They keep complete assistant messages, including supported reasoning, in the history sent to EXAONE without trimming or rebuilding responses. The framework may summarize or remove older messages during context management; the provider forwards the resulting history. Checkpoints last for this process only. Use a persistent checkpointer to keep threads across restarts.

The examples read `EXAONE_API_KEY` from the environment without loading `.env` files. To use another trusted compatible endpoint, set `EXAONE_BASE_URL` to its base URL, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [LangChain guide](https://docs.langchain.com/oss/python/langchain/agents) and [reference guide](https://docs.langchain.com/oss/python/langchain/structured-output), with the EXAONE provider passed explicitly in place of the default model.
