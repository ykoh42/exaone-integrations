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

Run each command separately; each makes EXAONE API requests and consumes your account quota.

```bash
.venv/bin/python chat.py
.venv/bin/python agent.py
.venv/bin/python structured_output.py
```

The requirements install this checkout. To use the published provider instead, replace `../..` with `langchain-exaone` in `requirements.txt`.

The examples print answer text only. Complete assistant messages, including supported reasoning, remain in the conversation sent to EXAONE. They do not trim or rebuild model responses. Framework context management can summarize or remove older messages; the provider forwards the history it receives. Checkpoints live only in this process; use a persistent checkpointer in your application to keep threads across restarts.

The examples read `EXAONE_API_KEY` from the environment and do not load `.env` files. `EXAONE_BASE_URL` optionally selects a trusted compatible endpoint, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [LangChain guide](https://docs.langchain.com/oss/python/langchain/agents) and [reference guide](https://docs.langchain.com/oss/python/langchain/structured-output); the provider is supplied explicitly instead of the guide's default model.
