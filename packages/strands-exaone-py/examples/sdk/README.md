# EXAONE with Strands SDK

Run a typed calculator tool, stream a two-turn conversation, or extract a schema-validated contact.

This example uses the community-maintained `strands-exaone` provider.

## Setup

Python 3.10 or newer is required. From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export EXAONE_API_KEY='your-api-key'
```

## Run

Run each command separately; each makes EXAONE API requests and consumes your account quota.

```bash
.venv/bin/python agent.py
.venv/bin/python streaming.py
.venv/bin/python structured_output.py
```

The requirements install this checkout. To use the published provider instead, replace `../..` with `strands-exaone` in `requirements.txt`.

The examples print answer text only. Complete assistant messages, including supported reasoning, remain in the conversation sent to EXAONE. They do not trim or rebuild model responses. Framework context management can summarize or remove older messages; the provider forwards the history it receives. `ExaoneRetryStrategy` handles rate-limit hints at the Agent level. Each script owns its own agent; the streaming script reuses one agent for both turns.

The examples read `EXAONE_API_KEY` from the environment and do not load `.env` files. `EXAONE_BASE_URL` optionally selects a trusted compatible endpoint, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [Strands SDK guide](https://strandsagents.com/docs/user-guide/quickstart/) and [reference guide](https://strandsagents.com/docs/user-guide/sdk/streaming/); the provider is supplied explicitly instead of the guide's default model.
