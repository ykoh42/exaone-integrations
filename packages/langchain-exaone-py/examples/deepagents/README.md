# EXAONE with Deep Agents

Ask a Deep Agent to plan a release brief, delegate to an EXAONE-backed reviewer, and save the brief in its virtual filesystem. The release notes are synthetic and require no search credentials.

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
.venv/bin/python agent.py
```

The requirements install this checkout. To use the published provider instead, replace `../..` with `langchain-exaone` in `requirements.txt`.

The examples print answer text only. Complete assistant messages, including supported reasoning, remain in the conversation sent to EXAONE. They do not trim or rebuild model responses. Framework context management can summarize or remove older messages; the provider forwards the history it receives.

The default StateBackend stores files in graph state, without reading or writing host files. The parent and named subagent use the same EXAONE model. Deep Agents owns tool orchestration and context management; the model chooses when to delegate or use filesystem tools. The in-memory checkpoint lasts for this process only. No shell-capable backend is configured.

The examples read `EXAONE_API_KEY` from the environment and do not load `.env` files. `EXAONE_BASE_URL` optionally selects a trusted compatible endpoint, including `/v1`.

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [Deep Agents guide](https://docs.langchain.com/oss/python/deepagents/customization) and [reference guide](https://docs.langchain.com/oss/python/deepagents/backends); the provider is supplied explicitly instead of the guide's default model.
