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

This command makes real EXAONE API requests and consumes your account quota.

```bash
.venv/bin/python agent.py
```

The requirements install the provider from this checkout. To use the published package, replace `../..` with `langchain-exaone` in `requirements.txt`.

The examples print only the answer text. They keep complete assistant messages, including supported reasoning, in the history sent to EXAONE without trimming or rebuilding responses. The framework may summarize or remove older messages during context management; the provider forwards the resulting history.

The default StateBackend keeps files in graph state and does not read or write host files. The parent and named subagent use the same EXAONE model. Deep Agents manages tools and context; the model decides when to delegate or use filesystem tools. The in-memory checkpoint lasts for this process only, and the example configures no shell-capable backend.

The examples read `EXAONE_API_KEY` from the environment without loading `.env` files. To use another trusted compatible endpoint, set `EXAONE_BASE_URL` to its base URL, including `/v1`.

## Optional tool-call recovery

Enable recovery if malformed tool arguments interrupt your agent runs. The example runs with recovery disabled by default.

```bash
.venv/bin/python agent.py --recover-tool-calls
```

The flag attaches `ToolCallRecoveryMiddleware` to the parent Deep Agent. It can request one correction for eligible invalid JSON before response output is delivered. See the [provider's recovery options](../../README.md#optional-malformed-tool-recovery).

See the [package README](../../README.md) for provider configuration and limitations. These examples follow the public APIs in the official [Deep Agents guide](https://docs.langchain.com/oss/python/deepagents/customization) and [reference guide](https://docs.langchain.com/oss/python/deepagents/backends), with the EXAONE provider passed explicitly in place of the default model.
