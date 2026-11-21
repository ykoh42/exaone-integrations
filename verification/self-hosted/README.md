# Self-hosted smoke checks

These opt-in checks connect the four providers to an **already-running** OpenAI-compatible Chat Completions endpoint. They do not start a model server or install a server parser. Use the model card and the server's own documentation to select its supported model, revision, build, reasoning parser and tool parser.

From the repository root, install the Python packages in separate environments as in [CONTRIBUTING.md](../../CONTRIBUTING.md), run `npm ci` and `npm run build`, and run the four commands **serially**. The example uses shell variables for brevity; replace every placeholder with the actual deployment details:

```bash
export SELF_HOSTED_API_KEY='EMPTY' # Replace only if the server requires a credential.
STRANDS_PY=/path/to/strands/venv/bin/python
LANGCHAIN_PY=/path/to/langchain/venv/bin/python
common=(
  --base-url http://127.0.0.1:8000/v1
  --model YOUR_SERVED_MODEL_NAME
  --server vllm
  --server-version YOUR_SERVER_VERSION
  --model-revision YOUR_MODEL_REVISION
  --reasoning-parser qwen3
  --tool-parser YOUR_TOOL_PARSER
  --server-options 'context length and other non-secret settings'
  --thinking on
)
"$STRANDS_PY" verification/self-hosted/smoke.py --framework strands "${common[@]}"
"$LANGCHAIN_PY" verification/self-hosted/smoke.py --framework langchain "${common[@]}"
node verification/self-hosted/smoke.mjs --framework strands "${common[@]}"
node verification/self-hosted/smoke.mjs --framework langchain "${common[@]}"
```

Repeat with `--thinking off` and the SGLang deployment when supported. The explicit thinking value is sent as `chat_template_kwargs.enable_thinking`; this does not assume that `reasoning_effort` has equivalent semantics. Add `--image` only for an image-capable EXAONE 4.5 server. The optional `--max-tokens` defaults to 1024. The provided key is sent to the selected endpoint; the scripts use the placeholder `EMPTY` if `SELF_HOSTED_API_KEY` is unset, never the hosted EXAONE key automatically.

The checks cover non-streaming and streaming text and reasoning, a two-turn conversation, automatic tool selection and tool-result history in both modes, and optionally a small synthetic red PNG. They make serial requests and do not perform load testing. Tool choice `required` and named tool choice, malformed-call recovery, comprehensive image understanding, caching, and every server/parser version are outside this smoke check. A model can fail the expected answer or tool choice without a provider defect; investigate server responses separately.

The scripts output status and caller-supplied deployment metadata, plus installed provider and SDK versions. They do not intentionally print prompts, model output, reasoning, keys or the endpoint URL. Do not put credentials or sensitive values in metadata arguments or commit output that contains sensitive information; servers and framework logging may retain additional data.

## Model-free verification of the checks

This starts a local deterministic HTTP fixture, runs all four smoke programs serially with thinking on/off and image input, and verifies that missing or unexpected reasoning, a wrong tool argument, truncated tool calls and malformed streamed arguments fail. It makes no model calls:

```bash
python verification/self-hosted/check-mock.py \
  --strands-python "$STRANDS_PY" --langchain-python "$LANGCHAIN_PY" --node node
```

The fixture validates the smoke programs and provider wiring only; it is not evidence of actual model inference or server compatibility. The same check is available in the manual model-free workflow.
