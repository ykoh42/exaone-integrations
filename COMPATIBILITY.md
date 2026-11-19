# Model and server compatibility

Hosted [K-EXAONE 2.5](https://docs.exaone.ai/ko/docs/k-exaone-2.5) is the text-only default and is verified with live API calls. The self-hosted recipes below are covered by provider fixtures and isolated server checks. End-to-end inference on these recipes still needs verification.

## Server versions and parser configurations

Use the model card to select the server build and configure its reasoning and tool-call parsers. Connect the provider to the running Chat Completions endpoint with its base URL, served model name, and credential.

| Model | Server | Reasoning parser | Tool-call parser |
| --- | --- | --- | --- |
| [K-EXAONE 2.0](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B) | vLLM | `qwen3` | `qwen3_xml` |
| K-EXAONE 2.0 | SGLang | `qwen3` | `qwen3_coder` |
| [EXAONE 4.5](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) | vLLM | `qwen3` | `hermes` |
| EXAONE 4.5 | SGLang | `qwen3` | `hermes` |

Upstream vLLM also lists K-EXAONE 2.0 among its [supported models](https://docs.vllm.ai/en/stable/models/supported_models/).

## Known server-side observations

- Affected SGLang `qwen3_coder` revisions can emit JSON null for the text `null` in a string argument. Check this case if your tools accept that literal string. [Parser source](https://github.com/sgl-project/sglang/blob/d9ec74eb72b856419c6eb44733e00193d28ed840/python/sglang/srt/function_call/qwen3_coder_detector.py#L112)
- With EXAONE 4.5 on vLLM 0.27.0, check image and text order when using string chat templates. [`--interleave-mm-strings`](https://docs.vllm.ai/en/v0.27.0/cli/serve/#interleave-mm-strings) preserves interleaving.
- With EXAONE 4.5 on the model-card SGLang fork, verify that multimodal tool turns retain earlier reasoning and that streamed tool calls with `auto` include their arguments. [SGLang parser issue](https://github.com/sgl-project/sglang/issues/31915)

See the [package guides](README.md#get-started) for SDK requirements and the [model-free probes](verification/model-free/README.md) to reproduce the isolated checks.
