# Model and server compatibility

These community-maintained integrations connect frameworks to an existing Chat Completions endpoint. They do not install or operate a model server. The server, model, template, parser configuration, and framework version all affect the result. The table records evidence, not a guarantee for other versions or configurations.

**Verification levels:** Provider fixtures use synthetic endpoint responses and the framework APIs. Isolated server-code checks execute selected server parsers or message-processing code with synthetic inputs and test doubles; they are not a running inference server. Real-model verification requires an actual model generating responses on a running endpoint.

| Model and endpoint | Provider fixtures | Isolated server-code checks | Real-model verification |
| --- | --- | --- | --- |
| Hosted K-EXAONE 2.5 API | All four packages | Not applicable | Previously checked against the hosted API, including reasoning and tool flows. |
| K-EXAONE 2.0, vLLM | All four packages | Model-card fork and selected upstream versions; reasoning and tool parsers | Not yet verified. |
| K-EXAONE 2.0, SGLang | All four packages | Model-card fork and selected upstream revisions; reasoning and tool parsers | Not yet verified. |
| EXAONE 4.5, vLLM | All four packages, including image input and tool history | vLLM 0.27.0 message normalization and template; selected Hermes parser revisions | Not yet verified. |
| EXAONE 4.5, SGLang | All four packages, including image input and tool history | Model-card fork's message processing and streaming handler; selected Hermes parser revisions | Not yet verified. |

The 2.5 hosted checks do not verify either self-hosted model. No full vLLM or SGLang inference server, model weights, actual self-hosted model generation, or production scheduling was exercised in the model-free checks. Framework-level synthetic fixtures also cannot verify what a server ultimately sends to a model.

## Server versions and parser configurations

The [K-EXAONE 2.0 model card](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B) shows `qwen3` reasoning with `qwen3_xml` tools for vLLM and `qwen3` reasoning with `qwen3_coder` tools for SGLang. The [EXAONE 4.5 model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) shows `qwen3` reasoning and `hermes` tools for both servers. These are server configurations, not provider options or a claim that every release supports each model.

| Model | Server code inspected with synthetic inputs | Scope |
| --- | --- | --- |
| K-EXAONE 2.0 | Model-card vLLM fork `33fa995`, official vLLM 0.27.0 `4bdc8a7`, upstream `1489434` | Selected Qwen3 reasoning and tool parsing code, with test doubles for unrelated dependencies. |
| K-EXAONE 2.0 | Model-card SGLang fork `918653d`, upstream `d9ec74e` | Selected `qwen3` reasoning and `qwen3_coder` tool detectors. |
| EXAONE 4.5 | Official vLLM 0.27.0 `4bdc8a7`, upstream `1489434` | Selected Hermes parser code; 0.27.0 message normalization and official template with a stub image tracker. |
| EXAONE 4.5 | Model-card SGLang fork `6757c9f` | Selected message-processing code, official template, streaming handler and Hermes parser using synthetic inputs and test doubles. The Hermes parser condition was also reproduced in isolation in official SGLang 0.5.23 `297e54f` and upstream `e4c3f00`; this does not establish model support in those revisions. |

The official model templates were pinned at K-EXAONE 2.0 `fec0c8d` and EXAONE 4.5 `570aa4b`. These observations were recorded on or before 2026-11-16. Server revisions and settings can change; check your deployed combination. Selected parser observations can be reproduced using the [model-free probes](verification/model-free/README.md).

## Known server-side observations

- The inspected SGLang [`qwen3_coder` fork](https://github.com/lkm2835/sglang/blob/918653d22b2f1ae853b81fc0f5539b7c9311487e/python/sglang/srt/function_call/qwen3_coder_detector.py#L104) and [upstream parser](https://github.com/sgl-project/sglang/blob/d9ec74eb72b856419c6eb44733e00193d28ed840/python/sglang/srt/function_call/qwen3_coder_detector.py#L112) converted `null` text to JSON null for a string parameter in isolated checks. This can fail framework tool validation.
- With the EXAONE 4.5 template and vLLM 0.27.0's automatic content-format detection, an isolated message-normalization check moved an image ahead of surrounding text by default. The [`--interleave-mm-strings`](https://docs.vllm.ai/en/v0.27.0/cli/serve/#interleave-mm-strings) setting preserved order in that check. The provider preserves order in its request; server settings determine what reaches the model.
- The EXAONE 4.5 template could omit earlier assistant reasoning when the first user turn contained an image followed by tool history. A preceding text-only user turn changed the result. With automatic tool choice, the inspected SGLang fork could emit a tool name without arguments when a synthetic call and stop arrived in the same final event. See [SGLang issue #31915](https://github.com/sgl-project/sglang/issues/31915) and [proposed fix #32167](https://github.com/sgl-project/sglang/pull/32167). Both observations come from isolated checks.

The providers do not repair these server-side behaviors. Consult the package READMEs for framework-specific input support and details, and verify the chosen server version, template and settings in your own deployment.

## Framework and language versions

The model-free compatibility work exercised selected combinations across Python 3.10–3.14, Strands Python SDK 1.63–1.65, LangChain Core Python 1.3.3 and 1.8.0, Node 20.3–24, Strands TypeScript SDK 1.24–1.25, and LangChain Core TypeScript 1.2.0 and 1.4.1. Not every cross-product was tested; Node 20 applies to LangChain, while Strands TypeScript requires Node 22 or newer. The package manifests specify dependency ranges; the evidence above does not certify all versions within those ranges. See [CONTRIBUTING.md](CONTRIBUTING.md) for current verification commands.
