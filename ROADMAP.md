# Roadmap

This roadmap tracks planned improvements to the community-maintained EXAONE integrations for Strands and LangChain in Python and TypeScript. Work can be merged independently as each issue is ready; the linked GitHub issues are the source of truth for status and acceptance criteria.

## 1. K-EXAONE 2.0

First, make [K-EXAONE 2.0 available on self-hosted endpoints](https://github.com/ykoh42/exaone-integrations/issues/6). The related implementation work can be merged one issue at a time:

- [Preserve reasoning from OpenAI-compatible server response fields](https://github.com/ykoh42/exaone-integrations/issues/2)
- [Preserve fragmented tool calls and framework execution semantics](https://github.com/ykoh42/exaone-integrations/issues/5)

These changes must preserve the existing hosted EXAONE 2.5 behavior. Protocol compatibility can be tested locally; real-model inference remains unverified until a suitable server is available.

## 2. EXAONE 4.5

Next, [support EXAONE 4.5 on self-hosted endpoints](https://github.com/ykoh42/exaone-integrations/issues/7), including image input through each framework's public message interface alongside text, reasoning, and tools.

## Server and parser coverage

For each model, evaluate vLLM and SGLang independently, including applicable upstream releases and model-card forks. Record the exact server version or revision and the configured reasoning and tool-call parsers. The model cards currently show these server examples:

| Model | Server | Reasoning parser | Tool-call parser |
| --- | --- | --- | --- |
| [K-EXAONE 2.0](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B) | vLLM | `qwen3` | `qwen3_xml` |
| K-EXAONE 2.0 | SGLang | `qwen3` | `qwen3_coder` |
| [EXAONE 4.5](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) | vLLM | `qwen3` | `hermes` |
| EXAONE 4.5 | SGLang | `qwen3` | `hermes` |

These are documented examples, not a guarantee for every server version or fork. Parser configuration belongs to the inference server. Track protocol fixtures, actual server execution, and real-model inference separately for each tested combination.

## Shared recovery and verification

Address these improvements as they are needed while bringing up each model, with normal responses kept unchanged:

- [Recover misplaced reasoning without changing normal assistant text](https://github.com/ykoh42/exaone-integrations/issues/3)
- [Use tool schemas to guide malformed argument recovery](https://github.com/ykoh42/exaone-integrations/issues/4)
- [Verify K-EXAONE 2.0 and EXAONE 4.5 on real inference servers](https://github.com/ykoh42/exaone-integrations/issues/8)
- [Document supported EXAONE models, endpoints, and verification levels](https://github.com/ykoh42/exaone-integrations/issues/9)

Real-model verification for K-EXAONE 2.0 and EXAONE 4.5 is pending a suitable inference environment. Protocol fixtures and hosted EXAONE API checks can support development, but they do not establish that a self-hosted model has been tested.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and the repository guidance for independent package builds, framework contracts, representative tests, and safe live checks. Do not commit credentials, real response text, or reasoning.
