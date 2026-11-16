# Roadmap

This roadmap tracks planned improvements to the community-maintained EXAONE integrations for Strands and LangChain in Python and TypeScript. Work can be merged independently as each issue is ready; the linked GitHub issues are the source of truth for status and acceptance criteria.

The integrations connect to a model already served at an endpoint supplied by the caller. The server operator chooses and runs the model, serving engine, chat template, and server-side parsers; the provider does not manage them. Provider options should respect the selected endpoint's contract without guessing server settings or silently translating between thinking controls. Keep the hosted EXAONE 2.5 API-key-only defaults unchanged.

## 1. K-EXAONE 2.0

First, support [connecting to K-EXAONE 2.0 on compatible self-hosted endpoints](https://github.com/ykoh42/exaone-integrations/issues/6). The related implementation work can be merged one issue at a time:

- [Preserve reasoning from OpenAI-compatible server response fields](https://github.com/ykoh42/exaone-integrations/issues/2) — completed
- [Preserve fragmented tool calls and framework execution semantics](https://github.com/ykoh42/exaone-integrations/issues/5)

These changes must preserve the existing hosted EXAONE 2.5 behavior. Protocol compatibility can be tested locally; real-model inference remains unverified until a suitable server is available.

## 2. EXAONE 4.5

Next, [support EXAONE 4.5 on self-hosted endpoints](https://github.com/ykoh42/exaone-integrations/issues/7), including image input through each framework's public message interface alongside text, reasoning, and tools.

## Server and parser coverage

For compatibility work, use the publisher's official model-card server examples as initial test references, not as requirements imposed on users. Evaluate vLLM and SGLang independently, then check upstream releases with confirmed model support and other relevant versions or forks where there is an identified need. Record the exact server version or revision and the configured reasoning and tool-call parsers for each verified combination. The model cards currently show these server examples:

| Model | Server | Reasoning parser | Tool-call parser |
| --- | --- | --- | --- |
| [K-EXAONE 2.0](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B) | vLLM | `qwen3` | `qwen3_xml` |
| K-EXAONE 2.0 | SGLang | `qwen3` | `qwen3_coder` |
| [EXAONE 4.5](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B) | vLLM | `qwen3` | `hermes` |
| EXAONE 4.5 | SGLang | `qwen3` | `hermes` |

These are documented examples, not a guarantee for every server version or fork. Parser configuration belongs to the inference server. The [compatibility matrix](COMPATIBILITY.md) tracks protocol fixtures, isolated server-code checks, and real-model inference separately.

## Shared recovery and verification

Address these improvements as they are needed while bringing up each model, with normal responses kept unchanged:

- [Recover misplaced reasoning without changing normal assistant text](https://github.com/ykoh42/exaone-integrations/issues/3)
- [Use tool schemas to guide malformed argument recovery](https://github.com/ykoh42/exaone-integrations/issues/4)
- [Verify K-EXAONE 2.0 and EXAONE 4.5 on real inference servers](https://github.com/ykoh42/exaone-integrations/issues/8)
- [Document supported EXAONE models, endpoints, and verification levels](https://github.com/ykoh42/exaone-integrations/issues/9) — completed

Real-model verification for K-EXAONE 2.0 and EXAONE 4.5 is pending a suitable inference environment. Protocol fixtures and hosted EXAONE API checks can support development, but they do not establish that a self-hosted model has been tested.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and the repository guidance for independent package builds, framework contracts, representative tests, and safe live checks. Do not commit credentials, real response text, or reasoning.
