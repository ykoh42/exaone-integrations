# Roadmap

This roadmap tracks planned improvements to the community-maintained EXAONE integrations for Strands and LangChain in Python and TypeScript. Work can be merged independently as each issue is ready; the linked GitHub issues are the source of truth for status and acceptance criteria.

1. [Preserve reasoning from OpenAI-compatible server response fields](https://github.com/ykoh42/exaone-integrations/issues/2)
2. [Recover misplaced reasoning without changing normal assistant text](https://github.com/ykoh42/exaone-integrations/issues/3)
3. [Use tool schemas to guide malformed argument recovery](https://github.com/ykoh42/exaone-integrations/issues/4)
4. [Preserve fragmented tool calls and framework execution semantics](https://github.com/ykoh42/exaone-integrations/issues/5)
5. [Support self-hosted EXAONE Chat Completions endpoints](https://github.com/ykoh42/exaone-integrations/issues/6)
6. [Support EXAONE 4.5 image input through framework-native messages](https://github.com/ykoh42/exaone-integrations/issues/7)
7. [Verify K-EXAONE 2.0 and EXAONE 4.5 on real inference servers](https://github.com/ykoh42/exaone-integrations/issues/8)
8. [Document supported EXAONE models, endpoints, and verification levels](https://github.com/ykoh42/exaone-integrations/issues/9)

Real-model verification for K-EXAONE 2.0 and EXAONE 4.5 is pending a suitable inference environment. Protocol fixtures and hosted EXAONE API checks can support development, but they do not establish that a self-hosted model has been tested.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and the repository guidance for independent package builds, framework contracts, representative tests, and safe live checks. Do not commit credentials, real response text, or reasoning.
