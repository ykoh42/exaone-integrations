# Changelog

## 0.1.3 (unreleased)

- Preserve SPDX license metadata with the minimum supported build backend.

- Respect call-level stop sequences over `model_kwargs` defaults.
- Restrict source and wheel distributions to intended package files and public examples.
- Reject non-text assistant content, reasoning, and refusal fields instead of discarding them.
- Stop reading streams at the completion marker in synchronous and asynchronous calls.
- Limit default model profile fields to those supported by the installed LangChain version.
- Reject response tool-call types that cannot be represented as function tools.
- Exclude local `.env`, `.envrc`, `.direnv`, `.netrc`, `.npmrc`, and `.pypirc` files from built distributions while retaining example templates.
- Support LangChain JSON Schema and JSON object structured-output methods and explicit tool strictness.

## 0.1.2 (2026-11-14)

- Support `langchain-core` 1.3.3 and newer in the 1.x line.
- Respect edited and removed LangChain tool calls in conversation history while retaining matching provider metadata.
- Preserve LangChain message names when sending system, user, assistant, and generic chat messages.
- Preserve reasoning and tool calls in LangChain v1 content-block conversation history, including streamed results.
- Normalize boolean `bind_tools()` choices according to LangChain's OpenAI-compatible convention.
- Aggregate request usage in LangChain `generate()` results and reject malformed candidates instead of dropping them.
- Redact raw and URL-encoded API keys from API and transport error diagnostics.
- Avoid retaining malformed response payloads in parser exceptions.
- Reject malformed tool-call shapes instead of silently dropping them.
- Preserve top-level API result codes when a nested error omits its code.
- Add temperature and medium reasoning support to the default model profile.
- Add package-specific registry links and parameter descriptions to the model schema.
