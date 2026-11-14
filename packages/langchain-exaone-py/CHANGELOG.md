# Changelog

## 0.1.2 (unreleased)

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
