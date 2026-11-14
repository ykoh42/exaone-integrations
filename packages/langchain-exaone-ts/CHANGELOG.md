# Changelog

## 0.1.2 (unreleased)

- Expose reasoning and tool calls through LangChain's standard content blocks and accept completed v1 history.
- Preserve invalid tool calls and validate Zod structured output with LangChain's tool parser.
- Support multiple non-streaming candidates through LangChain `generate()` and aggregate request usage.
- Redact raw and URL-encoded API keys from API and transport error diagnostics.
- Avoid retaining malformed response payloads in parser exceptions.
- Reject malformed tool-call shapes instead of silently dropping them.
- Preserve numeric EXAONE `result_code` values on typed API errors.
- Add package-specific registry links and editor documentation for public options.
