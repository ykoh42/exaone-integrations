# Changelog

## 0.1.5 (2026-11-20)

- Start with Deep Agents and include a runnable example of opt-in tool-call recovery.

## 0.1.4 (2026-11-19)

- Add optional `toolCallRecovery()` middleware for malformed tool arguments in LangChain agents.
- Forward explicit tool strictness while retaining per-tool settings when omitted.
- Align response types with numeric and string `matched_stop` values.
- Use LangChain's standard message type field and verify newer SDK versions.

## 0.1.3 (2026-11-16)

- Preserve reported cache token details in LangChain usage.
- Avoid duplicating response-level metadata and raw usage when streaming chunks are combined.
- Forward LangChain user image URLs and base64 data to compatible self-hosted endpoints.
- Keep refusal metadata and v1 refusal blocks separate from answer text during streaming and ordinary responses.
- Respect removal of reasoning blocks from LangChain v1 message history.
- Preserve `reasoning` from compatible server responses and accept HTTP endpoints supplied by the user.
- Keep hosted model facts and request limits scoped to the hosted endpoint; forward custom server settings.
- Process CR-only stream event boundaries and completion markers without waiting for more data.
- Reject non-text assistant content, reasoning, and refusal fields instead of discarding them.
- Preserve assistant annotations and audio in additional kwargs, and retain incomplete streamed usage in response metadata.
- Reject unsupported tool-call types and malformed streamed content or tool-call fields rather than dropping data.
- Forward reasoning, tool-call, and usage stream chunks to LangChain callbacks.
- Exclude local `.env`, `.envrc`, and `.direnv` state from built distributions while retaining example templates.
- Include TypeScript sources referenced by declaration maps in the npm package.
- Preserve LangChain `FunctionMessage` results in legacy function-call histories.
- Support LangChain JSON Schema and JSON object structured-output methods and explicit tool strictness.

## 0.1.2 (2026-11-14)

- Support Node.js 20.3 and `@langchain/core` 1.2.0 and newer in the 1.x line.
- Respect edited and removed LangChain tool calls in conversation history while retaining matching provider metadata.
- Expose reasoning and tool calls through LangChain's standard content blocks and accept completed v1 history.
- Preserve invalid tool calls and validate Zod structured output with LangChain's tool parser.
- Support multiple non-streaming candidates through LangChain `generate()` and aggregate request usage.
- Redact raw and URL-encoded API keys from API and transport error diagnostics.
- Avoid retaining malformed response payloads in parser exceptions.
- Reject malformed tool-call shapes instead of silently dropping them.
- Preserve numeric EXAONE `result_code` values on typed API errors.
- Add package-specific registry links and editor documentation for public options.
