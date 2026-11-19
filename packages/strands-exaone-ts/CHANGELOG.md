# Changelog

## 0.1.4 (2026-11-20)

- Add opt-in model-stage middleware to recover malformed tool arguments.
- Respect Strands SDK session blocks and expired injected context while validating the latest user turn.
- Include the Strands CLI example's TypeScript config in the npm package.

## 0.1.3 (2026-11-16)

- Expose reported cache read and write counts in Strands usage metadata.
- Forward Strands user image bytes or URLs and text in order to compatible self-hosted endpoints.
- Limit tool-argument syntax completion to the hosted 2.5 endpoint and pass blank arguments through to Strands.
- Preserve `reasoning` responses from compatible self-hosted servers as Strands reasoning.
- Handle the `stop` finish reason used by vLLM for named tool calls.
- Accept HTTP endpoints supplied by the user.
- Keep hosted model facts and request limits scoped to the hosted endpoint; forward custom server settings.
- Exclude local `.env`, `.envrc`, and `.direnv` state from built distributions while retaining example templates.
- Avoid applying the K-EXAONE context window to unknown model IDs.

## 0.1.2 (2026-11-14)

- Support Zod 4.5.4 and newer within the 4.x line, and verify Strands SDK 1.25 and Harness 0.4.
- Redact API keys from response metadata and parsed API error details.
- Keep API keys out of model object inspection.
- Retry response-read failures only before the first model event is exposed.
- Skip copying callback metadata when no callback is configured.
