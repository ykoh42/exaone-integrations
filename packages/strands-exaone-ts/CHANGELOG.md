# Changelog

## 0.1.2 (unreleased)

- Support Zod 4.5.4 and newer within the 4.x line, and verify Strands SDK 1.25 and Harness 0.4.

- Redact API keys from response metadata and parsed API error details.
- Keep API keys out of model object inspection.
- Retry response-read failures only before the first model event is exposed.
- Skip copying callback metadata when no callback is configured.
