# Changelog

## 0.1.2 (unreleased)

- Redact API keys from response metadata and parsed API error details.
- Keep API keys out of model object inspection.
- Retry response-read failures only before the first model event is exposed.
- Skip copying callback metadata when no callback is configured.
