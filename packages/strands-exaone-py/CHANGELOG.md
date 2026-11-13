# Changelog

## 0.1.2 (unreleased)

- Redact literal and URL-encoded API credentials from error details, response headers, and metadata callbacks.
- Prevent connection-error retries after a stream event has been exposed.
- Close an HTTP response when cancellation wins before it reaches the caller.
