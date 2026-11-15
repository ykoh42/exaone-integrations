# Changelog

## 0.1.3 (unreleased)

- Preserve SPDX license metadata with the minimum supported build backend.

- Restrict source and wheel distributions to intended package files and public examples.
- Exclude local `.env`, `.envrc`, `.direnv`, `.netrc`, `.npmrc`, and `.pypirc` files from built distributions while retaining example templates.
- Avoid applying the K-EXAONE context window to unknown model IDs.
- Reset retry state after a model call with no exception, matching the Strands hook contract.

## 0.1.2 (2026-11-14)

- Support Strands SDK 1.63 through 1.65 and test with Harness 0.4.
- Honor Strands' `is_retryable()` extension point while preventing retries after stream output is exposed.
- Document the SDK retry-delay event limitation of the custom retry strategy.
- Redact literal and URL-encoded API credentials from error details, response headers, and metadata callbacks.
- Prevent connection-error retries after a stream event has been exposed.
- Close an HTTP response when cancellation wins before it reaches the caller.
