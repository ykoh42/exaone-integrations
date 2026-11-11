# Repository guidance

- Keep each framework package independently buildable and installable. Test the built package when changing its packaging or public exports.
- Preserve EXAONE reasoning, content, and metadata that the framework can represent. Reject unsupported input or response shapes explicitly, and document framework and API limitations separately.
- Keep equivalent Python and TypeScript behavior aligned when their framework interfaces allow it.
- **Absolute SDK rule:** Treat the Strands Harness SDK and its underlying Strands SDK contracts as law, including behavior that appears surprising. Providers MUST follow their public APIs and conventions; adapt the provider, never modify, bypass, or “correct” an SDK. Document unsupported behavior as a provider limitation and propose changes upstream.
- Keep the provider's community-maintained status clear.
- Cover protocol changes with representative fixtures. Distinguish fixture coverage from live API observations and report unverified behavior.
- Run live EXAONE checks serially within the account's limits. Use synthetic inputs and avoid load tests. Never commit credentials, real response text, or reasoning; logprobs may contain generated tokens.
- Retry only before output is exposed; never replay a partially delivered stream or tool call.
- Record unexpected API behavior and its reproduction conditions in local `docs/`; keep credential-redacted request and response captures there when the user requests them. These working files must stay out of Git.
- See `CONTRIBUTING.md` and the package READMEs for current setup and verification commands.
