# EXAONE provider for Hermes Agent

A community-maintained model provider for using the EXAONE API Platform with Hermes Agent.

Install with `hermes plugins install 'ykoh42/exaone-integrations#packages/hermes-exaone'`. Set `EXAONE_API_KEY` and select **EXAONE API Platform** in `hermes model`. The provider uses `https://api.exaone.ai/v1` by default. `EXAONE_BASE_URL` can override the endpoint.

The K-EXAONE 2.5 profile preserves reasoning across turns and passes Hermes reasoning levels to the API. The API supports `low`, `medium`, `high`, and `max`; it always reasons, so selecting off does not disable reasoning. Sampling defaults are supplied by the API unless changed in Hermes.

Requests, including conversation content, tool schemas, and the API key, go to the configured EXAONE endpoint. This project is not affiliated with LG AI Research.

See the [EXAONE API documentation](https://docs.exaone.ai/en/docs/k-exaone-2.5) for model behavior and platform terms.
