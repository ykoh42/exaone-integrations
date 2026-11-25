"""EXAONE API Platform model provider for Hermes Agent."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_MODEL = "k-exaone_v2.5"
_EFFORTS = ("low", "medium", "high", "max")


class ExaoneProfile(ProviderProfile):
    """Map Hermes reasoning settings onto the EXAONE Chat Completions API."""

    def build_extra_body(self, *, model: str | None = None, **context: Any) -> dict[str, Any]:
        if model == _MODEL:
            return {"chat_template_kwargs": {"preserve_thinking": True}}
        return {}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if model != _MODEL:
            return {}, {}
        if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
            return {}, {}
        effort = reasoning_config.get("effort")
        return ({}, {"reasoning_effort": effort}) if effort and effort != "none" else ({}, {})


def _setup_reasoning_efforts(model: str) -> tuple[str, ...] | None:
    return _EFFORTS if model == _MODEL else None


register_provider(ExaoneProfile(
    name="exaone",
    display_name="EXAONE API Platform",
    description="K-EXAONE models via the EXAONE API Platform",
    signup_url="https://docs.exaone.ai/en/docs/overview",
    env_vars=("EXAONE_API_KEY", "EXAONE_BASE_URL"),
    base_url="https://api.exaone.ai/v1",
    auth_type="api_key",
    models_dev_id="exaone",
    fallback_models=(_MODEL,),
    model_capabilities={_MODEL: {
        "supports_reasoning": True,
        "supports_vision": False,
        "supports_tools": True,
        "context_window": 524288,
    }},
    setup_reasoning_efforts=_setup_reasoning_efforts,
))
