"""Model facts for the EXAONE API: https://docs.exaone.ai/ko/docs/k-exaone-2.5."""

from langchain_core.language_models.model_profile import ModelProfile


def get_model_profile(model: str) -> ModelProfile | None:
    """Return a fresh profile for a known model, or None when its facts are unknown."""
    if model != "k-exaone_v2.5":
        return None
    return {
        "max_input_tokens": 524_288,
        "text_inputs": True,
        "text_outputs": True,
        "image_inputs": False,
        "image_url_inputs": False,
        "image_tool_message": False,
        "pdf_tool_message": False,
        "pdf_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "reasoning_effort_levels": ["low", "medium", "high", "max"],
        "reasoning_effort_default": "max",
        "tool_calling": True,
        "tool_choice": True,
        "tool_call_streaming": True,
        "temperature": True,
    }
