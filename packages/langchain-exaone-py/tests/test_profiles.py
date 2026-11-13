"""Default model profiles and explicit LangChain overrides."""

import json

import httpx

from langchain_exaone import ChatEXAONE


def test_known_profile_and_unsupported_modalities() -> None:
    profile = ChatEXAONE(api_key="synthetic").profile
    assert profile is not None
    assert profile["max_input_tokens"] == 524_288
    assert profile["text_inputs"] and profile["text_outputs"]
    assert profile["reasoning_output"] and profile["tool_calling"]
    assert profile["tool_choice"] and profile["tool_call_streaming"]
    assert profile["reasoning_effort_levels"] == ["low", "high", "max"]
    assert profile["reasoning_effort_default"] == "max"
    assert all(
        profile[key] is False
        for key in (
            "image_inputs",
            "image_url_inputs",
            "image_tool_message",
            "pdf_tool_message",
            "pdf_inputs",
            "audio_inputs",
            "video_inputs",
            "image_outputs",
            "audio_outputs",
            "video_outputs",
        )
    )
    assert "max_output_tokens" not in profile
    assert "structured_output" not in profile


def test_unknown_models_have_no_inferred_profile() -> None:
    assert ChatEXAONE(model="unknown-model", api_key="synthetic").profile is None


def test_explicit_profile_replaces_defaults() -> None:
    custom = {"max_input_tokens": 4096, "tool_calling": False}
    assert ChatEXAONE(api_key="synthetic", profile=custom).profile == custom
    assert ChatEXAONE(api_key="synthetic", profile={}).profile == {}
    assert ChatEXAONE(api_key="synthetic", model="unknown-model", profile=custom).profile == custom


def test_default_profiles_are_not_shared_between_instances() -> None:
    first = ChatEXAONE(api_key="synthetic")
    second = ChatEXAONE(api_key="synthetic")
    first.profile["max_input_tokens"] = 4096
    first.profile["reasoning_effort_levels"].append("custom")
    assert second.profile["max_input_tokens"] == 524_288
    assert second.profile["reasoning_effort_levels"] == ["low", "high", "max"]


def test_profile_is_not_sent_as_an_api_parameter() -> None:
    bodies: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        model = ChatEXAONE(api_key="synthetic", http_client=client)
        assert model.invoke("Hello.").text == "Hello."
    assert "profile" not in bodies[0]
    assert "max_input_tokens" not in bodies[0]
