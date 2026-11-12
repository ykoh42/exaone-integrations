"""Collect response metadata without duplicating message or reasoning content."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_FIELDS = (
    "id",
    "object",
    "created",
    "model",
    "metadata",
    "service_tier",
    "system_fingerprint",
    "moderation",
    "result_code",
    "description",
)
_CHOICE_FIELDS = ("index", "finish_reason", "logprobs", "matched_stop")
_MESSAGE_FIELDS = ("refusal", "annotations", "audio", "function_call")


class ResponseMetadata:
    """Merge the metadata fields of consecutive Chat Completions chunks."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.response: dict[str, Any] = {}
        self._choice: dict[str, Any] = {}

    def add(self, chunk: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for key in _FIELDS:
            if key in chunk and (chunk[key] is not None or key not in self.response):
                self.response[key] = deepcopy(chunk[key])
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if isinstance(choice, dict):
                self._add_choice(choice)
        if choices:
            self.response["choices"] = [self._choice]

    def _add_choice(self, choice: dict[str, Any]) -> None:
        for key in _CHOICE_FIELDS:
            if key not in choice:
                continue
            value = choice[key]
            if key == "logprobs" and isinstance(value, dict):
                self._merge_logprobs(value)
            elif value is not None or key not in self._choice:
                self._choice[key] = deepcopy(value)
        message = choice.get("message")
        if isinstance(message, dict):
            self._add_message(message)
        delta = choice.get("delta")
        if isinstance(delta, dict):
            self._add_delta(delta)

    def _add_message(self, message: dict[str, Any]) -> None:
        fields = {key: deepcopy(message[key]) for key in _MESSAGE_FIELDS if key in message}
        if fields:
            self._choice["message"] = fields

    def _add_delta(self, delta: dict[str, Any]) -> None:
        message = self._choice.setdefault("message", {})
        for key in ("refusal", "annotations"):
            if key not in delta:
                continue
            value = delta[key]
            previous = message.get(key)
            if isinstance(previous, str) and isinstance(value, str) and key == "refusal":
                message[key] = previous + value
            elif isinstance(previous, list) and isinstance(value, list) and key == "annotations":
                previous.extend(deepcopy(value))
            elif value is not None or key not in message:
                message[key] = deepcopy(value)
        if not message:
            self._choice.pop("message", None)
        raw = {key: deepcopy(delta[key]) for key in ("audio", "function_call") if key in delta}
        if raw:
            self._choice.setdefault("message_deltas", []).append(raw)

    def _merge_logprobs(self, value: dict[str, Any]) -> None:
        previous = self._choice.get("logprobs")
        result: dict[str, Any] = previous if isinstance(previous, dict) else {}
        for key, item in value.items():
            if key in {"content", "refusal"} and isinstance(item, list) and isinstance(result.get(key), list):
                result[key].extend(deepcopy(item))
            elif item is not None or key not in result:
                result[key] = deepcopy(item)
        self._choice["logprobs"] = result
