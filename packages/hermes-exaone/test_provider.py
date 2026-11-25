"""Behavior checks against Hermes' real provider and transport interfaces."""

import copy
import unittest

from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


class ExaoneProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = get_provider_profile("exaone")
        self.assertIsNotNone(self.profile)
        self.transport = ChatCompletionsTransport()

    def build(self, messages, **kwargs):
        return self.transport.build_kwargs(
            "k-exaone_v2.5", messages, None, provider_profile=self.profile,
            base_url="https://api.exaone.ai/v1", **kwargs,
        )

    def test_reasoning_levels_and_preservation(self):
        for effort in ("low", "medium", "high", "max"):
            with self.subTest(effort=effort):
                request = self.build([{"role": "user", "content": "hi"}],
                                     reasoning_config={"enabled": True, "effort": effort})
                self.assertEqual(request["reasoning_effort"], effort)
                self.assertEqual(request["extra_body"]["chat_template_kwargs"], {"preserve_thinking": True})

    def test_explicit_template_override_and_no_unsupported_disable(self):
        request = self.build([{"role": "user", "content": "hi"}],
                             reasoning_config={"enabled": False},
                             request_overrides={"extra_body": {"chat_template_kwargs": {"preserve_thinking": False}}})
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(request["extra_body"]["chat_template_kwargs"], {"preserve_thinking": False})

    def test_reasoning_and_tool_history_is_replayed_without_mutating_input(self):
        messages = [
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": None, "reasoning_content": "Need the tool.",
             "tool_calls": [{"id": "call_test", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_test", "content": "found"},
            {"role": "user", "content": "summarize"},
        ]
        original = copy.deepcopy(messages)
        request = self.build(messages)
        self.assertEqual(messages, original)
        self.assertEqual(request["messages"], original)

    def test_reasoning_picker_and_unrelated_provider(self):
        from hermes_cli.main_provider_setup import _main_model_reasoning_efforts

        self.assertEqual(_main_model_reasoning_efforts("k-exaone_v2.5", "exaone"),
                         ["low", "medium", "high", "max"])
        other = get_provider_profile("fireworks")
        request = self.transport.build_kwargs(
            "accounts/fireworks/models/test", [{"role": "user", "content": "hi"}], None,
            provider_profile=other, base_url="https://api.fireworks.ai/inference/v1",
        )
        self.assertNotIn("chat_template_kwargs", request.get("extra_body", {}))

    def test_provider_applies_no_exaone_parameters_to_another_model(self):
        self.assertEqual(self.profile.build_extra_body(model="other"), {})
        body, kwargs = self.profile.build_api_kwargs_extras(model="other", reasoning_config={"effort": "high"})
        self.assertEqual(body, {})
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
