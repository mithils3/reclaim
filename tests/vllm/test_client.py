from __future__ import annotations

import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.vllm import client
from reclaim_vllm.vllm.endpoint import (
    ENV_API_KEY,
    ENV_CHAT_TEMPLATE_KWARGS,
    ENV_NO_TRUNCATE_PROMPT,
    ENV_OPENROUTER_PROVIDER,
    ENV_REASONING_EFFORT,
)

JSON_SCHEMA_RF = {
    "type": "json_schema",
    "json_schema": {"name": "v", "schema": {"type": "object"}},
}


class ApplyReasoningEffortTests(unittest.TestCase):
    def test_noop_when_unset(self) -> None:
        body = {"model": "muse-spark-1.2-contributor", "messages": []}
        with patch.dict("os.environ", {}, clear=True):
            client.apply_reasoning_effort(body)
        self.assertNotIn("reasoning_effort", body)

    def test_attaches_top_level_field(self) -> None:
        body = {"model": "muse-spark-1.2-contributor", "messages": []}
        with patch.dict("os.environ", {ENV_REASONING_EFFORT: "xhigh"}, clear=True):
            client.apply_reasoning_effort(body)
        self.assertEqual(body["reasoning_effort"], "xhigh")

    def test_does_not_clobber_existing(self) -> None:
        body = {"reasoning_effort": "low"}
        with patch.dict("os.environ", {ENV_REASONING_EFFORT: "xhigh"}, clear=True):
            client.apply_reasoning_effort(body)
        self.assertEqual(body["reasoning_effort"], "low")

    def test_ignores_whitespace_only(self) -> None:
        body = {"model": "m", "messages": []}
        with patch.dict("os.environ", {ENV_REASONING_EFFORT: "   "}, clear=True):
            client.apply_reasoning_effort(body)
        self.assertNotIn("reasoning_effort", body)

    def test_reaches_body_through_post_row(self) -> None:
        row = {"custom_id": "c1", "body": {"model": "m", "messages": []}}
        seen: dict = {}

        def _fake_post(base_url, body, timeout):
            seen["body"] = body
            return {"choices": []}

        with patch.dict("os.environ", {ENV_REASONING_EFFORT: "xhigh"}, clear=True):
            with patch.object(client, "post_vllm_chat_completion", _fake_post):
                client.post_chat_completion_row("http://h:8000", row, 30.0)
        self.assertEqual(seen["body"]["reasoning_effort"], "xhigh")


class DropTruncatePromptTokensTests(unittest.TestCase):
    def test_kept_when_unset(self) -> None:
        body = {"model": "m", "messages": [], "truncate_prompt_tokens": 967232}
        with patch.dict("os.environ", {}, clear=True):
            client.drop_truncate_prompt_tokens(body)
        self.assertEqual(body["truncate_prompt_tokens"], 967232)

    def test_dropped_when_set(self) -> None:
        body = {"model": "m", "messages": [], "truncate_prompt_tokens": 967232}
        with patch.dict("os.environ", {ENV_NO_TRUNCATE_PROMPT: "1"}, clear=True):
            client.drop_truncate_prompt_tokens(body)
        self.assertNotIn("truncate_prompt_tokens", body)

    def test_kept_for_falsey_spellings(self) -> None:
        for value in ("0", "false", "no", "", "  "):
            body = {"truncate_prompt_tokens": 42}
            with patch.dict("os.environ", {ENV_NO_TRUNCATE_PROMPT: value}, clear=True):
                client.drop_truncate_prompt_tokens(body)
            self.assertEqual(body["truncate_prompt_tokens"], 42, value)

    def test_absent_field_is_not_an_error(self) -> None:
        body = {"model": "m", "messages": []}
        with patch.dict("os.environ", {ENV_NO_TRUNCATE_PROMPT: "1"}, clear=True):
            client.drop_truncate_prompt_tokens(body)
        self.assertNotIn("truncate_prompt_tokens", body)

    def test_reaches_body_through_post_row(self) -> None:
        row = {
            "custom_id": "c1",
            "body": {"model": "m", "messages": [], "truncate_prompt_tokens": 967232},
        }
        seen: dict = {}

        def _fake_post(base_url, body, timeout):
            seen["body"] = body
            return {"choices": []}

        with patch.dict("os.environ", {ENV_NO_TRUNCATE_PROMPT: "1"}, clear=True):
            with patch.object(client, "post_vllm_chat_completion", _fake_post):
                client.post_chat_completion_row("http://h:8000", row, 30.0)
        self.assertNotIn("truncate_prompt_tokens", seen["body"])


class ApplyProviderRoutingTests(unittest.TestCase):
    def test_noop_when_unset(self) -> None:
        body = {"model": "deepseek/deepseek-v4-pro", "messages": []}
        with patch.dict("os.environ", {}, clear=True):
            client.apply_provider_routing(body, "https://openrouter.ai/api/v1")
        self.assertNotIn("provider", body)

    def test_pins_provider_when_set(self) -> None:
        body = {"model": "deepseek/deepseek-v4-pro", "messages": []}
        with patch.dict("os.environ", {ENV_OPENROUTER_PROVIDER: "deepseek"}, clear=True):
            client.apply_provider_routing(body, "https://openrouter.ai/api/v1")
        self.assertEqual(body["provider"], {"order": ["deepseek"], "allow_fallbacks": False})

    def test_does_not_clobber_existing_provider(self) -> None:
        body = {"provider": {"order": ["novita"]}}
        with patch.dict("os.environ", {ENV_OPENROUTER_PROVIDER: "deepseek"}, clear=True):
            client.apply_provider_routing(body, "https://openrouter.ai/api/v1")
        self.assertEqual(body["provider"], {"order": ["novita"]})

    def test_noop_on_a_non_openrouter_host(self) -> None:
        # A login shell exporting the knob must not aim `provider` at Meta/vLLM.
        body = {"model": "muse-spark-1.2-contributor", "messages": []}
        with patch.dict("os.environ", {ENV_OPENROUTER_PROVIDER: "deepseek"}, clear=True):
            client.apply_provider_routing(body, "https://api.meta.ai/v1")
        self.assertNotIn("provider", body)


class PostRowInjectsProviderTests(unittest.TestCase):
    def test_row_body_is_pinned_before_post(self) -> None:
        row = {"custom_id": "c1", "body": {"model": "m", "messages": []}}
        seen: dict = {}

        def _fake_post(base_url, body, timeout):
            seen["body"] = body
            return {"choices": []}

        with patch.dict("os.environ", {ENV_OPENROUTER_PROVIDER: "deepseek"}, clear=True):
            with patch.object(client, "post_vllm_chat_completion", _fake_post):
                client.post_chat_completion_row("https://openrouter.ai/api/v1", row, 1.0)

        self.assertEqual(
            seen["body"]["provider"], {"order": ["deepseek"], "allow_fallbacks": False}
        )


def _http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    exc = urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))
    exc.reclaim_body = body.decode()  # what retry.annotate_http_error stashes
    return exc


class PrepareStructuredOutputTests(unittest.TestCase):
    def test_marks_strict_and_requires_parameters_on_openrouter(self) -> None:
        body = {"response_format": JSON_SCHEMA_RF}
        with patch.dict("os.environ", {ENV_API_KEY: "sk-test"}, clear=True):
            client.prepare_structured_output(body, "https://openrouter.ai/api/v1")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertTrue(body["provider"]["require_parameters"])
        # the shared response_format constant must not be mutated in place
        self.assertNotIn("strict", JSON_SCHEMA_RF["json_schema"])

    def test_noop_on_a_keyless_vllm(self) -> None:
        body = {"response_format": JSON_SCHEMA_RF}
        with patch.dict("os.environ", {}, clear=True):
            client.prepare_structured_output(body, "http://gpu-node-01:8000")
        self.assertNotIn("provider", body)
        self.assertNotIn("strict", body["response_format"]["json_schema"])

    def test_noop_for_non_json_schema(self) -> None:
        body = {"response_format": {"type": "json_object"}}
        with patch.dict("os.environ", {ENV_API_KEY: "sk-test"}, clear=True):
            client.prepare_structured_output(body, "https://openrouter.ai/api/v1")
        self.assertNotIn("provider", body)

    def test_noop_on_a_keyed_non_openrouter_api(self) -> None:
        # Regression: the gate was `resolve_api_key() is not None`, so the
        # Meta Model API was handed a `provider` block and answered
        # 400 "unknown parameter `provider`" on the tools-off final pass of every run.
        body = {"response_format": JSON_SCHEMA_RF}
        with patch.dict("os.environ", {ENV_API_KEY: "meta-key"}, clear=True):
            client.prepare_structured_output(body, "https://api.meta.ai/v1")
        self.assertNotIn("provider", body)
        self.assertNotIn("strict", body["response_format"]["json_schema"])

    def test_final_pass_body_reaches_meta_clean(self) -> None:
        # End to end through the chokepoint: the exact shape that was 400ing.
        row = {"custom_id": "c1", "body": {"model": "m", "messages": [],
                                           "response_format": JSON_SCHEMA_RF}}
        seen: dict = {}

        def _fake_post(base_url, body, timeout):
            seen["body"] = body
            return {"choices": []}

        with patch.dict("os.environ", {ENV_API_KEY: "meta-key"}, clear=True):
            with patch.object(client, "post_vllm_chat_completion", _fake_post):
                client.post_chat_completion_row("https://api.meta.ai/v1", row, 1.0)
        self.assertNotIn("provider", seen["body"])
        self.assertEqual(seen["body"]["response_format"]["type"], "json_schema")


class DowngradeResponseFormatTests(unittest.TestCase):
    def test_downgrades_on_provider_reject_400(self) -> None:
        body = {"response_format": JSON_SCHEMA_RF, "provider": {"require_parameters": True}}
        exc = _http_error(400, b'{"error":{"message":"This response_format type is unavailable now"}}')
        out = client.downgrade_response_format_on_reject(body, exc)
        self.assertEqual(out["response_format"], {"type": "json_object"})
        self.assertNotIn("provider", out)  # require_parameters dropped, block emptied

    def test_downgrades_on_no_capable_provider_404(self) -> None:
        body = {"response_format": JSON_SCHEMA_RF, "provider": {"require_parameters": True}}
        exc = _http_error(404, b'{"error":{"message":"No endpoints found that can handle the requested parameters"}}')
        out = client.downgrade_response_format_on_reject(body, exc)
        self.assertEqual(out["response_format"], {"type": "json_object"})

    def test_keeps_pinned_provider_order_on_downgrade(self) -> None:
        body = {
            "response_format": JSON_SCHEMA_RF,
            "provider": {"order": ["deepseek"], "require_parameters": True},
        }
        exc = _http_error(404, b"No allowed providers are available")
        out = client.downgrade_response_format_on_reject(body, exc)
        self.assertEqual(out["provider"], {"order": ["deepseek"]})

    def test_no_downgrade_for_unrelated_400(self) -> None:
        body = {"response_format": JSON_SCHEMA_RF}
        exc = _http_error(400, b'{"error":{"message":"context length exceeded"}}')
        self.assertIsNone(client.downgrade_response_format_on_reject(body, exc))

    def test_no_downgrade_when_no_json_schema(self) -> None:
        body = {"response_format": {"type": "json_object"}}
        exc = _http_error(400, b"This response_format type is unavailable now")
        self.assertIsNone(client.downgrade_response_format_on_reject(body, exc))


class ApplyChatTemplateKwargsTests(unittest.TestCase):
    THINK_MAX = '{"thinking": true, "reasoning_effort": "max"}'

    def test_noop_when_unset(self) -> None:
        body = {"model": "m", "messages": []}
        with patch.dict("os.environ", {}, clear=True):
            client.apply_chat_template_kwargs(body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_attaches_think_max_when_set(self) -> None:
        body = {"model": "deepseek-ai/DeepSeek-V4-Flash", "messages": []}
        with patch.dict("os.environ", {ENV_CHAT_TEMPLATE_KWARGS: self.THINK_MAX}, clear=True):
            client.apply_chat_template_kwargs(body)
        self.assertEqual(
            body["chat_template_kwargs"], {"thinking": True, "reasoning_effort": "max"}
        )

    def test_does_not_clobber_existing(self) -> None:
        body = {"chat_template_kwargs": {"thinking": False}}
        with patch.dict("os.environ", {ENV_CHAT_TEMPLATE_KWARGS: self.THINK_MAX}, clear=True):
            client.apply_chat_template_kwargs(body)
        self.assertEqual(body["chat_template_kwargs"], {"thinking": False})

    def test_ignores_unparseable_json(self) -> None:
        body = {"model": "m", "messages": []}
        with patch.dict("os.environ", {ENV_CHAT_TEMPLATE_KWARGS: "not json"}, clear=True):
            client.apply_chat_template_kwargs(body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_reaches_body_through_post_row(self) -> None:
        row = {"custom_id": "c1", "body": {"model": "m", "messages": []}}
        seen: dict = {}

        def _fake_post(base_url, body, timeout):
            seen["body"] = body
            return {"choices": []}

        with patch.dict("os.environ", {ENV_CHAT_TEMPLATE_KWARGS: self.THINK_MAX}, clear=True):
            with patch.object(client, "post_vllm_chat_completion", _fake_post):
                client.post_chat_completion_row("http://h:8000", row, 1.0)

        self.assertEqual(
            seen["body"]["chat_template_kwargs"], {"thinking": True, "reasoning_effort": "max"}
        )


if __name__ == "__main__":
    unittest.main()
