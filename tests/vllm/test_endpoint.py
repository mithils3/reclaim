from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.vllm.endpoint import (
    ENV_API_KEY,
    is_openrouter,
    ENV_CONTEXT_LENGTH,
    ENV_ENDPOINT_FILE,
    ENV_OPENROUTER_PROVIDER,
    ENV_SERVED_MODEL,
    ENV_SERVER_URL,
    auth_headers,
    fetch_served_context_length,
    normalize_server_url,
    openrouter_provider_routing,
    resolve_api_key,
    resolve_served_model,
    resolve_server_url,
)


class NormalizeTests(unittest.TestCase):
    def test_strips_trailing_slash(self) -> None:
        self.assertEqual(normalize_server_url("http://h:8000/"), "http://h:8000")

    def test_strips_trailing_v1(self) -> None:
        self.assertEqual(normalize_server_url("http://h:8000/v1"), "http://h:8000")

    def test_strips_v1_and_slash(self) -> None:
        self.assertEqual(normalize_server_url("http://h:8000/v1/"), "http://h:8000")


class ResolvePrecedenceTests(unittest.TestCase):
    def test_cli_flag_wins(self) -> None:
        with patch.dict("os.environ", {ENV_SERVER_URL: "http://env:8000"}, clear=False):
            self.assertEqual(resolve_server_url("http://cli:8000/v1"), "http://cli:8000")

    def test_env_url_when_no_flag(self) -> None:
        with patch.dict("os.environ", {ENV_SERVER_URL: "http://env:8000/"}, clear=False):
            self.assertEqual(resolve_server_url(None), "http://env:8000")

    def test_endpoint_file_when_no_flag_or_env(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vllm_endpoint.json"
            path.write_text(json.dumps({"base_url": "http://file:8000"}), encoding="utf-8")
            env = {ENV_ENDPOINT_FILE: str(path)}
            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(resolve_server_url(None), "http://file:8000")

    def test_none_when_nothing_set(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(resolve_server_url(None))

    def test_none_when_endpoint_file_missing(self) -> None:
        env = {ENV_ENDPOINT_FILE: "/no/such/endpoint.json"}
        with patch.dict("os.environ", env, clear=True):
            self.assertIsNone(resolve_server_url(None))


class ResolveServedModelTests(unittest.TestCase):
    def _patch_models(self, ids: list[str]):
        return patch(
            "reclaim_vllm.vllm.endpoint.fetch_served_models",
            return_value=list(ids),
        )

    def test_picks_only_advertised_model(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self._patch_models(["m/A"]):
            self.assertEqual(resolve_served_model("http://h:8000"), "m/A")

    def test_picks_first_when_several(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self._patch_models(["m/A", "m/B"]):
            self.assertEqual(resolve_served_model("http://h:8000"), "m/A")

    def test_cli_override_wins_when_advertised(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self._patch_models(["m/A", "m/B"]):
            self.assertEqual(resolve_served_model("http://h:8000", "m/B"), "m/B")

    def test_env_override_when_no_flag(self) -> None:
        env = {ENV_SERVED_MODEL: "m/B"}
        with patch.dict("os.environ", env, clear=True), self._patch_models(["m/A", "m/B"]):
            self.assertEqual(resolve_served_model("http://h:8000"), "m/B")

    def test_flag_beats_env(self) -> None:
        env = {ENV_SERVED_MODEL: "m/B"}
        with patch.dict("os.environ", env, clear=True), self._patch_models(["m/A", "m/B"]):
            self.assertEqual(resolve_served_model("http://h:8000", "m/A"), "m/A")

    def test_override_not_served_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self._patch_models(["m/A"]):
            with self.assertRaises(RuntimeError):
                resolve_served_model("http://h:8000", "m/typo")

    def test_no_models_advertised_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self._patch_models([]):
            with self.assertRaises(RuntimeError):
                resolve_served_model("http://h:8000")


class AuthHeaderTests(unittest.TestCase):
    def test_no_key_means_no_header(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(resolve_api_key())
            self.assertEqual(auth_headers(), {})

    def test_reclaim_env_key(self) -> None:
        with patch.dict("os.environ", {ENV_API_KEY: "sk-or-env"}, clear=True):
            self.assertEqual(auth_headers(), {"Authorization": "Bearer sk-or-env"})

    def test_openrouter_env_is_a_fallback(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-2"}, clear=True):
            self.assertEqual(resolve_api_key(), "sk-or-2")

    def test_openai_key_is_never_used(self) -> None:
        # Guard against leaking an OpenAI key to a different provider's URL.
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai"}, clear=True):
            self.assertIsNone(resolve_api_key())
            self.assertEqual(auth_headers(), {})


class ProviderRoutingTests(unittest.TestCase):
    def test_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(openrouter_provider_routing())

    def test_blank_is_treated_as_unset(self) -> None:
        with patch.dict("os.environ", {ENV_OPENROUTER_PROVIDER: "  ,  "}, clear=True):
            self.assertIsNone(openrouter_provider_routing())

    def test_single_provider_pins_with_no_fallback(self) -> None:
        with patch.dict("os.environ", {ENV_OPENROUTER_PROVIDER: "deepseek"}, clear=True):
            self.assertEqual(
                openrouter_provider_routing(),
                {"order": ["deepseek"], "allow_fallbacks": False},
            )

    def test_comma_list_keeps_order(self) -> None:
        with patch.dict(
            "os.environ", {ENV_OPENROUTER_PROVIDER: "deepseek, novita"}, clear=True
        ):
            self.assertEqual(
                openrouter_provider_routing(),
                {"order": ["deepseek", "novita"], "allow_fallbacks": False},
            )


class FetchServedContextLengthTests(unittest.TestCase):
    """The input ceiling comes from the server, so a wrong read caps or overruns the run."""

    def _patch_cards(self, cards: list[dict]):
        return patch(
            "reclaim_vllm.vllm.endpoint.fetch_model_cards",
            return_value=cards,
        )

    def test_reads_vllm_max_model_len(self) -> None:
        with self._patch_cards([{"id": "m", "max_model_len": 1048576}]):
            self.assertEqual(fetch_served_context_length("http://b", "m"), 1048576)

    def test_reads_openai_proxy_context_length(self) -> None:
        with self._patch_cards([{"id": "m", "context_length": 262144}]):
            self.assertEqual(fetch_served_context_length("http://b", "m"), 262144)

    def test_picks_the_named_model_not_the_first(self) -> None:
        cards = [
            {"id": "other", "max_model_len": 8192},
            {"id": "wanted", "max_model_len": 393216},
        ]
        with self._patch_cards(cards):
            self.assertEqual(fetch_served_context_length("http://b", "wanted"), 393216)

    def test_raises_rather_than_guessing_a_window(self) -> None:
        # Inventing a default here is exactly how a 1M-context model got capped at 128K.
        with self._patch_cards([{"id": "m"}]):
            with self.assertRaises(RuntimeError):
                fetch_served_context_length("http://b", "m")

    def test_ignores_a_nonsense_window(self) -> None:
        with self._patch_cards([{"id": "m", "max_model_len": 0}]):
            with self.assertRaises(RuntimeError):
                fetch_served_context_length("http://b", "m")

    def test_env_override_wins_over_a_card(self) -> None:
        with self._patch_cards([{"id": "m", "max_model_len": 8192}]):
            with patch.dict("os.environ", {ENV_CONTEXT_LENGTH: "1000000"}, clear=True):
                self.assertEqual(fetch_served_context_length("http://b", "m"), 1000000)

    def test_env_override_rescues_a_card_with_no_window(self) -> None:
        # The Meta Model API advertises only id/object/created/owned_by.
        with self._patch_cards([{"id": "muse-spark-1.2-contributor"}]):
            with patch.dict("os.environ", {ENV_CONTEXT_LENGTH: "1000000"}, clear=True):
                self.assertEqual(
                    fetch_served_context_length("http://b", "muse-spark-1.2-contributor"),
                    1000000,
                )

    def test_env_override_must_be_a_positive_int(self) -> None:
        for bad in ("not-a-number", "0", "-5"):
            with self._patch_cards([{"id": "m"}]):
                with patch.dict("os.environ", {ENV_CONTEXT_LENGTH: bad}, clear=True):
                    with self.assertRaises(RuntimeError):
                        fetch_served_context_length("http://b", "m")


class IsOpenRouterTests(unittest.TestCase):
    def test_openrouter_urls(self) -> None:
        for url in ("https://openrouter.ai/api/v1", "https://openrouter.ai",
                    "https://gateway.openrouter.ai/api/v1"):
            self.assertTrue(is_openrouter(url), url)

    def test_everything_else(self) -> None:
        for url in ("https://api.meta.ai/v1", "http://gpu-node-01:8000",
                    "https://api.openai.com/v1", "https://notopenrouter.ai/v1", ""):
            self.assertFalse(is_openrouter(url), url)


if __name__ == "__main__":
    unittest.main()
