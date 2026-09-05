from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_serve import network
from reclaim_serve.profiles import resolve_profile

SAMPLE_IP_OUTPUT = (
    "2: fab0    inet 192.0.2.10/24 brd 192.0.2.255 scope global fab0\\       valid_lft forever\n"
)
LOOPBACK_ONLY = "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever\n"


class DiscoverIpv4Tests(unittest.TestCase):
    def test_parses_fabric_ip(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=SAMPLE_IP_OUTPUT)
        with patch("reclaim_serve.network.subprocess.run", return_value=completed):
            self.assertEqual(network.discover_ipv4("fab0"), "192.0.2.10")

    def test_skips_loopback(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=LOOPBACK_ONLY)
        with patch("reclaim_serve.network.subprocess.run", return_value=completed):
            self.assertIsNone(network.discover_ipv4("lo"))


class AdvertisedHostTests(unittest.TestCase):
    def test_explicit_wins(self) -> None:
        self.assertEqual(network.advertised_host("9.9.9.9", "fab0"), "9.9.9.9")

    def test_falls_back_to_iface(self) -> None:
        with patch("reclaim_serve.network.discover_ipv4", return_value="192.0.2.10"):
            self.assertEqual(network.advertised_host(None, "fab0"), "192.0.2.10")

    def test_tries_rest_of_fabric_family_when_first_has_no_ip(self) -> None:
        # fab0 has no address; fab1 does. Must still publish a routable IP.
        def fake(iface: str):
            return "192.0.2.11" if iface == "fab1" else None

        with (
            patch("reclaim_serve.network.FABRIC_IFACES", ["fab0", "fab1"]),
            patch("reclaim_serve.network.discover_ipv4", side_effect=fake),
        ):
            self.assertEqual(network.advertised_host(None, "fab0"), "192.0.2.11")

    def test_never_publishes_loopback(self) -> None:
        with (
            patch("reclaim_serve.network.discover_ipv4", return_value=None),
            patch("reclaim_serve.network.hostname_ipv4", return_value=None),
            patch("reclaim_serve.network.socket.gethostname", return_value="gpu-node-01"),
        ):
            self.assertEqual(network.advertised_host(None, "fab0"), "gpu-node-01")

    def test_base_url_format(self) -> None:
        self.assertEqual(network.base_url("1.2.3.4", 8000), "http://1.2.3.4:8000")


class ProfileTests(unittest.TestCase):
    def test_minimax_by_local_path_suffix(self) -> None:
        self.assertEqual(resolve_profile("/models/MiniMax-M2.7").name, "minimax_m2")

    def test_minimax_default(self) -> None:
        profile = resolve_profile("MiniMaxAI/MiniMax-M2.7")
        self.assertEqual(profile.tool_call_parser, "minimax_m2")
        self.assertEqual(profile.tensor_parallel_size, 4)
        self.assertIsNotNone(profile.compilation_config)

    def test_deepseek_v4_by_id(self) -> None:
        profile = resolve_profile("deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(profile.name, "deepseek_v4")
        self.assertEqual(profile.tensor_parallel_size, 2)
        self.assertEqual(profile.tool_call_parser, "deepseek_v4")
        self.assertEqual(profile.reasoning_parser, "deepseek_v4")
        self.assertEqual(profile.kv_cache_dtype, "fp8")
        # Think Max needs >= 393216; a smaller default would silently cap it.
        self.assertGreaterEqual(profile.max_model_len, 393216)

    def test_deepseek_v4_not_misrouted_to_minimax(self) -> None:
        # DeepSeek must NOT fall through to the minimax default (TP=4, wrong parsers).
        self.assertNotEqual(resolve_profile("deepseek-ai/DeepSeek-V4-Flash").name, "minimax_m2")


if __name__ == "__main__":
    unittest.main()
