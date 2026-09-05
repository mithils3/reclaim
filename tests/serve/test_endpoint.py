from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_serve.endpoint import build_payload, read_base_url, remove_endpoint, write_endpoint


def sample_payload() -> dict:
    return build_payload(
        base_url="http://1.2.3.4:8000",
        host="1.2.3.4",
        port=8000,
        served_model_name="MiniMaxAI/MiniMax-M2.7",
        model_path="/models/MiniMax-M2.7",
        node="gpu-node-01",
        job_id="100004",
    )


class PayloadTests(unittest.TestCase):
    def test_payload_keys_and_health(self) -> None:
        payload = sample_payload()
        self.assertEqual(payload["base_url"], "http://1.2.3.4:8000")
        self.assertEqual(payload["served_model_name"], "MiniMaxAI/MiniMax-M2.7")
        self.assertEqual(payload["health"], "ready")


class WriteReadRemoveTests(unittest.TestCase):
    def test_roundtrip_and_no_leftover_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vllm_endpoint.json"
            write_endpoint(path, sample_payload())
            self.assertEqual(read_base_url(path), "http://1.2.3.4:8000")
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())

    def test_remove_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vllm_endpoint.json"
            write_endpoint(path, sample_payload())
            remove_endpoint(path)
            self.assertFalse(path.exists())
            remove_endpoint(path)  # missing file must not raise

    def test_read_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(read_base_url(Path(d) / "absent.json"))

    def test_read_invalid_json_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(read_base_url(path))


if __name__ == "__main__":
    unittest.main()
