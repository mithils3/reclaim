from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_vllm.vllm.io import build_chat_completion_request, response_finish_reason
from reclaim_vllm.schema.output import FINAL_RESPONSE_FORMAT


def args(**overrides):
    defaults = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 8192,
        "max_input_tokens": 128000,
        # build_chat_completion_request reads these directly (no fallback): every
        # real arg namespace (repro cli_resolve, vllm cli_args) always sets them.
        "tools": [{"type": "function", "function": {"name": "dummy_tool"}}],
        "response_format": FINAL_RESPONSE_FORMAT,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class VllmBatchRequestTests(unittest.TestCase):
    def test_final_request_uses_response_format(self) -> None:
        request = build_chat_completion_request(
            "model",
            "2501.00001",
            [{"role": "user", "content": "prompt"}],
            args(),
            include_tools=False,
        )

        body = request["body"]
        self.assertEqual(body["response_format"], FINAL_RESPONSE_FORMAT)
        self.assertNotIn("tools", body)

    def test_tool_request_does_not_use_response_format(self) -> None:
        request = build_chat_completion_request(
            "model",
            "2501.00001",
            [{"role": "user", "content": "prompt"}],
            args(),
            include_tools=True,
        )

        self.assertIn("tools", request["body"])
        self.assertNotIn("response_format", request["body"])

    def test_sampling_omitted_when_unset(self) -> None:
        # Unset sampling -> fields omitted from the body, so the served model's
        # generation_config defaults apply.
        body = build_chat_completion_request(
            "model",
            "2501.00001",
            [],
            args(temperature=None, top_p=None, top_k=None),
            include_tools=True,
        )["body"]
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertNotIn("top_k", body)


class ResponseFinishReasonTests(unittest.TestCase):
    def test_normal_row(self) -> None:
        row = {"response": {"body": {"choices": [{"finish_reason": "length"}]}}}
        self.assertEqual(response_finish_reason(row), "length")

    def test_empty_choices(self) -> None:
        row = {"response": {"body": {"choices": []}}}
        self.assertIsNone(response_finish_reason(row))

    def test_missing_body(self) -> None:
        self.assertIsNone(response_finish_reason({"response": {}}))


if __name__ == "__main__":
    unittest.main()
