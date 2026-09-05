from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import postgrest


class _Resp:
    """Minimal context-manager stand-in for urlopen's return."""

    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))


class PostgrestRequestTests(unittest.TestCase):
    def test_success_returns_status_and_body(self):
        with mock.patch("reclaim_repro.postgrest.urllib.request.urlopen",
                        return_value=_Resp(201, b'[{"ok":true}]')) as urlopen:
            code, text = postgrest.request(
                "https://p/rest/v1/t", service_key="k", body=[{"a": 1}], timeout=8.0)
        self.assertEqual((code, text), (201, '[{"ok":true}]'))
        # Request carries the JSON body + auth headers.
        req = urlopen.call_args.args[0]
        self.assertEqual(json.loads(req.data.decode()), [{"a": 1}])
        self.assertEqual(req.get_header("Apikey"), "k")
        self.assertEqual(req.get_header("Authorization"), "Bearer k")
        self.assertIsNone(req.get_header("Prefer"))

    def test_patch_prefer_and_method(self):
        with mock.patch("reclaim_repro.postgrest.urllib.request.urlopen",
                        return_value=_Resp(204)) as urlopen:
            code, _ = postgrest.request(
                "https://p/x", service_key="k", method="PATCH", body={"s": 1},
                prefer="return=minimal", timeout=8.0)
        self.assertEqual(code, 204)
        req = urlopen.call_args.args[0]
        self.assertEqual(req.get_method(), "PATCH")
        self.assertEqual(req.get_header("Prefer"), "return=minimal")

    def test_http_error_returns_code_and_error_body(self):
        with mock.patch("reclaim_repro.postgrest.urllib.request.urlopen",
                        side_effect=_http_error(409, b'{"message":"dup"}')):
            code, text = postgrest.request("https://p/x", service_key="k", body={}, timeout=8.0)
        self.assertEqual((code, text), (409, '{"message":"dup"}'))

    def test_transport_error_returns_zero(self):
        with mock.patch("reclaim_repro.postgrest.urllib.request.urlopen",
                        side_effect=OSError("Name or service not known")):
            code, text = postgrest.request("https://p/x", service_key="k", body={}, timeout=8.0)
        self.assertEqual(code, 0)
        self.assertIn("Name or service", text)

    def test_raw_body_sent_verbatim_with_content_type(self):
        with mock.patch("reclaim_repro.postgrest.urllib.request.urlopen",
                        return_value=_Resp(200)) as urlopen:
            postgrest.request("https://p/storage", service_key="k", raw=b"\x00\x01raw",
                              content="text/plain", timeout=8.0)
        req = urlopen.call_args.args[0]
        self.assertEqual(req.data, b"\x00\x01raw")
        self.assertEqual(req.get_header("Content-type"), "text/plain")

    def test_no_body_sends_no_data(self):
        with mock.patch("reclaim_repro.postgrest.urllib.request.urlopen",
                        return_value=_Resp(200, b"[]")) as urlopen:
            postgrest.request("https://p/x", service_key="k", method="GET", timeout=8.0)
        self.assertIsNone(urlopen.call_args.args[0].data)

    def test_retries_transport_error_then_succeeds(self):
        seq = [OSError("boom"), _Resp(200, b"ok")]

        def fake(*_a, **_k):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch("reclaim_repro.postgrest.urllib.request.urlopen", side_effect=fake), \
                mock.patch("reclaim_repro.postgrest.time.sleep") as sleep:
            code, text = postgrest.request(
                "https://p/x", service_key="k", body={}, timeout=8.0, max_attempts=3)
        self.assertEqual((code, text), (200, "ok"))
        sleep.assert_called_once()  # one backoff before the successful retry


if __name__ == "__main__":
    unittest.main()
