from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.tools import output


class StripProgressTests(unittest.TestCase):
    def test_collapses_cr_rewrites_to_the_final_frame(self):
        tqdm = "\r 10%|█   | 10/100\r 50%|██  | 50/100\r100%|████| 100/100"
        text = f"loading\n{tqdm}\ndone: f1=0.87\n"
        shaped = output.strip_progress(text)
        self.assertIn("100%|████| 100/100", shaped)
        self.assertNotIn("10%", shaped)
        self.assertIn("done: f1=0.87", shaped)

    def test_crlf_files_are_not_mangled(self):
        self.assertEqual(output.strip_progress("a\r\nb\r\n"), "a\nb\n")

    def test_plain_text_is_untouched(self):
        self.assertEqual(output.strip_progress("x\ny\n"), "x\ny\n")


class ClipTests(unittest.TestCase):
    def test_short_text_is_returned_whole(self):
        text, clipped = output.clip("short", 100)
        self.assertEqual(text, "short")
        self.assertFalse(clipped)

    def test_long_text_keeps_head_and_tail(self):
        blob = "HEAD " + "x" * 10_000 + " TAIL: acc=0.9"
        text, clipped = output.clip(blob, 500, note=" — full log: /repro/evidence/gpu_step_0000.log")
        self.assertTrue(clipped)
        self.assertLess(len(text), 700)  # limit + the elision marker
        self.assertTrue(text.startswith("HEAD"))
        self.assertIn("TAIL: acc=0.9", text)  # the result line survives
        self.assertIn("chars elided", text)
        self.assertIn("gpu_step_0000.log", text)


class ShapeAndTailTests(unittest.TestCase):
    def test_shape_strips_then_clips(self):
        spam = "".join(f"\r{i}%" for i in range(1000))
        text, clipped = output.shape(f"{spam}\nresult=42\n", 200)
        self.assertFalse(clipped)  # the spam collapsed to one final frame
        self.assertIn("result=42", text)

    def test_tail_returns_last_chars_spam_stripped(self):
        blob = "\r1%\r2%\r99%\n" + "line\n" * 100 + "saved ckpt_7.pt"
        tail = output.tail(blob, 40)
        self.assertIn("ckpt_7.pt", tail)
        self.assertLessEqual(len(tail), 40)

    def test_shape_handles_none(self):
        self.assertEqual(output.shape(None, 10), ("", False))


if __name__ == "__main__":
    unittest.main()
