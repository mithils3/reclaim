from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro.reference import (
    materialize_reference,
    safe_target,
    write_paper,
)

# A synthetic bundle row. The supplement .py carries only ``content`` bytes (the
# builder never fills ``text`` for .py) -- exactly the case the materializer must
# handle by writing from bytes, not text.
ROW = {
    "arxiv_id": "2505.11483",
    "title": "Test paper",
    "paper_tex_files": [
        {"relative_path": "main.tex", "text": "\\documentclass{article}\\begin{document}x\\end{document}"},
        {"relative_path": "sec/intro.tex", "text": "intro body"},
    ],
    "supplement_files": [
        {"relative_path": "code/train.py", "content": b"print('hello')\n"},
        {"relative_path": "data/weights.bin", "content": b"\x00\x01\x02\x03"},
        {"relative_path": "README.txt", "text": "a readme"},
    ],
}


class WritePaperTests(unittest.TestCase):
    def test_materializes_latex_supplement_and_python_from_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "ref"
            counts = write_paper(ROW, dest)
            self.assertEqual(counts["latex_files"], 2)
            self.assertEqual(counts["supplement_files"], 3)
            self.assertEqual(
                (dest / "latex" / "main.tex").read_text(),
                ROW["paper_tex_files"][0]["text"],
            )
            self.assertEqual((dest / "latex" / "sec" / "intro.tex").read_text(), "intro body")
            # .py comes from raw content bytes, not the (absent) text field.
            self.assertEqual((dest / "supplement" / "code" / "train.py").read_bytes(), b"print('hello')\n")
            self.assertEqual((dest / "supplement" / "data" / "weights.bin").read_bytes(), b"\x00\x01\x02\x03")
            self.assertEqual((dest / "supplement" / "README.txt").read_bytes(), b"a readme")
            self.assertTrue((dest / "info.json").is_file())

    def test_manifest_lists_every_file(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "ref"
            write_paper(ROW, dest)
            manifest = (dest / "MANIFEST.txt").read_text()
            self.assertIn("latex/main.tex", manifest)
            self.assertIn("supplement/code/train.py", manifest)
            self.assertIn("bytes total", manifest)


class MaterializeReferenceTests(unittest.TestCase):
    def test_materialize_from_row_is_offline(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "reference"
            result = materialize_reference("2505.11483", dest, row=ROW)
            self.assertTrue(result["ok"])
            self.assertFalse(result["skipped"])
            self.assertTrue((dest / "supplement" / "code" / "train.py").is_file())
            self.assertTrue((dest / "MANIFEST.txt").is_file())

    def test_skips_already_materialized_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "reference"
            materialize_reference("2505.11483", dest, row=ROW)
            again = materialize_reference("2505.11483", dest, row=ROW)
            self.assertTrue(again["ok"])
            self.assertTrue(again["skipped"])


class SafeTargetTests(unittest.TestCase):
    def test_rejects_traversal_and_absolute(self):
        base = Path("/base")
        self.assertIsNone(safe_target(base, "../escape.tex"))
        self.assertIsNone(safe_target(base, "/etc/passwd"))
        self.assertIsNone(safe_target(base, ""))
        self.assertEqual(safe_target(base, "a/b.tex"), base / "a" / "b.tex")


if __name__ == "__main__":
    unittest.main()
