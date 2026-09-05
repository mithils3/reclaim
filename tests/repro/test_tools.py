from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import evidence
from reclaim_repro.context import ExecutionContext
from reclaim_repro.tools.files import edit_file, write_file
from reclaim_repro.tools.plan import update_plan
from reclaim_repro.tools.workspace_bash import workspace_bash


def _ctx(root: Path) -> ExecutionContext:
    ws, ref, ev = root / "workspace", root / "reference", root / "evidence"
    for p in (ws, ref, ev):
        p.mkdir(parents=True, exist_ok=True)
    evidence.init_evidence(ev)
    return ExecutionContext(arxiv_id="x", workspace=ws, reference=ref, evidence=ev)


class FileToolTests(unittest.TestCase):
    def test_reference_copy_is_never_writable(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.reference / "paper.tex").write_text("ref body")
            blocked = write_file({"path": str(ctx.reference / "paper.tex"), "content": "x"}, ctx)
            self.assertFalse(blocked["ok"])
            self.assertIn("writable", blocked["error"])

    def test_write_creates_file_in_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            res = write_file({"path": "src/model.py", "content": "x = 1\n"}, ctx)
            self.assertTrue(res["ok"])
            self.assertEqual((ctx.workspace / "src" / "model.py").read_text(), "x = 1\n")

    def test_write_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            self.assertFalse(write_file({"path": "../escape.txt", "content": "x"}, ctx)["ok"])
            self.assertFalse(write_file({"path": "/etc/passwd", "content": "x"}, ctx)["ok"])


class ContainerPathTranslationTests(unittest.TestCase):
    """The agent uses the short in-container /repro paths; the host-side file tools must
    translate them back to the episode's real dirs and keep the same confinement."""

    def test_repro_workspace_path_maps_to_host_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            res = write_file({"path": "/repro/workspace/src/m.py", "content": "x=1\n"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual((ctx.workspace / "src" / "m.py").read_text(), "x=1\n")

    def test_repro_evidence_path_is_writable(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            res = write_file({"path": "/repro/evidence/notes.md", "content": "hi\n"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual((ctx.evidence / "notes.md").read_text(), "hi\n")

    def test_repro_reference_path_stays_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            blocked = write_file({"path": "/repro/reference/paper.tex", "content": "x"}, ctx)
            self.assertFalse(blocked["ok"])

    def test_repro_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            evil = {"path": "/repro/workspace/../../etc/passwd", "content": "x"}
            self.assertFalse(write_file(evil, ctx)["ok"])


class EditFileTests(unittest.TestCase):
    def test_edit_applies_and_is_saved(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.workspace / "a.txt").write_text("a\nb\nc\n")
            res = edit_file({"path": "a.txt", "old_string": "b", "new_string": "B"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["replacements"], 1)
            self.assertEqual((ctx.workspace / "a.txt").read_text(), "a\nB\nc\n")
            self.assertEqual(len(list((ctx.evidence / "patches").glob("*.diff"))), 1)
            self.assertIn("edit_file", (ctx.evidence / "commands.log").read_text())

    def test_replace_all(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.workspace / "a.txt").write_text("x\nx\nx\n")
            res = edit_file({"path": "a.txt", "old_string": "x", "new_string": "y", "replace_all": True}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["replacements"], 3)
            self.assertEqual((ctx.workspace / "a.txt").read_text(), "y\ny\ny\n")

    def test_ambiguous_match_without_replace_all_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.workspace / "a.txt").write_text("x\nx\n")
            res = edit_file({"path": "a.txt", "old_string": "x", "new_string": "y"}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("2 times", res["error"])
            self.assertEqual((ctx.workspace / "a.txt").read_text(), "x\nx\n")

    def test_not_found_error_includes_nearby_text_hint(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.workspace / "f.py").write_text("def f():\n    return 41\n    # tail\n")
            # model thinks the body is 'return 99'; the real text is 'return 41'
            res = edit_file(
                {"path": "f.py", "old_string": "    return 99", "new_string": "    return 0"}, ctx,
            )
            self.assertFalse(res["ok"])
            self.assertIn("not found", res["error"])
            self.assertIn("return 41", res["error"])  # quotes the real line
            self.assertEqual((ctx.workspace / "f.py").read_text(), "def f():\n    return 41\n    # tail\n")

    def test_empty_new_string_deletes(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.workspace / "a.txt").write_text("keep-this-drop\n")
            res = edit_file({"path": "a.txt", "old_string": "-drop", "new_string": ""}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual((ctx.workspace / "a.txt").read_text(), "keep-this\n")

    def test_repo_relative_path_is_suffix_resolved(self):
        """The clone-into-workspace failure: agent passes a repo-relative path from
        inside the cloned repo instead of the workspace-relative one."""
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            nested = ctx.workspace / "repo" / "src"
            nested.mkdir(parents=True)
            (nested / "x.py").write_text("a = 1\n")
            res = edit_file({"path": "src/x.py", "old_string": "a = 1", "new_string": "a = 2"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual((nested / "x.py").read_text(), "a = 2\n")
            self.assertEqual(res["resolved_path"], str(nested / "x.py"))

    def test_container_absolute_path_is_suffix_resolved(self):
        """Same confusion with the container workspace prefix stapled on:
        /repro/workspace/<repo-relative path> that misses still gets located."""
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            nested = ctx.workspace / "repo" / "src"
            nested.mkdir(parents=True)
            (nested / "x.py").write_text("a = 1\n")
            res = edit_file(
                {"path": "/repro/workspace/src/x.py", "old_string": "a = 1", "new_string": "a = 2"}, ctx,
            )
            self.assertTrue(res["ok"], res)
            self.assertEqual((nested / "x.py").read_text(), "a = 2\n")

    def test_suffix_hit_never_escapes_the_roots(self):
        """A symlinked dir inside the workspace must not let a suffix hit resolve
        outside the writable roots (rglob may or may not follow the link by Python
        version; either way the edit must be refused)."""
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            outside = Path(d) / "outside" / "cfg"
            outside.mkdir(parents=True)
            (outside / "x.py").write_text("a = 1\n")
            (ctx.workspace / "link").symlink_to(outside.parent)
            res = edit_file({"path": "cfg/x.py", "old_string": "a = 1", "new_string": "a = 2"}, ctx)
            self.assertFalse(res["ok"])
            self.assertEqual((outside / "x.py").read_text(), "a = 1\n")

    def test_ambiguous_suffix_match_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.workspace / "a" / "src").mkdir(parents=True)
            (ctx.workspace / "b" / "src").mkdir(parents=True)
            (ctx.workspace / "a" / "src" / "x.py").write_text("a = 1\n")
            (ctx.workspace / "b" / "src" / "x.py").write_text("a = 1\n")
            res = edit_file({"path": "src/x.py", "old_string": "a = 1", "new_string": "a = 2"}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("Ambiguous", res["error"])

    def test_edit_escaping_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            res = edit_file({"path": "../evil.txt", "old_string": "a", "new_string": "b"}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("'..'", res["error"])

    def test_arg_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            (ctx.workspace / "a.txt").write_text("a\nb\nc\n")
            res = edit_file({"file_path": "a.txt", "old": "b", "new": "B"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual((ctx.workspace / "a.txt").read_text(), "a\nB\nc\n")


class UpdatePlanTests(unittest.TestCase):
    PLAN = [
        {"step": "Install torch + verify GPU", "status": "completed"},
        {"step": "Clone repo and install deps", "status": "in_progress"},
        {"step": "Run the MRE and score", "status": "pending"},
    ]

    def test_records_plan_on_context_and_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            res = update_plan({"plan": self.PLAN, "explanation": "deps next"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual(ctx.plan, self.PLAN)
            self.assertEqual(res["explanation"], "deps next")
            rendered = (ctx.evidence / "plan.md").read_text()
            self.assertIn("[x] Install torch + verify GPU", rendered)
            self.assertIn("[~] Clone repo and install deps", rendered)
            self.assertIn("[ ] Run the MRE and score", rendered)

    def test_replaces_previous_plan(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            update_plan({"plan": self.PLAN}, ctx)
            newer = [{"step": "Write report", "status": "in_progress"}]
            update_plan({"plan": newer}, ctx)
            self.assertEqual(ctx.plan, newer)

    def test_rejects_two_in_progress(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            bad = [
                {"step": "a", "status": "in_progress"},
                {"step": "b", "status": "in_progress"},
            ]
            res = update_plan({"plan": bad}, ctx)
            self.assertFalse(res["ok"])
            self.assertIn("in_progress", res["error"])
            self.assertEqual(ctx.plan, [])

    def test_rejects_bad_status_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            self.assertFalse(update_plan({"plan": []}, ctx)["ok"])
            self.assertFalse(update_plan({"plan": [{"step": "x", "status": "doing"}]}, ctx)["ok"])
            self.assertFalse(update_plan({"plan": [{"step": "", "status": "pending"}]}, ctx)["ok"])


class WorkspaceBashTests(unittest.TestCase):
    def test_runs_in_workspace_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = _ctx(Path(d))
            res = workspace_bash({"command": "echo hi > out.txt"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertEqual((ctx.workspace / "out.txt").read_text().strip(), "hi")
            self.assertIn("echo hi > out.txt", (ctx.evidence / "commands.log").read_text())

    def test_clone_local_repo(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / "src_repo"
            src.mkdir()
            (src / "hello.txt").write_text("hi\n")
            env = "-c user.email=t@t -c user.name=t"
            subprocess.run(["bash", "-lc", f"git init -q {src} && cd {src} && git {env} add -A && git {env} commit -qm init"], check=True)
            ctx = _ctx(root / "ep")
            res = workspace_bash({"command": f"git clone -q {src} code"}, ctx)
            self.assertTrue(res["ok"], res)
            self.assertTrue((ctx.workspace / "code" / "hello.txt").is_file())


if __name__ == "__main__":
    unittest.main()
