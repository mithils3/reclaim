from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import env, sandbox
from reclaim_repro.inputs import resolve_run_paths
from reclaim_repro.sandbox import (
    CONTAINER_EVIDENCE,
    CONTAINER_REFERENCE,
    CONTAINER_WORKSPACE,
    Bind,
    Sandbox,
    apptainer_usable,
    forward_env,
    from_run_paths,
    require_apptainer,
)
from reclaim_repro.workspace import create_layout

IMAGE = "/opt/images/pytorch_24.09-py3.sif"
# A real .sif to run the functional confinement test against; absent in most CI.
TEST_SIF = os.environ.get("REPRO_TEST_SIF")


class WrapArgvTests(unittest.TestCase):
    def _sandbox(self) -> Sandbox:
        return Sandbox(
            image=IMAGE,
            binds=(
                Bind("/tmp", "/tmp"),
                Bind("/host/ws", CONTAINER_WORKSPACE),
                Bind("/host/ref", CONTAINER_REFERENCE, ro=True),
            ),
            workdir=CONTAINER_WORKSPACE,
        )

    def test_prefix_pwd_and_remapped_binds(self):
        argv = self._sandbox().wrap_argv("cd /repro/workspace && echo hi")
        joined = " ".join(argv)
        # --pwd lands the step in the short container workspace; no `cd` to the long path
        self.assertEqual(argv[:6], ["apptainer", "exec", "--cleanenv", "--no-home", "--pwd", CONTAINER_WORKSPACE])
        self.assertEqual(argv[-3:], ["bash", "-lc", "cd /repro/workspace && echo hi"])
        self.assertEqual(argv[-4], IMAGE)  # image right before bash
        # node-local /tmp (same path); the episode dirs remapped onto short /repro paths
        self.assertIn("--bind /tmp", joined)
        self.assertIn("--bind /host/ws:/repro/workspace", joined)
        self.assertIn("--bind /host/ref:/repro/reference:ro", joined)

    def test_nv_only_when_requested(self):
        self.assertNotIn("--nv", self._sandbox().wrap_argv("echo hi"))  # CPU step
        self.assertIn("--nv", self._sandbox().wrap_argv("echo hi", nv=True))  # GPU step

    def test_gpu_steps_force_unbuffered_python(self):
        # GPU output streams to a teed evidence log; a block-buffered python would
        # lose its last chunk when SLURM kills the step at the --time wall.
        gpu = self._sandbox().wrap_argv("python t.py", nv=True)
        self.assertIn("PYTHONUNBUFFERED=1", gpu)
        self.assertEqual(gpu[gpu.index("PYTHONUNBUFFERED=1") - 1], "--env")
        self.assertNotIn("PYTHONUNBUFFERED=1", self._sandbox().wrap_argv("python t.py"))


class CpuCapTests(unittest.TestCase):
    def _sandbox(self, cpus, mem_gb=None) -> Sandbox:
        return Sandbox(
            image=IMAGE,
            binds=(Bind("/host/ws", CONTAINER_WORKSPACE),),
            workdir=CONTAINER_WORKSPACE,
            cpus=cpus,
            mem_gb=mem_gb,
        )

    def test_cpu_step_gets_taskset_prefix_and_max_jobs(self):
        with mock.patch.object(shutil, "which", return_value="/usr/bin/taskset"), \
                mock.patch.object(os, "sched_getaffinity", return_value=set(range(8)), create=True):
            argv = self._sandbox(2).wrap_argv("echo hi")
        self.assertEqual(argv[0], "taskset")
        self.assertEqual(argv[1], "-c")
        cores = argv[2].split(",")
        self.assertEqual(len(cores), 2)
        for c in cores:
            self.assertIn(int(c), range(8))
        self.assertIn("apptainer", argv)
        self.assertIn("--env", argv)
        idx = argv.index("MAX_JOBS=2")
        self.assertEqual(argv[idx - 1], "--env")

    def test_gpu_step_ignores_cpu_cap(self):
        with mock.patch.object(shutil, "which", return_value="/usr/bin/taskset"), \
                mock.patch.object(os, "sched_getaffinity", return_value=set(range(8)), create=True):
            argv = self._sandbox(2).wrap_argv("echo hi", nv=True)
        self.assertNotIn("taskset", argv)
        self.assertFalse(any(a.startswith("MAX_JOBS=") for a in argv))

    def test_cpus_none_keeps_shape_unchanged(self):
        with mock.patch.object(shutil, "which", return_value="/usr/bin/taskset"):
            argv = self._sandbox(None).wrap_argv("echo hi")
        self.assertNotIn("taskset", argv)
        self.assertFalse(any(a.startswith("MAX_JOBS=") for a in argv))
        self.assertEqual(argv[0], "apptainer")

    def test_missing_taskset_still_caps_build_env(self):
        with mock.patch.object(shutil, "which", return_value=None):
            argv = self._sandbox(2).wrap_argv("echo hi")
        self.assertNotIn("taskset", argv)
        self.assertIn("MAX_JOBS=2", argv)

    def test_wide_slice_keeps_build_fanout_at_four(self):
        # Compile RAM scales with job count; 12 cores speed non-build work but
        # builds stay at 4 jobs so they fit the 16G per-paper share.
        with mock.patch.object(shutil, "which", return_value="/usr/bin/taskset"), \
                mock.patch.object(os, "sched_getaffinity", return_value=set(range(72)), create=True):
            argv = self._sandbox(12).wrap_argv("echo hi")
        self.assertEqual(len(argv[2].split(",")), 12)
        self.assertIn("MAX_JOBS=4", argv)
        self.assertIn("MAKEFLAGS=-j4", argv)

    def test_mem_cap_prefixes_body_with_ulimit(self):
        with mock.patch.object(shutil, "which", return_value=None):
            argv = self._sandbox(2, mem_gb=16).wrap_argv("echo hi")
        self.assertEqual(argv[-1], f"ulimit -v {16 * 1024 * 1024} 2>/dev/null; echo hi")

    def test_gpu_step_body_is_never_ulimited(self):
        with mock.patch.object(shutil, "which", return_value=None):
            argv = self._sandbox(2, mem_gb=16).wrap_argv("echo hi", nv=True)
        self.assertEqual(argv[-1], "echo hi")


class RequireApptainerTests(unittest.TestCase):
    def test_require_raises_when_no_image(self):
        with self.assertRaises(SystemExit):
            require_apptainer(None)

    def test_require_raises_when_image_unusable(self):
        with mock.patch.object(sandbox, "_PROBE", {IMAGE: False}):
            with self.assertRaises(SystemExit):
                require_apptainer(IMAGE)

    def test_require_passes_when_usable(self):
        with mock.patch.object(sandbox, "_PROBE", {IMAGE: True}):
            require_apptainer(IMAGE)  # no raise


class ForwardEnvTests(unittest.TestCase):
    def test_hf_token_is_mirrored_for_cleanenv(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ["HF_TOKEN"] = "secret"
            forward_env()
            # forwarded via APPTAINERENV_* so --cleanenv keeps it (no token on argv)
            self.assertEqual(os.environ["APPTAINERENV_HF_TOKEN"], "secret")

    def test_uv_prefers_the_images_system_python(self):
        # So `uv venv --python 3.12` uses the bundled /usr/bin/python3.12 (headers at
        # /usr/include) rather than a managed download whose headers gcc can't find.
        with mock.patch.dict(os.environ, {}, clear=True):
            forward_env()
            self.assertEqual(os.environ["APPTAINERENV_UV_PYTHON_PREFERENCE"], "system")

    def test_home_writing_tool_dirs_redirect_into_bound_cache(self):
        # Under --no-home the container's $HOME is read-only, so uv's managed-Python
        # download dir is pointed at the rw-bound ~/.cache instead.
        with mock.patch.dict(os.environ, {}, clear=True):
            forward_env()
            self.assertTrue(
                os.environ["APPTAINERENV_UV_PYTHON_INSTALL_DIR"].endswith("/.cache/uv/python")
            )
            self.assertIn("/.cache", os.environ["APPTAINERENV_XDG_DATA_HOME"])


class FromRunPathsTests(unittest.TestCase):
    def test_remaps_episode_dirs_and_binds_caches(self):
        with tempfile.TemporaryDirectory() as d:
            paths = resolve_run_paths(Path(d) / "runs", "2505.11483", 8.0, run_id="RID")
            create_layout(paths)
            cache = Path(d) / "cache"
            sb = from_run_paths(paths, image=IMAGE, caches=[cache])
            self.assertEqual(sb.image, IMAGE)
            self.assertEqual(sb.workdir, CONTAINER_WORKSPACE)
            self.assertTrue(cache.is_dir())  # cache root created up front for the rw bind
            by_dst = {b.dst: b for b in sb.binds}
            # episode dirs remapped onto the short, stable container paths
            self.assertEqual(by_dst[CONTAINER_WORKSPACE].src, str(paths.workspace.resolve()))
            self.assertFalse(by_dst[CONTAINER_WORKSPACE].ro)
            self.assertEqual(by_dst[CONTAINER_EVIDENCE].src, str(paths.evidence.resolve()))
            self.assertTrue(by_dst[CONTAINER_REFERENCE].ro)  # reference is read-only
            # /tmp + the cache are bound at their own paths (rw)
            self.assertIn("/tmp", by_dst)
            self.assertEqual(by_dst[str(cache.resolve())].src, str(cache.resolve()))


@unittest.skipUnless(
    TEST_SIF and shutil.which("apptainer") and apptainer_usable(TEST_SIF),
    "no usable apptainer image (set REPRO_TEST_SIF to a real .sif to run)",
)
class FunctionalConfinementTests(unittest.TestCase):
    def test_writes_outside_the_episode_are_blocked(self):
        # `outside` lives under $HOME, which the container does NOT mount — a real
        # confinement check. The workspace is remapped to /repro/workspace and is the cwd,
        # so a relative write lands there; the out-of-bounds dir can't be reached.
        with tempfile.TemporaryDirectory() as wsroot, \
                tempfile.TemporaryDirectory(dir=Path.home()) as outside:
            ws = Path(wsroot) / "workspace"
            ws.mkdir()
            sb = Sandbox(
                image=TEST_SIF,
                binds=(Bind(str(ws.resolve()), CONTAINER_WORKSPACE),),
                workdir=CONTAINER_WORKSPACE,
            )
            cmd = (
                "echo ok > inside.txt; "
                f"(echo bad > {outside}/leak.txt && echo WROTE) || echo BLOCKED"
            )
            proc = subprocess.run(
                env.exec_argv(str(ws), cmd, sandbox=sb),
                capture_output=True, text=True, timeout=120,
            )
            # the workspace write lands (via the remapped bind); the unmounted $HOME is denied
            self.assertEqual((ws / "inside.txt").read_text().strip(), "ok")
            self.assertIn("BLOCKED", proc.stdout)
            self.assertFalse((Path(outside) / "leak.txt").exists())


if __name__ == "__main__":
    unittest.main()
