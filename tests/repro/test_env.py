from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import env
from reclaim_repro.sandbox import CONTAINER_WORKSPACE, Bind, Sandbox


class ExecArgvTests(unittest.TestCase):
    def test_bare_body_without_sandbox(self):
        # The pure builder shape: `cd <ws> && <cmd>`, no module load (the container
        # supplies the toolchain) and no container wrap (no sandbox passed).
        self.assertEqual(
            env.exec_argv("/ws", "git clone url"),
            ["bash", "-lc", "cd /ws && git clone url"],
        )

    def test_on_gpu_does_not_change_the_bare_body(self):
        # on_gpu only adds --nv *inside the sandbox*; with no sandbox the body is the
        # same — CUDA comes from the image, not a host `module load`.
        self.assertEqual(
            env.exec_argv("/ws", "python train.py", on_gpu=True),
            ["bash", "-lc", "cd /ws && python train.py"],
        )

    def test_sandbox_cds_to_container_workdir(self):
        # Under the sandbox the body cd's to the SHORT container workdir, not the long
        # host path passed in — the agent never sees the per-run path.
        sb = Sandbox(image="/img.sif", binds=(Bind("/host/ws", CONTAINER_WORKSPACE),))
        argv = env.exec_argv("/host/ws", "uv pip install x", on_gpu=True, sandbox=sb)
        self.assertEqual(argv[0], "apptainer")
        self.assertIn("--nv", argv)  # GPU step passes the device through
        self.assertEqual(argv[-4], "/img.sif")  # image right before bash
        self.assertEqual(argv[-3:], ["bash", "-lc", "cd /repro/workspace && uv pip install x"])

    def test_cpu_sandbox_step_omits_nv(self):
        sb = Sandbox(image="/img.sif", binds=(Bind("/host/ws", CONTAINER_WORKSPACE),))
        self.assertNotIn("--nv", env.exec_argv("/host/ws", "echo hi", sandbox=sb))


if __name__ == "__main__":
    unittest.main()
