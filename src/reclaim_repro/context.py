"""Per-episode execution state for the reproduction agent.

``ExecutionContext`` is the repro analog of the auditor's ``Paper``: one
instance per paper/episode, keyed by ``custom_id`` (the arXiv id) in the tool
loop. Where the auditor threads ``execute_tool_call(call, paper=paper)``, the
repro loop threads ``execute_repro_tool_call(call, ctx)`` so every tool acts on
*this* episode's mutable workspace, budget meter, allocation, and evidence dir.

The dataclass carries the minimal ``Budget`` surface the loop's compute-budget
guardrail needs (``remaining`` / ``exhausted`` / ``consume``). Setup fills in the
rest: ``workspace`` + ``evidence``, then ``budget`` metering with the hw-multiplier
table and ``allocation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reclaim_repro.cluster import Cluster
    from reclaim_repro.sandbox import Sandbox


@dataclass
class Budget:
    """Per-episode compute budget, denominated in H100-equivalent hours.

    This holds only the guardrail surface the loop reads. ``budget.py`` adds the
    ``hw_multiplier`` table (GH200/H200 to H100-equivalent) and the
    ``gpus x wallclock x hw_multiplier`` accounting that calls ``consume``.
    """

    total_h100_hours: float
    spent_h100_hours: float = 0.0

    def remaining(self) -> float:
        return self.total_h100_hours - self.spent_h100_hours

    def exhausted(self) -> bool:
        return self.remaining() <= 0

    def consume(self, h100_hours: float) -> float:
        """Charge ``h100_hours`` against the budget; return what remains."""
        self.spent_h100_hours += max(0.0, h100_hours)
        return self.remaining()


@dataclass
class GpuSession:
    """A held SLURM allocation the agent runs successive ``run_gpu`` steps into.

    The first ``run_gpu`` call acquires one allocation (``salloc --no-shell``) that
    stays up; every later call runs into it (``srun --jobid``) with no re-queue, so
    the agent stops paying a fresh queue wait between install → verify → run. It is
    released when the agent is done (``run_gpu`` with ``release=true``) or at episode
    teardown.

    Because the node is held across the agent's reasoning/install gaps too — not
    only while a command runs — the meter bills **wall clock**: ``gpus x (held
    seconds) x hw``, charged incrementally (``gpu_session.charge_accrued``) so the
    budget ceiling still fires mid-hold. ``last_charged`` advances on every charge so
    the running total equals the real held wall, never double-counting.
    """

    jobid: str
    gpus: int
    minutes: int                         # salloc --time: the hold's max lifetime + pre-auth
    hw: str
    started: float                       # time.monotonic() just after the allocation was granted
    last_charged: float                  # time.monotonic() at the most recent budget charge
    partition: str | None = None         # the pool this allocation was held on (default or model-picked)


@dataclass
class ExecutionContext:
    """Mutable per-episode state the repro tool loop dispatches against."""

    arxiv_id: str
    lockfile_row: dict[str, Any] = field(default_factory=dict)
    workspace: Path | None = None        # editable code clone + venv (rw)
    reference: Path | None = None        # read-only paper LaTeX + supplement (ro)
    budget: Budget | None = None         # metered compute budget
    session: "GpuSession | None" = None  # the live held run_gpu allocation, if any
    evidence: Path | None = None         # commands.log / trajectory.jsonl / ...
    run_dir: Path | None = None          # bundle root where report.json is written for the auditor
    cluster: "Cluster | None" = None     # GPU substrate run_gpu allocates on
    sandbox: "Sandbox | None" = None     # Apptainer container write-confinement applied to shell steps
    plan: list[dict[str, str]] = field(default_factory=list)  # latest update_plan checklist (survives compaction)
    last_prompt_tokens: int | None = None  # usage.prompt_tokens from the most recent response; drives the context tiers
