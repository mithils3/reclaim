"""``list_partitions`` — let the agent see the SLURM pools it can allocate on.

The cluster profile (``cluster.py``) pins a *default* partition per known cluster,
but the agent is no longer stuck with it: this tool runs ``sinfo`` on the
controller and reports every partition (node pool) the cluster exposes — node
counts by state, walltime limit, and GPU gres — alongside the built-in default for
each known cluster (``cluster.cluster_defaults``) and the one currently in effect.
The agent then passes a chosen ``partition`` to ``run_gpu`` to allocate there
instead of the default, for example swapping the batch pool for a faster-queueing
short pool on a brief step. Read-only: it never allocates anything.
"""

from __future__ import annotations

import subprocess
from typing import Any

from reclaim_vllm.config.config import function_tool

from reclaim_repro.cluster import cluster_defaults
from reclaim_repro.context import ExecutionContext

# One row per partition: name | avail | timelimit | nodes(A/I/O/T) | gres. ``%P``
# carries the ``*`` marker SLURM puts on the controller's default partition.
_SINFO_FORMAT = "%P|%a|%l|%F|%G"
_SINFO_TIMEOUT = 60


def list_partitions(arguments: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    """Enumerate the controller's partitions (``sinfo``) + the known-cluster defaults."""
    active = ctx.cluster.name if ctx.cluster is not None else None
    active_default = ctx.cluster.partition if ctx.cluster is not None else None
    partitions, note = _sinfo_partitions()
    result: dict[str, Any] = {
        "ok": True,
        "tool": "list_partitions",
        "active_cluster": active,
        "active_default_partition": active_default,
        "known_cluster_defaults": cluster_defaults(),
        "partitions": partitions,
        "hint": "Pass partition=<name> to run_gpu to allocate on a non-default pool; it "
        "takes effect on the call that starts a held session and is fixed until release.",
    }
    if note:
        result["note"] = note
    return result


def _sinfo_partitions() -> tuple[list[dict], str | None]:
    """Parse ``sinfo`` into one row per partition; degrade to a note if it is absent."""
    try:
        proc = subprocess.run(
            ["sinfo", "-h", "-o", _SINFO_FORMAT],
            capture_output=True,
            text=True,
            timeout=_SINFO_TIMEOUT,
        )
    except FileNotFoundError:
        return [], "sinfo not found on this host — only the known-cluster defaults are shown."
    except (subprocess.SubprocessError, OSError) as exc:
        return [], f"sinfo failed: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return [], f"sinfo exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}"

    rows: list[dict] = []
    seen: set[str] = set()
    for line in proc.stdout.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 5 or not cells[0]:
            continue
        raw_name, avail, timelimit, nodes, gres = cells[:5]
        name = raw_name.rstrip("*")
        if name in seen:  # sinfo emits one line per partition+state; collapse to the partition
            continue
        seen.add(name)
        rows.append(
            {
                "partition": name,
                "is_slurm_default": raw_name.endswith("*"),
                "available": avail,
                "timelimit": timelimit,
                "nodes_allocated_idle_other_total": nodes,
                "gres": gres,
            }
        )
    return rows, (None if rows else "sinfo returned no partitions.")


LIST_PARTITIONS_TOOLS = [
    function_tool(
        "list_partitions",
        "List the SLURM partitions (node pools) this cluster exposes — runs `sinfo` and "
        "returns each partition's node counts (allocated/idle/other/total), walltime "
        "limit, and GPU gres — plus the built-in default partition for each known cluster "
        "and the one currently in effect. Use it to choose a `partition` for run_gpu "
        "instead of the hardcoded default (e.g. a faster-queueing interactive pool).",
        {},
        [],
    ),
]

LIST_PARTITIONS_HANDLERS = {"list_partitions": list_partitions}
