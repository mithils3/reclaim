"""Host telemetry beacon: sample this node, push it to the telemetry service (opt-in).

One beacon per host. The sbatch master node runs ``--role master`` (tailing the
slurm ``.out`` so the viewer can watch the vLLM log); each held GPU allocation
gets a ``--role run`` beacon spawned into it by ``run_beacon``. Every cycle
appends one ``host_metrics`` time-series row and upserts this host's
``host_status`` snapshot; the master additionally self-prunes its own
``host_metrics`` rows older than 24 h, at most once an hour.

Same opt-in + swallow-all discipline as ``telemetry_sink``: disabled unless
``RECLAIM_TELEMETRY_URL`` + ``RECLAIM_TELEMETRY_KEY`` are set (exit 0, one line,
never an error — it's telemetry), and nothing may ever raise out of the sampling
loop.
Sampling is stdlib-only — /proc + ``nvidia-smi``, no psutil. The parsers take
*text*, not paths, so the tests run them on fixtures without a GPU.

    python -m reclaim_repro.metrics_beacon --role master --log-file slurm.out
    python -m reclaim_repro.metrics_beacon --role run --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable

from reclaim_repro import postgrest
from reclaim_repro.event_sink import credentials_from_env

HTTP_TIMEOUT = 8.0
NVSMI_TIMEOUT = 10.0
CPU_SAMPLE_GAP_S = 0.25        # between the two /proc/stat reads a busy% spans
TAIL_LINES = 120               # host_status.log_tail: last N lines ...
TAIL_BYTES = 16 * 1024         # ... capped at 16 KB
PRUNE_EVERY_S = 3600.0         # master self-prune cadence
PRUNE_KEEP_S = 24 * 3600.0     # rolling host_metrics window

_NVSMI_ARGV = [
    "nvidia-smi",
    "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
    "--format=csv,noheader,nounits",
]


# ---- pure parsers (text in, values out — unit-tested without /proc or a GPU) --

def parse_stat(text: str) -> tuple[float, float] | None:
    """Aggregate ``cpu`` line of /proc/stat -> ``(idle, total)`` jiffies."""
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        try:
            fields = [float(x) for x in line.split()[1:9]]  # user .. steal
        except ValueError:
            return None
        if len(fields) < 5:
            return None
        return fields[3] + fields[4], sum(fields)  # idle = idle + iowait
    return None


def cpu_pct_between(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    """Whole-node busy % between two ``parse_stat`` samples (0-100, 1 decimal)."""
    if a is None or b is None:
        return None
    idle_d, total_d = b[0] - a[0], b[1] - a[1]
    if total_d <= 0:
        return None
    return round(100.0 * (1.0 - idle_d / total_d), 1)


def parse_meminfo(text: str) -> tuple[float | None, float | None]:
    """/proc/meminfo -> ``(used_gb, total_gb)``; used = MemTotal - MemAvailable."""
    kb: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
            try:
                kb[parts[0]] = float(parts[1])
            except ValueError:
                return None, None
    total, avail = kb.get("MemTotal:"), kb.get("MemAvailable:")
    if total is None or avail is None:
        return None, None
    gib = 1024.0 * 1024.0  # kB -> GiB
    return round((total - avail) / gib, 1), round(total / gib, 1)


def parse_loadavg(text: str) -> float | None:
    """/proc/loadavg -> the 1-minute average (first field)."""
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return None


def _cell(raw: str, conv: Callable[[str], Any]) -> Any:
    """One nvidia-smi CSV cell -> value via ``conv``; N/A / unparseable -> None."""
    if not raw or raw.upper() in ("N/A", "[N/A]", "[NOT SUPPORTED]"):
        return None
    try:
        return conv(raw)
    except ValueError:
        return None


def parse_nvidia_csv(text: str) -> list[dict[str, Any]] | None:
    """``nvidia-smi ... --format=csv,noheader,nounits`` -> the contract's gpus json.

    Keys per GPU: ``i`` / ``util`` (%) / ``mem`` + ``mem_total`` (MiB -> GiB, 1
    decimal) / ``power`` (W) / ``temp`` (C). Any unexpected shape -> ``None``.
    """
    gpus: list[dict[str, Any]] = []
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 6:
            return None
        index = _cell(cells[0], int)
        if index is None:
            return None
        mib = lambda raw: round(float(raw) / 1024.0, 1)  # noqa: E731 — tiny cell conv
        gpus.append({
            "i": index,
            "util": _cell(cells[1], lambda raw: int(float(raw))),
            "mem": _cell(cells[2], mib),
            "mem_total": _cell(cells[3], mib),
            "power": _cell(cells[4], lambda raw: int(round(float(raw)))),
            "temp": _cell(cells[5], lambda raw: int(float(raw))),
        })
    return gpus or None


# ---- samplers (thin I/O over the parsers) ------------------------------------

def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def sample_gpus() -> list[dict[str, Any]] | None:
    """This host's GPUs, or ``None`` (no binary / nonzero rc / parse failure)."""
    try:
        proc = subprocess.run(_NVSMI_ARGV, capture_output=True, text=True, timeout=NVSMI_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return parse_nvidia_csv(proc.stdout or "")


def sample_host() -> dict[str, Any]:
    """One cycle's measurements (the busy % spans a short double /proc/stat read)."""
    before = parse_stat(_read("/proc/stat"))
    time.sleep(CPU_SAMPLE_GAP_S)
    after = parse_stat(_read("/proc/stat"))
    used, total = parse_meminfo(_read("/proc/meminfo"))
    return {
        "cpu_pct": cpu_pct_between(before, after),
        "mem_used_gb": used,
        "mem_total_gb": total,
        "load1": parse_loadavg(_read("/proc/loadavg")),
        "gpus": sample_gpus(),
    }


def tail_file(path: str) -> str | None:
    """Last ~``TAIL_LINES`` lines of ``path``, <= ``TAIL_BYTES``; None if unreadable."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - TAIL_BYTES))
            blob = fh.read(TAIL_BYTES)
    except OSError:
        return None
    return "\n".join(blob.decode("utf-8", errors="replace").splitlines()[-TAIL_LINES:])


# ---- the beacon ----------------------------------------------------------------

class Beacon:
    """The sampling loop: one host_metrics insert + one host_status upsert per cycle."""

    def __init__(self, args: argparse.Namespace, url: str | None, key: str | None) -> None:
        self.args = args
        self.url = url.rstrip("/") if url else None
        self.key = key
        self.host = socket.gethostname().split(".")[0]
        self.batch_id = args.batch_id or os.environ.get("REPRO_BATCH_ID") or None
        self._last_prune = 0.0

    def cycle(self) -> None:
        base = {"host": self.host, "role": self.args.role, "batch_id": self.batch_id,
                "run_id": self.args.run_id, **sample_host()}
        # One beacon per host, so a None log_tail (role=run has no log file)
        # never clobbers another writer's tail.
        status = {**base, "log_tail": tail_file(self.args.log_file) if self.args.log_file else None}
        if self.args.dry_run:
            print(json.dumps({"host_metrics": base, "host_status": status}, indent=2), flush=True)
            return
        self._post("POST", "/rest/v1/host_metrics", [base], prefer="return=minimal")
        self._post("POST", "/rest/v1/host_status?on_conflict=host", [status],
                   prefer="resolution=merge-duplicates,return=minimal")
        if self.args.role == "master":
            self._maybe_prune()

    def _maybe_prune(self) -> None:
        """Keep host_metrics a rolling 24 h window for this host, once an hour."""
        now = time.monotonic()
        if self._last_prune and now - self._last_prune < PRUNE_EVERY_S:
            return
        self._last_prune = now
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - PRUNE_KEEP_S))
        self._post("DELETE", f"/rest/v1/host_metrics?host=eq.{self.host}&created_at=lt.{cutoff}", None)

    def _post(self, method: str, path: str, body: Any, *, prefer: str | None = None) -> None:
        try:
            postgrest.request(
                self.url + path, service_key=self.key, method=method,
                body=body, prefer=prefer, timeout=HTTP_TIMEOUT)
        except Exception:  # noqa: BLE001 — best-effort; never propagate
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reclaim_repro.metrics_beacon",
        description="Sample this host (cpu/mem/gpus) and push it to the telemetry service.")
    parser.add_argument("--role", required=True, choices=("master", "run"))
    parser.add_argument("--run-id", default=None, help="stamped on rows (required when --role run)")
    parser.add_argument("--batch-id", default=os.environ.get("REPRO_BATCH_ID") or None,
                        help="sweep grouping (default: env REPRO_BATCH_ID)")
    parser.add_argument("--log-file", default=None,
                        help="slurm .out to tail into host_status.log_tail (master)")
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between samples")
    parser.add_argument("--once", action="store_true", help="single sample then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rows as JSON instead of POSTing (no telemetry env needed)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.role == "run" and not args.run_id:
        parser.error("--run-id is required when --role run")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # scancel/trap kill -> clean exit
    creds = credentials_from_env()
    if not args.dry_run and creds is None:
        # Opt-in telemetry: unset env is the normal off state, never an error.
        print("metrics_beacon: RECLAIM_TELEMETRY_URL / RECLAIM_TELEMETRY_KEY unset "
              "-> telemetry off", flush=True)
        return 0
    url, key = creds or (None, None)   # --dry-run prints rows instead of POSTing
    beacon = Beacon(args, url, key)
    while True:
        try:
            beacon.cycle()
        except Exception:  # noqa: BLE001 — telemetry may never raise out of its loop
            pass
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
