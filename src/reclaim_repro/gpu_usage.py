"""Per-run GPU-efficiency rollup of a run's ``host_metrics`` time series.

The metrics beacon (``metrics_beacon``) POSTs one ``host_metrics`` row per cycle
into each held GPU allocation, tagged with ``run_id``. At run finish the sink
reads that whole series back and boils it down to a handful of ``repro_runs``
columns plus a ``stats.json`` ``"gpu"`` section: how hard were the GPUs the agent
held actually driven?

Same seam as ``run_stats`` / ``telemetry_rows``: the arithmetic (:func:`aggregate`)
is pure and takes rows, so the tests run it on fixtures without a network. The
thin I/O (:func:`fetch_rows`, :func:`rollup`) wraps ``postgrest.request`` and
follows the telemetry swallow-all discipline — a rollup failure returns ``None``,
never raises, so it can never break a finishing run.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from reclaim_repro import postgrest

ACTIVE_UTIL_PCT = 10       # a sample is "active" when mean GPU util >= this
FETCH_LIMIT = 10000        # host_metrics rows pulled per rollup
TIMELINE_POINTS = 200      # max points kept in the stats.json sparkline
HTTP_TIMEOUT = 8.0
SAMPLE_GAP_S = 20          # beacon cadence; fallback offset when timestamps fail


# ---- pure helpers (rows in, values out — unit-tested without a network) -------

def _gpu_cells(gpus: Any) -> list[dict[str, Any]]:
    """The dict entries of a row's ``gpus`` list (tolerate None entries / shapes)."""
    if not isinstance(gpus, list):
        return []
    return [g for g in gpus if isinstance(g, dict)]


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _row_stats(gpus: Any) -> dict[str, float | None]:
    """One usable row -> ``mean_util`` / ``max_util`` / ``max_mem`` / ``mean_power``."""
    cells = _gpu_cells(gpus)
    utils = [c["util"] for c in cells if c.get("util") is not None]
    mems = [c["mem"] for c in cells if c.get("mem") is not None]
    mem_totals = [c["mem_total"] for c in cells if c.get("mem_total") is not None]
    powers = [c["power"] for c in cells if c.get("power") is not None]
    return {
        "mean_util": _mean(utils),
        "max_util": max(utils) if utils else None,
        "max_mem": max(mems) if mems else None,
        "max_mem_total": max(mem_totals) if mem_totals else None,
        "mean_power": _mean(powers),
    }


def _offsets(rows: list[dict[str, Any]]) -> list[float]:
    """Seconds since the first usable sample; index*SAMPLE_GAP_S if any TS fails."""
    parsed: list[datetime] = []
    for r in rows:
        ts = r.get("created_at")
        try:
            parsed.append(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
        except (ValueError, TypeError):
            return [i * SAMPLE_GAP_S for i in range(len(rows))]
    base = parsed[0]
    return [round((p - base).total_seconds(), 1) for p in parsed]


def _stride(n: int, k: int) -> list[int]:
    """Up to ``k`` evenly-spaced indices into ``range(n)`` (all of them when n<=k)."""
    if n <= k:
        return list(range(n))
    return [(i * n) // k for i in range(k)]


def aggregate(rows: list[dict]) -> tuple[dict | None, dict | None]:
    """``host_metrics`` rows -> ``(run_fields, stats_section)``; both None if empty.

    Pure. ``rows`` are ``host_metrics``-shaped dicts (``gpus`` + ``created_at``);
    rows with a falsy ``gpus`` are skipped. ``run_fields`` are the ``repro_runs``
    rollup columns; ``stats_section`` is those plus the active threshold and a
    down-sampled ``timeline`` for ``stats.json``.
    """
    usable = [r for r in rows if r.get("gpus")]
    if not usable:
        return None, None
    per = [_row_stats(r.get("gpus")) for r in usable]
    row_utils = [s["mean_util"] for s in per if s["mean_util"] is not None]
    row_powers = [s["mean_power"] for s in per if s["mean_power"] is not None]
    max_utils = [s["max_util"] for s in per if s["max_util"] is not None]
    max_mems = [s["max_mem"] for s in per if s["max_mem"] is not None]
    mem_totals = [s["max_mem_total"] for s in per if s["max_mem_total"] is not None]
    active = sum(1 for s in per if s["mean_util"] is not None and s["mean_util"] >= ACTIVE_UTIL_PCT)

    fields: dict[str, Any] = {
        "gpu_util_avg_pct": round(_mean(row_utils), 1) if row_utils else None,
        "gpu_util_max_pct": int(max(max_utils)) if max_utils else None,
        "gpu_active_pct": round(100.0 * active / len(usable), 1),
        "gpu_mem_peak_gb": round(max(max_mems), 1) if max_mems else None,
        "gpu_mem_total_gb": round(max(mem_totals), 1) if mem_totals else None,
        "gpu_samples": len(usable),
    }
    if row_powers:  # power.draw is often N/A; omit the column when never present
        fields["gpu_power_avg_w"] = int(round(_mean(row_powers)))

    offsets = _offsets(usable)
    timeline = []
    for j in _stride(len(usable), TIMELINE_POINTS):
        util = per[j]["mean_util"]
        mem = per[j]["max_mem"]
        timeline.append([
            offsets[j],
            round(util, 1) if util is not None else None,
            round(mem, 1) if mem is not None else None,
        ])
    stats = {**fields, "active_util_threshold_pct": ACTIVE_UTIL_PCT, "timeline": timeline}
    return fields, stats


# ---- thin I/O (swallow-all: telemetry may never break a finishing run) --------

def fetch_rows(url: str, service_key: str, run_id: str) -> list[dict] | None:
    """This run's ``host_metrics`` rows (util/mem series), oldest first; None on any failure."""
    path = (f"{url}/rest/v1/host_metrics?run_id=eq.{run_id}"
            f"&select=gpus,created_at&order=created_at.asc&limit={FETCH_LIMIT}")
    try:
        code, body = postgrest.request(path, service_key=service_key, method="GET", timeout=HTTP_TIMEOUT)
    except Exception:  # noqa: BLE001 — best-effort; never propagate
        return None
    if not code or code >= 300:
        return None
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, list) else None


def rollup(url: str, service_key: str, run_id: str) -> tuple[dict | None, dict | None]:
    """Fetch + aggregate this run's GPU series -> ``(run_fields, stats_section)``.

    ``(None, None)`` on any failure (no rows, HTTP/decode error, exception).
    Never raises.
    """
    try:
        rows = fetch_rows(url, service_key, run_id)
        if not rows:
            return None, None
        return aggregate(rows)
    except Exception:  # noqa: BLE001 — telemetry must never break a run
        return None, None
