"""gpu_usage: the pure host_metrics rollup + its thin PostgREST I/O.

aggregate() is exercised on hand-built host_metrics rows (multi-GPU means,
skipped null rows, tolerated null util cells, active-% math, timeline offsets
from ISO timestamps + the index-based fallback, and down-sampling). fetch_rows /
rollup are exercised with postgrest.request monkeypatched (success, non-2xx,
garbage body) — never a real network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reclaim_repro import gpu_usage


def _row(ts, gpus):
    return {"created_at": ts, "gpus": gpus}


def _gpu(i, util, mem, mem_total=97.8, power=610, temp=52):
    return {"i": i, "util": util, "mem": mem, "mem_total": mem_total, "power": power, "temp": temp}


# ---- aggregate: run_fields -------------------------------------------------

def test_empty_input_returns_none_none():
    assert gpu_usage.aggregate([]) == (None, None)


def test_rows_with_null_gpus_are_all_skipped():
    rows = [_row("2026-07-10T00:00:00Z", None), _row("2026-07-10T00:00:20Z", [])]
    assert gpu_usage.aggregate(rows) == (None, None)


def test_multi_gpu_means_and_maxes():
    rows = [
        _row("2026-07-10T00:00:00Z", [_gpu(0, 90, 70.0), _gpu(1, 10, 30.0)]),  # mean 50
        _row("2026-07-10T00:00:20Z", [_gpu(0, 100, 80.5), _gpu(1, 20, 40.0)]),  # mean 60
    ]
    fields, stats = gpu_usage.aggregate(rows)
    assert fields["gpu_util_avg_pct"] == 55.0        # mean of 50 and 60
    assert fields["gpu_util_max_pct"] == 100         # max single-GPU util
    assert fields["gpu_mem_peak_gb"] == 80.5         # max single-GPU mem
    assert fields["gpu_mem_total_gb"] == 97.8
    assert fields["gpu_power_avg_w"] == 610
    assert fields["gpu_samples"] == 2
    # stats section mirrors fields + adds the threshold + timeline
    assert stats["gpu_util_avg_pct"] == 55.0
    assert stats["active_util_threshold_pct"] == gpu_usage.ACTIVE_UTIL_PCT
    assert len(stats["timeline"]) == 2


def test_null_util_cells_tolerated():
    # one GPU reports N/A util; the mean uses only the populated cell
    rows = [_row("2026-07-10T00:00:00Z", [_gpu(0, None, 12.0), _gpu(1, 80, 34.0)])]
    fields, _ = gpu_usage.aggregate(rows)
    assert fields["gpu_util_avg_pct"] == 80.0
    assert fields["gpu_util_max_pct"] == 80


def test_active_percent_math():
    # 1 of 4 usable rows is active (mean util >= 10)
    rows = [
        _row("2026-07-10T00:00:00Z", [_gpu(0, 50, 10.0)]),   # active
        _row("2026-07-10T00:00:20Z", [_gpu(0, 5, 10.0)]),    # idle
        _row("2026-07-10T00:00:40Z", [_gpu(0, 0, 10.0)]),    # idle
        _row("2026-07-10T00:01:00Z", [_gpu(0, 2, 10.0)]),    # idle
    ]
    fields, _ = gpu_usage.aggregate(rows)
    assert fields["gpu_active_pct"] == 25.0
    assert fields["gpu_samples"] == 4


def test_power_column_omitted_when_never_present():
    rows = [_row("2026-07-10T00:00:00Z", [_gpu(0, 40, 10.0, power=None)])]
    fields, _ = gpu_usage.aggregate(rows)
    assert "gpu_power_avg_w" not in fields


# ---- aggregate: timeline ---------------------------------------------------

def test_timeline_offsets_from_iso_created_at():
    rows = [
        _row("2026-07-10T00:00:00Z", [_gpu(0, 40, 10.0)]),
        _row("2026-07-10T00:00:20Z", [_gpu(0, 60, 20.0)]),
        _row("2026-07-10T00:01:00Z", [_gpu(0, 80, 30.0)]),
    ]
    _, stats = gpu_usage.aggregate(rows)
    offsets = [pt[0] for pt in stats["timeline"]]
    assert offsets == [0.0, 20.0, 60.0]
    assert stats["timeline"][1] == [20.0, 60.0, 20.0]  # [offset, util, mem]


def test_timeline_offset_fallback_when_a_timestamp_is_garbage():
    rows = [
        _row("2026-07-10T00:00:00Z", [_gpu(0, 40, 10.0)]),
        _row("not-a-timestamp", [_gpu(0, 60, 20.0)]),
        _row("2026-07-10T00:01:00Z", [_gpu(0, 80, 30.0)]),
    ]
    _, stats = gpu_usage.aggregate(rows)
    offsets = [pt[0] for pt in stats["timeline"]]
    assert offsets == [0, gpu_usage.SAMPLE_GAP_S, 2 * gpu_usage.SAMPLE_GAP_S]


def test_timeline_downsamples_when_rows_exceed_cap():
    rows = [_row(f"2026-07-10T00:00:{i:02d}Z", [_gpu(0, i % 100, 10.0)])
            for i in range(gpu_usage.TIMELINE_POINTS + 50)]
    fields, stats = gpu_usage.aggregate(rows)
    assert fields["gpu_samples"] == gpu_usage.TIMELINE_POINTS + 50
    assert len(stats["timeline"]) == gpu_usage.TIMELINE_POINTS


# ---- fetch_rows / rollup (postgrest.request monkeypatched) -----------------

def test_fetch_rows_success(monkeypatch):
    captured = {}

    def fake_request(url, **kw):
        captured["url"] = url
        captured["method"] = kw.get("method")
        return 200, '[{"gpus": null, "created_at": "t"}]'

    monkeypatch.setattr(gpu_usage.postgrest, "request", fake_request)
    rows = gpu_usage.fetch_rows("https://telemetry.example", "k", "run123")
    assert rows == [{"gpus": None, "created_at": "t"}]
    assert captured["method"] == "GET"
    assert "run_id=eq.run123" in captured["url"]


def test_fetch_rows_non_2xx_returns_none(monkeypatch):
    monkeypatch.setattr(gpu_usage.postgrest, "request", lambda url, **kw: (500, "boom"))
    assert gpu_usage.fetch_rows("https://x", "k", "r") is None


def test_fetch_rows_garbage_body_returns_none(monkeypatch):
    monkeypatch.setattr(gpu_usage.postgrest, "request", lambda url, **kw: (200, "not json"))
    assert gpu_usage.fetch_rows("https://x", "k", "r") is None


def test_fetch_rows_transport_exception_returns_none(monkeypatch):
    def boom(url, **kw):
        raise OSError("dns")

    monkeypatch.setattr(gpu_usage.postgrest, "request", boom)
    assert gpu_usage.fetch_rows("https://x", "k", "r") is None


def test_rollup_success(monkeypatch):
    body = '[{"gpus": [{"i": 0, "util": 40, "mem": 10.0, "mem_total": 97.8, "power": 300, "temp": 50}], "created_at": "2026-07-10T00:00:00Z"}]'
    monkeypatch.setattr(gpu_usage.postgrest, "request", lambda url, **kw: (200, body))
    fields, stats = gpu_usage.rollup("https://x", "k", "r")
    assert fields["gpu_util_avg_pct"] == 40.0
    assert stats["timeline"][0][1] == 40.0


def test_rollup_no_rows_returns_none_none(monkeypatch):
    monkeypatch.setattr(gpu_usage.postgrest, "request", lambda url, **kw: (200, "[]"))
    assert gpu_usage.rollup("https://x", "k", "r") == (None, None)


def test_rollup_failure_returns_none_none(monkeypatch):
    monkeypatch.setattr(gpu_usage.postgrest, "request", lambda url, **kw: (503, "down"))
    assert gpu_usage.rollup("https://x", "k", "r") == (None, None)
