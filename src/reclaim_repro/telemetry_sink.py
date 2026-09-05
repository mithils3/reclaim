"""Best-effort live upload of the reproduce transcript to the telemetry service (opt-in).

Disabled unless ``RECLAIM_TELEMETRY_URL`` + ``RECLAIM_TELEMETRY_KEY`` are set, so
the default run is unchanged. When enabled, a background worker drains a bounded queue and batches
PostgREST writes (a full queue drops, never blocks). Mirrors ``live_log``'s
swallow-all discipline: no network failure may slow or crash the loop — the
on-disk ``agent.full.log`` stays the source of truth. Fed via
``live_log.register_sink(sink.on_event)``: each round becomes ``repro_events`` rows
(plus throttled ``repro_runs`` meter patches); the model's per-response token
``usage`` is summed onto the run row; on ``final`` the full log and a detailed
``stats.json`` are optionally pushed to Storage.

This module holds the reproduce-specific half: the pure PostgREST row builders and
the per-run token bookkeeping (``RunStats``). The queue + worker + HTTP plumbing it
shares with the auditor's sink lives in ``event_sink.PostgrestEventSink``.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reclaim_repro import gpu_usage, live_log
from reclaim_repro.event_sink import PostgrestEventSink, credentials_from_env, now_iso

BUCKET = "repro-logs"


# ---- pure PostgREST row builders -------------------------------------------
# Pure: a context / event payload in, a row ``dict`` out — no state, no network.
# The sink supplies the ``base`` dict (``run_id`` / ``seq`` / ``kind`` /
# ``round_index``) since the sequence counter is stateful.

STDOUT_CAP = 8000  # chars per stdout/stderr/text cell (full text -> agent.full.log)


def cap(text: Any, limit: int = STDOUT_CAP) -> tuple[str, bool]:
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s, False
    return s[:limit] + f"\n…(+{len(s) - limit} chars — see agent.full.log)", True


def run_id_of(ctx) -> str | None:
    ev = getattr(ctx, "evidence", None)
    return Path(ev).parent.name if ev else None


def budget_of(ctx) -> tuple[float | None, float | None]:
    b = getattr(ctx, "budget", None)
    if b is None:
        return None, None
    return b.total_h100_hours, b.remaining()


def message_row(base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    msg = payload.get("message") or {}
    reasoning, _ = cap(msg.get("reasoning") or msg.get("reasoning_content"), STDOUT_CAP * 2)
    content, _ = cap(msg.get("content"), STDOUT_CAP * 2)
    base.update({"role": "assistant", "reasoning": reasoning or None, "content": content or None})
    base["finish_reason"] = payload.get("finish_reason")
    if base.get("kind") == "final":
        base["exit_reason"] = payload.get("exit_reason") or None
    return base


def call_row(base: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    args = live_log.call_arguments(call)
    name = str((call.get("function") or {}).get("name") or "?")
    base["tool_name"] = name
    if "command" in args:
        base.update(detail_kind="command", command=str(args["command"]))
    elif "diff" in args:
        path = args.get("path")
        base.update(detail_kind="diff", command=f"(apply diff{' to ' + str(path) if path else ''})")
    elif "path" in args:
        base.update(detail_kind="path", command=str(args["path"]))
    elif args:
        base.update(detail_kind="json", args=args)
    return base


def result_row(base: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    out, t1 = cap(res.get("stdout"))
    err, t2 = cap(res.get("stderr"))
    base.update({
        "ok": res.get("ok"), "rc": res.get("returncode"), "duration_s": res.get("duration_s"),
        "cost_h100": res.get("cost_h100_hours"), "remaining_h100": res.get("remaining_h100_hours"),
        "error": (str(res["error"]).splitlines()[0] if res.get("error") else None),
        "path": res.get("path"), "stdout": out or None, "stderr": err or None,
        "truncated": bool(t1 or t2),
    })
    return base


# ---- per-run token bookkeeping ---------------------------------------------
# Pure in-memory bookkeeping: feed it each model response's ``usage`` and each
# tool call; it yields the ``repro_runs`` aggregate fields and writes the detailed
# ``stats.json`` document. No network, no globals.

_KEYS = ("prompt", "completion", "total", "cached", "reasoning")


def usage_fields(usage: Any) -> dict[str, int]:
    """OpenAI / vLLM ``usage`` object -> flat int counts (defensive about shape)."""
    if not isinstance(usage, dict):
        return {k: 0 for k in _KEYS}
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    tt = int(usage.get("total_tokens") or (pt + ct))
    ptd = usage.get("prompt_tokens_details")
    ctd = usage.get("completion_tokens_details")
    cached = int(ptd.get("cached_tokens") or 0) if isinstance(ptd, dict) else 0
    reasoning = int(ctd.get("reasoning_tokens") or 0) if isinstance(ctd, dict) else 0
    return {"prompt": pt, "completion": ct, "total": tt, "cached": cached, "reasoning": reasoning}


class RunStats:
    """Accumulates token usage + per-round records for a single run."""

    def __init__(self) -> None:
        self.tokens = {k: 0 for k in _KEYS}
        self.rounds: list[dict[str, Any]] = []
        self.tool_calls = 0

    def add_usage(self, round_index: int | None, kind: str | None, usage: Any) -> None:
        f = usage_fields(usage)
        for k in _KEYS:
            self.tokens[k] += f[k]
        self.rounds.append({
            "round_index": round_index, "kind": kind or "round",
            "ts": now_iso(), **f,
        })

    def add_tool_call(self) -> None:
        self.tool_calls += 1

    def run_fields(self) -> dict[str, Any]:
        """The aggregate columns patched onto the ``repro_runs`` row."""
        t = self.tokens
        return {
            "prompt_tokens": t["prompt"], "completion_tokens": t["completion"],
            "total_tokens": t["total"], "cached_tokens": t["cached"],
            "reasoning_tokens": t["reasoning"], "tool_calls": self.tool_calls,
        }

    def write_doc(self, path: Path, meta: dict[str, Any]) -> bool:
        """Write the detailed ``stats.json`` (meta + tokens + per-round). True on success."""
        doc = dict(meta)
        doc.update({"tokens": dict(self.tokens), "tool_calls": self.tool_calls, "rounds": self.rounds})
        try:
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            return True
        except OSError:
            return False


@dataclass
class SinkConfig:
    url: str
    service_key: str
    host: str
    upload_full_log: bool
    upload_stats: bool
    batch_id: str | None
    batch_label: str | None

    @classmethod
    def from_env(cls) -> "SinkConfig | None":
        creds = credentials_from_env()
        if creds is None:
            return None
        url, key = creds
        full = os.environ.get("REPRO_UPLOAD_FULL_LOG", "").lower() in ("1", "true", "yes")
        # stats.json is tiny and the whole point of enabling the sink, so it's on
        # by default whenever the sink is active; set REPRO_UPLOAD_STATS=0 to skip.
        stats = os.environ.get("REPRO_UPLOAD_STATS", "1").lower() not in ("0", "false", "no")
        # One sbatch sweep launches many `python -m reclaim_repro` processes; they
        # share REPRO_BATCH_ID (falling back to the SLURM job id) so the viewer can
        # show the whole sweep as a group. Empty strings count as unset.
        batch_id = os.environ.get("REPRO_BATCH_ID") or None
        if not batch_id:
            slurm_job = os.environ.get("SLURM_JOB_ID") or None
            batch_id = f"slurm-{slurm_job}" if slurm_job else None
        batch_label = os.environ.get("REPRO_BATCH_LABEL") or None
        return cls(url=url, service_key=key, host=socket.gethostname(),
                   upload_full_log=full, upload_stats=stats,
                   batch_id=batch_id, batch_label=batch_label)


class TelemetrySink(PostgrestEventSink):
    LABEL = "telemetry_sink"
    CLOSE_NOTE = " (local agent.full.log is the complete record)"
    ID_COLUMN = "run_id"
    EVENTS_PATH = "/rest/v1/repro_events"

    def __init__(self, cfg: SinkConfig) -> None:
        self.cfg = cfg
        self._total: dict[str, float | None] = {}
        self._lastpatch: dict[str, tuple] = {}
        self._stats: dict[str, RunStats] = {}
        super().__init__(cfg.url, cfg.service_key)

    def _stats_for(self, run_id: str) -> RunStats:
        s = self._stats.get(run_id)
        if s is None:
            s = self._stats[run_id] = RunStats()
        return s

    def upsert_run(self, ctx, *, model: str | None, status: str = "running") -> None:
        run_id = run_id_of(ctx)
        if not run_id:
            return
        total, remaining = budget_of(ctx)
        self._total[run_id] = total
        self._put("run_upsert", {
            "run_id": run_id, "arxiv_id": getattr(ctx, "arxiv_id", None), "model": model,
            "status": status, "host": self.cfg.host, "budget": total,
            "total_h100": total, "remaining_h100": remaining, "spent_h100": 0,
            "tool_rounds_used": 0, "started_at": now_iso(), "updated_at": now_iso(),
            "batch_id": self.cfg.batch_id, "batch_label": self.cfg.batch_label,
        })

    def on_event(self, kind: str, ctx, payload: dict[str, Any]) -> None:
        run_id = run_id_of(ctx)
        if not run_id:
            return
        if kind == "round_open" or kind == "final":
            idx = payload.get("round_index")
            self._note_round(run_id, idx)
            self._put("events", message_row(self._row_base(run_id, kind, idx), payload))
            if kind == "final":
                self._finish(ctx, run_id, payload.get("exit_reason") or "")
        elif kind == "usage":
            self._stats_for(run_id).add_usage(
                payload.get("round_index"), payload.get("kind"), payload.get("usage"))
            self._put("run_patch", (run_id, {**self._stats_for(run_id).run_fields(), "updated_at": now_iso()}))
        elif kind == "call_start":
            self._stats_for(run_id).add_tool_call()
            self._put("events", call_row(self._row_base(run_id, "call_start", self._round.get(run_id)), payload["call"]))
        elif kind == "call_result":
            self._put("events", result_row(self._row_base(run_id, "call_result", self._round.get(run_id)), payload["result"]))
            self._patch_meters(run_id, payload["result"])

    # ---- run-row meter / finalize --------------------------------------------
    def _patch_meters(self, run_id, res):
        rem = res.get("remaining_h100_hours")
        if rem is None:
            return
        rounds = self._rounds_seen.get(run_id, 0)
        key = (round(rem, 4), rounds)
        if self._lastpatch.get(run_id) == key:
            return
        self._lastpatch[run_id] = key
        fields = {"remaining_h100": round(rem, 4), "tool_rounds_used": rounds, "updated_at": now_iso()}
        total = self._total.get(run_id)
        if total is not None:
            fields["spent_h100"] = round(total - rem, 4)
        self._put("run_patch", (run_id, fields))

    def _finish(self, ctx, run_id, exit_reason):
        """Enqueue ONE finalize item; the worker does the GPU rollup (an HTTP GET
        that must not run on this event thread) then patches + uploads."""
        total, remaining = budget_of(ctx)
        fields = {"status": "finished", "exit_reason": exit_reason or None,
                  "finished_at": now_iso(), "updated_at": now_iso(),
                  "tool_rounds_used": self._rounds_seen.get(run_id, 0)}
        fields.update(self._stats_for(run_id).run_fields())
        if total is not None and remaining is not None:
            fields["spent_h100"] = round(total - remaining, 4)
            fields["remaining_h100"] = round(remaining, 4)
        meta = self._stats_meta(ctx, run_id, exit_reason)
        ev = getattr(ctx, "evidence", None)
        parent = Path(ev).parent if ev else None
        stats_path = str(parent / "stats.json") if parent and self.cfg.upload_stats else None
        log_path = str(parent / "agent.full.log") if parent and self.cfg.upload_full_log else None
        self._put("finalize", (run_id, fields, meta, stats_path, log_path))

    def _stats_meta(self, ctx, run_id, exit_reason) -> dict[str, Any]:
        total, remaining = budget_of(ctx)
        return {
            "run_id": run_id, "arxiv_id": getattr(ctx, "arxiv_id", None), "host": self.cfg.host,
            "exit_reason": exit_reason or None, "generated_at": now_iso(),
            "budget_h100": total, "remaining_h100": remaining,
            "spent_h100": (round(total - remaining, 4) if total is not None and remaining is not None else None),
            "tool_rounds_used": self._rounds_seen.get(run_id, 0),
        }

    def _finalize(self, run_id, fields, meta, stats_path, log_path):
        """Worker-thread run finalize: GPU rollup -> patch run row -> upload artifacts.
        A rollup failure still patches the row and uploads (swallow-all)."""
        run_fields, gpu_stats = gpu_usage.rollup(self.cfg.url, self.cfg.service_key, run_id)
        if run_fields:
            fields = {**fields, **run_fields}
        self._patch_run(run_id, fields)
        if stats_path:
            doc = {**meta, "gpu": gpu_stats} if gpu_stats else meta
            if self._stats_for(run_id).write_doc(Path(stats_path), doc):
                self._do_storage(run_id, stats_path, f"{run_id}-stats.json", "application/json", "stats_url")
        if log_path:
            self._do_storage(run_id, log_path, f"{run_id}.log", "text/plain", "full_log_url")

    # ---- worker-side item handling -------------------------------------------
    def _handle(self, kind, payload):
        if kind == "run_upsert":
            self._upsert_run(payload)
        elif kind == "run_patch":
            self._patch_run(payload[0], payload[1])
        elif kind == "finalize":
            self._finalize(*payload)

    def _upsert_run(self, row):
        self._post("POST", "/rest/v1/repro_runs?on_conflict=run_id", [row],
                   prefer="resolution=merge-duplicates,return=minimal")

    def _patch_run(self, run_id, fields):
        self._post("PATCH", f"/rest/v1/repro_runs?run_id=eq.{run_id}", fields, prefer="return=minimal")

    def _do_storage(self, run_id, path, object_name, content, url_field):
        """Upload a run artifact to the public bucket, then record its URL on the run."""
        try:
            data = Path(path).read_bytes()
        except OSError:
            return
        self._post("POST", f"/storage/v1/object/{BUCKET}/{object_name}", None, raw=data,
                   content=content, prefer=None)
        url = f"{self.cfg.url}/storage/v1/object/public/{BUCKET}/{object_name}"
        self._patch_run(run_id, {url_field: url})


def install(cfg: SinkConfig | None) -> "TelemetrySink | None":
    """Start the sink and wire it into the live_log seam. ``None`` cfg -> disabled."""
    if cfg is None:
        return None
    sink = TelemetrySink(cfg)
    live_log.register_sink(sink.on_event)
    return sink
