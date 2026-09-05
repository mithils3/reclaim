"""Best-effort live upload of the auditor transcript to the telemetry service (opt-in).

The reproduce mirror of this is ``reclaim_repro.telemetry_sink``; this is the same
machinery for the auditor so its run streams to the dashboard's Audits page in
real time. Disabled unless ``RECLAIM_TELEMETRY_URL`` + ``RECLAIM_TELEMETRY_KEY``
are set. A background worker drains a bounded queue and batches PostgREST writes;
a full queue
drops rather than blocking, and every network failure is swallowed — the loop and
the on-disk trace stay the source of truth.

Fed via ``live_events.register_sink(sink.on_event)``: each round becomes
``audit_events`` rows under one ``audit_runs`` row per graded paper. The row links
back to the reproduce run it graded via ``graded_run_id`` (``$RECLAIM_GRADED_RUN_ID``,
set by the pipeline), so the app can pair the audit with its run.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any

from reclaim_repro import live_log  # shared tool-call argument decoder
from reclaim_repro.event_sink import PostgrestEventSink, credentials_from_env, now_iso
from reclaim_vllm.runtime import live_events

STDOUT_CAP = 8000  # chars per stdout/stderr/text cell


# ---- pure PostgREST row builders ---------------------------------------------
# Mirrors ``reclaim_repro.telemetry_rows`` (the reproduce-run version) so the
# audit transcript lands in ``audit_events`` with the SAME column shape the
# viewer's ``rowsToRounds`` already understands — the dashboard renders an audit
# exactly like a run. Pure: an event payload in, a row ``dict`` out. The sink
# supplies ``base`` (``audit_run_id`` / ``seq`` / ``kind`` / ``round_index``).


def cap(text: Any, limit: int = STDOUT_CAP) -> tuple[str, bool]:
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s, False
    return s[:limit] + f"\n…(+{len(s) - limit} chars truncated)", True


def message_row(base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    msg = payload.get("message") or {}
    reasoning, _ = cap(msg.get("reasoning") or msg.get("reasoning_content"), STDOUT_CAP * 2)
    content, _ = cap(msg.get("content"), STDOUT_CAP * 2)
    base.update({"role": "assistant", "reasoning": reasoning or None, "content": content or None})
    if base.get("kind") == "final":
        base["exit_reason"] = payload.get("exit_reason") or None
    return base


def call_row(base: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    args = live_log.call_arguments(call)
    base["tool_name"] = str((call.get("function") or {}).get("name") or "?")
    if "command" in args:
        base.update(detail_kind="command", command=str(args["command"]))
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
        "error": (str(res["error"]).splitlines()[0] if res.get("error") else None),
        "path": res.get("path"), "stdout": out or None, "stderr": err or None,
        "truncated": bool(t1 or t2),
    })
    return base


@dataclass
class SinkConfig:
    url: str
    service_key: str
    graded_run_id: str | None
    model: str | None
    host: str
    # Discriminator for the audit's identity. The id is otherwise derived from the
    # graded run alone, so a SECOND grader of the same run (a different model, or a
    # re-grade) lands on the first audit's row: it overwrites the run row's model
    # and status while inheriting the old verdict, and every event insert collides
    # with the existing (audit_run_id, seq) rows and is dropped -- an audit that
    # appears finished with someone else's score and no transcript. Set this per
    # grading attempt to give each audit its own row.
    attempt: str | None = None

    @classmethod
    def from_env(cls, model: str | None = None) -> "SinkConfig | None":
        creds = credentials_from_env()
        if creds is None:
            return None
        url, key = creds
        return cls(url=url, service_key=key,
                   graded_run_id=os.environ.get("RECLAIM_GRADED_RUN_ID") or None,
                   model=model, host=socket.gethostname())


class AuditSink(PostgrestEventSink):
    LABEL = "audit_sink"
    ID_COLUMN = "audit_run_id"
    # Idempotent on (audit_run_id, seq), like the run upsert below. A plain insert
    # makes ONE duplicate row fatal to the whole batch: PostgREST rejects the
    # statement, so up to BATCH_MAX-1 perfectly good events are lost with it.
    # Duplicates are not hypothetical -- two graders launched in the same second
    # derive the same attempt stamp, hence the same audit_run_id, and the second
    # one collides on every seq it allocates. ignore-duplicates keeps the
    # transcript already on the page and lets the rest of the batch land.
    EVENTS_PATH = "/rest/v1/audit_events?on_conflict=audit_run_id,seq"
    EVENTS_PREFER = "resolution=ignore-duplicates,return=minimal"

    def __init__(self, cfg: SinkConfig) -> None:
        self.cfg = cfg
        self._seen: set[str] = set()
        self._base = now_iso()
        super().__init__(cfg.url, cfg.service_key)

    # ---- identity -----------------------------------------------------------
    def _audit_run_id(self, custom_id: str) -> str:
        base = self.cfg.graded_run_id or self._base
        attempt = f"-{self.cfg.attempt}" if self.cfg.attempt else ""
        return f"{base}-{custom_id}{attempt}-audit"

    def _ensure_run(self, custom_id: str) -> str:
        aid = self._audit_run_id(custom_id)
        if aid not in self._seen:
            self._seen.add(aid)
            # The verdict columns are cleared on start: re-grading into an existing
            # row must not display the previous run's score while this one works.
            self._put("run_upsert", {
                "audit_run_id": aid, "graded_run_id": self.cfg.graded_run_id,
                "arxiv_id": custom_id, "model": self.cfg.model, "status": "running",
                "host": self.cfg.host, "started_at": now_iso(), "updated_at": now_iso(),
                "score": None, "verdict": None, "reproduced": None,
                "has_high_cheat_flag": None, "exit_reason": None, "finished_at": None,
                "tool_rounds_used": 0,
            })
        return aid

    def on_event(self, kind: str, meta: dict[str, Any], payload: dict[str, Any]) -> None:
        custom_id = meta.get("custom_id")
        if not custom_id:
            return
        aid = self._ensure_run(custom_id)
        idx = meta.get("round_index")
        if kind in ("round_open", "final"):
            self._note_round(aid, idx)
            self._put("events", message_row(self._row_base(aid, kind, idx), payload))
            if kind == "final":
                self._finish(aid, payload)
        elif kind == "call_start":
            self._put("events", call_row(self._row_base(aid, "call_start", self._round.get(aid)), payload["call"]))
        elif kind == "call_result":
            self._put("events", result_row(self._row_base(aid, "call_result", self._round.get(aid)), payload["result"]))

    def _finish(self, aid: str, payload: dict[str, Any]) -> None:
        v = payload.get("verdict") or {}
        self._put("run_patch", (aid, {
            "status": "finished", "exit_reason": payload.get("exit_reason") or None,
            "score": v.get("score"), "verdict": v.get("verdict"),
            "reproduced": v.get("reproduced"), "has_high_cheat_flag": v.get("has_high_cheat_flag"),
            "tool_rounds_used": self._rounds_seen.get(aid, 0),
            "finished_at": now_iso(), "updated_at": now_iso(),
        }))

    # ---- worker-side item handling ------------------------------------------
    def _handle(self, kind: str, payload: Any) -> None:
        if kind == "run_upsert":
            self._post("POST", "/rest/v1/audit_runs?on_conflict=audit_run_id", [payload],
                       prefer="resolution=merge-duplicates,return=minimal")
        elif kind == "run_patch":
            self._post("PATCH", f"/rest/v1/audit_runs?audit_run_id=eq.{payload[0]}", payload[1],
                       prefer="return=minimal")

    def _note_failure(self, detail: str) -> None:
        """Count a failed write, and say so the FIRST time.

        Silence here reads as a working stream: a rejected insert (a colliding
        audit id, a schema drift) would otherwise leave the dashboard quietly
        empty while the audit runs to completion.
        """
        self.failed += 1
        if self.failed == 1:
            print(f"audit_sink: write rejected ({detail}); the Audits page will be incomplete",
                  flush=True)


def install(cfg: SinkConfig | None) -> "AuditSink | None":
    """Start the sink and wire it into the live_events seam. ``None`` cfg -> disabled."""
    if cfg is None:
        return None
    sink = AuditSink(cfg)
    live_events.register_sink(sink.on_event)
    return sink
