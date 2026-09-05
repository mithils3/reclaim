"""Shared queue + worker + batched-PostgREST plumbing for the live event sinks.

``telemetry_sink`` (reproduce runs) and ``reclaim_vllm.runtime.audit_sink`` (the
auditor) stream the same *shape* of transcript to two different table pairs. They
were hand-copied from each other, so the drop/batch/failure semantics existed twice
and drifted. This module owns the half that is genuinely identical — the bounded
queue, the drain worker, the sequence counter, the key-union padding every
PostgREST bulk insert needs, and the swallow-all HTTP wrapper.

What stays with each sink is what actually differs: the row builders (the two
event tables do not have the same columns), the run-row lifecycle, and the
conflict policy on the events insert. Subclasses declare their table paths and
handle their own non-``events`` queue items in :meth:`PostgrestEventSink._handle`.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any

from reclaim_repro import postgrest

QUEUE_MAX = 4000           # events; beyond this we drop rather than block the loop
BATCH_MAX = 50
HTTP_TIMEOUT = 8.0


def now_iso() -> str:
    """UTC stamp in the one format the DB rows, ``agent.log`` and evidence share."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def service_key_from_env() -> str | None:
    """The telemetry service key, or ``None`` when telemetry is off."""
    return os.environ.get("RECLAIM_TELEMETRY_KEY") or None


def credentials_from_env() -> tuple[str, str] | None:
    """``(base_url, service_key)`` from the environment, or ``None`` if unset.

    One place to answer "is telemetry configured?", so every writer no-ops on the
    same condition instead of testing the environment for itself.
    """
    url = os.environ.get("RECLAIM_TELEMETRY_URL")
    key = service_key_from_env()
    if not url or not key:
        return None
    return url.rstrip("/"), key


class PostgrestEventSink:
    """Bounded queue drained by one background thread into batched PostgREST writes.

    Never blocks the caller (a full queue drops) and never raises at it (every
    network failure is counted, not propagated): the on-disk transcript stays the
    source of truth. Subclasses set the table constants and implement
    :meth:`_handle` for the queue items that are not events.
    """

    LABEL = "postgrest_sink"      # worker thread name + close() prefix
    CLOSE_NOTE = ""               # appended to the close() summary line
    ID_COLUMN = "run_id"          # the events table's run-identity column
    EVENTS_PATH = ""              # e.g. "/rest/v1/repro_events"
    EVENTS_PREFER = "return=minimal"

    def __init__(self, url: str, service_key: str) -> None:
        self.url = url
        self.service_key = service_key
        self.q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=QUEUE_MAX)
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        self._round: dict[str, int | None] = {}
        self._rounds_seen: dict[str, int] = {}
        self.dropped = 0
        self.failed = 0
        self._stop = threading.Event()
        # Started last: subclasses call super().__init__() once their own state is
        # in place, so the worker never sees a half-built sink.
        self._worker = threading.Thread(target=self._run, name=self.LABEL, daemon=True)
        self._worker.start()

    # ---- enqueue (called from the loop / tool threads; never blocks) ----------
    def _put(self, kind: str, payload: Any) -> None:
        try:
            self.q.put_nowait((kind, payload))
        except queue.Full:
            self.dropped += 1

    def _next_seq(self, row_id: str) -> int:
        with self._lock:
            n = self._seq.get(row_id, 0)
            self._seq[row_id] = n + 1
            return n

    def _row_base(self, row_id: str, kind: str, idx: int | None) -> dict[str, Any]:
        return {self.ID_COLUMN: row_id, "seq": self._next_seq(row_id), "kind": kind,
                "round_index": idx}

    def _note_round(self, row_id: str, idx: int | None) -> None:
        """Remember the current round index and the high-water round count."""
        self._round[row_id] = idx
        if isinstance(idx, int):
            self._rounds_seen[row_id] = max(self._rounds_seen.get(row_id, 0), idx + 1)

    # ---- worker / HTTP -------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set() or not self.q.empty():
            try:
                item = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            batch = [item]
            while len(batch) < BATCH_MAX:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            self._flush(batch)
            for _ in batch:
                self.q.task_done()

    def _flush(self, batch: list[tuple[str, Any]]) -> None:
        events = []
        for kind, payload in batch:
            if kind == "events":
                events.append(payload)
            else:
                self._handle(kind, payload)
        if events:
            self._insert_events(events)

    def _handle(self, kind: str, payload: Any) -> None:
        """Dispatch one non-``events`` queue item on the worker thread."""
        raise NotImplementedError

    def _insert_events(self, events: list[dict[str, Any]]) -> None:
        # PostgREST bulk insert needs identical keys per row -> pad to the union.
        ks = set().union(*(e.keys() for e in events))
        self._post("POST", self.EVENTS_PATH, [{k: e.get(k) for k in ks} for e in events],
                   prefer=self.EVENTS_PREFER)

    def _post(self, method: str, path: str, body: Any, *, prefer: str | None = None,
              raw: bytes | None = None, content: str = "application/json") -> None:
        try:
            code, text = postgrest.request(
                self.url + path, service_key=self.service_key, method=method,
                body=body, raw=raw, prefer=prefer, content=content, timeout=HTTP_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — best-effort; never propagate
            self._note_failure(f"{type(exc).__name__}: {exc}")
            return
        if not code or code >= 300:
            self._note_failure(f"HTTP {code} {text[:200]}")

    def _note_failure(self, detail: str) -> None:
        """Count a failed write. Subclasses may also surface the first one."""
        self.failed += 1

    def close(self, timeout: float = 12.0) -> None:
        self._stop.set()
        self._worker.join(timeout=timeout)
        if self.dropped or self.failed:
            print(f"{self.LABEL}: {self.dropped} dropped, {self.failed} failed POSTs"
                  f"{self.CLOSE_NOTE}", flush=True)
