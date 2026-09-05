"""Retry transient network failures around a urllib request thunk.

On some clusters the orchestration host has rate-limited outbound DNS, so a single
``urllib.urlopen`` can raise ``gaierror`` ("Temporary failure in name resolution")
at random. Without a retry, one blip on any of the
hundreds of model calls in a multi-hour run propagates out of the worker future
and kills the whole budgeted run. ``with_retries`` wraps a request thunk in
bounded exponential backoff + jitter, retrying ONLY errors that are plausibly
transient -- caller errors (400/401/403/404/422, bad JSON) re-raise immediately
so we never burn budget spinning on something that can't succeed.

The defaults give a ~2 minute tolerance window (8 attempts, backoff capped at
60s) -- a single ``urlopen`` raising ``gaierror`` only costs the backoff sleep
since DNS resolution fails fast, so a wide window rides out a multi-second outage
cheaply while a genuinely-down server still bails in ~2 minutes.

Tunable via env: ``RECLAIM_HTTP_MAX_ATTEMPTS`` (default 8).
"""

from __future__ import annotations

import os
import random
import socket
import sys
import time
import urllib.error
from typing import Callable, TypeVar

T = TypeVar("T")

# Transient/server-side or rate-limit statuses worth retrying. Everything else in
# the 4xx range is a caller error that a retry cannot fix.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0


def _max_attempts() -> int:
    raw = os.environ.get("RECLAIM_HTTP_MAX_ATTEMPTS")
    if raw and raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return DEFAULT_MAX_ATTEMPTS


def _is_retryable(exc: BaseException) -> bool:
    # HTTPError is a subclass of URLError -- check it first and gate on status.
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_STATUS
    # URLError wraps gaierror (DNS), connection refused/reset, etc.
    return isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError))


def _retry_after_seconds(exc: BaseException) -> float | None:
    if isinstance(exc, urllib.error.HTTPError) and exc.headers is not None:
        raw = exc.headers.get("Retry-After")
        if raw and raw.strip().isdigit():
            return float(raw.strip())
    return None


def annotate_http_error(exc: urllib.error.HTTPError) -> str:
    """Fold an HTTPError's response body into its message; return the body text.

    ``str(HTTPError)`` renders as just ``HTTP Error 400: Bad Request`` -- the
    provider's JSON explanation of *why* the request was rejected (e.g.
    ``"This response_format type is unavailable now"``) sits unread in the
    response body. Read it once, stash it on the exception as ``reclaim_body``
    (so callers can branch on the cause without re-reading a consumed stream),
    and append it to ``msg`` so the surfaced error names the real problem instead
    of an opaque status line. Idempotent per exception and best-effort: a body
    that can't be read leaves the error unchanged.
    """
    body = getattr(exc, "reclaim_body", None)
    if body is not None:
        return body
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 -- the body is a diagnostic nicety, never fatal
        body = ""
    exc.reclaim_body = body
    if body:
        exc.msg = f"{exc.msg}: {body[:2000]}"
    return body


def with_retries(
    thunk: Callable[[], T],
    *,
    what: str = "request",
    max_attempts: int | None = None,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> T:
    """Call ``thunk`` retrying transient failures with backoff; return its result."""
    attempts = max_attempts if max_attempts is not None else _max_attempts()
    for attempt in range(1, attempts + 1):
        try:
            return thunk()
        except Exception as exc:  # noqa: BLE001 -- re-raised below unless transient
            if isinstance(exc, urllib.error.HTTPError):
                annotate_http_error(exc)  # in-place; enriches the message we raise/log
            if attempt >= attempts or not _is_retryable(exc):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                delay += random.uniform(0, delay * 0.25)  # jitter de-syncs workers
            print(
                f"[retry] {what}: {type(exc).__name__}: {exc} -- "
                f"attempt {attempt}/{attempts}, retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable: loop returns or raises")
