"""Shared urllib PostgREST/Storage transport for the telemetry writers.

``telemetry_sink``, ``audit_sink``, and ``audit_upload`` each hand-rolled the same
urllib request (headers with ``apikey``/``Authorization``/``Prefer``, JSON or raw
body, error handling). This is the one place that plumbing lives now; the callers
keep their own retry/timeout choices via parameters, so behaviour is unchanged:

* the two live sinks fire-and-forget — they only look at the status code and count
  a failure, so they wrap :func:`request` and treat ``code == 0`` (transport error,
  no HTTP response) or ``code >= 300`` as a failed POST;
* ``audit_upload`` needs the response body, so it uses the ``(code, text)`` return
  directly (an HTTP 4xx/5xx returns its PostgREST error body; a DNS/connection
  error returns ``(0, str(err))``).

Every caller has always been single-attempt; ``max_attempts`` is exposed so a
caller can opt into transient-error retries without changing the default.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def request(
    url: str,
    *,
    service_key: str,
    method: str = "POST",
    body: Any = None,
    raw: bytes | None = None,
    prefer: str | None = None,
    content: str = "application/json",
    timeout: float,
    max_attempts: int = 1,
    backoff: float = 0.5,
) -> tuple[int, str]:
    """Send one PostgREST/Storage request; return ``(status_code, body_text)``.

    ``raw`` bytes are sent verbatim (Storage uploads); otherwise ``body`` is
    JSON-encoded (``None`` -> no request body). An HTTP error response yields its
    ``(code, body)``; a transport error yields ``(0, str(err))`` after exhausting
    ``max_attempts`` (with a linear ``backoff`` between transport-error retries).
    """
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": content,
    }
    if prefer:
        headers["Prefer"] = prefer
    last: tuple[int, str] = (0, "")
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as exc:  # 4xx/5xx carry a useful body
            return exc.code, exc.read().decode()
        except OSError as exc:  # DNS/connection — transient, optionally retried
            last = (0, str(exc))
            if attempt + 1 < max_attempts:
                time.sleep(backoff * (attempt + 1))
    return last
