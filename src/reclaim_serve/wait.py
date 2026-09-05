"""Poll ``/health`` until the server is ready (or the process dies)."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request

from reclaim_serve.config import HEALTH_POLL_INTERVAL, SERVER_STARTUP_TIMEOUT


def wait_until_ready(
    health_url: str,
    process: subprocess.Popen,
    *,
    timeout: float = SERVER_STARTUP_TIMEOUT,
    interval: float = HEALTH_POLL_INTERVAL,
) -> None:
    """Block until ``health_url`` returns 2xx, raising if it never does.

    Health is polled on loopback (the serving node itself) even though the server
    binds 0.0.0.0, so this works before any peer can reach the fabric address.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode} before ready")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(f"vLLM server healthy at {health_url}", file=sys.stderr, flush=True)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(interval)
    raise TimeoutError(f"vLLM server did not become ready within {timeout:.0f}s ({health_url})")
