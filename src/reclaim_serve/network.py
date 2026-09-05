"""Resolve the address other nodes should dial.

The server binds ``0.0.0.0`` (every interface), but the URL we *publish* must be
the one peers can route to. On an HPC fabric that is the high-speed interface (see
``config.FABRIC_IFACES``), not loopback and not the management IP. If the interface
lookup falls through to loopback, a published ``127.0.0.1`` URL would be reachable
only from the serving node itself, so we fail loudly instead.
"""

from __future__ import annotations

import socket
import subprocess

from reclaim_serve.config import FABRIC_IFACES


def discover_ipv4(iface: str) -> str | None:
    """Return the IPv4 bound to ``iface`` (e.g. ``ib0``), or None if absent."""
    try:
        out = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", iface],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        # "... inet 10.0.0.5/16 ..." -> 10.0.0.5
        fields = line.split()
        if "inet" in fields:
            cidr = fields[fields.index("inet") + 1]
            ip = cidr.split("/")[0]
            if ip and not ip.startswith("127."):
                return ip
    return None


def hostname_ipv4() -> str | None:
    """Best-effort routable IPv4 via an outbound socket, skipping loopback."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            return ip if ip and not ip.startswith("127.") else None
    except OSError:
        return None


def advertised_host(explicit: str | None, iface: str | None) -> str:
    """The host to put in the published URL.

    Priority: explicit ``--advertise-ip`` > the requested fabric interface, then
    the rest of the configured fabric interfaces > outbound-socket IP > the node
    hostname. Loopback is never published.
    """
    if explicit:
        return explicit
    for candidate in _iface_candidates(iface):
        ip = discover_ipv4(candidate)
        if ip:
            return ip
    ip = hostname_ipv4()
    if ip:
        return ip
    return socket.gethostname()


def fabric_ipv4(iface: str | None) -> str | None:
    """This node's own fabric IPv4, or None, with no public-IP fallback.

    Used to pin vLLM's internal RPC / multiproc message queue to the same fabric
    NCCL uses. Unlike :func:`advertised_host`, this never falls back to the
    outbound-socket guess (which returns the public VLAN, unroutable
    compute-to-compute) or the explicit ``--advertise-ip`` (which is the *head*
    IP on every node); each node must resolve its own interface address.
    """
    for candidate in _iface_candidates(iface):
        ip = discover_ipv4(candidate)
        if ip:
            return ip
    return None


def _iface_candidates(iface: str | None) -> list[str]:
    """The requested interface first, then the configured fabric interfaces, deduped."""
    ordered: list[str] = []
    for candidate in [iface, *FABRIC_IFACES]:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"
