"""Serving defaults shared across the launcher modules.

These are the defaults the standalone, network-routable ``vllm serve`` uses for
each model; there is no in-process server left to mirror.
"""

from __future__ import annotations

import os

# Network defaults. Bind to all interfaces so other nodes can reach the server;
# the *advertised* address is resolved separately (see network.py) from the
# high-speed fabric interface so the published URL is the one peers should dial.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
# High-speed fabric interfaces, in preference order. Any one of them gives a
# routable IP for the API; the first with an address is the one published. NCCL data
# traffic is steered separately with an NCCL_SOCKET_IFNAME prefix that covers all of
# them. The default is empty, so the published address comes from the outbound-socket
# fallback in network.py. Set $RECLAIM_FABRIC_IFACES to a comma-separated list to name
# the interfaces on your fabric.
# e.g. RECLAIM_FABRIC_IFACES=ib0,ib1 on an InfiniBand fabric
ENV_FABRIC_IFACES = "RECLAIM_FABRIC_IFACES"
FABRIC_IFACES = [
    name.strip() for name in (os.environ.get(ENV_FABRIC_IFACES) or "").split(",") if name.strip()
]
DEFAULT_FABRIC_IFACE = FABRIC_IFACES[0] if FABRIC_IFACES else None

# How long to wait for /health after launch before giving up.
SERVER_STARTUP_TIMEOUT = 1800.0
HEALTH_POLL_INTERVAL = 5.0

# Must match reclaim_vllm.config.config.MAX_MODEL_LEN, and is restated rather than
# imported: this package must not import the agent packages (see the package
# docstring), so the dependency can only point the other way. Change both together.
MAX_MODEL_LEN = 196608
DEFAULT_GPU_MEMORY_UTILIZATION = 0.95

# The published endpoint contract. The reclaim runner reads ``base_url`` from
# this file when $RECLAIM_ENDPOINT_FILE points at it.
DEFAULT_ENDPOINT_FILENAME = "vllm_endpoint.json"

# Environment-variable names that form the cross-repo seam.
ENV_ENDPOINT_FILE = "RECLAIM_ENDPOINT_FILE"
