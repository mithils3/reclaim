"""Reproduction agent package.

The execution agent that takes one lockfile row and actually runs the paper's
MRE on the cluster.

This package imports only stable, mode-agnostic primitives from ``reclaim_vllm``
and forks the small loop body on purpose; it does not touch the auditor.
Submodules are imported lazily (not re-exported here) so ``--help`` and
the leaf utilities stay light.
"""

__version__ = "0.1.0"
