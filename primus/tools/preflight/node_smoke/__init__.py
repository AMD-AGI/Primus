###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Node-local preflight smoke test.

Goal: quickly identify which nodes have problems before launching a real
training job. Each node tests itself in parallel with no global rendezvous,
writes a per-node JSON verdict, and an aggregator (rank 0) reads all JSONs
and emits PASS/FAIL lists usable by SLURM ``--exclude=`` / ``--nodelist=``.

Subcommands
-----------

* ``run`` -- per-node entry. Runs Tier 1 (per-GPU sanity + reused
  host/gpu/network info collectors + a small dmesg scan) and, when
  ``--tier2-perf`` is set, Tier 2 perf sanity (GEMM TFLOPS, HBM bandwidth,
  and node-local RCCL all-reduce). Writes ``<dump>/smoke/<host>.json``.
  Always exits 0 when the JSON was written (the per-node verdict lives
  in the JSON's ``status`` field, not in the exit code) -- otherwise an
  intentionally-detected unhealthy node would make srun pollute its
  output with one ``error: ... task N: Exited with exit code 1`` per
  failing node, which is misleading: the smoke test is succeeding at
  identifying bad nodes, not failing.

* ``aggregate`` -- read all per-node JSONs and emit
  ``<dump>/smoke_report.md``, ``<dump>/passing_nodes.txt``, and
  ``<dump>/failing_nodes.txt``. Exits non-zero if any node FAILs or is
  missing -- this is the single CI-friendly cluster-health exit signal.

* ``_per_gpu`` -- internal subcommand spawned by ``run`` to test a single
  GPU in an isolated subprocess with a hard timeout. Not for direct use.

Why per-GPU subprocesses?
-------------------------

A stuck ``torch.cuda.set_device(i)`` cannot be aborted reliably with
``signal.alarm`` because the call may be inside a non-interruptible driver
syscall. By running each per-GPU test in its own subprocess we can SIGKILL
it on timeout without affecting the rest of the node's checks.

Package layout
--------------

This module is a sub-package -- the implementation is split across many
small files mirroring the Tier 1 A-G section structure. The single
public entry point exported here is :func:`main`, which is also the
target of ``python -m primus.tools.preflight.node_smoke``.
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
