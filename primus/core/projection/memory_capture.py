###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Memory capture utilities for projection benchmarking.

This module provides a small, dependency-free interface for snapshotting
per-rank GPU memory at well-defined moments during the projection
benchmark.  The captured data is later consumed by the memory-projection
extrapolation logic to predict the per-rank peak memory of a target
cluster (the OOM-relevant quantity).

Design notes
------------
- We capture both ``allocated`` and ``reserved`` byte counts.  OOM is
  gated by ``reserved`` (what the PyTorch caching allocator has actually
  carved out of VRAM), not ``allocated`` (live tensor bytes).  We track
  both so that the upper-bound projection can use ``reserved`` while the
  point estimate uses ``allocated``.
- We do *not* call ``torch.cuda.reset_peak_memory_stats`` between phases.
  ``max_memory_allocated`` / ``max_memory_reserved`` therefore accumulate
  the running high-water mark, which is exactly what we want for OOM
  prediction.  Per-phase deltas are recoverable from the ``current``
  snapshots.
- Every helper is a no-op when CUDA is unavailable, so simulate-only
  paths and unit tests on CPU continue to work.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401  (lazy import so non-GPU paths don't pay)
    except Exception:
        return False
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def capture_memory_snapshot(label: str, device: Optional[Any] = None) -> Dict[str, Any]:
    """Snapshot per-rank GPU memory at the moment of the call.

    Args:
        label: Free-form name (e.g. ``"post_setup"``, ``"post_layer_benchmark"``)
            attached to the snapshot for downstream reporting.
        device: Optional ``torch.device`` to query.  Defaults to the
            current CUDA device.

    Returns:
        A serializable dict of the form::

            {
                "label": "post_setup",
                "allocated_bytes": 12_345_678,        # current live
                "reserved_bytes":  20_000_000,        # current allocator reservation
                "max_allocated_bytes": 13_000_000,    # cumulative peak so far
                "max_reserved_bytes":  22_000_000,    # cumulative peak so far
                "free_bytes":  60_000_000_000,        # mem_get_info: free
                "total_bytes": 80_000_000_000,        # mem_get_info: total
            }

        When CUDA is unavailable the dict is returned with all numeric
        fields set to 0 (the label is preserved).
    """
    snap: Dict[str, Any] = {
        "label": label,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "max_allocated_bytes": 0,
        "max_reserved_bytes": 0,
        "free_bytes": 0,
        "total_bytes": 0,
    }

    if not _cuda_available():
        return snap

    import torch

    try:
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        snap["allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        snap["reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        snap["max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        snap["max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        try:
            free, total = torch.cuda.mem_get_info(device)
            snap["free_bytes"] = int(free)
            snap["total_bytes"] = int(total)
        except Exception:
            # mem_get_info can be unavailable on some HIP/older builds.
            pass
    except Exception as e:
        snap["error"] = repr(e)

    return snap


class MemoryBenchmarkRecorder:
    """Lightweight container for a sequence of memory snapshots.

    Usage::

        rec = MemoryBenchmarkRecorder(rank=0)
        rec.snapshot("pre_setup")
        # ... build trainer / model ...
        rec.snapshot("post_setup")
        # ... run layer benchmark ...
        rec.snapshot("post_layer_benchmark")
        payload = rec.to_payload()  # serializable dict for the JSON artifact

    The recorder only retains snapshots taken on rank 0 (other ranks are
    skipped to keep the artifact rank-symmetric and small).  Per-rank
    capture can be added later if asymmetric PP stages need it.
    """

    def __init__(self, rank: Optional[int] = None) -> None:
        self.rank = int(os.getenv("RANK", "0")) if rank is None else int(rank)
        self.snapshots: List[Dict[str, Any]] = []

    def snapshot(self, label: str) -> Optional[Dict[str, Any]]:
        if self.rank != 0:
            return None
        snap = capture_memory_snapshot(label)
        self.snapshots.append(snap)
        return snap

    def to_payload(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {}
        # Phase deltas: difference in current allocated between consecutive
        # snapshots.  Useful for "what did setup actually allocate?"
        deltas: List[Dict[str, Any]] = []
        for prev, cur in zip(self.snapshots, self.snapshots[1:]):
            deltas.append(
                {
                    "from": prev["label"],
                    "to": cur["label"],
                    "allocated_delta_bytes": cur["allocated_bytes"] - prev["allocated_bytes"],
                    "reserved_delta_bytes": cur["reserved_bytes"] - prev["reserved_bytes"],
                }
            )

        # Global high-water marks across all captured phases.
        global_peak_allocated = max(s["max_allocated_bytes"] for s in self.snapshots)
        global_peak_reserved = max(s["max_reserved_bytes"] for s in self.snapshots)

        return {
            "captured_on_rank": self.rank,
            "snapshots": self.snapshots,
            "phase_deltas": deltas,
            "global_peak_allocated_bytes": int(global_peak_allocated),
            "global_peak_reserved_bytes": int(global_peak_reserved),
        }


def format_bytes(b: int) -> str:
    """Pretty-print a byte count (GB)."""
    return f"{b / (1024**3):.3f} GB"
