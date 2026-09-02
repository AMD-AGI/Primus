###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Environment-variable helpers shared by this backend's patches.

Patches are conventionally gated on an environment variable so that a default
run is unaffected by them (see ``patches/__init__.py``). That makes parsing env
values a cross-cutting concern rather than a per-patch one, and parsing it
inconsistently is a real hazard: a patch that accepts only ``"1"`` silently
ignores ``PRIMUS_X=true`` and the feature appears not to work, with nothing
logged and nothing raised.

Keep this module dependency-free -- no torch, no AutoModel -- so that patch
conditions can be evaluated, and unit tests can run, without importing either.
"""
from __future__ import annotations

import os

# Accepted spellings of "on". Deliberately generous: the cost of accepting
# "True" is nothing, and the cost of rejecting it is a silently disabled feature.
TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, case-insensitively."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on anything unparseable.

    A malformed value returns the default rather than raising: these knobs are
    diagnostic, and taking down a training run over a typo in a profiler setting
    would be a worse outcome than profiling the wrong steps.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    """Read a string environment variable, treating empty as unset."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def current_rank() -> int:
    """Best-effort process rank, for per-rank output paths.

    Reads the launcher's environment first and only then asks torch.distributed,
    so this works before (and without) process-group initialization -- which is
    the situation patches run in.
    """
    for key in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        raw = os.getenv(key)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0
