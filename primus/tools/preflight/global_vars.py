###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

# Default warmup / iteration counts. These remain importable as module-level
# constants for backward compatibility, but new code should call
# `get_warmup()` / `get_iteration()` so that `--quick` (or any future override)
# can lower the counts at runtime.
WARMUP = 10
ITERATION = 50

# Internal mutable counters used by the accessors below.
_WARMUP = WARMUP
_ITERATION = ITERATION


def set_warmup(value: int) -> None:
    global _WARMUP
    _WARMUP = int(value)


def set_iteration(value: int) -> None:
    global _ITERATION
    _ITERATION = int(value)


def get_warmup() -> int:
    return _WARMUP


def get_iteration() -> int:
    return _ITERATION


_HOST_NAMES = [None]


def set_hostnames(hostnames):
    global _HOST_NAMES
    _HOST_NAMES[0] = hostnames


def get_hostnames():
    assert _HOST_NAMES[0] is not None, "_HOST_NAMES not initialized"
    return _HOST_NAMES[0]
