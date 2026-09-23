###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from Tencent HunyuanVideo-1.5 / HY-WorldPlay official implementation.

"""Single-process parallel-state stub for offline WorldPlay encoding."""

from dataclasses import dataclass


@dataclass
class ParallelDims:
    sp: int = 1
    world_size: int = 1
    sp_enabled: bool = False
    sp_group = None
    sp_mesh = None
    sp_rank: int = 0
    dp_enabled: bool = False


__parallel_dims = ParallelDims()


def initialize_parallel_state(sp: int = 1):
    global __parallel_dims
    if sp != 1:
        raise ValueError("WorldPlay offline encoding only supports sp=1")
    __parallel_dims = ParallelDims()
    return __parallel_dims


def get_parallel_state():
    return __parallel_dims
