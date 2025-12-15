###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import lru_cache

from .algorithms import *
from .algorithms.base import PipelineScheduleAlgo

__all__ = [
    "produce_schedule_instance",
]

pp_algorithm_map = {
    "1f1b": Schedule1F1B,
    "1f1b-interleaved": ScheduleInterleaved1F1B,
    "zero-bubble": ScheduleZeroBubble,
    "zbv-formatted": ScheduleZBVFormatted,
    "v-half": ScheduleZBVGreedy,
    "v-min": ScheduleZBVGreedy,
}


@lru_cache
def produce_schedule_instance(
    algorithm: str, pp_size: int, vpp_size: int, micro_batches: int, *args, **kwargs
) -> PipelineScheduleAlgo:
    if algorithm not in pp_algorithm_map:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    kwargs = {}
    if algorithm == "v-half":
        kwargs["memory_config"] = "half"
    elif algorithm == "v-min":
        kwargs["memory_config"] = "min"
    return pp_algorithm_map[algorithm](pp_size, vpp_size, micro_batches, *args, **kwargs)
