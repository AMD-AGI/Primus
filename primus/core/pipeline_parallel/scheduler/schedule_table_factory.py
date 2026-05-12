###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from .algorithms import *
from .algorithms.base import PipelineScheduleAlgo

__all__ = [
    "produce_schedule_instance",
]

pp_algorithm_map = {
    "1f1b": Schedule1F1B,
    "1f1b-interleaved": ScheduleInterleaved1F1B,
    "zero-bubble": ScheduleZeroBubble,
    "zero-bubble-heuristic": ScheduleZeroBubbleHeuristic,
    "zbv-formatted": ScheduleZBVFormatted,
    "v-half": ScheduleZBVGreedy,
    "v-min": ScheduleZBVGreedy,
}


_schedule_instance_cache: dict = {}


def produce_schedule_instance(
    algorithm: str, pp_size: int, vpp_size: int, micro_batches: int, *args, **kwargs
) -> PipelineScheduleAlgo:
    if algorithm not in pp_algorithm_map:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    if algorithm == "v-half":
        kwargs["memory_config"] = "half"
    elif algorithm == "v-min":
        kwargs["memory_config"] = "min"

    def _make_hashable(v):
        if isinstance(v, list):
            return tuple(v)
        return v

    cache_key = (
        algorithm,
        pp_size,
        vpp_size,
        micro_batches,
        tuple(_make_hashable(a) for a in args),
        tuple(sorted((k, _make_hashable(v)) for k, v in kwargs.items())),
    )
    if cache_key in _schedule_instance_cache:
        return _schedule_instance_cache[cache_key]

    instance = pp_algorithm_map[algorithm](pp_size, vpp_size, micro_batches, *args, **kwargs)
    _schedule_instance_cache[cache_key] = instance
    return instance
