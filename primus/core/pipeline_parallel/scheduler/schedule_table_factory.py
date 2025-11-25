from typing import List
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
    "zbv-greedy": ScheduleZBVGreedy,
}

def produce_schedule_instance(algorithm: str, pp_size: int, vpp_size: int, micro_batches: int, *args, **kwargs) -> PipelineScheduleAlgo:
    if algorithm not in pp_algorithm_map:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    return pp_algorithm_map[algorithm](pp_size, vpp_size, micro_batches, *args, **kwargs)