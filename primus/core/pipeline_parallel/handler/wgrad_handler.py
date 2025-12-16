###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Callable

from megatron.training.global_vars import get_args

from primus.core.pipeline_parallel.scheduler.scheduler_node import SchedulerNode
from primus.modules.trainer.megatron.utils import fwd_bwd_wrapper


class WGradRunningCache:

    cache = {}
    cur_minibatch = None
    cur_chunk = None

    @classmethod
    def set_current_minibatch_and_chunk(cls, minibatch: int, chunk: int):
        cls.cur_minibatch = minibatch
        cls.cur_chunk = chunk

    @classmethod
    def append(cls, wgrad_func: Callable):
        if cls.cur_minibatch is None or cls.cur_chunk is None:
            wgrad_func()
            return
        if cls.cur_minibatch not in cls.cache:
            cls.cache[cls.cur_minibatch] = {}
        if cls.cur_chunk not in cls.cache[cls.cur_minibatch]:
            cls.cache[cls.cur_minibatch][cls.cur_chunk] = []
        cls.cache[cls.cur_minibatch][cls.cur_chunk].append(wgrad_func)

    @classmethod
    def flush(cls, minibatch: int, chunk: int):
        assert minibatch in cls.cache, f"minibatch {minibatch} not found in cache"
        assert chunk in cls.cache[minibatch], f"chunk {chunk} not found in cache"

        for idx, wgrad_func in enumerate(cls.cache[minibatch][chunk]):
            wgrad_func()
            cls.cache[minibatch][chunk][idx] = None  # release memory

        del cls.cache[minibatch][chunk]

    @classmethod
    def is_empty(cls):
        for minibatch in cls.cache:
            if len(cls.cache[minibatch]) > 0:
                return False
        return True


WGRAD_RUNNING_CACHE = WGradRunningCache()


def default_wgrad_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    cal_stored_grad_func = WGRAD_RUNNING_CACHE.flush
    if get_args().dump_pp_data:
        cal_stored_grad_func = fwd_bwd_wrapper(
            WGRAD_RUNNING_CACHE.flush, "wgrad", minibatch=node.mini_batch, chunk=node.chunk
        )

    cal_stored_grad_func(node.mini_batch, node.chunk)
