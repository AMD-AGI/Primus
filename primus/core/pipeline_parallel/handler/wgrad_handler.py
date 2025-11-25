from typing import Callable

from primus.core.pipeline_parallel.scheduler.scheduler_node import SchedulerNode


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
        assert cls.cur_minibatch is not None, "current minibatch is not set"
        assert cls.cur_chunk is not None, "current chunk is not set"
        if cls.cur_minibatch not in cls.cache:
            cls.cache[cls.cur_minibatch] = {}
        if cls.cur_chunk not in cls.cache[cls.cur_minibatch]:
            cls.cache[cls.cur_minibatch][cls.cur_chunk] = []
        cls.cache[cls.cur_minibatch][cls.cur_chunk].append(wgrad_func)

    @classmethod
    def flush(cls, minibatch: int, chunk: int):
        assert minibatch in cls.cache, "minibatch not found in cache"
        assert chunk in cls.cache[minibatch], "chunk not found in cache"
        for wgrad_func in cls.cache[minibatch][chunk]:
            wgrad_func()
        del cls.cache[minibatch][chunk]  # release memory


WGRAD_RUNNING_CACHE = WGradRunningCache()


def default_wgrad_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    WGRAD_RUNNING_CACHE.flush(node.mini_batch, node.chunk)