from typing import Callable

from primus.core.projection.pipeline_simulation.scheduler.scheduler_node import (
    SchedulerNode,
)


class WGradRunningCache:
    def __init__(self):
        self.cache = {}
        self.cur_minibatch = None
        self.cur_chunk = None

    def set_current_minibatch_and_chunk(self, minibatch: int, chunk: int):
        self.cur_minibatch = minibatch
        self.cur_chunk = chunk

    def append(self, wgrad_func: Callable):
        assert self.cur_minibatch is not None, "current minibatch is not set"
        assert self.cur_chunk is not None, "current chunk is not set"
        if self.cur_minibatch not in self.cache:
            self.cache[self.cur_minibatch] = {}
        if self.cur_chunk not in self.cache[self.cur_minibatch]:
            self.cache[self.cur_minibatch][self.cur_chunk] = []
        self.cache[self.cur_minibatch][self.cur_chunk].append(wgrad_func)

    def flush(self, minibatch: int, chunk: int):
        assert minibatch in self.cache, "minibatch not found in cache"
        assert chunk in self.cache[self.cur_minibatch], "chunk not found in cache"
        for wgrad_func in self.cache[self.cur_minibatch][self.cur_chunk]:
            wgrad_func()
        del self.cache[self.cur_minibatch][self.cur_chunk]  # release memory


WGRAD_RUNNING_CACHE = WGradRunningCache()


def default_wgrad_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    WGRAD_RUNNING_CACHE.flush(node.mini_batch, node.chunk)
