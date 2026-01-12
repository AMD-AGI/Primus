###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType, SchedulerNode


def deallocate_gpu_tensor(tensor: torch.Tensor):
    """Pseudo-deallocate (i.e., set to scalar) the tensor's '.data' field.

    This method should be called right after the tensor has been
    sent to the next pipeline stage. At this point, the tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    """
    assert isinstance(tensor, torch.Tensor), "expected Tensor, found %s." % type(tensor).__name__
    assert tensor._base is None, "counter-productive to free a view of another tensor."
    tensor.data = torch.empty((1,), device=tensor.device, dtype=tensor.dtype)


class OffloadBuffer:

    def __init__(self, gpu_tensor: torch.Tensor):
        # key -> cpu_buffer
        self.cpu_tensor = None

        # key -> gpu_tensor
        self.gpu_tensor = gpu_tensor

        if not self.gpu_tensor.is_contiguous():
            self.gpu_tensor = self.gpu_tensor.contiguous()


    def offload(self):

        self.cpu_tensor = torch.empty_like(
            self.gpu_tensor.data, device="cpu", layout=self.gpu_tensor.data.layout, requires_grad=False, pin_memory=False
        )

        # todo: check if pinned memory is available
        self.cpu_tensor.copy_(self.gpu_tensor.data, non_blocking=True)

    
    def deallocate_gpu_tensor(self):
        deallocate_gpu_tensor(self.gpu_tensor)


    def reload(self):
        assert self.cpu_tensor is not None, "cpu_tensor is not initialized"

        self.gpu_tensor.data = torch.empty(
            self.cpu_tensor.shape, device="cuda", dtype=self.cpu_tensor.dtype
        )
        self.gpu_tensor.data.copy_(self.cpu_tensor, non_blocking=True)


    def release(self):
        # release holding refs
        self.cpu_tensor = None
        self.gpu_tensor = None


class OffloadBufferManager:

    def __init__(self):
        self.current_mini_batch = None
        self.current_chunk = None
        self.offload_stream = torch.cuda.Stream()
        self.reload_stream = torch.cuda.Stream()

        # minibatch-chunk -> tag -> offload_buffer
        self.offload_buffers = {}
        # minibatch-chunk -> cuda_event
        self.offload_events = {}
        self.reload_events = {}

        self.in_cache_minibatch_chunk = None

    def set_current_minibatch_chunk_info(self, mini_batch: int, chunk: int):
        self.current_mini_batch = mini_batch
        self.current_chunk = chunk

    def push_minibatch_chunk_to_cache(self):
        assert self.in_cache_minibatch_chunk is None, "in_cache_minibatch_chunk must be empty"

        self.in_cache_minibatch_chunk = (self.current_mini_batch, self.current_chunk)
        self.current_mini_batch = None
        self.current_chunk = None

    def try_pop_minibatch_chunk_from_cache(self):
        if self.in_cache_minibatch_chunk is None:
            return (None, None)

        mini_batch, chunk = self.in_cache_minibatch_chunk
        self.in_cache_minibatch_chunk = None
        return (mini_batch, chunk)

    def init_offload_buffer(self, tensor: torch.Tensor, tag: str):
        assert self.current_mini_batch is not None and self.current_chunk is not None, "current_mini_batch and current_chunk must be set before initializing offload buffer"

        mini_batch_chunk = f"{self.current_mini_batch}-{self.current_chunk}"
        if mini_batch_chunk not in self.offload_buffers:
            self.offload_buffers[mini_batch_chunk] = {}

        if tag not in self.offload_buffers[mini_batch_chunk]:
            self.offload_buffers[mini_batch_chunk][tag] = OffloadBuffer(tensor)
        return self.offload_buffers[mini_batch_chunk][tag]

    def offload_start(self, mini_batch: int, chunk: int):
        mini_batch_chunk = f"{mini_batch}-{chunk}"
        if mini_batch_chunk not in self.offload_buffers:
            return

        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            for tag in self.offload_buffers[mini_batch_chunk]:
                self.offload_buffers[mini_batch_chunk][tag].offload()
            event = torch.cuda.Event()
            event.record(self.offload_stream)
            self.offload_events[mini_batch_chunk] = event


    def wait_offload_done(self, mini_batch: int, chunk: int):
        mini_batch_chunk = f"{mini_batch}-{chunk}"
        if mini_batch_chunk not in self.offload_events:
            return
        torch.cuda.current_stream().wait_event(self.offload_events[mini_batch_chunk])
        del self.offload_events[mini_batch_chunk]

        for tag in self.offload_buffers[mini_batch_chunk]:
            self.offload_buffers[mini_batch_chunk][tag].deallocate_gpu_tensor()

    def reload_start(self, mini_batch: int, chunk: int):
        mini_batch_chunk = f"{mini_batch}-{chunk}"
        if mini_batch_chunk not in self.offload_buffers:
            return
        self.reload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.reload_stream):
            for tag in self.offload_buffers[mini_batch_chunk]:
                self.offload_buffers[mini_batch_chunk][tag].reload()
            event = torch.cuda.Event()
            event.record(self.reload_stream)
            self.reload_events[mini_batch_chunk] = event

    def wait_reload_done(self, mini_batch: int, chunk: int):
        mini_batch_chunk = f"{mini_batch}-{chunk}"
        if mini_batch_chunk not in self.reload_events:
            return
        torch.cuda.current_stream().wait_event(self.reload_events[mini_batch_chunk])
        del self.reload_events[mini_batch_chunk]
        for tag in self.offload_buffers[mini_batch_chunk]:
            self.offload_buffers[mini_batch_chunk][tag].release()
            self.offload_buffers[mini_batch_chunk][tag] = None
        del self.offload_buffers[mini_batch_chunk]


OFFLOAD_BUFFER_MANAGER = OffloadBufferManager()


def default_offload_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    OFFLOAD_BUFFER_MANAGER.wait_offload_done(node.mini_batch, node.chunk)

def default_reload_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    OFFLOAD_BUFFER_MANAGER.reload_start(node.mini_batch, node.chunk)

def offload_preprocess(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    ...
    # overlap offload and the compute node
    # if node.func_type in (FuncType.F, FuncType.B, FuncType.W, FuncType.BW):
    #     mini_batch, chunk = OFFLOAD_BUFFER_MANAGER.try_pop_minibatch_chunk_from_cache()
    #     if mini_batch is not None:
    #         assert chunk is not None
    #         OFFLOAD_BUFFER_MANAGER.offload_start(mini_batch, chunk)
