###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus.core.pipeline_parallel.scheduler.scheduler_node import SchedulerNode


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

    def __init__(self):
        # key -> cpu_buffer
        self.cpu_buffers = {}

        # key -> gpu_tensor
        self.gpu_tensors = {}

        # catogary -> cpu_buffer_pool[]
        self.cpu_buffer_pool = {}

        # key -> cuda_event
        self.offload_events = {}
        self.reload_events = {}

        self.offload_stream = torch.cuda.Stream()

    def async_offload(self, tensor: torch.Tensor, key: str, catogary: str):

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            self.gpu_tensors[key] = tensor
            if catogary not in self.cpu_buffer_pool:
                self.cpu_buffer_pool[catogary] = []
            if len(self.cpu_buffer_pool[catogary]) == 0:
                cpu_tensor = torch.empty_like(
                    tensor.data, device="cpu", layout=tensor.data.layout, requires_grad=False
                )
            else:
                cpu_tensor = self.cpu_buffer_pool[catogary].pop(0)

            cpu_tensor.copy_(tensor.data, non_blocking=False)
            self.cpu_buffers[key] = cpu_tensor

            event = torch.cuda.Event()
            event.record(self.offload_stream)
            self.offload_events[key] = event

    def wait_offload_done(self, key):
        if key not in self.offload_events:
            return
        torch.cuda.current_stream().wait_event(self.offload_events[key])
        del self.offload_events[key]

        deallocate_gpu_tensor(self.gpu_tensors[key])

    def reload_start(self, key):
        if key not in self.cpu_buffers:
            return

        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            self.gpu_tensors[key].data = torch.empty(
                self.cpu_buffers[key].shape, device="cuda", dtype=self.cpu_buffers[key].dtype
            )
            self.gpu_tensors[key].data.copy_(self.cpu_buffers[key], non_blocking=False)

            event = torch.cuda.Event()
            event.record(self.offload_stream)
            self.reload_events[key] = event

    def wait_reload_done(self, key):
        if key not in self.reload_events:
            return
        assert key in self.reload_events
        torch.cuda.current_stream().wait_event(self.reload_events[key])
        del self.reload_events[key]

        del self.cpu_buffers[key]
        del self.gpu_tensors[key]


OFFLOAD_BUFFER = OffloadBuffer()


def default_offload_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    key = f"{node.mini_batch}-{node.chunk}-input_tensor"
    OFFLOAD_BUFFER.wait_offload_done(key)


def default_reload_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    key = f"{node.mini_batch}-{node.chunk}-input_tensor"
    OFFLOAD_BUFFER.reload_start(key)
