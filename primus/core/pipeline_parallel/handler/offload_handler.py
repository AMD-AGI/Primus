###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

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


class CPUBufferPool:
    def __init__(self):
        self.pool = {}

    def get_cpu_buffer(self, key):
        if key not in self.pool or len(self.pool[key]) == 0:
            return None
        return self.pool[key].pop(0)

    def insert_cpu_buffer(self, key, cpu_buffer: torch.Tensor):
        assert cpu_buffer.device.type == "cpu", f"cpu_buffer must be on cpu. {cpu_buffer.device} != cpu"
        if key not in self.pool:
            self.pool[key] = []
        self.pool[key].append(cpu_buffer)


class OffloadBuffer:

    def __init__(self):
        # key -> cpu_buffer
        self.cpu_buffers = {}

        # key -> gpu_tensor
        self.gpu_tensors = {}

        self.key_list = {}

        # key -> cuda_event
        self.offload_events = {}
        self.reload_events = {}

        self.current_mini_batch = None
        self.current_chunk = None

        self.offload_stream = torch.cuda.Stream()
        self.use_pinned = os.environ.get("USE_PINNED_OFFLOAD", "0") != "0"

        self.record_offload_memory_info = os.environ.get("RECORD_OFFLOAD_MEMORY_INFO", "0") != "0"

        self.cpu_buffer_pool = CPUBufferPool()

        self.memory_info = {}
        self.cur_offloaded_memory = 0
        self.max_offloaded_memory = 0
        self.key_total_bytes = None

    def _record_memory_info(self, offloaded_memory, time_step: str):
        if not self.record_offload_memory_info:
            return

        offloaded_memory_delta = offloaded_memory
        self.cur_offloaded_memory += offloaded_memory_delta
        self.max_offloaded_memory = max(self.max_offloaded_memory, self.cur_offloaded_memory)
        self.memory_info[time_step] = (
            self.cur_offloaded_memory / (1024 * 1024 * 1024),
            offloaded_memory_delta / (1024 * 1024 * 1024),
        )

    def print_offload_memory_info(self, file):
        if not self.record_offload_memory_info:
            return
        self.record_offload_memory_info = False
        file.write("==================== Offload Memory Info dump ====================\n")
        for time_step, memory_info in self.memory_info.items():

            file.write(f"{time_step} {memory_info[0]} {memory_info[1]}\n")
        file.write("==================== Offload Memory Info dump ====================\n")
        file.write(f"Max offloaded memory: {self.max_offloaded_memory / (1024 * 1024 * 1024)} GB\n")
        if self.key_total_bytes is not None:
            for key in self.key_total_bytes:
                file.write(
                    f"{key}: {self.key_total_bytes[key] / (1024 * 1024 * 1024)} GB\t{self.key_total_bytes[key] / self.key_total_bytes['total'] * 100:.2f}%\n"
                )
        file.flush()

    def set_current_mini_batch_and_chunk(self, mini_batch: Optional[int], chunk: Optional[int]):
        self.current_mini_batch = mini_batch
        self.current_chunk = chunk

    def add_offload_tensor(self, key: str, tensor: torch.Tensor):
        if self.current_mini_batch is None or self.current_chunk is None:
            return

        if tensor._base is not None:
            # print(f"{key} base is not None")
            return
        if self.current_mini_batch not in self.key_list:
            self.key_list[self.current_mini_batch] = {}
            self.gpu_tensors[self.current_mini_batch] = {}
            self.cpu_buffers[self.current_mini_batch] = {}
        if self.current_chunk not in self.key_list[self.current_mini_batch]:
            self.key_list[self.current_mini_batch][self.current_chunk] = set()
            self.gpu_tensors[self.current_mini_batch][self.current_chunk] = {}  # key -> tensor_list
            self.cpu_buffers[self.current_mini_batch][self.current_chunk] = {}  # key -> cpu_buffer_list

        if key not in self.gpu_tensors[self.current_mini_batch][self.current_chunk]:
            self.gpu_tensors[self.current_mini_batch][self.current_chunk][key] = []
            self.cpu_buffers[self.current_mini_batch][self.current_chunk][key] = []

        self.key_list[self.current_mini_batch][self.current_chunk].add(key)

        self.gpu_tensors[self.current_mini_batch][self.current_chunk][key].append(tensor)

    def async_offload(self, mini_batch: int, chunk: int):
        if mini_batch not in self.key_list or chunk not in self.key_list[mini_batch]:
            return

        self.offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.offload_stream):
            for key in self.key_list[mini_batch][chunk]:
                gpu_tensor_list = self.gpu_tensors[mini_batch][chunk][key]

                assert (
                    len(self.cpu_buffers[mini_batch][chunk][key]) == 0
                ), f"cpu_buffers[{mini_batch}][{chunk}][{key}] must be empty. {len(self.cpu_buffers[mini_batch][chunk][key])} != 0"

                for gpu_tensor in gpu_tensor_list:
                    cpu_buffer = self.cpu_buffer_pool.get_cpu_buffer(key)
                    if cpu_buffer is None:
                        cpu_buffer = torch.empty_like(
                            gpu_tensor.data,
                            device="cpu",
                            layout=gpu_tensor.data.layout,
                            requires_grad=False,
                            pin_memory=self.use_pinned,
                        )

                    self.cpu_buffers[mini_batch][chunk][key].append(cpu_buffer)
                    cpu_buffer.copy_(gpu_tensor.data, non_blocking=self.use_pinned)

            event = torch.cuda.Event()
            event.record(self.offload_stream)
            self.offload_events[f"{mini_batch}-{chunk}"] = event

    def wait_offload_done(self, mini_batch: int, chunk: int):
        if f"{mini_batch}-{chunk}" not in self.offload_events:
            return
        torch.cuda.current_stream().wait_event(self.offload_events[f"{mini_batch}-{chunk}"])
        del self.offload_events[f"{mini_batch}-{chunk}"]

        cur_stream = torch.cuda.current_stream()
        offload_total_bytes = 0
        record_key_total_bytes = False
        if self.key_total_bytes is None:
            self.key_total_bytes = dict()
            record_key_total_bytes = True

        for key in self.key_list[mini_batch][chunk]:
            if record_key_total_bytes:
                self.key_total_bytes[key] = 0

            for gpu_tensor in self.gpu_tensors[mini_batch][chunk][key]:
                gpu_tensor.record_stream(cur_stream)
                offload_total_bytes += gpu_tensor.nbytes
                if record_key_total_bytes:
                    self.key_total_bytes[key] += gpu_tensor.nbytes
                deallocate_gpu_tensor(gpu_tensor)
        if record_key_total_bytes:
            self.key_total_bytes["total"] = offload_total_bytes

        self._record_memory_info(offload_total_bytes, f"{mini_batch}-{chunk}-offload_done")

    def reload_start(self, mini_batch: int, chunk: int):
        if mini_batch not in self.key_list or chunk not in self.key_list[mini_batch]:
            return

        with torch.cuda.stream(self.offload_stream):
            for key in self.key_list[mini_batch][chunk]:
                cpu_buffer_list = self.cpu_buffers[mini_batch][chunk][key]
                assert len(cpu_buffer_list) == len(
                    self.gpu_tensors[mini_batch][chunk][key]
                ), f"cpu_buffer_list and gpu_tensors_list must have the same length. {mini_batch}-{chunk}-{key} {len(cpu_buffer_list)} != {len(self.gpu_tensors[mini_batch][chunk][key])}"
                for i, cpu_buffer in enumerate(cpu_buffer_list):

                    device = self.gpu_tensors[mini_batch][chunk][key][i].device
                    self.gpu_tensors[mini_batch][chunk][key][i].data = torch.empty(
                        cpu_buffer.shape, device=device, dtype=cpu_buffer.dtype
                    )
                    self.gpu_tensors[mini_batch][chunk][key][i].data.copy_(
                        cpu_buffer, non_blocking=self.use_pinned
                    )

            event = torch.cuda.Event()
            event.record(self.offload_stream)
            self.reload_events[f"{mini_batch}-{chunk}"] = event

    def wait_reload_done(self, mini_batch: int, chunk: int):
        if f"{mini_batch}-{chunk}" not in self.reload_events:
            return
        torch.cuda.current_stream().wait_event(self.reload_events[f"{mini_batch}-{chunk}"])
        del self.reload_events[f"{mini_batch}-{chunk}"]

        total_reload_bytes = 0
        for key in self.key_list[mini_batch][chunk]:
            for cpu_buffer in self.cpu_buffers[mini_batch][chunk][key]:
                self.cpu_buffer_pool.insert_cpu_buffer(key, cpu_buffer)
                total_reload_bytes += cpu_buffer.nbytes

            for i in range(len(self.gpu_tensors[mini_batch][chunk][key])):
                self.gpu_tensors[mini_batch][chunk][key][i] = None
                self.cpu_buffers[mini_batch][chunk][key][i] = None

        self._record_memory_info(-1 * total_reload_bytes, f"{mini_batch}-{chunk}-reload_done")

        del self.cpu_buffers[mini_batch][chunk]
        del self.gpu_tensors[mini_batch][chunk]

        del self.key_list[mini_batch][chunk]

    def check_empty(self):
        for mini_batch in self.gpu_tensors:
            assert (
                len(self.gpu_tensors[mini_batch]) == 0
            ), f"gpu_tensors[{mini_batch}] must be empty. {len(self.gpu_tensors[mini_batch])} != 0"

        for mini_batch in self.key_list:
            assert (
                len(self.key_list[mini_batch]) == 0
            ), f"key_list[{mini_batch}] must be empty. {len(self.key_list[mini_batch])} != 0"

        for mini_batch in self.cpu_buffers:
            assert (
                len(self.cpu_buffers[mini_batch]) == 0
            ), f"cpu_buffers[{mini_batch}] must be empty. {len(self.cpu_buffers[mini_batch])} != 0"


OFFLOAD_BUFFER = OffloadBuffer()


def default_offload_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    OFFLOAD_BUFFER.wait_offload_done(node.mini_batch, node.chunk)


def default_reload_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    OFFLOAD_BUFFER.reload_start(node.mini_batch, node.chunk)
