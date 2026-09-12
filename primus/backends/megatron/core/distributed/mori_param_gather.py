###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MORI async SDMA parameter all-gather support for Megatron ZeRO-1."""

import atexit
import os
import warnings
from typing import Callable, Optional, Sequence, Tuple

import torch

_MIB = 1024 * 1024
_mori_runtime = None
_mori_init_error: Optional[Exception] = None
_mori_debug_messages: set[str] = set()


def _debug_mori_path(message: str) -> None:
    if os.getenv("MEGATRON_MORI_DEBUG", "0") == "1" and message not in _mori_debug_messages:
        _mori_debug_messages.add(message)
        warnings.warn(f"MORI param all-gather: {message}")


def _strict_mori() -> bool:
    return os.getenv("MEGATRON_MORI_STRICT", "0") == "1"


class _WaitableHandle:
    """Small ``torch.distributed.Work``-compatible wait handle."""

    def __init__(self, wait_fn: Optional[Callable[[], None]] = None, work=None):
        self._wait_fn = wait_fn
        self._work = work
        self._done = False

    def wait(self):
        if self._done:
            return True
        if self._work is not None:
            self._work.wait()
        if self._wait_fn is not None:
            self._wait_fn()
        self._done = True
        return True


def _rccl_all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup],
    async_op: bool,
):
    work = torch.distributed.all_gather_into_tensor(
        output_tensor, input_tensor, group=group, async_op=async_op
    )
    return _WaitableHandle(work=work) if work is not None else _WaitableHandle()


class _MoriRuntime:
    """One lifetime-stable MORI communicator for the world-sized DP group."""

    def __init__(self):
        import mori.ccl as ccl
        import mori.shmem as shmem

        if os.getenv("MORI_ENABLE_SDMA", "0") != "1":
            raise RuntimeError("MORI_ENABLE_SDMA=1 must be set before MORI initialization")
        if "MORI_SHMEM_HEAP_SIZE" not in os.environ:
            raise RuntimeError(
                "MORI_SHMEM_HEAP_SIZE must be set explicitly; it must hold all "
                "ZeRO-1 parameter buffers plus the MORI input transit buffer"
            )

        self.shmem = shmem
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        bootstrap_group = torch.distributed.new_group(ranks=list(range(self.world_size)), backend="gloo")
        self.bootstrap_group = bootstrap_group
        bootstrap_name = "primus_mori_world"
        torch._C._distributed_c10d._register_process_group(bootstrap_name, bootstrap_group)
        shmem.shmem_torch_process_group_init(bootstrap_name)

        max_input_bytes = int(os.getenv("MEGATRON_MORI_MAX_INPUT_BYTES", str(288 * _MIB)))
        self.handle = ccl.AllgatherSdma(
            shmem.shmem_mype(),
            shmem.shmem_npes(),
            input_buffer_size=max_input_bytes,
            output_buffer_size=4,
            copy_output_to_user=False,
        )
        self.max_input_bytes = max_input_bytes
        self.comm_stream = torch.cuda.Stream()
        self._registered_buffers: dict[int, torch.Tensor] = {}
        atexit.register(finalize_mori_runtime)

    def supports_group(self, group) -> bool:
        return (
            torch.distributed.get_world_size(group=group) == self.world_size
            and torch.distributed.get_rank(group=group) == self.rank
        )

    def allocate_param_buffer(self, shape, dtype) -> torch.Tensor:
        return self.shmem.mori_shmem_create_tensor(shape, dtype)

    def register_param_buffer(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        if ptr in self._registered_buffers:
            return
        self.handle.register_output_buffer(tensor)
        if not self.handle.is_output_registered(tensor):
            raise RuntimeError("MORI did not retain ZeRO-1 parameter-buffer registration")
        self._registered_buffers[ptr] = tensor
        _debug_mori_path(f"registered uncached param_data ptr=0x{ptr:x}, bytes={tensor.nbytes}")

    def start(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor) -> None:
        if input_tensor.nbytes > self.max_input_bytes:
            raise RuntimeError(
                f"MORI input shard is {input_tensor.nbytes} bytes, larger than "
                f"MEGATRON_MORI_MAX_INPUT_BYTES={self.max_input_bytes}"
            )
        if not self.handle.is_output_registered(output_tensor):
            raise RuntimeError("MORI output is not inside a registered parameter buffer")
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            started = self.handle.start_async(
                input_tensor, output_tensor, input_tensor.numel(), self.comm_stream
            )
        if not started:
            raise RuntimeError("MORI start_async returned false")

    def wait(self) -> None:
        with torch.cuda.stream(self.comm_stream):
            self.handle.wait_async(self.comm_stream)
        torch.cuda.current_stream().wait_stream(self.comm_stream)


def initialize_mori_runtime() -> Optional[_MoriRuntime]:
    """Initialize MORI collectively once, with optional RCCL fallback."""
    global _mori_init_error, _mori_runtime
    if _mori_runtime is not None:
        return _mori_runtime
    if _mori_init_error is not None:
        if _strict_mori():
            raise RuntimeError("MORI initialization previously failed") from _mori_init_error
        return None
    try:
        _mori_runtime = _MoriRuntime()
        _debug_mori_path(f"initialized world communicator, max input={_mori_runtime.max_input_bytes} bytes")
    except Exception as exc:
        _mori_init_error = exc
        if _strict_mori():
            raise
        warnings.warn(f"MORI param all-gather disabled; RCCL fallback: {exc!r}")
    return _mori_runtime


def finalize_mori_runtime() -> None:
    """Deregister MORI outputs and finalize SHMEM collectively."""
    global _mori_runtime
    runtime = _mori_runtime
    if runtime is None:
        return
    torch.cuda.synchronize()
    for tensor in runtime._registered_buffers.values():
        runtime.handle.deregister_output_buffer(tensor)
    runtime._registered_buffers.clear()
    runtime.handle = None
    runtime.shmem.shmem_finalize()
    torch.distributed.destroy_process_group(runtime.bootstrap_group)
    _mori_runtime = None


def remap_param_buffer_to_mori(buffer) -> bool:
    """Move a completed Megatron param buffer to MORI uncached storage."""
    runtime = initialize_mori_runtime()
    old_data = getattr(buffer, "param_data", None)
    if runtime is None or old_data is None or not buffer.ddp_config.use_distributed_optimizer:
        return False
    if not runtime.supports_group(buffer.data_parallel_group):
        message = "data-parallel group is not the rank-ordered default world group"
        if _strict_mori():
            raise RuntimeError(message)
        _debug_mori_path(f"RCCL fallback: {message}")
        return False
    if hasattr(buffer, "shared_buffer"):
        raise RuntimeError("MORI param remap does not support shared MXFP8 param/grad buffers")

    new_data = runtime.allocate_param_buffer(old_data.shape, old_data.dtype)
    new_data.copy_(old_data)

    old_begin = old_data.data_ptr()
    old_end = old_begin + old_data.nbytes
    for param, (start, end, _bucket_id) in buffer.param_index_map.items():
        param_ptr = param.data.data_ptr()
        if not old_begin <= param_ptr < old_end:
            continue
        param.data = new_data[start:end].view(param.data.shape)

    for bucket, (start, end) in zip(buffer.buckets, buffer.bucket_indices):
        if bucket.param_data is not None:
            bucket.param_data = new_data[start:end]

    buffer.param_data = new_data
    buffer._primus_mori_param_data_owner = new_data
    runtime.register_param_buffer(new_data)
    del old_data
    torch.cuda.empty_cache()
    _debug_mori_path(f"remapped ZeRO-1 param_data to {new_data.nbytes} uncached bytes")
    return True


class _MoriBatchHandle(_WaitableHandle):
    """Serialize a bucket group's operations through one MORI communicator."""

    def __init__(self, runtime: _MoriRuntime, operations: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
        self._runtime = runtime
        self._operations = list(operations)
        self._next = 0
        super().__init__(wait_fn=self._wait_all)
        if self._operations:
            self._start_next()

    def _start_next(self) -> None:
        output_tensor, input_tensor = self._operations[self._next]
        self._runtime.start(output_tensor, input_tensor)
        self._next += 1

    def _wait_all(self) -> None:
        while self._next:
            self._runtime.wait()
            if self._next >= len(self._operations):
                break
            self._start_next()


def start_mori_all_gathers(
    operations: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    group: torch.distributed.ProcessGroup,
    async_op: bool,
):
    """Start one or more all-gathers, falling back as a complete batch."""
    runtime = initialize_mori_runtime()
    if runtime is None or not runtime.supports_group(group):
        handles = [
            _rccl_all_gather(output, input_tensor, group, async_op) for output, input_tensor in operations
        ]
        return _WaitableHandle(wait_fn=lambda: [handle.wait() for handle in handles])

    _debug_mori_path("using async SDMA path for rank-ordered world DP group")
    handle = _MoriBatchHandle(runtime, operations)
    if not async_op:
        handle.wait()
    return handle
