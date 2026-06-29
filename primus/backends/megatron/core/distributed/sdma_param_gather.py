###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SDMA (copy-engine) param all-gather helpers for the distributed optimizer.

Migrated from the source patch ``megatron_sdma_allgather.patch`` (mpo branch).

These helpers implement a ``torch.distributed.all_gather_into_tensor``-compatible
all-gather that routes the gather through Primus-Turbo symmetric memory + HIP
copy-engine (SDMA) memcpys instead of RCCL kernels. The intent is to free up
CU/compute resources by performing the param all-gather purely on the copy
engine, overlapping it with forward compute.

The implementation is consumed by
:mod:`primus.backends.megatron.patches.parallelism.sdma_param_all_gather_patches`,
which swaps Megatron's ``_ParamAndGradBucketGroup.start_param_sync`` distributed
optimizer path to dispatch per-bucket all-gathers through
:func:`all_gather_into_tensor_sdma`.

Activation:
    ``ENABLE_SDMA_ALLGATHER=1`` (gated by the patch). When the required
    Primus-Turbo / ``hip`` primitives are unavailable, every call falls back to
    ``torch.distributed.all_gather_into_tensor`` so behaviour is preserved.
"""

import importlib
import os
import warnings
from typing import Callable, Dict, Optional, Tuple

import torch

# Minimum symmetric-memory workspace size (bytes) requested per group. The
# workspace is reused across calls, so it is sized for the largest expected
# bucket. Overridable via env for A/B sizing.
_SDMA_SYMM_MEM_MIN_BYTES = int(os.getenv("MEGATRON_SDMA_SYMM_MEM_MIN_BYTES", str(749887296)))


class _WaitableHandle:
    """Lightweight waitable handle that mimics ``torch.distributed.Work``."""

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


def _all_gather_into_tensor_waitable_fallback(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup] = None,
    async_op: bool = False,
):
    """Fallback to ``torch.distributed.all_gather_into_tensor`` with a waitable handle."""
    work = torch.distributed.all_gather_into_tensor(
        output_tensor, input_tensor, group=group, async_op=async_op
    )
    if async_op and work is not None:
        return _WaitableHandle(work=work)
    return _WaitableHandle()


def _get_sdma_peer_copy_stream_count(world_size: int) -> int:
    """Resolve the number of peer-copy streams for SDMA param all-gather."""
    max_streams = max(world_size - 1, 1)
    default_streams = min(max_streams, 8)
    env_value = os.getenv("MEGATRON_SDMA_PEER_COPY_STREAMS")
    if env_value is None:
        return default_streams

    try:
        configured_streams = int(env_value)
    except ValueError:
        warnings.warn(
            f"Invalid MEGATRON_SDMA_PEER_COPY_STREAMS={env_value!r}; using default "
            f"{default_streams}."
        )
        return default_streams

    if configured_streams < 1:
        warnings.warn(
            f"MEGATRON_SDMA_PEER_COPY_STREAMS must be >= 1, got {configured_streams}; "
            f"using default {default_streams}."
        )
        return default_streams

    return min(configured_streams, max_streams)


class _SDMAGroupRuntime:
    """Shared SDMA runtime for a process group (comm + peer-copy streams)."""

    def __init__(self, world_size: int):
        # Barrier/publish/local-copy run on one shared communication stream.
        self.comm_stream = torch.cuda.Stream()
        # Limit peer-copy fan-out by default to reduce SDMA/memory-system
        # pressure and improve overlap with forward compute.
        peer_copy_stream_count = _get_sdma_peer_copy_stream_count(world_size)
        self.peer_copy_streams = [torch.cuda.Stream() for _ in range(peer_copy_stream_count)]


_sdma_group_runtime_cache: Dict[Tuple[str, int], _SDMAGroupRuntime] = {}


def _get_sdma_group_runtime(group_name: str, world_size: int) -> _SDMAGroupRuntime:
    cache_key = (group_name, world_size)
    runtime = _sdma_group_runtime_cache.get(cache_key)
    if runtime is None:
        runtime = _SDMAGroupRuntime(world_size=world_size)
        _sdma_group_runtime_cache[cache_key] = runtime
    return runtime


def all_gather_into_tensor_sdma(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup] = None,
    async_op: bool = False,
):
    """SDMA all-gather with ``all_gather_into_tensor``-compatible signature.

    Follows the Primus-Turbo async-tp style (symmetric memory + DMA copies) and
    always returns a ``.wait()``-able handle. Falls back to
    ``torch.distributed.all_gather_into_tensor`` when SDMA primitives are
    unavailable.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before all_gather_into_tensor_sdma")

    if group is None:
        group = torch.distributed.distributed_c10d._get_default_group()

    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)

    if output_tensor.numel() != input_tensor.numel() * world_size:
        raise ValueError(
            "output_tensor.numel() must equal input_tensor.numel() * world_size "
            "for all_gather_into_tensor"
        )
    if output_tensor.dtype != input_tensor.dtype:
        raise ValueError(
            "output_tensor.dtype must match input_tensor.dtype for SDMA all_gather_into_tensor"
        )

    # SDMA path is CUDA-only and expects a contiguous flattened layout.
    if (
        not output_tensor.is_cuda
        or not input_tensor.is_cuda
        or output_tensor.device != input_tensor.device
        or output_tensor.device.index != torch.cuda.current_device()
        or not output_tensor.is_contiguous()
    ):
        return _all_gather_into_tensor_waitable_fallback(
            output_tensor, input_tensor, group=group, async_op=async_op
        )
    assert input_tensor.is_contiguous(), "SDMA all_gather_into_tensor requires contiguous input_tensor"

    try:
        hip = importlib.import_module("hip").hip
        get_amd_symm_mem_workspace = importlib.import_module(
            "primus_turbo.pytorch.kernels.async_tp.amd_symmetric_memory"
        ).get_amd_symm_mem_workspace
        hip_check = importlib.import_module(
            "primus_turbo.pytorch.kernels.async_tp.common_ops"
        ).hip_check
    except Exception:
        return _all_gather_into_tensor_waitable_fallback(
            output_tensor, input_tensor, group=group, async_op=async_op
        )

    group_name = getattr(group, "group_name", None)
    if group_name is None:
        return _all_gather_into_tensor_waitable_fallback(
            output_tensor, input_tensor, group=group, async_op=async_op
        )

    input_nbytes = input_tensor.nbytes
    output_flat = output_tensor.view(-1)

    memcpy_comm_kind = getattr(
        hip.hipMemcpyKind, "hipMemcpyDeviceToDeviceNoCU", hip.hipMemcpyKind.hipMemcpyDeviceToDevice
    )

    # Allocate/reuse symmetric workspace and expose each rank's local shard buffer.
    symm_mem = get_amd_symm_mem_workspace(
        group_name, min_size=max(input_nbytes, _SDMA_SYMM_MEM_MIN_BYTES)
    )
    gather_buffers = [
        symm_mem.get_buffer(r, input_tensor.shape, input_tensor.dtype) for r in range(world_size)
    ]
    runtime = _get_sdma_group_runtime(group_name=group_name, world_size=world_size)

    current_stream = torch.cuda.current_stream()
    # Per-call barrier event so each handle tracks its own all-gather point.
    barrier_event = torch.cuda.Event()

    # Ensure input is ready on the communication stream.
    runtime.comm_stream.wait_stream(current_stream)

    with torch.cuda.stream(runtime.comm_stream):
        # Barrier (workspace safety), publish local shard, post-publish barrier.
        symm_mem.barrier()
        gather_buffers[rank].copy_(input_tensor, True)
        symm_mem.barrier()
        # Peer copies may start after the publish barrier completes.
        barrier_event.record(runtime.comm_stream)
        # Copy local shard into its slot in the output.
        hip_check(
            hip.hipMemcpyAsync(
                output_flat.data_ptr() + rank * input_nbytes,
                input_tensor.data_ptr(),
                input_nbytes,
                memcpy_comm_kind,
                runtime.comm_stream.cuda_stream,
            )
        )

    for peer_idx, src_rank in enumerate(r for r in range(world_size) if r != rank):
        src_buf = gather_buffers[src_rank]
        copy_stream = runtime.peer_copy_streams[peer_idx % len(runtime.peer_copy_streams)]
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(barrier_event)
            hip_check(
                hip.hipMemcpyAsync(
                    output_flat.data_ptr() + src_rank * input_nbytes,
                    src_buf.data_ptr(),
                    input_nbytes,
                    memcpy_comm_kind,
                    copy_stream.cuda_stream,
                )
            )

    def _wait_impl():
        wait_stream = torch.cuda.current_stream()
        wait_stream.wait_stream(runtime.comm_stream)
        for copy_stream in runtime.peer_copy_streams:
            wait_stream.wait_stream(copy_stream)

    handle = _WaitableHandle(wait_fn=_wait_impl)
    if not async_op:
        handle.wait()
    return handle
