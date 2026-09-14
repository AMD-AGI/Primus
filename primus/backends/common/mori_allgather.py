###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MORI-backed FSDP2 all-gather adapter.

The implementation mirrors the public FSDP2 adapter shape from
ROCm/mori's ``examples/fsdp_sdma/mori_allgather.py`` while keeping the
Primus backend patches small. It is intentionally all-gather only:
FSDP reduce-scatter stays on the framework default path.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from primus.core.utils.module_utils import log_rank_0

try:
    from torch.distributed.fsdp._fully_shard._fsdp_api import (
        AllGather as _FSDPAllGather,
    )
except Exception as e:  # pragma: no cover - depends on torch internal version
    _FSDPAllGather = object
    _FSDP_ALL_GATHER_IMPORT_ERROR = e
else:
    _FSDP_ALL_GATHER_IMPORT_ERROR = None

_MORI_SHMEM_INITIALIZED = False
_MIB = 1 << 20
_GIB = 1 << 30
_DEFAULT_SLICE_MIN_BYTES = 8 * _MIB


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _env_nonnegative_int(name: str, default: int = 0) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _compact_workspace_sizes(
    cap_bytes: int,
    world_size: int,
    ranks_per_node: int,
    slice_min_bytes: int,
) -> tuple[int, int]:
    """Return MORI's input/output capacities for its compact direct path.

    Large messages use MORI's sliced path: the inter-node ring holds one shard
    per node, and the intra-node phase writes directly to the FSDP output.
    Messages below ``slice_min_bytes`` may use the non-sliced fallback, whose
    full-world output must also fit. The pinned MORI implementation uses the
    larger capacity for both fused transits, so this is conservative while
    avoiding full-layer allocations.
    """
    if cap_bytes <= 0:
        raise ValueError(f"cap_bytes must be positive, got {cap_bytes}")
    if ranks_per_node <= 0 or world_size % ranks_per_node != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ranks_per_node ({ranks_per_node})"
        )
    if slice_min_bytes < 0:
        raise ValueError(f"slice_min_bytes must be non-negative, got {slice_min_bytes}")

    num_nodes = world_size // ranks_per_node
    fallback_per_rank = min(cap_bytes, slice_min_bytes)
    sliced_ring_bytes = num_nodes * cap_bytes
    fallback_output_bytes = world_size * fallback_per_rank
    workspace_bytes = max(sliced_ring_bytes, fallback_output_bytes)
    return fallback_per_rank, workspace_bytes


def _auto_shmem_heap_bytes(
    input_buffer_size: int,
    output_buffer_size: int,
    world_size: int,
    ranks_per_node: int,
    *,
    host_proxy: bool = False,
    host_proxy_sdma: bool = False,
) -> int:
    """Size MORI's static heap from the collective workspaces it will own."""
    num_nodes = world_size // ranks_per_node
    if host_proxy:
        data_bytes = output_buffer_size + output_buffer_size // max(num_nodes, 1) if host_proxy_sdma else 0
    elif num_nodes >= 2:
        intra_bytes = max(ranks_per_node * input_buffer_size, output_buffer_size)
        data_bytes = intra_bytes + output_buffer_size
    else:
        data_bytes = input_buffer_size + output_buffer_size

    margin_bytes = max(512 * _MIB, data_bytes // 4)
    required_bytes = data_bytes + margin_bytes
    rounded_bytes = ((required_bytes + _GIB - 1) // _GIB) * _GIB
    return max(2 * _GIB, rounded_bytes)


def _safe_log_rank_0(message: str) -> None:
    """Log through Primus when initialized; otherwise fall back to print."""
    try:
        log_rank_0(message)
    except Exception:
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            print(message, flush=True)


def mori_all_gather_enabled() -> bool:
    """Return whether Primus should install the MORI FSDP all-gather backend."""
    return os.environ.get("FSDP_ALL_GATHER_BACKEND", "") == "mori"


def ensure_mori_shmem_initialized(pg_name: str = "default") -> None:
    """Initialize MORI SHMEM from torchrun's c10d store once.

    MORI's convenience process-group initializer broadcasts its unique ID via
    the process-group backend. For NCCL groups that eagerly creates a second
    cross-node transport before MORI is ready. Use the rendezvous TCPStore
    instead so MORI remains the only RDMA data path.
    """
    global _MORI_SHMEM_INITIALIZED

    if _MORI_SHMEM_INITIALIZED:
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("MORI FSDP all-gather requires torch.distributed to be initialized")

    shmem = importlib.import_module("mori.shmem")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    store = dist.distributed_c10d._get_default_store()
    uid_key = f"primus_mori_shmem_uid_{pg_name}_{world_size}"
    if rank == 0:
        store.set(uid_key, shmem.shmem_get_unique_id())
    uid = store.get(uid_key)
    shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
        rank,
        world_size,
        uid,
    )

    if shmem.shmem_mype() != rank or shmem.shmem_npes() != world_size:
        raise RuntimeError(
            "MORI SHMEM PE mapping must match the FSDP process group: "
            f"rank/world_size={rank}/{world_size}, "
            f"mype/npes={shmem.shmem_mype()}/{shmem.shmem_npes()}"
        )

    _MORI_SHMEM_INITIALIZED = True
    _safe_log_rank_0("[MORI:FSDP] initialized MORI SHMEM from torch process group")


class _CudaEventWork:
    """Small Work-like object for async FSDP all-gather calls."""

    def __init__(self, event: torch.cuda.Event, device: torch.device) -> None:
        self._event = event
        self._device = device
        self._waited = False

    def wait(self) -> bool:
        if not self._waited:
            torch.cuda.current_stream(self._device).wait_event(self._event)
            self._waited = True
        return True


class _DeviceDeferredEventWork(dist.distributed_c10d.Work):
    """Insert a device-side dependency when FSDP consumes the MORI result."""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        event: torch.cuda.Event | None = None,
    ) -> None:
        super().__init__()
        self._stream = stream
        self._device = device
        self._event = event
        self._done = False

    def wait(self, timeout=None) -> bool:  # noqa: ARG002
        if not self._done:
            consumer_stream = torch.cuda.current_stream(self._device)
            if self._event is not None:
                consumer_stream.wait_event(self._event)
            else:
                consumer_stream.wait_stream(self._stream)
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done


class _HostProxyDeferredWork(dist.distributed_c10d.Work):
    """Defer host-proxy completion to FSDP's wait/copy-out point."""

    def __init__(self, collective: Any, handle: Any, drain: bool = False) -> None:
        super().__init__()
        self._collective = collective
        self._handle = handle
        self._drain = drain
        self._done = False

    def wait(self, timeout=None) -> bool:  # noqa: ARG002
        if not self._done:
            self._collective._complete(self._handle)
            if self._drain:
                self._handle["stream"].synchronize()
            self._collective._pending = None
            self._done = True
        return True

    def is_completed(self) -> bool:
        return self._done


class MoriAllGather(_FSDPAllGather):
    """FSDP2 custom all-gather backed by ``mori.ccl.HierAllGather``."""

    def __init__(self, ranks_per_node: int | None = None) -> None:
        if _FSDP_ALL_GATHER_IMPORT_ERROR is not None:
            raise ImportError(
                "MORI FSDP all-gather requires PyTorch FSDP2's internal " "AllGather API"
            ) from _FSDP_ALL_GATHER_IMPORT_ERROR

        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
        os.environ.setdefault("MORI_HIER_CUDA_GRAPH", "0")
        if "MORI_SOCKET_IFNAME" not in os.environ and "NCCL_SOCKET_IFNAME" in os.environ:
            os.environ["MORI_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"].lstrip("=")

        self._ranks_per_node = ranks_per_node
        self._collective: Any | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
        self._cap_bytes = 0
        self._observed_max_shard_bytes = 0
        self._output_buffer: torch.Tensor | None = None

        world = int(os.environ.get("WORLD_SIZE", "0") or "0")
        if world > 0:
            rpn = self._ranks_per_node_value(world)
            num_nodes = world // rpn if rpn else 1
            if num_nodes >= 2:
                # These defaults are copied from MORI's FSDP example and only
                # apply if the user did not explicitly tune the same variables.
                setdefault = os.environ.setdefault
                setdefault("MORI_HIER_FUSE_LOCAL", "1")
                setdefault("MORI_HIER_FUSE_REMOTE", "1")
                setdefault("MORI_HIER_LOCAL_PUSHONLY", "1")
                if rpn < 8:
                    setdefault("MORI_HIER_DEEP_PIPE", "auto")
                    setdefault("MORI_SDMA_NUM_CHANNELS", "8")
                else:
                    setdefault("MORI_HIER_DEBUG_SYNC", "0")
                    setdefault("MORI_HIER_CUDA_GRAPH", "0")
                    setdefault("MORI_FSDP_DEFER_HOSTSYNC", "1")
                    setdefault("MORI_FSDP_EVENT_FENCE", "1")
                    setdefault("MORI_FSDP_FWD_PREFETCH", "1")

        self._host_proxy = os.environ.get("MORI_FSDP_HOST_PROXY", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        self._hostproxy_async = os.environ.get("MORI_HOSTPROXY_ASYNC", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        if self._hostproxy_async:
            os.environ.setdefault("MORI_HOSTPROXY_ASYNC_DRAIN", "1")
            os.environ.setdefault("MORI_HOSTPROXY_ASYNC_RING", "2")
        self._hostproxy_async_drain = os.environ.get("MORI_HOSTPROXY_ASYNC_DRAIN", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        self._defer_hostsync = os.environ.get("MORI_FSDP_DEFER_HOSTSYNC", "") not in (
            "",
            "0",
            "false",
            "False",
        )
        self._event_fence = os.environ.get("MORI_FSDP_EVENT_FENCE", "") not in (
            "",
            "0",
            "false",
            "False",
        )

    def observe_fsdp_param_group(self, param_group: Any) -> int:
        """Record a conservative all-gather shard size before training starts."""
        param_dtype = getattr(getattr(param_group, "mp_policy", None), "param_dtype", None)
        shard_bytes = 0
        for fsdp_param in getattr(param_group, "fsdp_params", ()):
            tensor = getattr(fsdp_param, "_sharded_param_data", None)
            if tensor is None:
                continue
            dtype = tensor.dtype
            if param_dtype is not None and dtype.is_floating_point:
                dtype = param_dtype
            shard_bytes += tensor.numel() * torch.empty((), dtype=dtype).element_size()

        if shard_bytes > 0:
            shard_bytes = ((shard_bytes + _MIB - 1) // _MIB) * _MIB
            self._observed_max_shard_bytes = max(self._observed_max_shard_bytes, shard_bytes)
        return shard_bytes

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        numel = 1
        for dim in size:
            numel *= int(dim)
        if (
            self._output_buffer is not None
            and self._output_buffer.dtype == dtype
            and self._output_buffer.device == device
            and self._output_buffer.numel() >= numel
        ):
            return self._output_buffer.narrow(0, 0, numel)
        self._output_buffer = torch.empty(numel, dtype=dtype, device=device)
        return self._output_buffer

    def _registration_output(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """Use the persistent backing extent for device-path IPC registration."""
        backing = self._output_buffer
        if (
            self._host_proxy
            or backing is None
            or backing.dtype != output_tensor.dtype
            or backing.device != output_tensor.device
            or backing.data_ptr() != output_tensor.data_ptr()
            or backing.numel() < output_tensor.numel()
        ):
            return output_tensor
        return backing

    def _ranks_per_node_value(self, world_size: int) -> int:
        if self._ranks_per_node is not None:
            return self._ranks_per_node
        env_value = os.environ.get("LOCAL_WORLD_SIZE")
        if env_value:
            return int(env_value)
        return min(torch.cuda.device_count(), world_size)

    def _get_collective(self, group: dist.ProcessGroup, per_rank_bytes: int) -> Any:
        rank, world_size = group.rank(), group.size()
        cap_floor = self._observed_max_shard_bytes
        if self._host_proxy:
            hostproxy_floor = _env_nonnegative_int("MORI_FSDP_HOSTPROXY_CAP_MB", 160) * _MIB
            cap_floor = max(cap_floor, hostproxy_floor)
        required_cap = max(per_rank_bytes, cap_floor)
        if (
            self._collective is not None
            and self._rank == rank
            and self._world_size == world_size
            and self._cap_bytes >= required_cap
        ):
            return self._collective

        cap = max(required_cap, self._cap_bytes)
        ranks_per_node = self._ranks_per_node_value(world_size)
        num_nodes = world_size // ranks_per_node
        if self._host_proxy:
            input_buffer_size = cap
            output_buffer_size = cap * world_size
            slice_min_bytes = _DEFAULT_SLICE_MIN_BYTES
            compact = False
        else:
            compact = num_nodes >= 2 and _env_flag("MORI_FSDP_COMPACT_WORKSPACE", True)
            if compact:
                slice_min_bytes = _env_nonnegative_int(
                    "MORI_FSDP_SLICE_MIN_MB", _DEFAULT_SLICE_MIN_BYTES // _MIB
                ) * _MIB
                input_buffer_size, output_buffer_size = _compact_workspace_sizes(
                    cap,
                    world_size,
                    ranks_per_node,
                    slice_min_bytes,
                )
            else:
                slice_min_bytes = _DEFAULT_SLICE_MIN_BYTES
                input_buffer_size = cap
                output_buffer_size = cap * world_size

        heap_bytes = _auto_shmem_heap_bytes(
            input_buffer_size,
            output_buffer_size,
            world_size,
            ranks_per_node,
            host_proxy=self._host_proxy,
            host_proxy_sdma=_env_flag("MORI_HOSTPROXY_SDMA_INTRA", False),
        )
        if "MORI_SHMEM_HEAP_SIZE" not in os.environ:
            os.environ["MORI_SHMEM_HEAP_SIZE"] = f"{heap_bytes // _GIB}G"
            _safe_log_rank_0(
                f"[MORI:FSDP] auto-sized MORI SHMEM heap to {heap_bytes // _GIB} GiB"
            )

        ensure_mori_shmem_initialized("default")

        shmem = importlib.import_module("mori.shmem")
        ccl = importlib.import_module("mori.ccl")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        if my_pe != rank or npes != world_size:
            raise RuntimeError(
                "MORI FSDP HierAllGather requires the FSDP process group to match "
                f"SHMEM PEs, got rank/world_size={rank}/{world_size} and "
                f"my_pe/npes={my_pe}/{npes}"
            )

        if self._host_proxy:
            if self._collective is not None:
                raise RuntimeError(
                    "HostProxyHierAllGather built with cap "
                    f"{self._cap_bytes} B but requires {required_cap} B; "
                    "raise MORI_FSDP_HOSTPROXY_CAP_MB"
                )
            collective = ccl.HostProxyHierAllGather(
                rank,
                world_size,
                ranks_per_node,
                output_buffer_size=output_buffer_size,
            )
        else:
            _safe_log_rank_0(
                "[MORI:FSDP] building HierAllGather "
                f"cap={cap} B input_workspace={input_buffer_size} B "
                f"output_workspace={output_buffer_size} B compact={compact}"
            )
            collective = ccl.HierAllGather(
                my_pe,
                npes,
                input_buffer_size=input_buffer_size,
                output_buffer_size=output_buffer_size,
                copy_output_to_user=True,
                ranks_per_node=ranks_per_node,
                slice_min_bytes=slice_min_bytes,
                slice_direct=True if compact else None,
            )

        self._collective = collective
        self._rank = rank
        self._world_size = world_size
        self._cap_bytes = cap
        return collective

    def _validate(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> None:
        if not input_tensor.is_cuda or not output_tensor.is_cuda:
            raise RuntimeError("MORI FSDP HierAllGather requires CUDA tensors")
        if input_tensor.device != output_tensor.device:
            raise RuntimeError("MORI FSDP HierAllGather requires tensors on the same device")
        if input_tensor.dtype != output_tensor.dtype:
            raise RuntimeError("MORI FSDP HierAllGather requires matching dtypes")
        expected = input_tensor.numel() * group.size()
        if output_tensor.numel() != expected:
            raise RuntimeError(
                f"MORI FSDP HierAllGather expected output numel {expected}, " f"got {output_tensor.numel()}"
            )
        if (input_tensor.numel() * input_tensor.element_size()) % 4 != 0:
            raise RuntimeError("MORI FSDP HierAllGather requires 4-byte-aligned input bytes")

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Any | None:
        self._validate(output_tensor, input_tensor, group)
        per_rank_bytes = input_tensor.numel() * input_tensor.element_size()
        collective = self._get_collective(group, per_rank_bytes)
        device = input_tensor.device
        stream = torch.cuda.current_stream(device)

        input_tensor.record_stream(stream)
        output_tensor.record_stream(stream)

        if self._host_proxy and self._hostproxy_async:
            pending = getattr(collective, "_pending", None)
            if pending is not None:
                pending.wait()
            handle = collective.call_async(input_tensor, output_tensor, input_tensor.numel(), stream=stream)
            if handle is None:
                return None
            work = _HostProxyDeferredWork(collective, handle, drain=self._hostproxy_async_drain)
            collective._pending = work
            return work

        registration_output = self._registration_output(output_tensor)
        if registration_output is not output_tensor:
            registration_output.record_stream(stream)
        ok = collective(input_tensor, registration_output, input_tensor.numel(), stream=stream)
        if not ok:
            raise RuntimeError("MORI HierAllGather call failed")

        if self._defer_hostsync:
            event = None
            if self._event_fence:
                event = torch.cuda.Event()
                event.record(stream)
            return _DeviceDeferredEventWork(stream, device, event)

        if async_op:
            event = torch.cuda.Event()
            event.record(stream)
            return _CudaEventWork(event, device)
        return None
