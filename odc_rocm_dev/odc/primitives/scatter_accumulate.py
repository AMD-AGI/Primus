import logging
import math
import os
import queue as _queue
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from threading import Thread
from typing import List, Mapping, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing
import triton
import triton.language as tl

from odc.primitives import (
    SHMEM_EXTERN_LIBS,
    __syncthreads,
    get_ipc_handle,
    int_p,
    int_p_remote,
    int_wait_until_equals,
    int_wait_until_equals_remote,
    putmem_nbi_block,
    quiet,
    reconstruct_tensor,
    tid,
)
from odc.primitives.utils import (
    BufferSplitter,
    SymmBufferRegistry,
    get_comm_stream,
    get_local_world_size,
    sync_cta,
)

logger = logging.getLogger(__name__)

# rocSHMEM RO (host-driven) cross-node backend for reduce-scatter. When active,
# the cross-node scatter-accumulate handshake (push my segment to a remote
# peer's staging buffer, signal its watcher, wait for the ack) is driven from
# the CPU with blocking ``rs_putmem`` / ``rs_int_p`` / ``rs_getmem`` instead of
# the device-initiated MORI kernels. The same-node path (XGMI peer
# copy + on-device int_p/wait handshake) and the watcher subprocess are reused
# unchanged. The mori device kernel path is untouched.
from odc.primitives.utils import _USE_ROCSHMEM  # noqa: E402

if _USE_ROCSHMEM:
    from odc.primitives import _rocshmem_backend as _rs
else:
    _rs = None


def _ro_active():
    return _rs is not None and _rs.ro_enabled()


def _gda_active():
    return _rs is not None and _rs.gda_enabled()


def _single_device_reduce():
    """ODC_SINGLE_DEVICE_REDUCE=1 -> single-node, watcher-FREE reduce-scatter
    accumulate. This is the gated grey path that lets us evaluate dropping the
    ReductionWatcher subprocess + tensor_ipc (HIP GPU-IPC) on a single node.

    Mechanism (owner-side PULL + on-chip sum over XGMI peer views):
      * Each PE stages its (locally pre-accumulated) full grad into a symmetric
        buffer whose same-node peer views are already resolved via ``rs_ptr``
        (rocshmem) / ``mori`` -- the SAME XGMI peer-view machinery gather.py and
        the watcher push already use.
      * After a rendezvous barrier, PE r reads every same-node peer p's segment
        destined for r (a plain XGMI ``.copy_`` peer read) and sums it into its
        fp32 accumulator on-chip. There are NO cross-rank writes -> no device
        atomics -> no MI300X write-visibility/L2-staleness hazard; only XGMI
        reads gated by a barrier (identical guarantee to the gather pull).
      * No second process -> no ``get_ipc_handle`` / ``reconstruct_tensor`` and
        no ``ReductionWatcher``/``server_loop`` at all on this path.

    Deadlock safety: like the multi-node GDA path, the per-micro-batch call only
    pre-accumulates LOCALLY (no collective), and the ONE barriered reduce runs
    per-group at ``get_accumulation`` (barrier count == #param-groups, matched
    across ranks). So variable-length (nopad) micro-batch counts cannot mismatch
    a collective barrier -> no rendezvous deadlock.

    Default ("0") preserves the watcher + tensor_ipc path byte-for-byte; nothing
    is deleted. Only same-node reduce groups take this path (see the guard in
    ``scatter_accumulate``); multi-node GDA/RO is untouched.
    """
    return (
        os.environ.get("ODC_SINGLE_DEVICE_REDUCE", "0") == "1"
        and not _gda_active()
        and not _ro_active()
    )


def _official_push():
    """ODC_OFFICIAL_PUSH=1 -> faithfully reproduce the reference single-sided
    "push + fire-and-forget" scatter-accumulate, bypassing every ROCm-specific
    addition we layered on (NONE are deleted -- they are only skipped via
    ``if not _official_push()`` while this switch is on):

      * intra-node IPC path: keep the direct peer write (``peer_buf.copy_``) +
        the watcher request signal (owner-side accumulation), but DROP the
        per-call ack/settle kernel (``shmem_wait_accumulation_same_node_kernel``)
        and DROP the trailing ``wait_stream`` serialization so the push overlaps
        the backward compute (fire-and-forget). A single light minibatch-end
        ``sync()`` (watcher task-count rendezvous) guarantees arrival.
      * inter-node non-GDA path already IS the official single-sided push
        (``shmem_cross_node_scatter`` -> device ``putmem_nbi`` + periodic
        ``quiet``), so it is used unchanged.
      * GDA path: skip our ``_rs.barrier()`` per-call, strided/full warmup
        settle, overlap side-stream and DEFER block (collective reduce-scatter
        runs bare -- closest reproduction of official no-settle).

    Default ("0") preserves all current behaviour. Env-gated; nothing removed.
    """
    return os.environ.get("ODC_OFFICIAL_PUSH", "0") == "1"


def _settle_defer():
    """ODC_SETTLE_DEFER=1 -> move the per-layer settle (the
    ``shmem_wait_accumulation_same_node_kernel`` ack-wait) OFF the backward
    critical path to raise compute/settle overlap, WITHOUT going fully
    fire-and-forget (which deadlocks, see ``_official_push``).

    What it changes (intra-node same-node path only):
      * KEEP the per-layer push + per-layer ack/settle kernel (nothing deleted)
        -- they stay queued on the per-peer ``rank_streams`` whose serial order
        still guards staging-buffer reuse across layers.
      * SKIP the trailing per-layer ``current_stream().wait_stream(rank_streams)``
        so backward compute does not block on each layer's settle.
      * A SINGLE aggregate join in ``sync()`` (per minibatch, before the
        optimizer reads the grads) collects all side-stream settles -> grads
        are guaranteed landed before optimizer.step.
      * ``record_stream`` keeps the source grad alive until the side-stream push
        copy consumed it (cheap protection against source-buffer reuse).

    Default ("0") preserves all current behaviour byte-for-byte. Env-gated.
    """
    return os.environ.get("ODC_SETTLE_DEFER", "0") == "1"


def _settle_watchdog_sec():
    """Fail-fast timeout (seconds) for the minibatch-end watcher rendezvous when
    ODC_SETTLE_DEFER is on; 0 disables. Avoids a permanent GPU spin if the
    watcher task-count never converges."""
    try:
        return float(os.environ.get("ODC_SETTLE_WATCHDOG_SEC", "120"))
    except ValueError:
        return 120.0


MAX_REQUEST_COUNT = 2 * 100000


@triton.jit(do_not_specialize=[])
def shmem_scatter_kernel(
    input_tensor_ptr,
    rank_input_size,
    input_segment_start,
    chunk_buffer,
    output_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    num_ranks_per_node,
    world_size: tl.constexpr,
    chunk_size: tl.constexpr,
    signal_next_expected,
    signal_ptr,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    # np = tl.num_programs(axis=0)
    assert num_ranks_per_node == tl.num_programs(axis=0)
    np = num_ranks_per_node
    assert world_size % np == 0
    num_nodes = world_size // np
    expected = signal_next_expected
    chunk_buffer_seg = chunk_buffer + pid * chunk_size

    # Use different kernel for the ranks in the same node.
    for i in range(1, num_nodes):
        peer_node = (i + rank // np) % num_nodes
        peer = (pid + peer_node * np) % world_size

        num_chunks = tl.cdiv(elem_per_rank, chunk_size)
        for chunk in range(num_chunks):
            this_chunk_size = chunk_size
            if chunk == num_chunks - 1:
                this_chunk_size = elem_per_rank - chunk * chunk_size
            chunk_offsets = tl.arange(0, chunk_size)
            input_start = peer * rank_input_size + input_segment_start + (chunk * chunk_size)
            mask = chunk_offsets < this_chunk_size
            input_chunk_data = tl.load(input_tensor_ptr + input_start + chunk_offsets, mask=mask)
            tl.store(chunk_buffer_seg + chunk_offsets, input_chunk_data, mask=mask)
            # As we initialize the symmetric runtime on the global process
            # group, we need to use the global rank to access the peer tensor.
            putmem_nbi_block(
                output_tensor_ptr + (chunk * chunk_size),
                chunk_buffer_seg,
                this_chunk_size * size_per_elem,
                peer,
            )

            expected += np
            sync_cta(signal_ptr, expected)
            if tidx == 0 and pid == 0:
                quiet()
            __syncthreads()

            expected += np
            sync_cta(signal_ptr, expected)

    expected += np
    sync_cta(signal_ptr, expected)

    if pid == 0:
        if tidx == 0:
            quiet()
        __syncthreads()

    return expected


@triton.jit(do_not_specialize=["rank", "peer", "next_request_id"])
def shmem_cross_node_scatter(
    input_tensor_ptr,
    rank_input_size,
    chunk_buffer,
    trans_buffer,
    size_per_elem,
    rank,
    num_ranks_per_node,
    world_size: tl.constexpr,
    output_size,
    local_buf_size,
    chunk_size: tl.constexpr,
    signal_ptr,
    # client request
    request_buffer_ptr,
    response_buffer_ptr,
    response_scratch_ptr,
    rank_start_same_node,
    rank_end_same_node,
    accumulation_command,
    next_request_id,
):
    signal_next_expected = 0
    for start in range(0, output_size, local_buf_size):
        size = min(local_buf_size, output_size - start)
        signal_next_expected = shmem_scatter_kernel(
            input_tensor_ptr=input_tensor_ptr,
            rank_input_size=rank_input_size,
            input_segment_start=start,
            chunk_buffer=chunk_buffer,
            output_tensor_ptr=trans_buffer,
            elem_per_rank=size,
            size_per_elem=size_per_elem,
            rank=rank,
            num_ranks_per_node=num_ranks_per_node,
            world_size=world_size,
            chunk_size=chunk_size,
            signal_next_expected=signal_next_expected,
            signal_ptr=signal_ptr,
        )

        shmem_request_accumulation_remote_node_kernel(
            request_buffer_ptr=request_buffer_ptr,
            rank=rank,
            rank_start_same_node=rank_start_same_node,
            rank_end_same_node=rank_end_same_node,
            world_size=world_size,
            accumulation_command=accumulation_command,
        )
        shmem_wait_accumulation_remote_node_kernel(
            response_buffer_ptr=response_buffer_ptr,
            scratch_ptr=response_scratch_ptr,
            rank=rank,
            rank_start_same_node=rank_start_same_node,
            rank_end_same_node=rank_end_same_node,
            world_size=world_size,
            next_request_id=next_request_id,
        )
        next_request_id += 1

        # All CTAs need to wait for the first CTA to finish the accumulation request.
        if start + local_buf_size < output_size:
            signal_next_expected += num_ranks_per_node
            sync_cta(signal_ptr, signal_next_expected)


@triton.jit(do_not_specialize=["rank", "peer", "accumulation_command"])
def shmem_request_accumulation_same_node_kernel(
    request_buffer_ptr,
    rank,
    peer,
    accumulation_command,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0:
        if tidx == 0:
            int_p(request_buffer_ptr + rank, accumulation_command, peer)
        __syncthreads()


@triton.jit(do_not_specialize=["rank", "peer", "next_request_id"])
def shmem_wait_accumulation_same_node_kernel(
    response_buffer_ptr,
    rank,
    peer,
    next_request_id,
):
    # ROCm fix: a naive loop `while r != next_request_id: quiet(); int_g(...)`
    # hangs on MI300X because a repeated in-kernel volatile load of a peer
    # address never observes the watcher's ack (GPU L2 stale). int_wait_until_equals
    # uses a system-scope atomic load that bypasses the L2.
    pid = tl.program_id(axis=0)
    if pid == 0:
        int_wait_until_equals(response_buffer_ptr + rank, next_request_id, peer)
        __syncthreads()


@triton.jit(
    do_not_specialize=[
        "rank",
        "rank_start_same_node",
        "rank_end_same_node",
        "world_size",
        "accumulation_command",
    ]
)
def shmem_request_accumulation_remote_node_kernel(
    request_buffer_ptr,
    rank,
    rank_start_same_node,
    rank_end_same_node,
    world_size,
    accumulation_command,
):
    pid = tl.program_id(axis=0)
    tidx = tid(axis=0)
    if pid == 0:
        if tidx == 0:
            for peer in range(world_size):
                if peer < rank_start_same_node or peer >= rank_end_same_node:
                    # cross-node peer -> RDMA single-int put (int_p_remote).
                    # This kernel is only ever compiled on multi-node runs, so
                    # the RDMA device symbol never enters a single-node kernel.
                    int_p_remote(request_buffer_ptr + rank, accumulation_command, peer)
        __syncthreads()


@triton.jit(
    do_not_specialize=[
        "rank",
        "rank_start_same_node",
        "rank_end_same_node",
        "world_size",
        "next_request_id",
    ]
)
def shmem_wait_accumulation_remote_node_kernel(
    response_buffer_ptr,
    scratch_ptr,
    rank,
    rank_start_same_node,
    rank_end_same_node,
    world_size,
    next_request_id,
):
    # Cross-node ack wait. Unlike the same-node kernel (which polls the peer's
    # XGMI-mapped response slot), a remote-node peer has no local alias for its
    # response buffer, so we actively pull the slot over RDMA with
    # ``int_wait_until_equals_remote`` (getmem-spin into a local symmetric
    # scratch).
    pid = tl.program_id(axis=0)
    if pid == 0:
        for peer in range(world_size):
            if peer < rank_start_same_node or peer >= rank_end_same_node:
                int_wait_until_equals_remote(
                    scratch_ptr + rank,
                    response_buffer_ptr + rank,
                    next_request_id,
                    peer,
                )
        __syncthreads()


@dataclass
class ClientContext:
    request_buffer: torch.Tensor
    response_buffer: torch.Tensor
    next_request_id: int
    local_next_request_id: int


@dataclass
class ServerContext:
    request_buffer: torch.Tensor
    response_buffer: torch.Tensor
    next_request_id: list[int]
    accumulation_start: Mapping[Tuple[int, int], int]


def ack(server_context, client_rank):
    server_context.request_buffer[client_rank] = 0
    server_context.response_buffer[client_rank] = server_context.next_request_id[client_rank]
    server_context.next_request_id[client_rank] += 1
    if server_context.next_request_id[client_rank] > MAX_REQUEST_COUNT:
        server_context.next_request_id[client_rank] = 1


def server_loop(server_context, dispatch_func, exit_predicate, client_mask=None):
    if client_mask is None:
        client_mask = set()
    request_buffer_cpu = torch.empty_like(server_context.request_buffer, device="cpu").pin_memory()
    # ODC_DEBUG_WATCHER=1 turns on once-per-N-iter diagnostics inside the
    # watcher subprocess. Useful when the client appears to send requests
    # but the watcher never processes them (typical symptom of a missing
    # memory fence in the int_p path).
    debug = bool(int(os.environ.get("ODC_DEBUG_WATCHER", "0")))
    # Tuning knob (single-node rocSHMEM scatter-accumulate hand-shake latency):
    # the watcher D2H-polls request_buffer in a tight loop. Default 1/10000 s
    # (== 0.0001) preserves the original byte-for-byte behaviour (mori path
    # unchanged). Set ODC_WATCHER_SLEEP_S=0 to busy-spin (lowest dispatch
    # latency, burns one CPU core) or a smaller value to poll more often.
    watcher_sleep_s = float(os.environ.get("ODC_WATCHER_SLEEP_S", "0.0001"))
    iter_count = 0
    started_at = time.time()
    while True:
        request_buffer_cpu.copy_(server_context.request_buffer)
        nonzeros = torch.nonzero(request_buffer_cpu, as_tuple=False).squeeze(1).tolist()
        if debug and (iter_count % 100000 == 0):
            buf_preview = request_buffer_cpu.tolist()
            elapsed = time.time() - started_at
            print(
                f"[watcher pid={os.getpid()}] iter={iter_count} elapsed={elapsed:.1f}s "
                f"request_buffer={buf_preview} nonzeros={nonzeros}",
                flush=True,
            )
        if nonzeros:
            time.time()
        if watcher_sleep_s > 0:
            time.sleep(watcher_sleep_s)
        for client_rank in nonzeros:
            if len(client_mask) > 0 and client_rank not in client_mask:
                continue
            command = request_buffer_cpu[client_rank].item()
            assert isinstance(client_rank, int)
            assert isinstance(command, int)
            if debug:
                print(
                    f"[watcher pid={os.getpid()}] dispatching client_rank={client_rank} cmd={command}",
                    flush=True,
                )
            acked = dispatch_func(client_rank, command)
            if not acked:
                with torch.cuda.nvtx.range(f"ack {client_rank} cmd {command}"):
                    ack(server_context, client_rank)
        if exit_predicate():
            break
        iter_count += 1


class DistLock:
    def __init__(self):
        self.world_size = torch.distributed.get_world_size()
        self.request_buffer = SymmBufferRegistry.get_instance().allocate_symm_buffer(
            "request_buffer", (self.world_size,), torch.int32
        )
        self.response_buffer = SymmBufferRegistry.get_instance().allocate_symm_buffer(
            "response_buffer", (self.world_size,), torch.int32
        )
        self.request_buffer.fill_(0)
        self.response_buffer.fill_(0)
        self.client_context = ClientContext(self.request_buffer, self.response_buffer, 1, 1)


class ReductionWatcher:
    def __init__(
        self,
        world_size,
        accumulations: List[torch.Tensor],
        buffers: List[torch.Tensor],
        request_buffer: torch.Tensor,
        response_buffer: torch.Tensor,
    ):
        assert len(accumulations) == 0
        assert len(buffers) == 0
        self.accumulations = accumulations
        self.group_world_sizes = []
        self.buffers = buffers
        self.request_buffer = request_buffer
        self.response_buffer = response_buffer
        self.running = True
        self.task_count = 0
        self.server_context = ServerContext(
            self.request_buffer, self.response_buffer, [1] * world_size, defaultdict(lambda: 0)
        )

    def stop(self):
        self.running = False

    def wait_and_reset_task_count(self, expected):
        while self.task_count < expected:
            time.sleep(0)
        self.task_count = 0

    def add_buffer(self, buffers):
        self.buffers.append([tensor_from_handle(*buffer) for buffer in buffers])

    def add_accumulation(self, accumulations, group_world_size):
        tensors = [tensor_from_handle(*acc) for acc in accumulations]
        # Ensure the watcher-side view starts from zeros (epoch 0 safety).
        for tensor in tensors:
            tensor.zero_()
        self.accumulations.append(tensors)
        self.group_world_sizes.append(group_world_size)

    def run(self):
        def dispatch_func(client_rank, command):
            if command == -1:
                # client_mask.add(client_rank)
                return False
            else:
                buffer_id = command >> 16
                accumulation_id = command & 0xFFFF

                acc = self.accumulations[accumulation_id - 1][0]
                # Only support one unified group_world_size for scatter-accumulate calls for now.
                client_group_rank = client_rank % self.group_world_sizes[accumulation_id - 1]
                buf = self.buffers[buffer_id][client_group_rank]
                start = self.server_context.accumulation_start[(buffer_id, client_rank)]
                size = min(buf.numel(), acc.numel() - start)
                with torch.cuda.nvtx.range(
                    f"add client {client_rank} buffer {buffer_id} accumulation {accumulation_id}"
                ):
                    acc[start : start + size].add_(buf[:size])
                if start + size >= acc.numel():
                    assert start + size == acc.numel()
                    self.server_context.accumulation_start[(buffer_id, client_rank)] = 0
                else:
                    self.server_context.accumulation_start[(buffer_id, client_rank)] += size
                torch.cuda.current_stream().synchronize()
                self.task_count += 1
                # client_mask.remove(client_rank)
                return False

        def exit_predicate():
            return not self.running

        client_mask = set()
        server_loop(self.server_context, dispatch_func, exit_predicate, client_mask)


def tensor_from_handle(handle, size, dtype):
    return reconstruct_tensor(handle, (size,), dtype)


def reduction_watcher_function(
    device_id,
    world_size,
    accumulations,
    buffers,
    request_buffer,
    response_buffer,
    cmd_queue,
    response_queue,
):
    torch.cuda.set_device(device_id)

    # torch.cuda.cudart().cudaProfilerStart()
    buffers = [tensor_from_handle(*buffer) for buffer in buffers]
    accumulations = [tensor_from_handle(*acc) for acc in accumulations]
    request_buffer = tensor_from_handle(*request_buffer)
    response_buffer = tensor_from_handle(*response_buffer)

    watcher = ReductionWatcher(world_size, accumulations, buffers, request_buffer, response_buffer)

    def cmd_thread():
        torch.cuda.set_device(device_id)
        while True:
            data = cmd_queue.get()
            cmd = data[0]
            args = data[1:]
            response_queue.put(getattr(watcher, cmd)(*args))
            if cmd == "stop":
                break

    cmd_thread = Thread(target=cmd_thread)
    cmd_thread.start()
    watcher.run()
    cmd_thread.join()


def start_reduction_watcher(accumulations, buffers, request_buffer, response_buffer):
    rank = torch.distributed.get_rank()
    current_device = torch.cuda.current_device()
    local_world_size = get_local_world_size()

    logger.debug(
        f"Rank {rank} start_reduction_watcher: "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')}, "
        f"current_device={current_device}, "
        f"local_world_size={local_world_size}"
    )

    ctx = torch.multiprocessing.get_context("spawn")
    cmd_queue = ctx.Queue()
    response_queue = ctx.Queue()
    device_id = torch.distributed.get_rank() % get_local_world_size()
    world_size = torch.distributed.get_world_size()

    logger.debug(f"Rank {rank} spawning reduction_watcher with device_id={device_id}")
    process = ctx.Process(
        target=reduction_watcher_function,
        args=(
            device_id,
            world_size,
            accumulations,
            buffers,
            request_buffer,
            response_buffer,
            cmd_queue,
            response_queue,
        ),
        # ROCm/MI355X teardown fix: the watcher's server_loop spins forever until
        # it receives an explicit "stop". If the process exits without ODC teardown
        # (crash, KeyboardInterrupt, or a host that skips stop()), a non-daemon
        # watcher keeps the parent rank blocked on join() and holds the GPU. Marking
        # it daemon lets the Python runtime reap it on parent exit so the GPU frees.
        daemon=True,
    )
    process.start()
    return cmd_queue, response_queue


def call_watcher(watcher_handle, cmd, *args):
    cmd_queue, response_queue = watcher_handle
    cmd_queue.put((cmd, *args))
    return response_queue.get()


def get_shmem_handle(tensor):
    logger.info(
        f"Rank {torch.distributed.get_rank()} get_shmem_handle {tensor.data_ptr()} with shape {tensor.shape} and dtype {tensor.dtype}",
    )
    handle = get_ipc_handle(tensor)
    return handle, tensor.numel(), tensor.dtype


class ReductionService:
    def __init__(self, accumulation_dtype=None):
        self.accumulations = []
        self.buffers = []
        self.lock = None
        self.reduction_watcher = None
        self.accumulation_indices = {}
        self.buffer_indices = {}
        self.shared_buffer = {}
        self.input_buffer = {}
        self.dispatched_tasks = 0
        self.accumulation_dtype = accumulation_dtype
        self.buffer_splitter = BufferSplitter()
        self.rank_streams = defaultdict(torch.cuda.Stream)
        self.chunk_size_bytes = 2**20

    def get_chunk_size(self, buffer_dtype):
        return self.chunk_size_bytes // buffer_dtype.itemsize

    def register(self, key, output_tensor_shape, grad_dtype, reduction_dtype, pg: dist.ProcessGroup):
        if self.reduction_watcher is None:
            self.lock = DistLock()
            request_buffer_handle = get_shmem_handle(self.lock.request_buffer)
            response_buffer_handle = get_shmem_handle(self.lock.response_buffer)

            # Make sure changes are visible to all reduction watchers
            torch.distributed.barrier()
            torch.cuda.synchronize()

            self.reduction_watcher = start_reduction_watcher(
                [], [], request_buffer_handle, response_buffer_handle
            )

        accumulation_key = f"rs_accumulation_{key}"
        assert len(output_tensor_shape) == 1
        registry = SymmBufferRegistry.get_instance()
        assert not registry.has_key(accumulation_key)
        # assert self.reduction_watcher is None, "Reduction watcher is already running"
        group_world_size = torch.distributed.get_world_size(pg)

        def create_and_register_accumulation(key, shape, dtype, add_func):
            buffer = registry.allocate_symm_buffer(key, shape, dtype)
            call_watcher(self.reduction_watcher, add_func, [get_shmem_handle(buffer)], group_world_size)
            return buffer

        def create_and_register_buffer(key, shape, dtype, add_func):
            buffers = []
            for rank in range(group_world_size):
                buffer = registry.allocate_symm_buffer(f"{key}_rank_{rank}", shape, dtype)
                buffers.append(buffer)
            call_watcher(self.reduction_watcher, add_func, [get_shmem_handle(b) for b in buffers])
            return buffers

        acc = create_and_register_accumulation(
            accumulation_key, output_tensor_shape, reduction_dtype, "add_accumulation"
        )
        self.accumulation_indices[key] = len(self.accumulations)
        self.accumulations.append(acc)

        buffer_size = self.buffer_splitter.get_local_buffer_size(output_tensor_shape, group_world_size)
        output_size = reduce(lambda x, y: x * y, output_tensor_shape)
        logger.info(
            f"buffer_size: {buffer_size} output_size: {output_size} num_split: {math.ceil(output_size / buffer_size)}"
        )
        buffer_shape = (buffer_size,)

        shared_buffer_key = (grad_dtype, buffer_shape)
        if shared_buffer_key not in self.shared_buffer:
            output_size = reduce(lambda x, y: x * y, output_tensor_shape)
            logger.info(
                f"Rank {torch.distributed.get_rank()} create buffer: output_size: {output_size} num_sub_buffers: {math.ceil(output_size / buffer_size)} buffer_size: {buffer_size}",
            )
            cnt = len(self.shared_buffer)
            buffers = create_and_register_buffer(
                f"shared_buffer_{cnt}", buffer_shape, grad_dtype, "add_buffer"
            )
            self.shared_buffer[shared_buffer_key] = (cnt, buffers)

            self.buffers.append(buffers)
        self.buffer_indices[key] = self.shared_buffer[shared_buffer_key][0]

        # Make sure changes are visible to all reduction watchers
        torch.distributed.barrier()
        torch.cuda.synchronize()

    def clear_accumulations(self):
        for acc in self.accumulations:
            acc.fill_(0)
        if hasattr(self, "_gda_deferred"):  # Deliverable 23: reset per-minibatch deferred grads
            self._gda_deferred = {}
            self._gda_deferred_pg = {}
        if hasattr(self, "_sdr_deferred"):  # ODC_SINGLE_DEVICE_REDUCE: reset per-minibatch deferred grads
            self._sdr_deferred = {}
            self._sdr_deferred_pg = {}

    def infer_output_shape(self, input_tensor, pg: dist.ProcessGroup):
        assert len(input_tensor.shape) == 1
        assert input_tensor.shape[0] % dist.get_world_size(pg) == 0
        return (input_tensor.shape[0] // dist.get_world_size(pg),)

    def _gda_scatter_accumulate(self, key, input_tensor, pg: dist.ProcessGroup):
        """GPU-direct pull-based reduce-scatter accumulate (race-free, no watcher).

        Stages this rank's full input into a symmetric buffer, barriers, then a
        device kernel pulls every PE's contribution to MY output shard and sums
        it on-chip into the fp32 accumulation buffer (acc += reduce_scatter(input)).
        """
        gws = torch.distributed.get_world_size(pg)
        assert (
            gws == _rs._n_pes
        ), f"GDA reduce-scatter requires a full-world group: gws={gws} n_pes={_rs._n_pes}"
        assert input_tensor.numel() % gws == 0, f"{input_tensor.numel()=} % {gws=}"
        shard_elems = input_tensor.numel() // gws
        dt = input_tensor.dtype
        es = input_tensor.element_size()
        reg = SymmBufferRegistry.get_instance()

        if key not in self.accumulation_indices:
            acc = reg.get_or_create_symm_buffer(f"gda_acc_{key}", (shard_elems,), torch.float32)
            acc.fill_(0)
            self.accumulation_indices[key] = len(self.accumulations)
            self.accumulations.append(acc)
        acc = self.accumulations[self.accumulation_indices[key]]

        in_key = ("gda_in", dt, input_tensor.numel())
        if in_key not in self.input_buffer:
            self.input_buffer[in_key] = reg.get_or_create_symm_buffer(
                f"gda_in_{dt}_{input_tensor.numel()}", (input_tensor.numel(),), dt
            )
        input_sym = self.input_buffer[in_key]

        # GRID geometry sweep (Deliverable 15): nblk = reduce-scatter kernel grid (one block
        # per disjoint shard chunk -> # concurrent cross-node getmem_wg = QP/NIC
        # parallelism lever). Scratch auto-resizes (sc_key includes chunk*nblk), so
        # this stays correct for any nblk. Default 64 (current behavior).
        nblk = int(os.environ.get("ODC_GDA_RS_BLOCKS", "64"))
        if nblk < 1:
            nblk = 1
        # PIPE (Deliverable 17): peer-pipeline batch depth. The pipelined rs_acc needs `pipe`
        # scratch slots PER BLOCK (issues `pipe` peers' nbi getmem concurrently), so
        # scratch grows pipe x and the main call passes scratch_stride = pipe*chunk.
        pipe = int(os.environ.get("ODC_GDA_PIPE", "1"))
        if pipe < 1:
            pipe = 1
        chunk = (shard_elems + nblk - 1) // nblk
        sc_slots = nblk * pipe
        sc_key = ("gda_scr", dt, chunk * sc_slots)
        if sc_key not in self.input_buffer:
            self.input_buffer[sc_key] = reg.get_or_create_symm_buffer(
                f"gda_scr_{dt}_{chunk * sc_slots}", (chunk * sc_slots,), dt
            )
        scratch = self.input_buffer[sc_key]

        rank = torch.distributed.get_rank(pg)
        official = _official_push()
        import time as _t

        _prof = os.environ.get("ODC_GDA_PROFILE", "0") == "1"
        # Cross-node write-visibility strategy for the just-staged grad (see the
        # warm-up note below for why this is needed at all):
        #   "full" (default) - torch copy_ stage, then a FULL-shard throwaway
        #            reduce-scatter + barrier primes every NIC/page so the real
        #            RS reads fresh data. Correct but ~doubles the RS (~59% of RS).
        #   "hdp"            - stage with torch copy_, then flush this GPU's HDP
        #            via the HDP_MEM_FLUSH_CNTL register (gda_hdp_flush): an O(1)
        #            MMIO write that makes the staged symmetric write NIC-visible
        #            across ALL pages/NICs. The proper GPUDirect-RDMA primitive;
        #            NO throwaway reduce-scatter.
        #   "fence"          - stage via gda_stage_fence: a device copy that ends
        #            in __threadfence_system(). NOTE: empirically INSUFFICIENT on
        #            this mlx5/GDA path (grad spikes return) -- kept for reference.
        # Default "strided": page-strided tiny throwaway READ (one element per
        # ODC_GDA_STRIDE_BYTES page x all PEs) -- keeps full-warmup's deterministic
        # "read-triggered settle" (validated 0 grad spikes, loss == single-node)
        # but ~9-10% faster (skips the full-shard throwaway reduce-scatter). Modes:
        #   "strided" (default) - 0 spikes, ~9-10% faster than full; stride via
        #                         ODC_GDA_STRIDE_BYTES (default 65536 = 64KB).
        #   "full"              - full-shard throwaway RS: most robust, slowest.
        #   "hdp"               - O(1) HDP-flush register write: fastest, but a
        #                         50-iter run showed intermittent spikes (nopad 6/50);
        #                         opt-in. Auto-falls back to "full" if no HDP register.
        _warm_mode = os.environ.get("ODC_GDA_WARMUP_MODE", "strided")
        if _warm_mode == "hdp" and getattr(self, "_hdp_fallback", False):
            _warm_mode = "full"
        if not getattr(self, "_logged_warm_mode", False):
            logger.warning(
                "[GDA] reduce-scatter warm-up mode=%s stride_bytes=%s",
                _warm_mode,
                os.environ.get("ODC_GDA_STRIDE_BYTES", "65536"),
            )
            self._logged_warm_mode = True
        # OVERLAP (ODC_GDA_OVERLAP=1): the previous group's reduce-scatter was
        # launched on a side stream and NOT synced, so it overlapped the backward
        # compute that ran since. Now, before we re-stage into the SHARED input_sym
        # (and before its acc is needed), wait it -> guards the buffer reuse + acc.
        _overlap = os.environ.get("ODC_GDA_OVERLAP", "0") == "1"
        # ODC_GDA_BUCKET=1: drop the redundant post-strided-warmup barrier (barrier#2).
        # The strided touch only primes THIS rank's NIC read paths to peers' already-
        # staged data (barrier#1 after staging guaranteed all peers staged); it writes
        # nothing, so a second cross-rank barrier before the real reduce-scatter is
        # unnecessary -> halves rocshmem_barrier_all from ~62 to ~31 per step.
        _bucket = os.environ.get("ODC_GDA_BUCKET", "0") == "1"
        if official:
            # Official single-sided push has no comm/compute overlap side-stream;
            # force the synchronous bare reduce-scatter (no overlap collect).
            _overlap = False
        if _overlap:
            _rs.gda_rs_overlap_sync()
        _ts0 = _t.perf_counter()
        if _warm_mode in ("fence", "hdpfence"):
            # Ensure the grad (input_tensor) is fully produced before copy+fence,
            # then copy into the symmetric staging buffer with a trailing
            # system-scope fence (gda_stage_fence self-syncs the device).
            # Deliverable 19 phase 2: "hdpfence" = device system-scope fence (orders the staged
            # writes to system scope) PLUS an HDP register flush (pushes this GPU's
            # HDP cache so the remote NIC's RDMA read sees fresh data) -- a read-FREE
            # visibility guarantee combining BOTH writer-side primitives, to skip the
            # strided read settle. Correctness gate decides if it's deterministic.
            torch.cuda.current_stream().synchronize()
            _flat = input_tensor.view(-1)
            _rs.gda_stage_fence(input_sym.data_ptr(), _flat.data_ptr(), _flat.numel() * es)
            if _warm_mode == "hdpfence":
                if not getattr(self, "_hdp_inited", False) and not getattr(self, "_hdp_fallback", False):
                    rc = _rs.gda_hdp_init()
                    if rc == 0:
                        self._hdp_inited = True
                    else:
                        logger.warning("gda_hdp_init failed rc=%d -> hdpfence falls back to fence-only", rc)
                        self._hdp_fallback = True
                if getattr(self, "_hdp_inited", False):
                    _rs.gda_hdp_flush()
        else:
            get_comm_stream().wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(get_comm_stream()):
                input_sym.copy_(input_tensor.view(-1))
            # Deliverable 20: the staging only needs the COPY (on comm stream) done before the
            # settle/RS read input_sym; a FULL torch.cuda.synchronize() also waits the
            # gather-async kernels on their DEDICATED stream (which are already waited
            # by gather.py's own wait_stream at consumption) -> redundant, it absorbs
            # the gather RDMA wait into scatter. STAGE_STREAMSYNC=1 syncs only the comm
            # stream so async gather truly overlaps scatter. Default (0) = full sync.
            if os.environ.get("ODC_GDA_STAGE_STREAMSYNC", "0") == "1":
                get_comm_stream().synchronize()
            else:
                torch.cuda.synchronize()
            if _warm_mode == "hdp":
                if not getattr(self, "_hdp_inited", False):
                    rc = _rs.gda_hdp_init()
                    if rc != 0:
                        logger.warning(
                            "gda_hdp_init failed rc=%d -> falling back to full-shard warm-up RS", rc
                        )
                        self._hdp_fallback = True
                        _warm_mode = "full"
                    else:
                        self._hdp_inited = True
                if _warm_mode == "hdp":
                    _rs.gda_hdp_flush()
        _t_stage = _t.perf_counter() - _ts0
        _tb0 = _t.perf_counter()
        # ODC_OFFICIAL_PUSH: official single-sided push does NOT barrier per call
        # (no cross-PE rendezvous before the reduce-scatter reads peers' staged
        # grad) -- skip it. On this mlx5/GDA fabric the just-staged write may not
        # be NIC-visible -> stale read -> grad spike: that is the documented risk.
        if not official:
            _rs.barrier()
        _t_bar = _t.perf_counter() - _tb0

        # ROOT-CAUSE PROBE (ODC_GDA_VERIFY=1): run the SAME reduce-scatter twice
        # from the identical staged input_sym into two fresh buffers and compare.
        # diff>0 => non-deterministic => cross-node visibility/race (stale read);
        # diff==0 => deterministic (wiring bug, compare vs reference separately).
        if os.environ.get("ODC_GDA_VERIFY", "0") == "1":
            t1 = torch.zeros(shard_elems, dtype=torch.float32, device="cuda")
            t2 = torch.zeros(shard_elems, dtype=torch.float32, device="cuda")
            _rs.gda_reduce_scatter_acc(
                t1.data_ptr(),
                input_sym.data_ptr(),
                rank * shard_elems * es,
                shard_elems,
                _rs._n_pes,
                scratch.data_ptr(),
                chunk * es,
                _rs.dtype_code(dt),
                nblk,
            )
            _rs.barrier()
            _rs.gda_reduce_scatter_acc(
                t2.data_ptr(),
                input_sym.data_ptr(),
                rank * shard_elems * es,
                shard_elems,
                _rs._n_pes,
                scratch.data_ptr(),
                chunk * es,
                _rs.dtype_code(dt),
                nblk,
            )
            torch.cuda.synchronize()
            d = (t1 - t2).abs().max().item()
            if rank == 0:
                logger.warning(
                    "[GDA-VERIFY] key=%s n=%d max|run1-run2|=%.4e run1_norm=%.4e",
                    str(key),
                    shard_elems,
                    d,
                    t1.norm().item(),
                )

        # Cross-node write-visibility settle (FIX for the intermittent stale-read
        # grad spikes): a throwaway "warm-up" reduce-scatter + barrier BEFORE the
        # real one. A single barrier after staging is insufficient on this
        # mlx5/GDA path (peer's just-staged GPU write isn't yet NIC-visible to the
        # first device getmem -> stale read -> huge wrong gradient ~half the time).
        # The warm-up getmem + barrier forces the staged data visible; the real
        # reduce-scatter then reads fresh data. (Verified: eliminates spikes,
        # grad norms normal, loss matches single-node. Reduce-scatter is a small
        # fraction of the step, so the extra pass is cheap vs the gather.)
        # Opt1: the warm-up only needs to settle cross-node write-visibility (drain
        # the HDP so peers' staged writes are NIC-visible), NOT move the full
        # shard. A tiny getmem touch per peer + barrier achieves that at a
        # fraction of the cost (full-shard warm-up was ~9s/iter of pure overhead).
        # warm-up settles cross-node write-visibility (HDP) before the real RS.
        # Single-NIC: a 1024-elem touch suffices. Multi-NIC: each peer routes via a
        # different NIC and HDP is per-NIC/per-page, so a tiny touch leaves most
        # pages stale -> spikes; use the FULL shard (default) so every NIC/page is
        # settled. The full warm-up is itself parallelized across NICs (cheap).
        if official:
            # ODC_OFFICIAL_PUSH: no warm-up / strided / settle reduce-scatter and
            # no settle barrier. The real reduce-scatter reads peers' staged grad
            # directly (single-sided), exactly like upstream. Stale-read spikes
            # here are the documented risk we are reproducing, not fixing.
            _t_warm = 0.0
        elif _warm_mode in ("fence", "hdp", "hdpfence"):
            # The staged write was already made NIC-visible (HDP flush / fence),
            # so the throwaway warm-up reduce-scatter (the ~59%-of-RS overhead) is
            # skipped entirely. The barrier after staging still rendezvouses PEs.
            _t_warm = 0.0
        elif _warm_mode == "strided":
            # Page-strided tiny throwaway READ: keeps full-warmup's deterministic
            # "read-triggered settle" (covers every page of my segment on every PE
            # -> all 8 NICs) but at minimal volume (one touch per ODC_GDA_STRIDE_BYTES
            # page, not the whole shard). Staging above used plain copy_ (no flush).
            stride_b = int(os.environ.get("ODC_GDA_STRIDE_BYTES", "65536"))
            touch_b = int(es)
            seg_bytes = int(shard_elems * es)
            npages = (seg_bytes + stride_b - 1) // stride_b
            total_touch = _rs._n_pes * max(npages, 1)
            sstride = 256  # bytes/throwaway scratch slot (>= touch_b; avoids collide)
            scratch_cap = scratch.numel() * scratch.element_size()
            nblk_t = min(int(total_touch), 4096, max(1, scratch_cap // sstride))
            _tw0 = _t.perf_counter()
            _rs.gda_strided_touch(
                input_sym.data_ptr(),
                rank * shard_elems * es,
                seg_bytes,
                _rs._n_pes,
                stride_b,
                touch_b,
                scratch.data_ptr(),
                sstride,
                int(nblk_t),
            )
            if not _bucket:
                _rs.barrier()  # barrier#2 (redundant in bucket mode; see _bucket note)
            _t_warm = _t.perf_counter() - _tw0
        else:
            if os.environ.get("ODC_GDA_WARMUP_TINY", "0") == "1":
                n_warm, w_nblk, w_stride = min(int(shard_elems), 1024), 1, None
            else:
                n_warm, w_nblk, w_stride = int(shard_elems), nblk, chunk * es
            if w_stride is None:
                w_stride = n_warm * es
            wkey = ("gda_warmup", n_warm)
            if wkey not in self.input_buffer:
                self.input_buffer[wkey] = torch.zeros(n_warm, dtype=torch.float32, device="cuda")
            warmup = self.input_buffer[wkey]
            _tw0 = _t.perf_counter()
            warmup.zero_()
            _rs.gda_reduce_scatter_acc(
                warmup.data_ptr(),
                input_sym.data_ptr(),
                rank * shard_elems * es,
                n_warm,
                _rs._n_pes,
                scratch.data_ptr(),
                w_stride,
                _rs.dtype_code(dt),
                w_nblk,
            )
            _rs.barrier()
            _t_warm = _t.perf_counter() - _tw0
        _tr0 = _t.perf_counter()
        if _overlap:
            # Launch reduce-scatter on the side stream and RETURN without syncing,
            # so the next backward compute overlaps this RDMA-bound kernel. The next
            # scatter call's start-sync (and sync() at step end) collect it. acc and
            # input_sym are both guarded by those waits -> no stale read / race.
            _rs.gda_reduce_scatter_acc_async(
                acc.data_ptr(),
                input_sym.data_ptr(),
                rank * shard_elems * es,
                shard_elems,
                _rs._n_pes,
                scratch.data_ptr(),
                chunk * es,
                _rs.dtype_code(dt),
                nblk,
            )
        else:
            _rs.gda_reduce_scatter_acc(
                acc.data_ptr(),
                input_sym.data_ptr(),
                rank * shard_elems * es,
                shard_elems,
                _rs._n_pes,
                scratch.data_ptr(),
                pipe * chunk * es,
                _rs.dtype_code(dt),
                nblk,
            )
            torch.cuda.synchronize()
        _t_real = _t.perf_counter() - _tr0
        self.dispatched_tasks += 1
        if _prof and rank == 0:
            logger.warning(
                "[GDA-PROF scatter] shard=%d stage=%.3f barrier=%.3f warmup_rs=%.3f real_rs=%.3f",
                shard_elems,
                _t_stage,
                _t_bar,
                _t_warm,
                _t_real,
            )

    def _single_device_scatter_accumulate(self, key, input_tensor, pg: dist.ProcessGroup):
        """Single-node, watcher-FREE device-side reduce-scatter accumulate.

        Owner-side PULL + on-chip sum over same-node XGMI peer views (see
        ``_single_device_reduce`` for the full rationale). ``input_tensor`` is
        this PE's full (locally pre-accumulated) grad, laid out as
        ``[shard_0 | shard_1 | ... | shard_{gws-1}]``; PE r owns output shard r.

        Steps (all same-node, no subprocess, no IPC handle, no atomics):
          1. Stage my full grad into a symmetric fp32 buffer (peer views resolved
             on allocation, exactly like the watcher push buffer).
          2. cuda sync + collective barrier -> every peer's stage is retired and
             visible before any peer reads it over XGMI.
          3. For each same-node peer p, XGMI-``.copy_`` p's segment destined for
             MY shard into a local temp, then acc += temp (fp32 on-chip sum).
          4. cuda sync + collective barrier -> all reads done before the shared
             staging buffer can be reused by the next key/minibatch.
        """
        gws = torch.distributed.get_world_size(pg)
        lws = get_local_world_size()
        assert gws == lws, f"single-device reduce is same-node only: gws={gws} lws={lws}"
        assert input_tensor.numel() % gws == 0, f"{input_tensor.numel()=} % {gws=}"
        shard_elems = input_tensor.numel() // gws
        reg = SymmBufferRegistry.get_instance()
        rank = torch.distributed.get_rank(pg)  # single node -> group rank == local pos

        # fp32 accumulator = MY output shard (matches GDA's fp32 acc semantics).
        if key not in self.accumulation_indices:
            acc = reg.get_or_create_symm_buffer(f"sdr_acc_{key}", (shard_elems,), torch.float32)
            acc.fill_(0)
            self.accumulation_indices[key] = len(self.accumulations)
            self.accumulations.append(acc)
        acc = self.accumulations[self.accumulation_indices[key]]

        # Symmetric fp32 staging for my full grad, WITH same-node peer views. One
        # per grad-numel; reused across keys/minibatches (guarded by the trailing
        # barrier so no peer reuses it mid-read).
        in_key = ("sdr_in", input_tensor.numel())
        if in_key not in self.input_buffer:
            self.input_buffer[in_key] = reg.get_or_create_symm_buffer(
                f"sdr_in_{input_tensor.numel()}", (input_tensor.numel(),), torch.float32
            )
        input_sym = self.input_buffer[in_key]
        peer_inputs = reg.get_peer_tensors(input_sym)
        assert len(peer_inputs) == lws

        tmp_key = ("sdr_tmp", shard_elems)
        if tmp_key not in self.input_buffer:
            self.input_buffer[tmp_key] = torch.empty(shard_elems, dtype=torch.float32, device="cuda")
        tmp = self.input_buffer[tmp_key]

        input_sym.copy_(input_tensor.view(-1).to(torch.float32))
        torch.cuda.synchronize()
        torch.distributed.barrier(group=pg)

        lo = rank * shard_elems
        hi = lo + shard_elems
        # Rotate the peer order by local rank so all 8 PEs don't hammer the same
        # peer's HBM first (matches the gather/scatter round-robin peer ordering).
        for off in range(lws):
            p = (rank + off) % lws
            tmp.copy_(peer_inputs[p][lo:hi])
            acc.add_(tmp)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=pg)
        self.dispatched_tasks += 1

    def scatter_accumulate(self, key, input_tensor, pg: dist.ProcessGroup):
        official = _official_push()
        if _gda_active():
            # Deliverable 23: DEFER the cross-node reduce to once-per-minibatch. The per-call
            # _gda_scatter_accumulate does rocshmem_barrier_all (collective); calling it
            # per-microbatch deadlocks under nopad (ranks have different micro-batch
            # counts -> mismatched barrier counts vs pre_optimizer_step's NCCL barrier,
            # py-spy confirmed). Official ODC is single-sided push (no per-call barrier).
            # Fix (matches official's per-minibatch sync): accumulate the unsharded grad
            # LOCALLY here (no comm/barrier -> backward is lockstep-free -> ranks run
            # variable micro-batch counts without deadlock AND bubbles are saved), then
            # do ONE barriered reduce-scatter per group at get_accumulation (count =
            # #groups, matched across ranks). Deterministic strided settle preserved.
            # ODC_OFFICIAL_PUSH skips DEFER (official does no local pre-accumulation).
            # Multi-node DEFAULT = defer. The per-microbatch path issues a collective
            # rocshmem_barrier_all; under variable-length (nopad) packing, ranks on
            # different nodes get UNEQUAL micro-batch counts -> unequal barrier counts
            # -> cross-node rendezvous DEADLOCK (reproduced dual-node: DEFER=0 hangs,
            # DEFER=1 runs 25/25 with 0 nan). Deferring does ONE barriered reduce-scatter
            # per group (count == #groups, matched across ranks) -> deadlock-free.
            # Single-node (n_pes <= local_world_size) keeps the per-call path (no
            # cross-node barrier to mismatch). The env var still overrides both.
            defer_default = "1" if _rs._n_pes > get_local_world_size() else "0"
            if os.environ.get("ODC_GDA_DEFER_REDUCE", defer_default) == "1" and not official:
                if not hasattr(self, "_gda_deferred"):
                    self._gda_deferred = {}
                    self._gda_deferred_pg = {}
                cur = self._gda_deferred.get(key)
                if cur is None:
                    self._gda_deferred[key] = input_tensor.detach().clone()
                else:
                    cur.add_(input_tensor)
                self._gda_deferred_pg[key] = pg
                return
            return self._gda_scatter_accumulate(key, input_tensor, pg)
        if _single_device_reduce() and torch.distributed.get_world_size(pg) == get_local_world_size():
            # ODC_SINGLE_DEVICE_REDUCE grey path (single node only). DEFER exactly
            # like the multi-node GDA path: pre-accumulate this micro-batch's grad
            # LOCALLY (no collective -> nopad-safe), then run ONE barriered
            # owner-side pull-sum per group at get_accumulation. Pre-accumulate in
            # fp32 so cross-micro-batch accumulation matches the watcher's fp32 acc.
            if not hasattr(self, "_sdr_deferred"):
                self._sdr_deferred = {}
                self._sdr_deferred_pg = {}
            cur = self._sdr_deferred.get(key)
            if cur is None:
                self._sdr_deferred[key] = input_tensor.detach().to(torch.float32)
            else:
                cur.add_(input_tensor)
            self._sdr_deferred_pg[key] = pg
            return
        output_tensor_shape = self.infer_output_shape(input_tensor, pg)
        accum_dtype = self.accumulation_dtype if self.accumulation_dtype is not None else input_tensor.dtype
        if key not in self.accumulation_indices:
            self.register(key, output_tensor_shape, input_tensor.dtype, accum_dtype, pg)

        group_world_size = torch.distributed.get_world_size(pg)
        assert group_world_size in (
            torch.distributed.get_world_size(),
            get_local_world_size(),
        ), f"{group_world_size=} {torch.distributed.get_world_size()=} {get_local_world_size()=}"
        local_buf_size = self.buffer_splitter.get_local_buffer_size(output_tensor_shape, group_world_size)
        output_size = reduce(lambda x, y: x * y, output_tensor_shape)

        chunk_size = self.get_chunk_size(input_tensor.dtype)
        grid_size = get_local_world_size()
        input_tensor_symm_shape = (chunk_size * grid_size,)
        rank = torch.distributed.get_rank()
        if (input_tensor_symm_shape, input_tensor.dtype) not in self.input_buffer:
            self.input_buffer[
                (input_tensor_symm_shape, input_tensor.dtype)
            ] = SymmBufferRegistry.get_instance().allocate_symm_buffer(
                f"rs_buffer_{input_tensor_symm_shape}_{input_tensor.dtype}",
                input_tensor_symm_shape,
                input_tensor.dtype,
            )
        input_tensor_symm = self.input_buffer[(input_tensor_symm_shape, input_tensor.dtype)]

        buffer_id = self.buffer_indices[key]
        group_rank = torch.distributed.get_rank(pg)
        buffer = self.buffers[buffer_id][group_rank]
        accumulation_id = self.accumulation_indices[key] + 1

        accumulation_command = (buffer_id << 16) | accumulation_id
        assert buffer.nbytes % (2**6) == 0, f"better align to 64 for efficiency. Found {buffer.nbytes} bytes"

        get_comm_stream().wait_stream(torch.cuda.current_stream())
        rank_start_same_node = rank - rank % get_local_world_size()
        rank_end_same_node = rank_start_same_node + get_local_world_size()
        for local_peer in range(rank_start_same_node, rank_end_same_node):
            self.rank_streams[local_peer].wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(get_comm_stream()):
            signal_ptr = torch.empty(1, dtype=torch.int32, device="cuda")
        num_requests = math.ceil(output_size / local_buf_size)
        assert group_world_size % 8 == 0 or group_world_size < 8
        assert group_world_size % grid_size == 0
        _, rank_input_size = input_tensor.view(-1).view(group_world_size, -1).shape

        for start in range(0, output_size, local_buf_size):
            size = min(local_buf_size, output_size - start)
            assert local_buf_size == buffer.numel(), f"{local_buf_size=} {buffer.numel()=}"
            buf = buffer[:size]
            # Use mem-copy for the ranks in the same node.
            same_node_tensors = SymmBufferRegistry.get_instance().get_peer_tensors(buffer)
            for i in range(get_local_world_size()):
                peer_idx = (rank % get_local_world_size() + i) % get_local_world_size()
                local_peer = rank_start_same_node + peer_idx
                if group_world_size == get_local_world_size():
                    rank_input_start = peer_idx * rank_input_size + start
                else:
                    rank_input_start = local_peer * rank_input_size + start
                same_node_peer_buffer = same_node_tensors[peer_idx]
                peer_buf = same_node_peer_buffer[:size]
                stream = self.rank_streams[local_peer]
                with torch.cuda.stream(stream):
                    peer_buf.copy_(
                        input_tensor[rank_input_start : rank_input_start + size],
                        non_blocking=True,
                    )

            for local_peer in range(rank_start_same_node, rank_end_same_node):
                with torch.cuda.stream(self.rank_streams[local_peer]):
                    shmem_request_accumulation_same_node_kernel[(1,)](
                        request_buffer_ptr=self.lock.request_buffer,
                        rank=rank,
                        peer=local_peer,
                        accumulation_command=accumulation_command,
                        num_warps=1,
                        extern_libs=SHMEM_EXTERN_LIBS,
                    )
                    # ODC_OFFICIAL_PUSH: fire-and-forget. The push (peer_buf.copy_)
                    # + request signal are enough for the owner's watcher to
                    # accumulate; DROP the per-call ack/settle wait so the push
                    # overlaps backward compute. Arrival is guaranteed by the
                    # single minibatch-end sync() (watcher task-count rendezvous).
                    # NOTE: this removes the buffer-reuse handshake -> a later call
                    # reusing this staging slot before the watcher consumed it can
                    # stale/lose a gradient (the documented divergence risk).
                    if not official:
                        shmem_wait_accumulation_same_node_kernel[(1,)](
                            response_buffer_ptr=self.lock.response_buffer,
                            rank=rank,
                            peer=local_peer,
                            next_request_id=self.lock.client_context.local_next_request_id,
                            num_warps=1,
                            extern_libs=SHMEM_EXTERN_LIBS,
                        )
            self.lock.client_context.local_next_request_id += 1
            if self.lock.client_context.local_next_request_id > MAX_REQUEST_COUNT:
                self.lock.client_context.local_next_request_id = 1

        if group_world_size > get_local_world_size():
            assert group_world_size % get_local_world_size() == 0
            # Cross-node only: a symmetric int32 scratch for the RDMA getmem-spin
            # ack wait (one slot per global rank; we only read [rank]). Allocated
            # lazily and collectively on the FIRST cross-node scatter — every
            # rank enters this branch together because group_world_size >
            # local_world_size is a global property, so symmetric offsets stay
            # consistent. On a single node this branch is never taken, so no
            # extra symmetric buffer is allocated and behaviour is unchanged.
            world_size = torch.distributed.get_world_size()
            response_scratch = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
                "rs_response_scratch", (world_size,), torch.int32
            )
            if _ro_active():
                self._ro_cross_node_scatter(
                    input_tensor=input_tensor,
                    buffer=buffer,
                    rank_input_size=rank_input_size,
                    output_size=output_size,
                    local_buf_size=local_buf_size,
                    accumulation_command=accumulation_command,
                    response_scratch=response_scratch,
                    rank=rank,
                    rank_start_same_node=rank_start_same_node,
                    rank_end_same_node=rank_end_same_node,
                    group_world_size=group_world_size,
                )
            else:
                with torch.cuda.stream(get_comm_stream()):
                    signal_ptr.fill_(0)
                    shmem_cross_node_scatter[(grid_size,)](
                        input_tensor_ptr=input_tensor,
                        rank_input_size=rank_input_size,
                        chunk_buffer=input_tensor_symm,
                        trans_buffer=buf,
                        size_per_elem=buf.element_size(),
                        rank=rank,
                        num_ranks_per_node=grid_size,
                        world_size=group_world_size,
                        output_size=output_size,
                        local_buf_size=local_buf_size,
                        chunk_size=chunk_size,
                        signal_ptr=signal_ptr,
                        # client request
                        request_buffer_ptr=self.lock.request_buffer,
                        response_buffer_ptr=self.lock.response_buffer,
                        response_scratch_ptr=response_scratch,
                        rank_start_same_node=rank_start_same_node,
                        rank_end_same_node=rank_end_same_node,
                        accumulation_command=accumulation_command,
                        next_request_id=self.lock.client_context.next_request_id,
                        num_warps=32,
                        extern_libs=SHMEM_EXTERN_LIBS,
                    )
        self.lock.client_context.next_request_id += num_requests
        if self.lock.client_context.next_request_id > MAX_REQUEST_COUNT:
            self.lock.client_context.next_request_id = 1
        self.dispatched_tasks += num_requests

        # ODC_OFFICIAL_PUSH: do NOT serialize the current stream onto the per-peer
        # push streams / comm stream (that trailing wait_stream is what forces the
        # push to complete before backward continues). Skipping it lets the push
        # truly overlap the backward compute (fire-and-forget); the minibatch-end
        # sync() is the single light rendezvous that guarantees completion.
        if not official:
            if _settle_defer():
                # ODC_SETTLE_DEFER: do NOT serialize backward onto the per-peer
                # settle here -- keep push+ack queued on rank_streams (their
                # serial order still guards staging-buffer reuse) and let the
                # single sync() join collect them before optimizer.step. Keep the
                # source grad alive until the side-stream copy consumed it.
                for local_peer in range(rank_start_same_node, rank_end_same_node):
                    input_tensor.record_stream(self.rank_streams[local_peer])
            else:
                for local_peer in range(rank_start_same_node, rank_end_same_node):
                    torch.cuda.current_stream().wait_stream(self.rank_streams[local_peer])
                torch.cuda.current_stream().wait_stream(get_comm_stream())

    def _ro_cross_node_scatter(
        self,
        *,
        input_tensor,
        buffer,
        rank_input_size,
        output_size,
        local_buf_size,
        accumulation_command,
        response_scratch,
        rank,
        rank_start_same_node,
        rank_end_same_node,
        group_world_size,
    ):
        """Host-driven RO cross-node scatter-accumulate (rocSHMEM RO backend).

        Mirrors ``shmem_cross_node_scatter`` but drives the transfer from the
        CPU: for each output chunk and each remote peer ``r``, push my segment
        destined for ``r`` into ``buffer[my_rank]`` on ``r`` (which ``r``'s
        watcher accumulates), signal ``r``'s request slot, and wait for the ack.
        Same-node contributions were already handshaked on-device before this.
        """
        es = buffer.element_size()
        isize = 4  # int32 bytes
        # Symmetric staging for the putmem source (must be a registered
        # symmetric address). Sized to one local buffer chunk, reused per peer
        # (rs_putmem is blocking on the source, so reuse is safe).
        staging_key = ("ro_scatter_staging", buffer.dtype, int(local_buf_size))
        if staging_key not in self.input_buffer:
            self.input_buffer[staging_key] = SymmBufferRegistry.get_instance().get_or_create_symm_buffer(
                f"ro_scatter_staging_{buffer.dtype}_{local_buf_size}",
                (int(local_buf_size),),
                buffer.dtype,
            )
        staging = self.input_buffer[staging_key]

        dst_ptr = buffer.data_ptr()
        req_ptr = self.lock.request_buffer.data_ptr()
        resp_ptr = self.lock.response_buffer.data_ptr()
        scratch_ptr = response_scratch.data_ptr()
        base = self.lock.client_context.next_request_id
        remote_peers = [
            r for r in range(group_world_size) if not (rank_start_same_node <= r < rank_end_same_node)
        ]
        comm_stream = get_comm_stream()
        flat_input = input_tensor.view(-1)

        prof = os.environ.get("ODC_RO_PROFILE", "0") == "1"
        import time as _t

        t_put = t_quiet = t_sig = t_ack = 0.0
        ack_polls = 0
        t0_total = _t.perf_counter()

        chunk_idx = 0
        for start in range(0, output_size, local_buf_size):
            size = min(local_buf_size, output_size - start)
            nbytes = size * es
            # 1. push my segment for each remote peer into peer's buffer[my_rank]
            _tp = _t.perf_counter()
            for r in remote_peers:
                seg_start = r * rank_input_size + start
                with torch.cuda.stream(comm_stream):
                    staging[:size].copy_(flat_input[seg_start : seg_start + size])
                comm_stream.synchronize()
                _rs.putmem(dst_ptr, staging.data_ptr(), nbytes, r)
            t_put += _t.perf_counter() - _tp
            _tq = _t.perf_counter()
            _rs.quiet()
            t_quiet += _t.perf_counter() - _tq
            # 2. signal each remote peer's watcher (request_buffer[my_rank]=cmd)
            _ts = _t.perf_counter()
            for r in remote_peers:
                _rs.int_p(req_ptr + rank * isize, int(accumulation_command), r)
            _rs.quiet()
            t_sig += _t.perf_counter() - _ts
            # 3. wait for each remote peer's ack (response_buffer[my_rank]==expected)
            expected = base + chunk_idx
            _ta = _t.perf_counter()
            for r in remote_peers:
                while True:
                    _rs.getmem(scratch_ptr + rank * isize, resp_ptr + rank * isize, isize, r)
                    _rs.quiet()
                    ack_polls += 1
                    if int(response_scratch[rank].item()) == expected:
                        break
            t_ack += _t.perf_counter() - _ta
            chunk_idx += 1

        if prof and rank == 0:
            logger.info(
                "[RO-PROF scatter] out=%d chunks=%d peers=%d total=%.3fs | put=%.3f quiet=%.3f "
                "signal=%.3f ackwait=%.3f ack_polls=%d",
                output_size,
                chunk_idx,
                len(remote_peers),
                _t.perf_counter() - t0_total,
                t_put,
                t_quiet,
                t_sig,
                t_ack,
                ack_polls,
            )

    def get_accumulation(self, key):
        # Deliverable 23 DEFER: run the single per-minibatch cross-node reduce-scatter now
        # (once per group, matched barrier count across ranks -> deadlock-free).
        if _gda_active() and getattr(self, "_gda_deferred", None) is not None:
            pending = self._gda_deferred.get(key)
            if pending is not None:
                self._gda_deferred[key] = None
                self._gda_scatter_accumulate(key, pending, self._gda_deferred_pg[key])
        # ODC_SINGLE_DEVICE_REDUCE DEFER: run the single per-minibatch owner-side
        # pull-sum now (once per group, matched barrier count across ranks).
        if _single_device_reduce() and getattr(self, "_sdr_deferred", None) is not None:
            pending = self._sdr_deferred.get(key)
            if pending is not None:
                self._sdr_deferred[key] = None
                self._single_device_scatter_accumulate(key, pending, self._sdr_deferred_pg[key])
        acc = self.accumulations[self.accumulation_indices[key]]
        return acc

    def sync(self, pg: dist.ProcessGroup):
        if _gda_active():
            # OVERLAP: collect any in-flight side-stream reduce-scatters before the
            # optimizer reads acc (final correctness barrier for the overlap path).
            # ODC_OFFICIAL_PUSH forces synchronous RS (no side stream), so nothing
            # to collect here.
            if os.environ.get("ODC_GDA_OVERLAP", "0") == "1" and not _official_push():
                _rs.gda_rs_overlap_sync()
            # GDA path is fully synchronous (device kernels + barriers); no watcher
            # task accounting needed.
            torch.cuda.synchronize()
            torch.distributed.barrier(group=pg)
            self.dispatched_tasks = 0
            return
        if _single_device_reduce():
            # Single-node watcher-free path: the owner-side pull-sum already ran
            # (synchronously, with its own barriers) inside get_accumulation. No
            # watcher subprocess and no task-count rendezvous exists on this path;
            # just a final cuda sync + barrier before the optimizer reads acc.
            torch.cuda.synchronize()
            torch.distributed.barrier(group=pg)
            self.dispatched_tasks = 0
            return
        if _settle_defer():
            # ODC_SETTLE_DEFER: single aggregate join of every deferred per-peer
            # settle stream (replaces the per-layer wait_stream we skipped) so all
            # gradient accumulations are landed before the optimizer reads them.
            with torch.cuda.nvtx.range("settle_defer_join"):
                for stream in list(self.rank_streams.values()):
                    torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.current_stream().wait_stream(get_comm_stream())
                torch.cuda.current_stream().synchronize()
        dispatched_task_list = [None for _ in range(dist.get_world_size(pg))]
        with torch.cuda.nvtx.range("task_all_gather"):
            torch.distributed.all_gather_object(dispatched_task_list, self.dispatched_tasks, group=pg)
        target = sum(dispatched_task_list)

        watchdog = _settle_watchdog_sec() if _settle_defer() else 0.0
        if watchdog > 0:
            # Fail-fast rendezvous: a missing watcher convergence must raise
            # instead of spinning the GPU forever.
            cmd_queue, response_queue = self.reduction_watcher
            cmd_queue.put(("wait_and_reset_task_count", target))
            try:
                response_queue.get(timeout=watchdog)
            except _queue.Empty:
                raise RuntimeError(
                    f"[ODC_SETTLE_DEFER] watcher rendezvous timed out after "
                    f"{watchdog}s (target={target}); failing fast to avoid GPU hang"
                )
        else:
            call_watcher(self.reduction_watcher, "wait_and_reset_task_count", target)
        self.dispatched_tasks = 0

    def stop(self):
        if self.reduction_watcher is not None:
            call_watcher(self.reduction_watcher, "stop")
            torch.distributed.barrier()
            torch.cuda.synchronize()
