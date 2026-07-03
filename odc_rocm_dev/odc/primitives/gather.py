import logging
import math

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from odc.primitives import (
    NVSHMEM_EXTERN_LIBS,
    __syncthreads,
    getmem_nbi_block,
    quiet,
    tid,
)
from odc.primitives.utils import (
    BufferSplitter,
    SymmBufferRegistry,
    get_comm_stream,
    get_local_world_size,
    sync_cta,
)
from torch import Tensor

logger = logging.getLogger(__name__)

# rocSHMEM RO (host-driven) cross-node backend. When active, the cross-node
# all-gather pulls each remote peer's input shard with a blocking host
# ``rs_getmem`` from the CPU (no device-initiated RDMA -> avoids the MORI /
# NVSHMEM device-completion hang on this fabric). Same-node peers still use the
# XGMI peer-tensor copy. The mori / nvshmem device kernel path is untouched.
from odc.primitives.utils import _USE_ROCSHMEM  # noqa: E402

if _USE_ROCSHMEM:
    from odc.primitives import _rocshmem_backend as _rs
else:
    _rs = None


def _ro_active():
    return _rs is not None and _rs.ro_enabled()


def _gda_active():
    return _rs is not None and _rs.gda_enabled()


def _official_push():
    """ODC_OFFICIAL_PUSH=1 -> use upstream NVSHMEM ODC's single-sided device-get
    all-gather and skip our host-driven RO get + the GDA gather barrier/async
    extras. Default ("0") keeps current behaviour. Env-gated; nothing deleted.
    (Gather is read-only on stable params, so this is the lower-risk side; the
    primary alignment target is scatter-accumulate.)"""
    import os as _os

    return _os.environ.get("ODC_OFFICIAL_PUSH", "0") == "1"


@triton.jit
def nvshmem_device_producer_gather_2d_get_block_kernel_chunked_synced(
    remote_tensor_ptr,
    target_tensor_ptr,
    elem_per_rank,
    size_per_elem,
    rank,
    num_ranks_per_node,
    world_size,
    chunk_size,
    signal_ptr,
):
    pid = tl.program_id(axis=0)
    # np = tl.num_programs(axis=0)
    assert num_ranks_per_node == tl.num_programs(axis=0)
    np = num_ranks_per_node
    num_nodes = world_size // np

    tidx = tid(axis=0)
    expected = 0
    for i in range(1, num_nodes):
        peer_node = (i + rank // np) % num_nodes
        peer = (pid + peer_node * np) % world_size
        # chunk_size = elem_per_rank // num_chunks
        num_chunks = tl.cdiv(elem_per_rank, chunk_size)
        for chunk in range(num_chunks):
            this_chunk_size = chunk_size
            if chunk == num_chunks - 1:
                this_chunk_size = elem_per_rank - chunk * chunk_size
            getmem_nbi_block(
                target_tensor_ptr + peer * elem_per_rank + (chunk * chunk_size),
                remote_tensor_ptr + (chunk * chunk_size),
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


class GatherService:
    def __init__(self):
        self.shaped_buffer = {}
        self.buffer_splitter = BufferSplitter()
        self.chunk_size_bytes = 2**20

    def get_chunk_size(self, buffer_dtype):
        return self.chunk_size_bytes // buffer_dtype.itemsize

    def gather_into_tensor(self, output_tensor: Tensor, input_tensor: Tensor, pg: dist.ProcessGroup):
        buf_size = self.buffer_splitter.get_global_buffer_size(output_tensor.shape)
        buffer_shape = (buf_size,)
        output_size = output_tensor.numel()
        assert output_size >= buf_size, f"output_size: {output_size} < buf_size: {buf_size}"

        rank = torch.distributed.get_rank()
        if (buffer_shape, output_tensor.dtype) not in self.shaped_buffer:
            logger.info(
                f"Rank {rank} create buffer: output_size: {output_size} num_sub_buffers: {math.ceil(output_size / buf_size)} buf_size: {buf_size}"
            )
            self.shaped_buffer[
                (buffer_shape, output_tensor.dtype)
            ] = SymmBufferRegistry.get_instance().allocate_symm_buffer(
                f"ag_buffer_{buffer_shape}_{output_tensor.dtype}",
                buffer_shape,
                output_tensor.dtype,
            )
        target_tensor = self.shaped_buffer[(buffer_shape, output_tensor.dtype)]

        assert (input_tensor.numel() * input_tensor.element_size()) % (
            2**6
        ) == 0 or input_tensor.numel() < 2**6, "better align to 64 for efficiency"
        chunk_size = self.get_chunk_size(input_tensor.dtype)
        # assert input_tensor.numel() % chunk_size == 0

        registry = SymmBufferRegistry.get_instance()
        peer_tensors = registry.get_peer_tensors(input_tensor)

        group_world_size = torch.distributed.get_world_size(pg)
        local_world_size = get_local_world_size()
        assert group_world_size in (
            torch.distributed.get_world_size(),
            local_world_size,
        ), f"{group_world_size=} {torch.distributed.get_world_size()=} {local_world_size=}"

        get_comm_stream().wait_stream(torch.cuda.current_stream())
        # GPU-direct (GDA): IPC is off, so there are no XGMI peer views; ALL ranks
        # (incl. same-node and self) are pulled via device rocshmem_getmem. Handle
        # the whole cross-group gather here and return.
        gda = _gda_active() and local_world_size != group_world_size
        if gda:
            self._gda_gather_into_tensor(
                output_tensor,
                input_tensor,
                target_tensor,
                buf_size,
                group_world_size,
                rank,
            )
            return

        with torch.cuda.stream(get_comm_stream()):
            output_tensor_split = output_tensor.view(group_world_size, -1)
            assert local_world_size == len(peer_tensors)
            local_rank = rank % local_world_size
            rank_same_node_start = rank - local_rank
            rank_same_node_end = rank_same_node_start + local_world_size
            for r_offset in range(local_world_size):
                src_local_rank = (local_rank + r_offset) % local_world_size
                if group_world_size == local_world_size:
                    output_tensor_split[src_local_rank].copy_(peer_tensors[src_local_rank])
                else:
                    src_rank = rank_same_node_start + src_local_rank
                    output_tensor_split[src_rank].copy_(peer_tensors[src_local_rank])

            assert buf_size % group_world_size == 0
            local_buf_size = buf_size // group_world_size
            signal_ptr = torch.empty(1, dtype=torch.int32, device="cuda")
            # ODC_OFFICIAL_PUSH: bypass the host-driven RO get (and its barrier) and
            # use the upstream single-sided device-get kernel instead.
            ro = _ro_active() and local_world_size != group_world_size and not _official_push()
            import os as _os
            import time as _t

            _prof = ro and _os.environ.get("ODC_RO_PROFILE", "0") == "1"
            _g_get = _g_quiet = _g_bar = 0.0
            if ro:
                # Cross-node host-driven RO gather: every PE must have its input
                # shard resident on the symmetric heap and be at the same point
                # before any peer pulls it. rs_barrier() (rocshmem_barrier_all)
                # provides the cross-PE rendezvous + completes pending ops.
                _tb = _t.perf_counter()
                torch.cuda.synchronize()
                _rs.barrier()
                _g_bar = _t.perf_counter() - _tb
            for start in range(0, input_tensor.numel(), local_buf_size):
                if local_world_size == group_world_size:
                    continue
                size = min(local_buf_size, input_tensor.numel() - start)
                sub_input_tensor = input_tensor.view(-1)[start : start + size]
                assert (sub_input_tensor.numel() * sub_input_tensor.element_size()) % (
                    2**6
                ) == 0 or sub_input_tensor.numel() < 2**6, "better align to 64 for efficiency"
                target_buf_size = size * group_world_size
                assert target_buf_size <= buf_size
                target_tensor_split = target_tensor[:target_buf_size].view(group_world_size, size)

                if ro:
                    # Host-driven RO: for each cross-node peer r, pull r's input
                    # piece [start:start+size] (at the symmetric address of MY
                    # sub_input_tensor, which maps to r's matching shard) into
                    # target_tensor_split[r]. Same-node ranks were already copied
                    # via XGMI peer-tensors above.
                    src_ptr = sub_input_tensor.data_ptr()
                    nbytes = sub_input_tensor.numel() * sub_input_tensor.element_size()
                    _tg = _t.perf_counter()
                    for r in range(group_world_size):
                        if rank_same_node_start <= r < rank_same_node_end:
                            continue
                        _rs.getmem(target_tensor_split[r].data_ptr(), src_ptr, nbytes, r)
                    _g_get += _t.perf_counter() - _tg
                    _tq = _t.perf_counter()
                    _rs.quiet()
                    torch.cuda.synchronize()
                    _g_quiet += _t.perf_counter() - _tq
                else:
                    signal_ptr.fill_(0)
                    assert group_world_size % 8 == 0 or group_world_size < 8
                    # grid_size = 8 if world_size == 32 else world_size
                    grid_size = local_world_size
                    nvshmem_device_producer_gather_2d_get_block_kernel_chunked_synced[(grid_size,)](
                        remote_tensor_ptr=sub_input_tensor,
                        target_tensor_ptr=target_tensor_split.view(-1),
                        elem_per_rank=sub_input_tensor.numel(),
                        size_per_elem=sub_input_tensor.element_size(),
                        rank=rank,
                        num_ranks_per_node=local_world_size,
                        world_size=group_world_size,
                        chunk_size=chunk_size,
                        signal_ptr=signal_ptr,
                        num_warps=32,
                        extern_libs=NVSHMEM_EXTERN_LIBS,
                    )
                if buf_size == output_size:
                    local_world_data_size = size * local_world_size
                    local_world_idx = rank // local_world_size
                    data_start_idx = local_world_data_size * local_world_idx
                    data_end_idx = data_start_idx + local_world_data_size
                    output_tensor[:data_start_idx].copy_(target_tensor[:data_start_idx])
                    output_tensor[data_end_idx:].copy_(target_tensor[data_end_idx:])
                    # output_tensor.copy_(target_tensor)
                else:
                    for r in range(group_world_size):
                        if rank_same_node_start <= r < rank_same_node_end:
                            continue
                        output_tensor_split[r, start : start + size].copy_(target_tensor_split[r, :])
        torch.cuda.current_stream().wait_stream(get_comm_stream())
        if _prof and rank == 0:
            logger.info(
                "[RO-PROF gather] in_numel=%d ws=%d | barrier=%.3fs getmem=%.3fs quiet=%.3fs",
                input_tensor.numel(),
                group_world_size,
                _g_bar,
                _g_get,
                _g_quiet,
            )

    def _gda_gather_into_tensor(
        self, output_tensor, input_tensor, target_tensor, buf_size, group_world_size, rank
    ):
        """GPU-direct all-gather: every rank's shard (incl same-node and self) is
        pulled via a single device rocshmem_getmem kernel into target slot[r],
        then copied into output_tensor_split[r]. No XGMI peer views / host loop."""
        import os as _os
        import time as _t

        _prof = _os.environ.get("ODC_GDA_PROFILE", "0") == "1"
        _g_bar = _g_get = _g_copy = 0.0
        es = input_tensor.element_size()
        assert buf_size % group_world_size == 0
        local_buf_size = buf_size // group_world_size
        output_split = output_tensor.view(group_world_size, -1)
        peers = list(range(group_world_size))
        # 方案1 (ODC_GDA_GATHER_ASYNC=1): launch the all-gather kernel on the comm
        # stream WITHOUT a host hipDeviceSynchronize, so FSDP2's prefetch (it issues
        # layer L+1's all-gather during layer L compute, async_op=True) actually
        # overlaps. Gather reads STABLE params (written last step, read-only) -> no
        # settle/barrier needed; correctness via stream ordering: the reassembly
        # copies run on the SAME stream after the gather, and the consumer waits the
        # comm stream (current_stream().wait_stream + FSDP async_op event).
        # ODC_OFFICIAL_PUSH forces the synchronous single-sided device get (no
        # prefetch side-stream); the gather barrier below also stays off.
        _gather_async = _os.environ.get("ODC_GDA_GATHER_ASYNC", "0") == "1" and not _official_push()
        if _gather_async:
            # Phase 2 fix: run the async gather on a DEDICATED stream (not the
            # SHARED get_comm_stream, which scatter-staging etc. also use -> the
            # original gather-async race: a prefetched gather / other comm-stream
            # op overwrote the shared `target` scratch mid-use). A dedicated stream
            # keeps gather+reassembly serial & isolated; the consumer is ordered via
            # current_stream().wait_stream below (+ FSDP async_op event). No host sync.
            if not hasattr(self, "_gda_gather_stream") or self._gda_gather_stream is None:
                self._gda_gather_stream = torch.cuda.Stream()
            comm = self._gda_gather_stream
        else:
            comm = get_comm_stream()
        comm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(comm):
            _tb = _t.perf_counter()
            if not _gather_async:
                torch.cuda.synchronize()
            # Opt2: NO per-gather rocshmem_barrier_all here. Gather reads PARAMS
            # (symmetric shard written by the optimizer a full step earlier and
            # read-only through fwd/bwd) -> peers' data is already stable/visible,
            # so the cross-PE rendezvous is unnecessary (unlike scatter's
            # just-written staging). Saves ~2.6s/iter. (Gated revert: set
            # ODC_GDA_GATHER_BARRIER=1 to restore the barrier.)
            if _os.environ.get("ODC_GDA_GATHER_BARRIER", "0") == "1" and not _official_push():
                _rs.barrier()
            _g_bar += _t.perf_counter() - _tb
            for start in range(0, input_tensor.numel(), local_buf_size):
                size = min(local_buf_size, input_tensor.numel() - start)
                sub_input = input_tensor.view(-1)[start : start + size]
                tb = size * group_world_size
                tsplit = target_tensor[:tb].view(group_world_size, size)
                _tg = _t.perf_counter()
                if _gather_async:
                    # launch on comm stream, NO sync -> overlaps with compute
                    _rs.gda_gather_async(
                        target_tensor.data_ptr(),
                        sub_input.data_ptr(),
                        size * es,
                        peers,
                        size * es,
                        comm.cuda_stream,
                    )
                else:
                    _rs.gda_gather(
                        target_tensor.data_ptr(), sub_input.data_ptr(), size * es, peers, size * es
                    )
                    torch.cuda.synchronize()
                _g_get += _t.perf_counter() - _tg
                _tc = _t.perf_counter()
                # reassembly on the SAME comm stream -> ordered AFTER the gather kernel
                for r in range(group_world_size):
                    output_split[r, start : start + size].copy_(tsplit[r])
                _g_copy += _t.perf_counter() - _tc
        torch.cuda.current_stream().wait_stream(comm)
        if _prof and rank == 0:
            logger.warning(
                "[GDA-PROF gather] in_numel=%d barrier=%.3f gda_gather=%.3f reassembly=%.3f",
                input_tensor.numel(),
                _g_bar,
                _g_get,
                _g_copy,
            )
