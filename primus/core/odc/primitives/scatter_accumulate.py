# Adapted from ODC (https://github.com/sail-sg/odc), which is distributed under
# the MIT License per its package metadata (pyproject.toml / setup.py
# classifiers). The upstream repository ships no LICENSE file or per-file
# copyright headers; upstream copyright is held by the ODC authors (Sea AI Lab).
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# See LICENSE for license information.

import logging
import os

import torch
import torch.distributed as dist

from odc.primitives.utils import (
    SymmBufferRegistry,
    get_comm_stream,
    get_local_world_size,
)

logger = logging.getLogger(__name__)

from odc.primitives.utils import _USE_ROCSHMEM  # noqa: E402

if _USE_ROCSHMEM:
    from odc.primitives import _rocshmem_backend as _rs
else:
    _rs = None


def _gda_active():
    return _rs is not None and _rs.gda_enabled()


def _official_push():
    """ODC_OFFICIAL_PUSH=1 -> reproduce the reference single-sided push on the
    GDA path: skip our per-call ``_rs.barrier()``, the strided/full warm-up
    settle, the overlap side-stream and the DEFER block, so the collective
    reduce-scatter runs bare (the closest reproduction of the upstream
    no-settle behaviour). Default ("0") preserves the current GDA behaviour.
    """
    return os.environ.get("ODC_OFFICIAL_PUSH", "0") == "1"


class ReductionService:
    """Reduce-scatter-accumulate service (device-side, no host-polling subprocess).

    Two paths, both device-side and free of any host-polling subprocess:
      * single node  -> owner-side XGMI PULL + on-chip fp32 sum
        (``_single_device_scatter_accumulate``).
      * multi node    -> rocSHMEM GPU-direct (GDA) reduce-scatter
        (``_gda_scatter_accumulate``).

    Both DEFER the collective reduce to once per param-group at
    ``get_accumulation`` (matched barrier count across ranks -> deadlock-free
    under variable-length / nopad micro-batching), pre-accumulating each
    micro-batch's grad locally in fp32.
    """

    def __init__(self, accumulation_dtype=None):
        self.accumulations = []
        self.accumulation_indices = {}
        self.input_buffer = {}
        self.dispatched_tasks = 0
        # Accepted for API compatibility with the FSDP1 caller; the device and
        # GDA reduce paths always accumulate in fp32.
        self.accumulation_dtype = accumulation_dtype

    def clear_accumulations(self):
        for acc in self.accumulations:
            acc.fill_(0)
        if hasattr(self, "_gda_deferred"):  # reset per-minibatch deferred grads
            self._gda_deferred = {}
            self._gda_deferred_pg = {}
        if hasattr(self, "_sdr_deferred"):  # reset per-minibatch deferred grads
            self._sdr_deferred = {}
            self._sdr_deferred_pg = {}

    def _gda_scatter_accumulate(self, key, input_tensor, pg: dist.ProcessGroup):
        """GPU-direct pull-based reduce-scatter accumulate (race-free, no host-polling subprocess).

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
        """Single-node device-side reduce-scatter accumulate (no host-polling subprocess).

        Mechanism (owner-side PULL + on-chip fp32 sum over same-node XGMI peer
        views): each PE stages its (locally pre-accumulated) full grad into a
        symmetric fp32 buffer whose same-node peer views are already resolved on
        allocation (the same XGMI peer-view machinery gather.py uses); after a
        rendezvous barrier, PE r reads every same-node peer's segment destined
        for its shard (a plain XGMI ``.copy_`` peer read) and sums it on-chip.
        There are NO cross-rank writes -> no device atomics -> no MI300X
        write-visibility hazard; only XGMI reads gated by a barrier. No second
        process -> no IPC handle and no host-side reduction subprocess.

        ``input_tensor`` is this PE's full (locally pre-accumulated) grad, laid
        out as ``[shard_0 | shard_1 | ... | shard_{gws-1}]``; PE r owns output
        shard r.

        Steps (all same-node, no subprocess, no IPC handle, no atomics):
          1. Stage my full grad into a symmetric fp32 buffer (peer views resolved
             on allocation).
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
        if _gda_active():
            official = _official_push()
            # DEFER the cross-node reduce to once-per-minibatch. The per-call
            # _gda_scatter_accumulate does rocshmem_barrier_all (collective);
            # calling it per-microbatch deadlocks under nopad (ranks have
            # different micro-batch counts -> mismatched barrier counts). Fix:
            # accumulate the unsharded grad LOCALLY here (no comm/barrier ->
            # backward is lockstep-free), then do ONE barriered reduce-scatter
            # per group at get_accumulation (count == #groups, matched across
            # ranks). Multi-node DEFAULT = defer; single-node (n_pes <=
            # local_world_size) keeps the per-call path. ODC_OFFICIAL_PUSH skips
            # DEFER (official does no local pre-accumulation). Env var overrides.
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
        # Single-node device-side reduce (replaces the removed host-side
        # reduction subprocess path). Cross-node reduce always goes through the
        # rocSHMEM GDA path above.
        assert torch.distributed.get_world_size(pg) == get_local_world_size(), (
            f"non-GDA reduce-scatter is single-node only "
            f"(gws={torch.distributed.get_world_size(pg)} lws={get_local_world_size()}); "
            f"multi-node requires the rocSHMEM GDA backend"
        )
        # DEFER exactly like the multi-node GDA path: pre-accumulate this
        # micro-batch's grad LOCALLY (no collective -> nopad-safe), then run ONE
        # barriered owner-side pull-sum per group at get_accumulation.
        # Pre-accumulate in fp32 so cross-micro-batch accumulation matches the
        # fp32 device accumulator.
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

    def get_accumulation(self, key):
        # GDA DEFER: run the single per-minibatch cross-node reduce-scatter now
        # (once per group, matched barrier count across ranks -> deadlock-free).
        if _gda_active() and getattr(self, "_gda_deferred", None) is not None:
            pending = self._gda_deferred.get(key)
            if pending is not None:
                self._gda_deferred[key] = None
                self._gda_scatter_accumulate(key, pending, self._gda_deferred_pg[key])
        # Single-node device DEFER: run the single per-minibatch owner-side
        # pull-sum now (once per group, matched barrier count across ranks).
        if not _gda_active() and getattr(self, "_sdr_deferred", None) is not None:
            pending = self._sdr_deferred.get(key)
            if pending is not None:
                self._sdr_deferred[key] = None
                self._single_device_scatter_accumulate(key, pending, self._sdr_deferred_pg[key])
        acc = self.accumulations[self.accumulation_indices[key]]
        return acc

    def sync(self, pg: dist.ProcessGroup):
        if _gda_active():
            # OVERLAP: collect any in-flight side-stream reduce-scatters before
            # the optimizer reads acc. ODC_OFFICIAL_PUSH forces synchronous RS
            # (no side stream), so nothing to collect here.
            if os.environ.get("ODC_GDA_OVERLAP", "0") == "1" and not _official_push():
                _rs.gda_rs_overlap_sync()
            # GDA path is fully synchronous (device kernels + barriers).
            torch.cuda.synchronize()
            torch.distributed.barrier(group=pg)
            self.dispatched_tasks = 0
            return
        # Single-node device path: the owner-side pull-sum already ran
        # (synchronously, with its own barriers) inside get_accumulation. Just a
        # final cuda sync + barrier before the optimizer reads acc.
        torch.cuda.synchronize()
        torch.distributed.barrier(group=pg)
        self.dispatched_tasks = 0

    def stop(self):
        # The device and GDA reduce paths run no host-polling subprocess, so there is
        # nothing to tear down. Kept for API compatibility with the callers.
        pass
