###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Explicit inference communication model (feature B: custom collective ops).

The training profiler folds tensor-parallel AllReduce and expert-parallel
AllToAll *into* the per-layer forward time.  For serving we want comm to be a
first-class, reportable quantity so users can:

  * see a per-phase **communication breakdown** (TP AllReduce, EP AllToAll,
    PP send/recv, and KV-cache transfer for disaggregation),
  * **force a collective algorithm** (ring / one-shot / two-shot /
    hierarchical) instead of the auto-selected fastest,
  * **overlap** comm with compute (a fraction hidden behind GEMMs), and
  * apply a **custom fused-op** speedup (e.g. AllReduce+RMSNorm fusion or
    DeepEP-style overlapped dispatch/combine).

All times are returned in **milliseconds**.  At default knob values this model
reproduces the implicit cost computed inside ``transformer_layer`` so totals
stay consistent; the delta only appears when a knob is changed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from primus.core.projection.module_profilers import collective_model as cm
from primus.core.projection.module_profilers.collective_args import get_default_args
from primus.core.projection.training_config import (
    InferenceCollectiveConfig,
    ModelConfig,
    ModelParallelConfig,
)

_PROTOCOLS = ("simple", "ll", "ll64", "ll128")


def deepep_overlap_efficiency(model_config) -> float:
    """Fraction of EP All-to-All hidden behind compute under DeepEP/SyncFree.

    DeepEP issues the dispatch/combine asynchronously so the All-to-All
    overlaps with grouped-GEMM expert compute. SyncFree stages progressively
    remove CPU sync stalls and push the overlap higher. Returns ``0.0`` when
    neither ``use_turbo_deepep`` nor ``turbo_sync_free_moe_stage`` is set.

    The efficiency ladder mirrors the training projection's
    ``_get_deepep_overlap_efficiency`` so serving and training agree.
    """
    sync_free_stage = getattr(model_config, "turbo_sync_free_moe_stage", 0) or 0
    if not (getattr(model_config, "use_turbo_deepep", False) or sync_free_stage > 0):
        return 0.0
    if sync_free_stage >= 3:
        return 0.85
    if sync_free_stage >= 2:
        return 0.80
    if sync_free_stage >= 1:
        return 0.75
    return 0.65


def _min_over_protocols(fn, args, msg_size, gpus, groups) -> float:
    """Run a low-level collective ``fn`` over every protocol, return the min (us)."""
    best = float("inf")
    for p in _PROTOCOLS:
        try:
            t = fn(args, msg_size, gpus, groups=groups, protocol=p)
        except TypeError:
            # Some helpers don't take ``groups`` (e.g. single_shot_*).
            t = fn(args, msg_size, gpus, protocol=p)
        if t < best:
            best = t
    return best


def _allreduce_overhead_us(args, msg_size, gpus) -> float:
    """Fixed AllReduce overhead matching ``collective_model.allreduce``.

    Forced-algorithm paths must add the same RCCL setup + NIC warmup cost the
    auto-selected path adds, otherwise *choosing* an algorithm spuriously
    appears faster than ``auto`` (which is the min over algorithms + overhead).
    """
    overhead = getattr(args, "rccl_overhead_us", 0.0)
    if gpus > args.node_size:
        nics = getattr(args, "nics_per_node", None) or args.node_size
        num_nodes = gpus // args.node_size
        node_steps = max(1, int(np.ceil(np.log2(num_nodes))))
        per_nic_bytes = msg_size / (nics * node_steps)
        warmup = getattr(args, "nic_warmup_bytes", 32 * 1024 * 1024)
        ratio = min(1.0, per_nic_bytes / warmup) if warmup > 0 else 1.0
        if ratio < 1.0:
            setup_us = getattr(args, "nic_rdma_setup_us", 0.0) * node_steps
            overhead += setup_us * (1.0 - ratio ** 2.5)
    return overhead


def _alltoall_overhead_us(args, gpus) -> float:
    """Fixed AllToAll overhead matching ``collective_model.alltoall``."""
    gpus_per_node = args.node_size
    intra_node_peers = min(gpus - 1, gpus_per_node - 1)
    inter_node_peers = max(0, gpus - gpus_per_node)
    intra_sync = getattr(args, "a2a_intra_sync_overhead", 50.0)
    intra_per_peer = getattr(args, "a2a_intra_node_peer_lat", 2.5)
    intra_overhead = (intra_sync + intra_per_peer * intra_node_peers) if intra_node_peers > 0 else 0.0
    inter_per_peer = getattr(args, "a2a_peer_lat", 0.45)
    inter_overhead = inter_per_peer * inter_node_peers
    return intra_overhead + inter_overhead + getattr(args, "a2a_rccl_overhead_us", 0.0)


@dataclass
class CommBreakdown:
    """Per-forward communication time (ms), split by collective op."""

    tp_allreduce_ms: float = 0.0
    ep_a2a_ms: float = 0.0
    pp_p2p_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.tp_allreduce_ms + self.ep_a2a_ms + self.pp_p2p_ms


class InferenceCollectiveModel:
    """Explicit per-forward communication model for inference."""

    def __init__(
        self,
        model_config: ModelConfig,
        mp_config: ModelParallelConfig,
        coll_config: InferenceCollectiveConfig,
        *,
        num_nodes: int | None = None,
        gpus_per_node: int | None = None,
    ):
        self.mc = model_config
        self.mp = mp_config
        self.cc = coll_config

        self.tp = max(1, mp_config.tensor_model_parallel_size)
        self.pp = max(1, mp_config.pipeline_model_parallel_size)
        self.ep = max(1, getattr(mp_config, "expert_model_parallel_size", 1) or 1)
        self.cp = max(1, getattr(mp_config, "context_model_parallel_size", 1) or 1)
        self.hidden = model_config.hidden_size
        self.topk = getattr(model_config, "moe_router_topk", 2) or 2

        # DeepEP / SyncFree overlap the EP All-to-All behind expert compute;
        # the exposed A2A cost is scaled down by this fraction (0 = disabled).
        self._deepep_overlap = deepep_overlap_efficiency(model_config)

        gpn = gpus_per_node if gpus_per_node else int(os.environ.get("GPUS_PER_NODE", "8"))
        nn = num_nodes if num_nodes else int(os.environ.get("NNODES", "1"))
        self._args = get_default_args(
            num_nodes=nn,
            gpus_per_node=gpn,
            tp=self.tp,
            pp=self.pp,
            ep=self.ep,
            cp=self.cp,
            hardware_config=coll_config.hardware_config,
        )

    # -- TP AllReduce ----------------------------------------------------------

    def tp_allreduce_ms(self, batch: int, tokens: int) -> float:
        """2 AllReduces per layer (post-attention + post-MLP), forward only."""
        if self.tp <= 1:
            return 0.0
        msg = max(1, batch * tokens * self.hidden * 2 // self.cp)
        algo = (self.cc.tp_allreduce_algo or "auto").lower()
        if algo == "auto":
            # cm.allreduce already includes the fixed RCCL/NIC overheads.
            us = cm.allreduce(self._args, msg, self.tp, groups=["tp"])
        else:
            if algo == "ring":
                raw = _min_over_protocols(cm.RingAllreduce, self._args, msg, self.tp, ["tp"])
            elif algo == "one_shot":
                raw = _min_over_protocols(cm.single_shot_allreduce, self._args, msg, self.tp, ["tp"])
            elif algo == "two_shot":
                rs = _min_over_protocols(cm.run_reduce_scatter, self._args, msg, self.tp, ["tp"])
                ag = _min_over_protocols(cm.run_allgather, self._args, msg, self.tp, ["tp"])
                raw = rs + ag
            elif algo == "hierarchical":
                raw = _min_over_protocols(cm.hierarchical_allreduce, self._args, msg, self.tp, ["tp"])
            else:
                raw = _min_over_protocols(cm.RingAllreduce, self._args, msg, self.tp, ["tp"])
            # Forced algorithms must carry the same fixed overhead as ``auto``.
            us = raw + _allreduce_overhead_us(self._args, msg, self.tp)
        # 2 AllReduces per layer in forward.
        return 2.0 * (us / 1000.0) * float(self.cc.tp_allreduce_efficiency)

    # -- EP AllToAll -----------------------------------------------------------

    def ep_a2a_ms(self, batch: int, tokens: int) -> float:
        """Dispatch + combine AllToAll per MoE layer, forward only."""
        if self.ep <= 1:
            return 0.0
        msg = max(1, batch * tokens * self.hidden * self.topk * 2)
        algo = (self.cc.ep_a2a_algo or "auto").lower()
        if algo == "auto":
            # cm.alltoall already includes the fixed RCCL/peer overheads.
            one = cm.alltoall(self._args, msg, self.ep, groups=["ep"])
        else:
            if algo == "direct":
                raw = _min_over_protocols(cm.run_alltoall, self._args, msg, self.ep, ["ep"])
            elif algo == "single_shot":
                raw = _min_over_protocols(cm.single_shot_alltoall, self._args, msg, self.ep, ["ep"])
            elif algo == "hierarchical":
                raw = _min_over_protocols(cm.hierarchical_alltoall, self._args, msg, self.ep, ["ep"])
            else:
                raw = _min_over_protocols(cm.run_alltoall, self._args, msg, self.ep, ["ep"])
            # Forced algorithms must carry the same fixed overhead as ``auto``.
            one = raw + _alltoall_overhead_us(self._args, self.ep)
        # dispatch + combine, scaled by the custom-op efficiency and reduced by
        # the DeepEP/SyncFree compute-overlap fraction (exposed A2A only).
        return (
            2.0
            * (one / 1000.0)
            * float(self.cc.ep_a2a_efficiency)
            * (1.0 - self._deepep_overlap)
        )

    # -- Pipeline P2P ----------------------------------------------------------

    def pp_p2p_ms(self, batch: int, tokens: int) -> float:
        """Activation send/recv across (pp-1) stage boundaries per forward."""
        if self.pp <= 1 or not self.cc.include_pp_p2p:
            return 0.0
        msg = max(1, batch * tokens * self.hidden * 2 // self.cp)
        us = cm.sendrecv(self._args, msg)
        return (self.pp - 1) * (us / 1000.0)

    # -- KV-cache transfer (disaggregation) ------------------------------------

    def kv_transfer_ms(self, kv_bytes: float, *, bw_gbps: float | None = None, latency_us: float = 0.0) -> float:
        """Time to move ``kv_bytes`` of KV cache prefill→decode worker.

        Uses the inter-node (pod) bandwidth from the collective args unless an
        explicit ``bw_gbps`` override is supplied.  ``bw`` is GB/s, so bytes
        are converted to GB; result in ms.
        """
        if kv_bytes <= 0:
            return 0.0
        bw = bw_gbps if (bw_gbps and bw_gbps > 0) else self._args.pod_bw
        bw = max(bw, 1e-6)
        gb = kv_bytes / (1024.0 ** 3)
        return (gb / bw) * 1000.0 + latency_us / 1000.0

    # -- combined per-layer ----------------------------------------------------

    def layer_comm_ms(self, batch: int, tokens: int, *, is_moe: bool) -> CommBreakdown:
        """Per-*layer* comm (TP AllReduce always; EP A2A only on MoE layers)."""
        return CommBreakdown(
            tp_allreduce_ms=self.tp_allreduce_ms(batch, tokens),
            ep_a2a_ms=self.ep_a2a_ms(batch, tokens) if is_moe else 0.0,
            pp_p2p_ms=0.0,
        )
