###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Memory projection — analytical extrapolation primitives.

Given a bench measurement at a small (sub-)node and analytical models of
the bench and target configurations, project per-rank peak memory at the
target cluster.  The result is split into a *point estimate* and an
*upper bound* (for OOM-fits decisions).

Decomposition::

    target_peak ≈ Σ analytical_components(target_cfg)
                + framework_overhead          # bench-measured invariant
                + live_tensor_excess          # bench-measured live-tensor under-count
                + safety_margin × (framework_overhead + live_tensor_excess)   (upper bound only)

    framework_overhead     = max(0, bench_peak_reserved
                                    − bench_peak_allocated
                                    − analytical_bench.comm_buffers)
    live_tensor_excess     = max(0, bench_peak_allocated − Σ analytical_components(bench_cfg))

Anything that scales with cluster shape (parameters, gradients,
distributed-optimizer state, per-layer activations, PP/VPP/GA factors,
DeepEP buffers, rough O(N) communicator cost) lives on the *analytical*
side of the equation, evaluated at the target config.

The residual is split into two well-defined, measurable terms:

* ``framework_overhead`` captures the gap between reserved and live VRAM
  at the bench peak — allocator pages, NCCL/RCCL buffers, kernel
  workspaces, autograd graphs.  Largely invariant in bench↔target
  scaling.  The analytical comm-buffer baseline at the bench world size
  is subtracted to avoid double-counting it against the target comm
  estimate.
* ``live_tensor_excess`` captures any live-tensor memory the analytical
  model under-counted at the bench scope.  Clamps to zero when the
  analytical model already over-predicts at bench (which it commonly
  does at ``num_layers=1`` due to embedding/output replication).

This is more robust than the older single-term ``bench_peak −
analytical_at_bench``, which collapsed to zero whenever the analytical
model was conservative at the small bench scope, hiding the framework
overhead signal entirely.

Why decompose rather than scale a single bench number:
    A 1-GPU bench has TP=PP=EP=1; DeepEP/communicator/grad-buffer costs
    are zero.  Scaling the bench peak directly to a 256-GPU target would
    omit those terms entirely, guaranteeing under-prediction.  Pulling
    them out and re-adding them analytically at target keeps the math
    honest even at extreme bench/target ratios.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StaticBreakdown:
    """Per-rank, per-config static (non-activation) memory in bytes."""

    params_bytes: int = 0
    grads_bytes: int = 0
    optimizer_bytes: int = 0

    @property
    def total(self) -> int:
        return self.params_bytes + self.grads_bytes + self.optimizer_bytes


@dataclass
class ActivationBreakdown:
    """Per-rank activation memory in bytes (already includes PP/VPP/GA)."""

    transformer_layers_bytes: int = 0
    embedding_bytes: int = 0
    output_bytes: int = 0
    loss_bytes: int = 0

    @property
    def total(self) -> int:
        return self.transformer_layers_bytes + self.embedding_bytes + self.output_bytes + self.loss_bytes


@dataclass
class AnalyticalBreakdown:
    """Analytical components of the per-rank memory budget at one config."""

    static: StaticBreakdown = field(default_factory=StaticBreakdown)
    activations: ActivationBreakdown = field(default_factory=ActivationBreakdown)
    deepep_buffers_bytes: int = 0
    comm_buffers_bytes: int = 0

    @property
    def total(self) -> int:
        return (
            self.static.total + self.activations.total + self.deepep_buffers_bytes + self.comm_buffers_bytes
        )


@dataclass
class BenchMeasurement:
    """What we measured at the bench config (rank 0)."""

    global_peak_allocated_bytes: int = 0
    global_peak_reserved_bytes: int = 0
    # Per-layer activation memory (bytes), keyed by layer type ("dense", "moe").
    # Comes from profiling_results[layer_idx]["activation_memory_bytes"].
    per_layer_activation_bytes: Dict[str, int] = field(default_factory=dict)
    # Per-component activation at bench (embedding / output if measured).
    embedding_activation_bytes: int = 0
    output_activation_bytes: int = 0


@dataclass
class PerRankProjection:
    """Projected per-rank peak memory at the target cluster."""

    point_estimate_bytes: int
    upper_bound_bytes: int
    breakdown: Dict[str, Any]
    diagnostics: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Env-var override helper (so we can ask a profiler for "what would rank 0
# see in a target config that needs N nodes" without actually being on that
# rank).
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def env_overrides(**overrides: Any) -> Iterator[None]:
    """Temporarily set env vars; restore on exit.

    ``LanguageModelProfiler.get_dp_size`` and ``estimated_activation_memory``
    read ``RANK``, ``NNODES``, and ``GPUS_PER_NODE`` from the environment.
    This context manager lets us evaluate analytical estimates for an
    arbitrary cluster shape from any process.
    """
    saved: Dict[str, Optional[str]] = {}
    try:
        for key, val in overrides.items():
            saved[key] = os.environ.get(key)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(val)
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


# ─────────────────────────────────────────────────────────────────────────────
# Static memory (params / grads / optimizer) — derived from the existing
# profiler primitives so that distributed-optimizer + ZeRO sharding rules
# are inherited automatically.
# ─────────────────────────────────────────────────────────────────────────────


def static_breakdown_for_profiler(profiler, *, rank: int = 0) -> StaticBreakdown:
    """Compute per-rank static memory (bf16 params + bf16 grads + fp32 optim).

    The profiler's ``get_num_bytes_per_param`` already encodes the convention::

        bytes_per_param = 4 (bf16 params + bf16 grads)
                        + 10 / dp_size (fp32 main params + fp32 mom1 + fp32 mom2)

    We re-decompose that here so the report can show params/grads/optimizer
    individually.  ``estimated_num_params(rank=rank)`` gives the number of
    parameters held by ``rank`` (TP + PP + EP sharded).
    """
    n_params = int(profiler.estimated_num_params(rank=rank))
    dp_size = max(1, int(profiler.get_dp_size()))
    return StaticBreakdown(
        params_bytes=n_params * 2,  # bf16 params held on this rank
        grads_bytes=n_params * 2,  # bf16 grads (Megatron grad buffer is fp32; see below)
        optimizer_bytes=int(n_params * 10 / dp_size),
    )


# Note on grads: Megatron's distributed_optimizer uses an FP32 gradient
# buffer (4 bytes/param/dp_local).  The 4-byte "params + grads" multiplier
# above lumps bf16-params (2) + bf16-grads (2).  When the user runs
# distributed_optimizer (the common case), grads are FP32 and DP-sharded,
# which is roughly 4 bytes / param / dp_size.  Both end up close in
# practice; we keep the existing convention (matches simulate mode) and
# accept a small (~2%) bias here that the residual will absorb.


# ─────────────────────────────────────────────────────────────────────────────
# Activation memory
# ─────────────────────────────────────────────────────────────────────────────


def per_layer_analytical_activation(profiler, layer_type: str, batch_size: int, seq_len: int) -> int:
    """Analytical per-(one-)layer activation in bytes for the given type."""
    key = f"{layer_type}_transformer_layer"
    sub = profiler.sub_profilers.get(key)
    if sub is None:
        return 0
    return int(sub.estimated_activation_memory(batch_size, seq_len))


def total_analytical_activation(profiler, batch_size: int, seq_len: int) -> int:
    """Analytical per-rank activation in bytes (PP/VPP/GA already applied)."""
    return int(profiler.estimated_activation_memory(batch_size, seq_len))


def corrected_activation_bytes(
    *,
    target_profiler,
    bench_profiler,
    bench: BenchMeasurement,
    batch_size: int,
    seq_len: int,
) -> Dict[str, Any]:
    """Apply a per-layer-type empirical correction to the analytical activation.

    For each layer type (dense, moe) we compute::

        correction[type] = measured_per_layer[type] / analytical_per_layer_at_bench[type]

    and use it to scale that type's contribution to the analytical
    activation total at the *target* config.  The correction captures
    real-world overhead (FA workspaces, GroupedGEMM scratch, FP8 amax,
    Turbo grouped-MLP buffers, etc.) that the closed-form formula misses.
    The analytical-at-target structure then handles all the
    cluster-shape-dependent scaling (TP/CP/PP/VPP/GA/recompute).

    Returns a dict::

        {
            "uncorrected_activation_bytes":   <analytical at target>,
            "corrected_activation_bytes":     <after measurement scaling>,
            "correction_factors": {"dense": 1.07, "moe": 1.12},
            "applied_correction": True | False,    # False ⇒ no measurement to anchor on
        }
    """
    uncorrected = total_analytical_activation(target_profiler, batch_size, seq_len)

    if not bench.per_layer_activation_bytes:
        return {
            "uncorrected_activation_bytes": uncorrected,
            "corrected_activation_bytes": uncorrected,
            "correction_factors": {},
            "applied_correction": False,
        }

    # Compute per-layer-type corrections from bench measurements.
    factors: Dict[str, float] = {}
    for layer_type, measured in bench.per_layer_activation_bytes.items():
        analytical_bench = per_layer_analytical_activation(bench_profiler, layer_type, batch_size, seq_len)
        if measured > 0 and analytical_bench > 0:
            factors[layer_type] = measured / analytical_bench

    if not factors:
        return {
            "uncorrected_activation_bytes": uncorrected,
            "corrected_activation_bytes": uncorrected,
            "correction_factors": {},
            "applied_correction": False,
        }

    # Compute the target activation contribution per layer type, weighted
    # by the analytical share that type contributes at target.
    moe_pattern = list(target_profiler.config.model_config.moe_pattern)
    num_dense_target = sum(1 for x in moe_pattern if x == 0)
    num_moe_target = sum(1 for x in moe_pattern if x == 1)

    analytical_dense_per_layer = per_layer_analytical_activation(
        target_profiler, "dense", batch_size, seq_len
    )
    analytical_moe_per_layer = per_layer_analytical_activation(target_profiler, "moe", batch_size, seq_len)

    # Scale per-layer type contributions by the per-type correction factor.
    weighted_correction = num_dense_target * analytical_dense_per_layer * factors.get(
        "dense", 1.0
    ) + num_moe_target * analytical_moe_per_layer * factors.get("moe", 1.0)
    weighted_uncorrected = (
        num_dense_target * analytical_dense_per_layer + num_moe_target * analytical_moe_per_layer
    )

    if weighted_uncorrected <= 0:
        return {
            "uncorrected_activation_bytes": uncorrected,
            "corrected_activation_bytes": uncorrected,
            "correction_factors": factors,
            "applied_correction": False,
        }

    # Apply that ratio to the *transformer-layer* portion of analytical
    # target activation only.  Embedding/output/loss are kept analytical
    # since the bench may not cover them on every rank.
    transformer_correction_ratio = weighted_correction / weighted_uncorrected
    # Approximate the transformer-layer share of the analytical total.
    # The analytical total at target = (Σ per_layer × pp × vpp × ga) + emb + output + loss.
    # We don't have the breakdown without re-deriving the formula, so we
    # apply the correction conservatively to the whole total.  This
    # slightly over-corrects emb/output (they typically contribute < 5%);
    # the residual absorbs the small bias and the upper-bound margin
    # covers it for OOM purposes.
    corrected = int(uncorrected * transformer_correction_ratio)

    return {
        "uncorrected_activation_bytes": uncorrected,
        "corrected_activation_bytes": corrected,
        "correction_factors": factors,
        "applied_correction": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DeepEP / communicator buffer estimates
# ─────────────────────────────────────────────────────────────────────────────


# DeepEP overprovisions the per-rank dispatch buffer to handle worst-case
# routing skew (multiple experts on this rank receiving the same token).
# We use a conservative *upper bound* matching how Primus Turbo's DeepEP
# token dispatcher allocates its receive buffer:
#
#   num_max_dispatch_tokens_per_rank ≈ tokens_per_rank × topk × OVERPROVISION
#
# where ``tokens_per_rank = mbs × seq_len / (TP × CP)``.  A separate
# combine buffer of equal size is allocated for the reverse path.
#
# The factor exists because MoE token routing is stochastic and
# unbalanced (top-k routing has no hard cap on how many tokens go to a
# given expert; multiple of a token's top-k experts can live on the
# same rank), and DeepEP's RDMA/shared-memory buffer is allocated once
# at startup — any iteration whose actual received-token count exceeds
# the buffer crashes the dispatcher.  DeepEP / DeepSeek's reference
# implementation therefore exposes ``num_max_dispatch_tokens_per_rank``
# and recommends sizing it at 1.0–2.5× the average load depending on
# topk and EP (higher topk → wider distribution → bigger factor).
#
# ``_DEEPEP_OVERPROVISION`` is the value this projection uses when
# producing a closed-form *upper-bound* DeepEP footprint for OOM
# prediction.  Setting it to 2.0 mirrors DeepEP's own recommended
# ``num_max_dispatch_tokens_per_rank`` sizing at the conservative end
# of the empirical 1.0–2.5 range — over-predicting is the safe
# direction for OOM detection (predicting "won't fit" when it actually
# fits is preferable to predicting "fits" and then OOMing in
# production).  For high-topk MoE (e.g. DeepSeek V3 with topk=8) the
# real worst case is closer to ~1.5×; 2.0× over-predicts the DeepEP
# footprint slightly there but stays on the safe side of the OOM
# boundary.
_DEEPEP_OVERPROVISION = 2.0
_BF16_BYTES = 2


def estimate_deepep_buffer_bytes(training_config) -> int:
    """Closed-form upper bound on DeepEP per-rank token-buffer footprint.

    Returns 0 when DeepEP/Turbo is disabled in ``training_config``.

    The estimate is intentionally conservative: it caps the dispatch +
    combine buffers at the largest plausible footprint given the
    per-rank token count, top-k routing, and hidden size.  Real DeepEP
    builds may allocate less; over-estimation is the safe direction for
    OOM prediction.
    """
    mp_cfg = getattr(training_config, "model_parallel_config", None)
    mc = getattr(training_config, "model_config", None)
    if mp_cfg is None or mc is None:
        return 0

    use_deepep = bool(getattr(mc, "use_turbo_deepep", False)) or bool(
        getattr(mp_cfg, "use_turbo_deepep", False)
    )
    if not use_deepep:
        return 0

    hidden = int(getattr(mc, "hidden_size", 0) or 0)
    topk = int(getattr(mc, "moe_router_topk", 0) or 0)
    if hidden <= 0 or topk <= 0:
        return 0

    rt = getattr(training_config, "runtime_config", None)
    mbs = int(getattr(rt, "micro_batch_size", 1) or 1) if rt is not None else 1
    seq_len = int(getattr(rt, "sequence_length", 0) or 0) if rt is not None else 0
    tp = int(getattr(mp_cfg, "tensor_model_parallel_size", 1) or 1)
    cp = int(getattr(mp_cfg, "context_model_parallel_size", 1) or 1)
    if seq_len <= 0:
        return 0

    tokens_per_rank = max(1, mbs * seq_len // max(1, tp * cp))
    num_max_dispatch_tokens = int(tokens_per_rank * topk * _DEEPEP_OVERPROVISION)

    # Dispatch + combine buffers.  Each token slot holds `hidden`
    # BF16 elements.  The combine path mirrors the dispatch buffer.
    per_buffer = num_max_dispatch_tokens * hidden * _BF16_BYTES
    return int(2 * per_buffer)


# RCCL/NCCL communicator footprint per rank.  Empirically (MI300/MI355
# RCCL):
#   - ~64-128 MB base allocation per process at small world sizes
#     (channels, bootstrap, peer-mem registrations).
#   - Slow linear growth with world_size: a few MB per peer for ring
#     buffers and connection metadata.
#   - Larger configs can hit 500 MB-1 GB at thousand-GPU scale.
#
# We model this as ``base + slope × world_size`` capped at a hard
# ceiling.  Slightly more aggressive than the original placeholder so we
# don't under-predict OOM at large world sizes.
_COMM_BASE_BYTES = 96 * 1024 * 1024  # 96 MB
_COMM_PER_RANK_BYTES = 2 * 1024 * 1024  # 2 MB / peer
_COMM_CEILING_BYTES = 1 * 1024**3  # 1 GB


def estimate_comm_buffer_bytes(target_world_size: int) -> int:
    """Conservative estimate of NCCL/RCCL communicator footprint per rank."""
    if target_world_size <= 1:
        return 0
    base = _COMM_BASE_BYTES
    grow = _COMM_PER_RANK_BYTES * target_world_size
    return int(min(base + grow, _COMM_CEILING_BYTES))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level extrapolator
# ─────────────────────────────────────────────────────────────────────────────


def compute_analytical_at_config(
    profiler,
    *,
    rank: int,
    nnodes: int,
    gpus_per_node: int,
    batch_size: int,
    seq_len: int,
    training_config,
    target_world_size_for_comm: Optional[int] = None,
) -> AnalyticalBreakdown:
    """Compute the full analytical breakdown for one (rank, cluster) shape.

    Sets the env vars the profiler reads, evaluates each component, and
    restores the environment.
    """
    if target_world_size_for_comm is None:
        target_world_size_for_comm = max(1, nnodes * gpus_per_node)

    with env_overrides(RANK=rank, NNODES=nnodes, GPUS_PER_NODE=gpus_per_node):
        static = static_breakdown_for_profiler(profiler, rank=rank)
        act_total = total_analytical_activation(profiler, batch_size, seq_len)
        deepep = estimate_deepep_buffer_bytes(training_config)
        comm = estimate_comm_buffer_bytes(target_world_size_for_comm)

    return AnalyticalBreakdown(
        static=static,
        activations=ActivationBreakdown(transformer_layers_bytes=int(act_total)),
        deepep_buffers_bytes=int(deepep),
        comm_buffers_bytes=int(comm),
    )


def extrapolate_per_rank_peak(
    *,
    bench_profiler,
    target_profiler,
    bench: BenchMeasurement,
    bench_training_config,
    target_training_config,
    bench_nnodes: int,
    bench_gpus_per_node: int,
    target_nnodes: int,
    target_gpus_per_node: int,
    batch_size: int,
    seq_len: int,
    safety_margin: float = 0.05,
) -> PerRankProjection:
    """Project per-rank peak memory at the target cluster.

    Args:
        bench_profiler / target_profiler: ``LanguageModelProfiler``
            instances built from the bench and target training configs.
        bench: Measured peak + per-layer activation from the bench run.
        bench_training_config / target_training_config: ``TrainingConfig``
            objects providing parallelism + DeepEP flags.
        bench_nnodes / bench_gpus_per_node: Bench cluster shape.
        target_nnodes / target_gpus_per_node: Target cluster shape.
        batch_size / seq_len: Microbatch shape (assumed equal at bench
            and target — the simulate path enforces this today).
        safety_margin: Multiplier applied to residual for the upper bound
            (default 5%).

    Returns:
        :class:`PerRankProjection` with the breakdown for both the point
        estimate and the OOM-relevant upper bound.
    """
    # ── Analytical components at bench and target ──
    bench_world_size = max(1, bench_nnodes * bench_gpus_per_node)
    target_world_size = max(1, target_nnodes * target_gpus_per_node)

    analytical_bench = compute_analytical_at_config(
        bench_profiler,
        rank=0,
        nnodes=bench_nnodes,
        gpus_per_node=bench_gpus_per_node,
        batch_size=batch_size,
        seq_len=seq_len,
        training_config=bench_training_config,
        target_world_size_for_comm=bench_world_size,
    )
    analytical_target = compute_analytical_at_config(
        target_profiler,
        rank=0,
        nnodes=target_nnodes,
        gpus_per_node=target_gpus_per_node,
        batch_size=batch_size,
        seq_len=seq_len,
        training_config=target_training_config,
        target_world_size_for_comm=target_world_size,
    )

    # ── Apply per-layer measurement correction to target activation ──
    with env_overrides(
        RANK=0,
        NNODES=target_nnodes,
        GPUS_PER_NODE=target_gpus_per_node,
    ):
        correction = corrected_activation_bytes(
            target_profiler=target_profiler,
            bench_profiler=bench_profiler,
            bench=bench,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    corrected_target_activation = correction["corrected_activation_bytes"]
    analytical_target_corrected = AnalyticalBreakdown(
        static=analytical_target.static,
        activations=ActivationBreakdown(
            transformer_layers_bytes=int(corrected_target_activation),
        ),
        deepep_buffers_bytes=analytical_target.deepep_buffers_bytes,
        comm_buffers_bytes=analytical_target.comm_buffers_bytes,
    )

    # ── Residual decomposition ──
    #
    # framework_overhead: VRAM reserved by the allocator/comm runtime
    # that does NOT back live tensors at the bench peak.  Largely
    # invariant in bench↔target scaling, so we forward it directly to
    # the target.  Subtract the analytical comm baseline at the bench
    # world size to avoid double-counting that part against the target
    # comm estimate.
    framework_overhead = max(
        0,
        int(bench.global_peak_reserved_bytes)
        - int(bench.global_peak_allocated_bytes)
        - int(analytical_bench.comm_buffers_bytes),
    )
    # live_tensor_excess: any live-tensor bytes at the bench peak that
    # the analytical model missed at bench scope.  Clamps to 0 when the
    # analytical model already over-predicts (common at num_layers=1).
    live_tensor_excess = max(
        0,
        int(bench.global_peak_allocated_bytes) - int(analytical_bench.total),
    )
    residual_total = framework_overhead + live_tensor_excess

    # ── Final totals ──
    # Point estimate: analytical components + measured residual terms.
    # Upper bound: residual gets the safety-margin treatment; the
    # analytical part is already the conservative point estimate.
    point_total = analytical_target_corrected.total + residual_total
    upper_total = analytical_target_corrected.total + int(residual_total * (1.0 + safety_margin))

    breakdown = {
        "analytical_at_bench": asdict(analytical_bench),
        "analytical_at_target_uncorrected": asdict(analytical_target),
        "analytical_at_target_corrected": asdict(analytical_target_corrected),
        "activation_correction": correction,
        # New, well-defined residual decomposition.
        "framework_overhead_bytes": int(framework_overhead),
        "live_tensor_excess_bytes": int(live_tensor_excess),
        "residual_total_bytes": int(residual_total),
        # Legacy field names kept for back-compat with consumers that
        # read the older keys.  The semantics are mapped to the new
        # terms so callers see "live-tensor excess" for *_allocated and
        # "framework + excess" for *_reserved.
        "residual_allocated_bytes": int(live_tensor_excess),
        "residual_reserved_bytes": int(residual_total),
        "safety_margin": float(safety_margin),
        "point_estimate_bytes": int(point_total),
        "upper_bound_bytes": int(upper_total),
    }

    diagnostics = {
        "bench_world_size": bench_world_size,
        "target_world_size": target_world_size,
        "bench_global_peak_allocated_bytes": int(bench.global_peak_allocated_bytes),
        "bench_global_peak_reserved_bytes": int(bench.global_peak_reserved_bytes),
        "framework_overhead_bytes": int(framework_overhead),
        "live_tensor_excess_bytes": int(live_tensor_excess),
        "analytical_share": (analytical_target_corrected.total / point_total if point_total > 0 else 0.0),
        "residual_share": residual_total / point_total if point_total > 0 else 0.0,
    }
    if bench_world_size * 8 < target_world_size:
        diagnostics["warning"] = (
            "Bench world size is much smaller than target; analytical share "
            "is high. The residual term covers only invariants captured at "
            "bench. Validate against a real run before relying on the "
            "upper-bound for OOM-fits decisions."
        )

    return PerRankProjection(
        point_estimate_bytes=int(point_total),
        upper_bound_bytes=int(upper_total),
        breakdown=breakdown,
        diagnostics=diagnostics,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bench-measurement extraction from a profiling-results dict
# ─────────────────────────────────────────────────────────────────────────────


def extract_bench_measurement(profiling_results: Dict[Any, Any]) -> BenchMeasurement:
    """Build a :class:`BenchMeasurement` from a profiling-results dict.

    The dict shape is what :func:`_run_layer_benchmark` returns (and what
    ``_load_profiling_results`` reconstructs from a saved artifact):
    integer keys for transformer layers, ``"embedding"`` / ``"output"``,
    and a ``"_memory_benchmark"`` block with phase snapshots.
    """
    mem = profiling_results.get("_memory_benchmark", {}) or {}
    bm = BenchMeasurement(
        global_peak_allocated_bytes=int(mem.get("global_peak_allocated_bytes", 0)),
        global_peak_reserved_bytes=int(mem.get("global_peak_reserved_bytes", 0)),
    )

    # Pick one representative measurement per layer type from the
    # profiled layers.  Bench typically profiles 1 dense + 1 moe.
    seen_types: Dict[str, int] = {}
    for key, val in profiling_results.items():
        if not isinstance(key, int) or not isinstance(val, dict):
            continue
        layer_type = val.get("type")
        if layer_type not in ("dense", "moe"):
            continue
        if layer_type in seen_types:
            continue
        act = int(val.get("activation_memory_bytes", 0))
        if act > 0:
            seen_types[layer_type] = act
    bm.per_layer_activation_bytes = seen_types

    emb = profiling_results.get("embedding")
    if isinstance(emb, dict):
        bm.embedding_activation_bytes = int(emb.get("activation_memory_bytes", 0))
    out = profiling_results.get("output")
    if isinstance(out, dict):
        bm.output_activation_bytes = int(out.get("activation_memory_bytes", 0))

    return bm
