###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.module_profilers import collective_model as cm
from primus.core.projection.module_profilers.collective_args import get_default_args
from primus.core.projection.module_profilers.language_model import (
    LanguageModelProfiler,
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.performance_projection.simulator import (
    SchedulerSimulationRunner,
)
from primus.core.projection.simulation_backends.factory import (
    get_gemm_simulation_backend,
    get_sdpa_simulation_backend,
)
from primus.core.projection.training_config import (
    convert_primus_config_to_projection_config,
)

# NOTE: MegatronPretrainTrainer is imported lazily inside _run_layer_benchmark()
# to avoid pulling in the megatron dependency when running in pure simulation mode
# (--profiling-mode simulate).

_MAX_EXPERT_PARALLEL_SIZE = 8
_BYTES_PER_GB = 1024**3

# HBM bandwidth (GB/s) by GPU architecture — used for optimizer step estimation
_HBM_BANDWIDTH_GBPS: Dict[str, float] = {
    "mi300x": 5300.0,
    "gfx942": 5300.0,
    "mi325x": 6000.0,
    "mi355x": 8000.0,
    "gfx950": 8000.0,
}


def _estimate_optimizer_step_ms(
    training_config,
    dp_size: int = 1,
    gpu_arch: Optional[str] = None,
) -> float:
    """
    Estimate the optimizer step time (Adam/AdamW) for one training iteration.

    The optimizer step is HBM-bandwidth-bound. For each parameter,
    Adam reads/writes the following (mixed-precision training with BF16
    forward, FP32 master weights):

        Read:  FP32 master param (4B) + FP32 grad (4B) + m (4B) + v (4B) = 16 B
        Write: FP32 master param (4B) + m (4B) + v (4B) + BF16 param (2B) = 14 B
        Total: 30 bytes per parameter

    With distributed_optimizer or FSDP, optimizer state is sharded across
    DP ranks, so each GPU only updates ``N_params / dp_size`` parameters.

    Returns:
        Optimizer step time in milliseconds.
    """
    model_config = training_config.model_config
    mp_config = training_config.model_parallel_config

    # --- Count total parameters per GPU (post TP/PP/EP sharding) ----------
    hidden = model_config.hidden_size
    ffn_hidden = model_config.ffn_hidden_size or (hidden * 4)
    moe_ffn = model_config.moe_ffn_hidden_size or ffn_hidden
    num_layers = model_config.num_layers
    num_experts = model_config.num_experts or 0
    moe_pattern = model_config.moe_pattern  # list of 0/1
    num_moe_layers = sum(1 for p in moe_pattern if p == 1)
    num_dense_layers = num_layers - num_moe_layers

    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    ep = getattr(mp_config, "expert_model_parallel_size", 1) or 1

    # Attention params: Q, K, V, O -> 4 * h * h (per layer, sharded by TP)
    attn_params_per_layer = 4 * hidden * hidden // tp
    # Dense MLP: gate, up, down -> 3 * h * ffn (per layer, sharded by TP)
    dense_mlp_params_per_layer = 3 * hidden * ffn_hidden // tp
    # Expert MLP params per expert: 3 * h * moe_ffn (NOT sharded by TP normally)
    expert_tp = getattr(mp_config, "expert_tensor_parallel_size", None) or 1
    expert_mlp_params_per_expert = 3 * hidden * moe_ffn // expert_tp

    # Non-expert params across all layers (sharded by TP, PP)
    non_expert_params = (
        num_layers * attn_params_per_layer
        + num_dense_layers * dense_mlp_params_per_layer
    )
    # Expert params (sharded by EP, expert_TP, PP)
    expert_params = (
        num_moe_layers * num_experts * expert_mlp_params_per_expert // max(ep, 1)
    )

    # Shared experts (if any)
    shared_sz = getattr(model_config, "moe_shared_expert_intermediate_size", 0) or 0
    shared_expert_params = 0
    if shared_sz and num_moe_layers > 0:
        shared_expert_params = num_moe_layers * 3 * hidden * shared_sz // tp

    total_params_per_gpu = (
        non_expert_params + expert_params + shared_expert_params
    ) // pp

    # Embedding + output layer params (only on first / last PP rank, amortise)
    vocab_size = getattr(model_config, "vocab_size", 0) or 0
    if vocab_size and pp > 0:
        embedding_params = vocab_size * hidden // tp
        output_params = vocab_size * hidden // tp
        # Amortise across PP ranks (only 1 rank holds each)
        total_params_per_gpu += (embedding_params + output_params) // pp

    # --- Distributed optimizer / FSDP sharding ---
    use_distributed_optimizer = getattr(mp_config, "use_distributed_optimizer", False)
    use_fsdp = getattr(mp_config, "use_torch_fsdp2", False)

    if use_distributed_optimizer or use_fsdp:
        # Optimizer state is sharded across DP ranks
        params_for_optim = total_params_per_gpu // max(dp_size, 1)
    else:
        params_for_optim = total_params_per_gpu

    # --- Compute time ---
    bytes_per_param = 30  # Adam read+write (mixed precision)
    total_bytes = params_for_optim * bytes_per_param

    # Look up HBM bandwidth
    arch = (gpu_arch or os.getenv("PRIMUS_GPU_ARCH", "mi300x")).lower().strip()
    hbm_bw_gbps = _HBM_BANDWIDTH_GBPS.get(arch, 5300.0)
    hbm_bw_bytes_per_ms = hbm_bw_gbps * 1e9 / 1e3  # bytes/ms

    optimizer_time_ms = total_bytes / hbm_bw_bytes_per_ms

    return optimizer_time_ms


# =============================================================================
# Hardware and Communication Functions (moved from multinode_projection)
# =============================================================================


def load_hardware_config(config_path: str) -> Dict[str, Any]:
    """Load hardware configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("hardware_config", {})


def calculate_collective_communication_time(
    training_config,
    num_nodes: int,
    gpus_per_node: int,
    tp: int,
    pp: int,
    ep: int,
    cp: int,
    dp: int,
    hardware_config: Dict[str, Any] = None,
) -> Tuple[float, Dict[str, float], Dict[str, Any], list]:
    """
    Calculate collective communication time for given configuration.

    Returns:
        (total_comm_time_ms, breakdown_dict, message_info_dict, per_layer_info_list)
    """
    model_config = training_config.model_config
    runtime_config = training_config.runtime_config

    # Setup collective args
    coll_args = get_default_args(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
        hardware_config=hardware_config,
    )

    # Check if DeepEP is enabled and pass it to coll_args
    mp_config = training_config.model_parallel_config
    moe_enable_deepep = getattr(mp_config, "moe_enable_deepep", False)
    use_turbo_deepep = getattr(mp_config, "use_turbo_deepep", False)
    coll_args.moe_enable_deepep = moe_enable_deepep
    coll_args.use_turbo_deepep = use_turbo_deepep

    # Model parameters
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_layers
    moe_router_topk = model_config.moe_router_topk
    moe_pattern = model_config.moe_pattern
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length

    # Count MoE layers
    num_moe_layers = sum(1 for p in moe_pattern if p == 1)
    num_dense_layers = num_layers - num_moe_layers

    # Calculate per-rank parameters — MoE-aware
    # For MoE models, expert params are much larger than the dense approximation.
    # Dense layers: attention (4*h^2) + MLP (3*h*ffn) per layer
    # MoE layers: attention (4*h^2) + num_experts * MLP (3*h*moe_ffn) per layer
    ffn_hidden = model_config.ffn_hidden_size or hidden_size * 4
    moe_ffn = model_config.moe_ffn_hidden_size or ffn_hidden
    num_experts = model_config.num_experts or 1
    attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    dense_mlp_params = 3 * hidden_size * ffn_hidden  # gate, up, down
    expert_mlp_params = 3 * hidden_size * moe_ffn  # per expert

    # Non-expert params are replicated across EP; expert params are partitioned by EP
    non_expert_params = num_layers * attn_params + num_dense_layers * dense_mlp_params
    expert_total_params = num_moe_layers * num_experts * expert_mlp_params
    # Per-GPU: non-expert (full copy) + expert (sharded by EP)
    params_per_gpu = non_expert_params + (expert_total_params // max(ep, 1))
    num_params_per_rank = params_per_gpu // (tp * pp)  # Further sharded by TP/PP

    breakdown = {}
    message_info = {}
    per_layer_info = []  # Store per-layer communication details

    # Get model parallel config (already retrieved above for DeepEP check)
    mp_config = training_config.model_parallel_config
    # Note: use_torch_fsdp2 = True means actual FSDP (shards weights, uses all-gather/reduce-scatter)
    # use_distributed_optimizer = True means ZeRO-1 style (shards optimizer state only, uses all-reduce)
    # These are DIFFERENT! Only FSDP2 replaces gradient all-reduce with reduce-scatter.
    use_fsdp = getattr(mp_config, "use_torch_fsdp2", False)

    # 1. Gradient AllReduce (DP group) - ONLY if NOT using FSDP2
    # With FSDP2, gradient sync is handled by reduce-scatter, not all-reduce
    # With distributed_optimizer (ZeRO-1), we still need gradient all-reduce
    if dp > 1 and not use_fsdp:
        # For MoE with EP > 1, expert gradient allreduce is across dp_replicas
        # (1 GPU per node), which is bandwidth-limited by single inter-node link.
        # Non-expert gradient allreduce is across full DP group.
        dp_replicas = dp // max(ep, 1)  # data-parallel replicas (excluding EP)

        if ep > 1 and num_moe_layers > 0 and dp_replicas > 1:
            # Expert gradient allreduce: across dp_replicas GPUs (1 per node)
            expert_params_per_gpu = (expert_total_params // ep) // (tp * pp)
            expert_grad_size = expert_params_per_gpu * 4  # FP32
            # These GPUs span different nodes → use inter-node bandwidth
            # Ring allreduce: 2 * (N-1)/N * msg / BW
            pod_bw = getattr(coll_args, "pod_bw", 50.0)
            bw_eff = getattr(coll_args, "bw_eff", 0.91)
            inter_bw = pod_bw * bw_eff  # GB/s per link
            msg_scale = (dp_replicas - 1) / dp_replicas
            expert_ar_time_ms = (
                2 * expert_grad_size * msg_scale / (inter_bw * 1e9) * 1e3
            )

            # Non-expert gradient allreduce: across full DP group
            non_expert_per_rank = non_expert_params // (tp * pp)
            non_expert_grad_size = non_expert_per_rank * 4  # FP32
            non_expert_ar_time = cm.allreduce(
                coll_args, non_expert_grad_size, dp, groups=["dp"]
            )
            non_expert_ar_ms = non_expert_ar_time / 1000

            total_ar_ms = expert_ar_time_ms + non_expert_ar_ms
            total_grad_size = expert_grad_size + non_expert_grad_size

            breakdown["gradient_allreduce"] = total_ar_ms
            message_info["gradient_allreduce_size"] = total_grad_size
            message_info["gradient_allreduce_size_mb"] = total_grad_size / (1024 * 1024)
            message_info["expert_ar_dp_replicas"] = dp_replicas
            message_info["expert_ar_time_ms"] = expert_ar_time_ms
            message_info["non_expert_ar_time_ms"] = non_expert_ar_ms
            # MoE all2all barriers prevent gradient allreduce from overlapping
            message_info["moe_ar_no_overlap"] = True
        else:
            grad_size = num_params_per_rank * 4  # FP32 gradients
            ar_time_dp = cm.allreduce(coll_args, grad_size, dp, groups=["dp"])
            breakdown["gradient_allreduce"] = ar_time_dp / 1000  # Convert to ms
            message_info["gradient_allreduce_size"] = grad_size
            message_info["gradient_allreduce_size_mb"] = grad_size / (1024 * 1024)
            message_info["moe_ar_no_overlap"] = False
    else:
        breakdown["gradient_allreduce"] = 0.0
        message_info["gradient_allreduce_size"] = 0
        message_info["gradient_allreduce_size_mb"] = 0.0
        message_info["moe_ar_no_overlap"] = False

    # 2. MoE All-to-All (EP group)
    # With TP > 1 and sequence parallelism, each GPU holds S/TP tokens.
    # The A2A dispatches these S/TP tokens; AG(TP) recovers full S after A2A.
    if ep > 1 and num_moe_layers > 0:
        tokens_per_gpu = seq_len * batch_size // max(tp, 1)  # S/TP with seq parallel
        dispatch_size = tokens_per_gpu * hidden_size * moe_router_topk * 2  # BF16

        a2a_dispatch = cm.alltoall(coll_args, dispatch_size, ep, groups=["ep"])
        a2a_combine = cm.alltoall(coll_args, dispatch_size, ep, groups=["ep"])

        total_a2a_fwd = (a2a_dispatch + a2a_combine) * num_moe_layers / 1000  # ms
        total_a2a_bwd = total_a2a_fwd

        breakdown["moe_a2a_fwd"] = total_a2a_fwd
        breakdown["moe_a2a_bwd"] = total_a2a_bwd
        message_info["moe_a2a_size"] = dispatch_size
        message_info["moe_a2a_size_mb"] = dispatch_size / (1024 * 1024)
        message_info["moe_a2a_per_layer_fwd"] = (a2a_dispatch + a2a_combine) / 1000
        message_info["num_moe_layers"] = num_moe_layers
    else:
        breakdown["moe_a2a_fwd"] = 0.0
        breakdown["moe_a2a_bwd"] = 0.0
        message_info["moe_a2a_size"] = 0
        message_info["moe_a2a_size_mb"] = 0.0
        message_info["moe_a2a_per_layer_fwd"] = 0.0
        message_info["num_moe_layers"] = 0

    # Note: TP AllReduce is already included in the benchmarked run, so we don't add it here
    message_info["num_layers"] = num_layers

    # 3. FSDP Communication (if enabled)
    # FSDP shards weights across DP ranks. Each layer needs:
    #   - Forward: All-gather to reconstruct full weights
    #   - Backward: Reduce-scatter to distribute gradients back to shards
    #
    # Recompute correction: with recompute_granularity="full", every backward
    # layer recomputes its forward pass, requiring a SECOND AllGather per
    # layer to re-fetch the sharded weights.
    # Note: use_fsdp and mp_config already defined above

    if use_fsdp and dp > 1:
        # Per-layer weight size (simplified estimate)
        # Dense layer: ~12 * hidden^2 params (qkv_proj, o_proj, mlp up/down/gate)
        # MoE layer: similar attention + num_experts * expert_params
        ffn_hidden = model_config.ffn_hidden_size or hidden_size * 4
        params_per_dense_layer = (
            hidden_size * hidden_size * 4 + hidden_size * ffn_hidden * 3
        )  # attn + MLP
        params_per_dense_layer = (
            params_per_dense_layer // tp
        )  # Divide by TP (params are TP-sharded)

        # Weight size in bytes (BF16 = 2 bytes)
        weight_size_per_layer = params_per_dense_layer * 2

        # All-gather: each rank sends its shard (1/DP), receives full weights
        # Total data moved = weight_size * (DP-1)/DP per rank
        ag_time_per_layer_us = cm.allgather(
            coll_args, weight_size_per_layer, dp, groups=["dp"]
        )

        # Reduce-scatter: each rank sends full gradients, receives its shard
        grad_size_per_layer = (
            params_per_dense_layer * 2
        )  # BF16 gradients for communication
        rs_time_per_layer_us = cm.reduce_scatter(
            coll_args, grad_size_per_layer, dp, groups=["dp"]
        )

        # --- Recompute correction ---
        # With recompute_granularity="full", during the backward pass each layer
        # re-runs its forward pass.  This means the weights must be AllGathered
        # AGAIN for each recomputed layer (the first AG result was freed after
        # the initial forward).  The ReduceScatter count is unchanged (1 per
        # layer backward).
        recompute_gran = getattr(mp_config, "recompute_granularity", None)
        recomp_n_layers = getattr(mp_config, "recompute_num_layers", 0) or 0
        ag_multiplier = 1  # default: AG once per layer (forward)
        if recompute_gran == "full" and recomp_n_layers > 0:
            # Each recomputed layer needs a second AG in backward
            recomp_ratio = min(recomp_n_layers, num_layers) / num_layers
            ag_multiplier = 1 + recomp_ratio  # e.g. 2.0 when all layers recomputed

        # Calculate total FSDP time for all layers
        total_fsdp_ag_fwd = (
            ag_time_per_layer_us * num_layers * ag_multiplier
        ) / 1000  # ms
        total_fsdp_rs_bwd = (rs_time_per_layer_us * num_layers) / 1000  # ms

        breakdown["fsdp_allgather_fwd"] = total_fsdp_ag_fwd
        breakdown["fsdp_reducescatter_bwd"] = total_fsdp_rs_bwd
        message_info["fsdp_weight_size_per_layer_mb"] = weight_size_per_layer / (
            1024 * 1024
        )
        message_info["fsdp_ag_per_layer_ms"] = ag_time_per_layer_us / 1000
        message_info["fsdp_rs_per_layer_ms"] = rs_time_per_layer_us / 1000
        message_info["fsdp_ag_multiplier"] = ag_multiplier
        message_info["fsdp_enabled"] = True
    else:
        breakdown["fsdp_allgather_fwd"] = 0.0
        breakdown["fsdp_reducescatter_bwd"] = 0.0
        message_info["fsdp_enabled"] = False

    # Note: PP P2P communication is NOT calculated here because it's already
    # accounted for in the pipeline scheduler simulator (simulator.py).
    # The simulator handles send/receive synchronization and bubble time.

    # Build per-layer communication information
    for layer_idx in range(num_layers):
        layer_comm = {
            "layer_idx": layer_idx,
            "layer_type": "MoE" if moe_pattern[layer_idx] == 1 else "Dense",
            "communications": [],
        }

        # MoE All-to-All (if EP > 1 and this is a MoE layer)
        # Note: TP AllReduce is already included in benchmarked run, so not added here
        if ep > 1 and moe_pattern[layer_idx] == 1:
            layer_comm["communications"].append(
                {
                    "type": "MoE All-to-All (fwd+bwd)",
                    "time_ms": (a2a_dispatch + a2a_combine) * 2 / 1000,  # fwd + bwd
                    "message_size_mb": dispatch_size / (1024 * 1024),
                    "group_size": ep,
                }
            )

        per_layer_info.append(layer_comm)

    total_comm_time = sum(breakdown.values())

    # Check if gradient all-reduce should be overlapped
    overlap_grad_reduce = getattr(
        mp_config, "overlap_grad_reduce", True
    )  # Default to True

    # If overlapped and NOT MoE-no-overlap, don't add to critical path
    moe_no_overlap = message_info.get("moe_ar_no_overlap", False)
    if overlap_grad_reduce and not moe_no_overlap and "gradient_allreduce" in breakdown:
        total_comm_time -= breakdown["gradient_allreduce"]
        message_info["gradient_allreduce_overlapped"] = True
    else:
        message_info["gradient_allreduce_overlapped"] = False

    if use_fsdp and dp > 1:
        overlap_fsdp = getattr(mp_config, "use_torch_fsdp2", False)
        if overlap_fsdp:
            recompute_gran = getattr(mp_config, "recompute_granularity", None)
            recomp_n = getattr(mp_config, "recompute_num_layers", 0) or 0
            has_recompute = recompute_gran == "full" and recomp_n > 0

            total_fsdp_ag = breakdown.get("fsdp_allgather_fwd", 0)
            total_fsdp_rs = breakdown.get("fsdp_reducescatter_bwd", 0)

            if has_recompute:
                # With full recompute the AG total already includes the 2×
                # multiplier.  Split into forward AG and backward (recomp) AG.
                recomp_ratio = min(recomp_n, num_layers) / num_layers
                ag_multiplier_val = message_info.get(
                    "fsdp_ag_multiplier", 1 + recomp_ratio
                )
                fwd_ag_total = total_fsdp_ag / ag_multiplier_val
                bwd_ag_total = total_fsdp_ag - fwd_ag_total
            else:
                fwd_ag_total = total_fsdp_ag
                bwd_ag_total = 0.0

            # Overlap factor applied uniformly to all FSDP
            # communication - (AllGather fwd, AllGather recompute, ReduceScatter).
            FSDP_OVERLAP = 0.93

            total_fsdp = total_fsdp_ag + total_fsdp_rs
            total_hidden = total_fsdp * FSDP_OVERLAP
            total_comm_time -= total_hidden
            message_info["fsdp_overlapped"] = True
            message_info["fsdp_overlap"] = FSDP_OVERLAP
            message_info["fsdp_overall_overlap"] = FSDP_OVERLAP
            message_info["fsdp_exposed_ms"] = total_fsdp - total_hidden
        else:
            message_info["fsdp_overlapped"] = False

    return total_comm_time, breakdown, message_info, per_layer_info


def extract_single_node_time_from_profiling(
    profiling_results: dict, training_config
) -> float:
    """
    Extract total single-node time from profiling results.

    The profiling phase only benchmarks a few representative layers (1 dense + 1 MoE) to save time.
    This function extrapolates those results to the full model by calculating averages and scaling.

    Args:
        profiling_results: Dict with integer keys for layers (0, 1, ...) and "embedding", "output"
        training_config: Training configuration containing model config

    Returns:
        Total single-node time in milliseconds for the full model
    """
    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    if is_rank_0:
        print(
            "[Primus:Performance Projection] Extracting timing from benchmark results..."
        )
        print("-" * 100)

    model_config = training_config.model_config
    mp_config = training_config.model_parallel_config
    moe_pattern = model_config.moe_pattern  # Full model pattern (e.g., 27 layers)

    # Get recomputation settings
    recompute_granularity = getattr(mp_config, "recompute_granularity", None)
    recompute_num_layers = getattr(mp_config, "recompute_num_layers", 0) or 0
    num_total_layers = len(moe_pattern)

    # Get profiled layer indices
    profiled_layer_indices = sorted(
        [k for k in profiling_results.keys() if isinstance(k, int)]
    )
    if is_rank_0:
        print(f"  Profiled layers: {profiled_layer_indices}")
        print(f"  Full model has {num_total_layers} transformer layers")
        if recompute_granularity == "full" and recompute_num_layers > 0:
            print(
                f"  Recomputation: {recompute_num_layers} layers (granularity={recompute_granularity})"
            )

    total_time_ms = 0.0

    # Embedding layer
    if "embedding" in profiling_results:
        emb = profiling_results["embedding"]
        emb_time = emb.get("forward_time_ms", 0) + emb.get("backward_time_ms", 0)
        total_time_ms += emb_time
        if is_rank_0:
            print(f"  Embedding: {emb_time:.2f} ms")

    # Analyze profiled transformer layers - track forward times separately for recompute
    profiled_dense_times = []
    profiled_dense_fwd_times = []
    profiled_moe_times = []
    profiled_moe_fwd_times = []

    for layer_idx in profiled_layer_indices:
        if layer_idx < len(moe_pattern):
            layer_data = profiling_results[layer_idx]
            fwd_time = layer_data.get("forward_time_ms", 0)
            bwd_time = layer_data.get("backward_time_ms", 0)
            layer_time = fwd_time + bwd_time

            if moe_pattern[layer_idx] == 0:
                profiled_dense_times.append(layer_time)
                profiled_dense_fwd_times.append(fwd_time)
            else:
                profiled_moe_times.append(layer_time)
                profiled_moe_fwd_times.append(fwd_time)

    # Calculate averages from profiled layers
    avg_dense_time = (
        sum(profiled_dense_times) / len(profiled_dense_times)
        if profiled_dense_times
        else 0
    )
    avg_dense_fwd = (
        sum(profiled_dense_fwd_times) / len(profiled_dense_fwd_times)
        if profiled_dense_fwd_times
        else 0
    )
    avg_moe_time = (
        sum(profiled_moe_times) / len(profiled_moe_times) if profiled_moe_times else 0
    )
    avg_moe_fwd = (
        sum(profiled_moe_fwd_times) / len(profiled_moe_fwd_times)
        if profiled_moe_fwd_times
        else 0
    )

    # Count total dense and MoE layers in full model
    num_dense_layers = sum(1 for x in moe_pattern if x == 0)
    num_moe_layers = sum(1 for x in moe_pattern if x == 1)

    # Extrapolate to full model
    total_dense_time = avg_dense_time * num_dense_layers
    total_moe_time = avg_moe_time * num_moe_layers
    total_transformer_time = total_dense_time + total_moe_time

    total_time_ms += total_transformer_time

    # Print detailed breakdown
    if is_rank_0:
        if profiled_dense_times:
            print(
                f"  Dense Layers: {len(profiled_dense_times)} profiled → {num_dense_layers} total"
            )
            print(
                f"    Avg per layer: {avg_dense_time:.2f} ms (fwd={avg_dense_fwd:.2f} ms)"
            )
            print(f"    Total time: {total_dense_time:.2f} ms")

        if profiled_moe_times:
            print(
                f"  MoE Layers: {len(profiled_moe_times)} profiled → {num_moe_layers} total"
            )
            print(
                f"    Avg per layer: {avg_moe_time:.2f} ms (fwd={avg_moe_fwd:.2f} ms)"
            )
            print(f"    Total time: {total_moe_time:.2f} ms")

    # Output layer
    if "output" in profiling_results:
        out = profiling_results["output"]
        out_time = out.get("forward_time_ms", 0) + out.get("backward_time_ms", 0)
        total_time_ms += out_time
        if is_rank_0:
            print(f"  Output Layer: {out_time:.2f} ms")

    # Add recomputation overhead
    # With recompute_granularity="full", during backward pass the forward is re-run for recomputed layers
    # This adds approximately 1x forward time per recomputed layer
    recompute_overhead_ms = 0.0
    if recompute_granularity == "full" and recompute_num_layers > 0:
        # Calculate how many dense vs MoE layers are recomputed
        # Typically recompute_num_layers applies to all transformer layers
        recompute_ratio = min(recompute_num_layers, num_total_layers) / num_total_layers

        # Recompute overhead = forward time for recomputed layers
        recompute_dense_layers = int(num_dense_layers * recompute_ratio)
        recompute_moe_layers = int(num_moe_layers * recompute_ratio)

        recompute_overhead_ms = (avg_dense_fwd * recompute_dense_layers) + (
            avg_moe_fwd * recompute_moe_layers
        )
        total_time_ms += recompute_overhead_ms

        if is_rank_0:
            print(f"  Recomputation Overhead: {recompute_overhead_ms:.2f} ms")
            print(
                f"    ({recompute_dense_layers} dense + {recompute_moe_layers} MoE layers recomputed)"
            )

    if is_rank_0:
        print("-" * 100)
        print(
            f"[Primus:Performance Projection] Extrapolated Baseline Time: {total_time_ms:.2f} ms/iteration"
        )
        if recompute_overhead_ms > 0:
            print(f"  (Includes {recompute_overhead_ms:.2f} ms recomputation overhead)")
        print(
            f"  (Based on {len(profiled_layer_indices)} profiled layers → {num_total_layers} total layers)"
        )
        print("=" * 100)

    return total_time_ms


# =============================================================================
# Layer Configuration Functions
# =============================================================================


def _has_dense_layers(moe_layer_freq):
    """Best-effort detection of whether the original config contains dense layers."""
    if moe_layer_freq is None:
        return True
    if isinstance(moe_layer_freq, int):
        return moe_layer_freq != 1  # 1 => every layer is MoE
    if isinstance(moe_layer_freq, (list, tuple)):
        return any(layer_flag == 0 for layer_flag in moe_layer_freq)
    if isinstance(moe_layer_freq, str):
        evaluated = eval(moe_layer_freq, {}, {})
        if isinstance(evaluated, (list, tuple)):
            return any(layer_flag == 0 for layer_flag in evaluated)
    return True


def _limit_layers_for_projection(module_config):
    """
    Restrict the transformer stack to a small number of layers for profiling.
    Using more layers provides better accuracy by capturing inter-layer effects
    and reducing per-layer overhead percentage.
    """
    has_moe = getattr(module_config, "num_experts", None)
    has_moe = has_moe is not None and module_config.num_experts > 0
    original_layers = getattr(module_config, "num_layers", 1) or 1
    original_moe_layout = getattr(module_config, "moe_layer_freq", None)
    dense_layers_present = _has_dense_layers(original_moe_layout)
    # Use 1 layer for fast profiling - results are extrapolated to full model
    # Increase to 2-4 for better accuracy if needed
    max_layers = 1
    target_layers = max(1, min(original_layers, max_layers))
    module_config.num_layers = target_layers

    if has_moe:
        if not dense_layers_present:
            module_config.moe_layer_freq = [1] * target_layers
        elif target_layers == 1:
            module_config.moe_layer_freq = [1]
        else:
            dense_then_moe = [0, 1]
            if target_layers > 2:
                dense_then_moe.extend([0] * (target_layers - 2))
            module_config.moe_layer_freq = dense_then_moe
    else:
        module_config.moe_layer_freq = [0] * target_layers

    # disable pipeline model parallelism
    module_config.pipeline_model_parallel_size = 1
    for attr in (
        "num_layers_per_virtual_pipeline_stage",
        "num_virtual_stages_per_pipeline_rank",
    ):
        if hasattr(module_config, attr):
            setattr(module_config, attr, None)


def _rescale_expert_parallelism(module_config):
    """
    Cap expert_model_parallel_size so that EP * TP * CP <= 8 and adjust num_experts.
    """
    expert_mp_size = getattr(module_config, "expert_model_parallel_size", None)
    if expert_mp_size is None or expert_mp_size <= _MAX_EXPERT_PARALLEL_SIZE:
        current_tp = getattr(module_config, "tensor_model_parallel_size", 1) or 1
        current_cp = getattr(module_config, "context_parallel_size", 1) or 1
        if expert_mp_size is None:
            expert_mp_size = 1
        if expert_mp_size * current_tp * current_cp <= _MAX_EXPERT_PARALLEL_SIZE:
            return None

    num_experts = getattr(module_config, "num_experts", None)
    current_tp = getattr(module_config, "tensor_model_parallel_size", 1) or 1
    current_cp = getattr(module_config, "context_parallel_size", 1) or 1
    total_parallel_product = max(1, current_tp * current_cp)
    max_ep_allowed = max(1, _MAX_EXPERT_PARALLEL_SIZE // total_parallel_product)
    new_expert_mp = min(expert_mp_size, _MAX_EXPERT_PARALLEL_SIZE, max_ep_allowed)

    if new_expert_mp == expert_mp_size:
        print(
            "[Primus:Performance Projection] Expert parallelism already within limit "
            f"(EP={expert_mp_size}, TP={current_tp}, CP={current_cp})."
        )
        return None

    prev_num_experts = num_experts
    if num_experts is not None:
        experts_per_rank = math.ceil(num_experts / expert_mp_size)
        module_config.num_experts = max(new_expert_mp * experts_per_rank, new_expert_mp)

    module_config.expert_model_parallel_size = new_expert_mp
    print(
        "[Primus:Performance Projection] Rescaled expert parallelism "
        f"(EP {expert_mp_size} -> {new_expert_mp}, TP={current_tp}, CP={current_cp})."
    )
    if prev_num_experts is not None:
        print(
            "[Primus:Performance Projection] Adjusted num_experts "
            f"{prev_num_experts} -> {module_config.num_experts} "
            "(preserving experts per rank)."
        )
    return {
        "ep_before": expert_mp_size,
        "ep_after": new_expert_mp,
        "tp": current_tp,
        "cp": current_cp,
        "num_experts_before": prev_num_experts,
        "num_experts_after": getattr(module_config, "num_experts", None),
    }


def _calculate_single_node_config(original_config, gpus_per_node=8):
    """
    Calculate a reduced parallelism configuration that fits on a single node.

    Strategy:
    1. Reduce PP to 1 (easiest to add back communication overhead)
    2. If still doesn't fit, rescale EP to fit on single node
    3. Keep TP, CP unchanged
    4. Return the adjustment info for later baseline correction

    Args:
        original_config: Original module config
        gpus_per_node: Number of GPUs per node (default 8)

    Returns:
        dict with keys:
            'adjusted': bool - whether adjustment was needed
            'original_pp': int - original PP value
            'benchmark_pp': int - PP for benchmarking
            'original_nodes_required': int - original minimum nodes
            'original_ep': int - original EP value
            'benchmark_ep': int - EP for benchmarking
    """
    tp = getattr(original_config, "tensor_model_parallel_size", 1) or 1
    pp = getattr(original_config, "pipeline_model_parallel_size", 1) or 1
    ep = getattr(original_config, "expert_model_parallel_size", 1) or 1
    cp = getattr(original_config, "context_parallel_size", 1) or 1
    num_experts = getattr(original_config, "num_experts", None)

    gpus_required = tp * pp * ep * cp
    nodes_required = (gpus_required + gpus_per_node - 1) // gpus_per_node

    # If already fits on 1 node, no adjustment needed
    if nodes_required <= 1:
        return {
            "adjusted": False,
            "original_pp": pp,
            "benchmark_pp": pp,
            "original_nodes_required": nodes_required,
            "original_tp": tp,
            "original_ep": ep,
            "benchmark_ep": ep,
            "original_cp": cp,
            "original_num_experts": num_experts,
            "benchmark_num_experts": num_experts,
        }

    # Step 1: Reduce PP to 1
    benchmark_pp = 1
    benchmark_gpus_required = tp * benchmark_pp * ep * cp

    # Step 2: If still doesn't fit, rescale EP
    benchmark_ep = ep
    benchmark_num_experts = num_experts
    if benchmark_gpus_required > gpus_per_node:
        print(
            f"[Primus:Performance Projection] After reducing PP to 1, "
            f"config still requires {benchmark_gpus_required} GPUs (TP={tp}, EP={ep}, CP={cp})."
        )
        print(
            f"[Primus:Performance Projection] Rescaling EP to fit on {gpus_per_node} GPUs..."
        )

        # Rescale EP to fit
        rescale_info = _rescale_expert_parallelism(original_config)
        if rescale_info:
            benchmark_ep = rescale_info["ep_after"]
            benchmark_num_experts = rescale_info.get("num_experts_after", num_experts)
            benchmark_gpus_required = tp * benchmark_pp * benchmark_ep * cp

            if benchmark_gpus_required > gpus_per_node:
                raise ValueError(
                    f"[Primus:Performance Projection] Cannot reduce to single node."
                    f"Even with PP=1 and EP={benchmark_ep}, configuration requires {benchmark_gpus_required} GPUs "
                    f"(TP={tp}, EP={benchmark_ep}, CP={cp})."
                    f"Single node has only {gpus_per_node} GPUs."
                    f"Please reduce TP or CP in your configuration."
                )
        else:
            # Rescaling didn't help or wasn't needed
            raise ValueError(
                f"[Primus:Performance Projection] Cannot reduce to single node."
                f"Even with PP=1, configuration requires {benchmark_gpus_required} GPUs "
                f"(TP={tp}, EP={ep}, CP={cp})."
                f"Single node has only {gpus_per_node} GPUs."
                f"Please reduce TP, EP, or CP in your configuration."
            )

    # Modify the config
    original_config.pipeline_model_parallel_size = benchmark_pp

    # Also disable virtual pipeline stages (already done in _limit_layers_for_projection)
    for attr in (
        "num_layers_per_virtual_pipeline_stage",
        "num_virtual_stages_per_pipeline_rank",
    ):
        if hasattr(original_config, attr):
            setattr(original_config, attr, None)

    return {
        "adjusted": True,
        "original_pp": pp,
        "benchmark_pp": benchmark_pp,
        "original_nodes_required": nodes_required,
        "original_tp": tp,
        "original_ep": ep,
        "benchmark_ep": benchmark_ep,
        "original_cp": cp,
        "original_num_experts": num_experts,
        "benchmark_num_experts": benchmark_num_experts,
    }


def _estimate_pp_communication_overhead(
    training_config, pp_size, hardware_config_dict=None
):
    """
    Estimate the PP P2P communication overhead for a given PP size.

    Args:
        training_config: Training configuration
        pp_size: Pipeline parallelism size
        hardware_config_dict: Optional hardware config

    Returns:
        float: Estimated PP communication time in ms per iteration
    """
    if pp_size <= 1:
        return 0.0

    mp_config = training_config.model_parallel_config
    model_config = training_config.model_config
    runtime_config = training_config.runtime_config

    tp = mp_config.tensor_model_parallel_size
    ep = getattr(mp_config, "expert_model_parallel_size", 1)
    cp = getattr(mp_config, "context_model_parallel_size", 1)

    # Get hardware setup
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    gpus_required = tp * pp_size * ep * cp
    num_nodes = (gpus_required + gpus_per_node - 1) // gpus_per_node

    # Get collective model args
    coll_args = get_default_args(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp_size,
        ep=ep,
        cp=cp,
        hardware_config=hardware_config_dict,
    )

    # Calculate PP P2P communication
    hidden_size = model_config.hidden_size
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length

    # Activation size for P2P
    p2p_size = batch_size * seq_len * hidden_size * 2  # BF16

    # Number of microbatches
    global_batch_size = runtime_config.global_batch_size
    data_parallel_size = (num_nodes * gpus_per_node) // (tp * pp_size * ep * cp)
    num_microbatches = global_batch_size // (batch_size * data_parallel_size)

    # P2P time: 2 * (PP-1) sends per microbatch (forward + backward)
    # Using sendrecv as approximation (no groups parameter for sendrecv)
    p2p_time_per_transfer = cm.sendrecv(coll_args, p2p_size)

    # Total P2P time per iteration
    # Forward: (PP-1) sends, Backward: (PP-1) sends
    # Times number of microbatches
    total_p2p_time_ms = (
        2 * (pp_size - 1) * num_microbatches * p2p_time_per_transfer / 1000
    )

    return total_p2p_time_ms


def _compute_ep_mlp_scale(
    model_config,
    benchmark_ep,
    original_ep,
    original_num_experts=None,
    benchmark_num_experts=None,
):
    """
    Compute the MLP time scaling factor when EP changes, accounting for
    shared experts (EP-independent) vs routed experts.

    Key insight — per-GPU routed compute is EP-invariant
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    In Megatron MoE each GPU in the EP group processes the **same**
    micro-batch.  After A2A token redistribution every GPU ends up
    computing ``batch_tokens × topk`` total token-expert pairs,
    regardless of EP.  Therefore:

    * When ``_rescale_expert_parallelism`` adjusts ``num_experts``
      proportionally (preserving ``experts_per_rank``), the profiled /
      simulated MLP time already reflects the correct per-rank workload
      → **no scaling needed** (returns 1.0).

    * When EP changes but ``num_experts`` stays fixed (hypothetical —
      our rescaling always preserves experts_per_rank), the GEMM shapes
      change (fewer, larger GEMMs vs. more, smaller GEMMs) but total
      FLOPs remain identical.  We conservatively return 1.0 because the
      simple ``benchmark_ep / original_ep`` ratio does not capture GEMM-
      efficiency differences.

    Shared expert compute is constant regardless of EP (no A2A needed).

    Args:
        model_config: Model configuration.
        benchmark_ep: EP used during profiling / simulation.
        original_ep: EP of the target deployment.
        original_num_experts: Total expert count in the target config.
        benchmark_num_experts: Total expert count after rescaling
            (may differ from ``original_num_experts`` if
            ``_rescale_expert_parallelism`` adjusted it).

    Returns:
        float: Scale factor to apply to the profiled MLP time.
              1.0 when experts_per_rank is preserved (the common case).
    """
    if benchmark_ep == original_ep:
        return 1.0

    # Determine whether experts_per_rank was preserved by rescaling.
    # When it is, per-GPU routed compute is identical — no scaling.
    if original_num_experts is not None and benchmark_num_experts is not None:
        orig_epr = original_num_experts / original_ep
        bench_epr = benchmark_num_experts / benchmark_ep
        if abs(orig_epr - bench_epr) < 0.5:
            # experts_per_rank preserved — per-GPU routed compute unchanged.
            return 1.0

    # Fallback: per-GPU MoE routed compute is EP-invariant (A2A
    # redistributes tokens so each GPU processes batch_tokens × topk).
    # The simple benchmark_ep / original_ep ratio is NOT correct because
    # total FLOPs are constant; only GEMM shapes differ.  We return 1.0
    # rather than an inaccurate heuristic.
    return 1.0


def _estimate_ep_communication_overhead(
    training_config, original_ep, benchmark_ep, hardware_config_dict=None
):
    """
    Estimate the additional EP All-to-All communication overhead when scaling
    from benchmark_ep to original_ep.

    Args:
        training_config: Training configuration
        original_ep: Original expert parallelism size (e.g., 16)
        benchmark_ep: Benchmark expert parallelism size (e.g., 8)
        hardware_config_dict: Optional hardware config

    Returns:
        tuple: (forward_overhead_ms, backward_overhead_ms) - additional time per MoE layer
    """
    if original_ep <= benchmark_ep:
        return 0.0, 0.0

    mp_config = training_config.model_parallel_config
    model_config = training_config.model_config
    runtime_config = training_config.runtime_config

    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    cp = getattr(mp_config, "context_model_parallel_size", 1)

    # Get hardware setup
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))

    # Calculate nodes required for original EP
    gpus_required_original = tp * pp * original_ep * cp
    num_nodes_original = (gpus_required_original + gpus_per_node - 1) // gpus_per_node

    # Calculate nodes for benchmark EP (should be 1)
    gpus_required_benchmark = tp * pp * benchmark_ep * cp
    num_nodes_benchmark = (gpus_required_benchmark + gpus_per_node - 1) // gpus_per_node

    # Get collective model args for original EP configuration
    coll_args_original = get_default_args(
        num_nodes=num_nodes_original,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        ep=original_ep,
        cp=cp,
        hardware_config=hardware_config_dict,
    )

    # Get collective model args for benchmark EP configuration
    coll_args_benchmark = get_default_args(
        num_nodes=num_nodes_benchmark,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        ep=benchmark_ep,
        cp=cp,
        hardware_config=hardware_config_dict,
    )

    # Calculate All-to-All message size for MoE layers
    # With TP > 1 and sequence parallelism, each GPU holds S/TP tokens.
    # The AlltoAll dispatcher sends these S/TP tokens across the EP group,
    # then AllGather(TP) recovers full S tokens after A2A (see workflow in
    # MoEAlltoAllTokenDispatcher: step 3 = A2A(EP), step 4 = AG(TP)).
    hidden_size = model_config.hidden_size
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length
    moe_router_topk = getattr(model_config, "moe_router_topk", 2)

    tokens_per_gpu = seq_len * batch_size // max(tp, 1)  # S/TP with seq parallel
    dispatch_size = tokens_per_gpu * hidden_size * moe_router_topk * 2  # BF16

    # Calculate All-to-All time for original EP (dispatch + combine)
    a2a_dispatch_original = cm.alltoall(
        coll_args_original, dispatch_size, original_ep, groups=["ep"]
    )
    a2a_combine_original = cm.alltoall(
        coll_args_original, dispatch_size, original_ep, groups=["ep"]
    )
    a2a_time_original_fwd = (a2a_dispatch_original + a2a_combine_original) / 1000  # ms

    # Calculate All-to-All time for benchmark EP (dispatch + combine)
    a2a_dispatch_benchmark = cm.alltoall(
        coll_args_benchmark, dispatch_size, benchmark_ep, groups=["ep"]
    )
    a2a_combine_benchmark = cm.alltoall(
        coll_args_benchmark, dispatch_size, benchmark_ep, groups=["ep"]
    )
    a2a_time_benchmark_fwd = (
        a2a_dispatch_benchmark + a2a_combine_benchmark
    ) / 1000  # ms

    # The overhead is the difference (original is larger due to inter-node communication)
    fwd_overhead_per_layer = a2a_time_original_fwd - a2a_time_benchmark_fwd
    bwd_overhead_per_layer = fwd_overhead_per_layer  # Same for backward

    return fwd_overhead_per_layer, bwd_overhead_per_layer


def _extract_layer_type_timings(layer_results: dict) -> Dict[str, dict[str, float]]:
    if not layer_results:
        return {}
    type_timings: Dict[str, dict[str, float]] = {}
    for result in layer_results.values():
        layer_type = result.get("type")
        if layer_type not in ("dense", "moe"):
            continue
        if layer_type in type_timings:
            continue
        forward = float(result.get("forward_time_ms", 0.0) or 0.0)
        backward = float(result.get("backward_time_ms", 0.0) or 0.0)
        activation = (
            float(result.get("activation_memory_bytes", 0.0) or 0.0) / _BYTES_PER_GB
        )
        type_timings[layer_type] = {
            "forward": forward,
            "backward": backward,
            # wgrad is already included in the benchmarked backward time,
            # so set to 0 to avoid double-counting in the simulator
            "wgrad": 0.0,
            "activation": activation,
        }
    return type_timings


def _add_io_layer_timings(chunk_timings: List[list[dict]], profiling_results: dict):
    if not chunk_timings:
        return

    embedding = profiling_results.get("embedding")
    if embedding and chunk_timings[0]:
        first_chunk = chunk_timings[0][0]
        first_chunk["fwd"] += embedding.get("forward_time_ms", 0.0) or 0.0
        emb_bwd = embedding.get("backward_time_ms", 0.0) or 0.0
        first_chunk["bwd"] += emb_bwd
        # wgrad already included in backward, don't add again
        first_chunk["activation"] += (
            embedding.get("activation_memory_bytes", 0.0) or 0.0
        ) / _BYTES_PER_GB

    output = profiling_results.get("output")
    if output and chunk_timings[-1]:
        last_chunk = chunk_timings[-1][-1]
        last_chunk["fwd"] += output.get("forward_time_ms", 0.0) or 0.0
        out_bwd = output.get("backward_time_ms", 0.0) or 0.0
        last_chunk["bwd"] += out_bwd
        # wgrad already included in backward, don't add again
        last_chunk["activation"] += (
            output.get("activation_memory_bytes", 0.0) or 0.0
        ) / _BYTES_PER_GB


def _build_chunk_time_matrix(
    training_config, layer_results: dict
) -> Optional[List[List[dict]]]:
    model_cfg = getattr(training_config, "model_config", None)
    mp_cfg = getattr(training_config, "model_parallel_config", None)
    if model_cfg is None or mp_cfg is None:
        return None

    total_layers = getattr(model_cfg, "num_layers", 0) or 0
    if total_layers <= 0:
        return None

    layer_type_pattern = getattr(model_cfg, "moe_pattern", None)
    if (
        not isinstance(layer_type_pattern, (list, tuple))
        or len(layer_type_pattern) != total_layers
    ):
        layer_type_pattern = [0] * total_layers
    type_timings = _extract_layer_type_timings(layer_results)
    if not type_timings:
        return None

    pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1
    vpp_size = getattr(mp_cfg, "virtual_pipeline_model_parallel_size", 1) or 1
    tp_size = getattr(mp_cfg, "tensor_model_parallel_size", 1) or 1
    cp_size = getattr(mp_cfg, "context_model_parallel_size", 1) or 1
    ep_size = getattr(mp_cfg, "expert_model_parallel_size", 1) or 1

    decoder_first = getattr(mp_cfg, "decoder_first_pipeline_num_layers", None)
    decoder_last = getattr(mp_cfg, "decoder_last_pipeline_num_layers", None)

    mp_group = tp_size * cp_size * ep_size
    chunk_timings: List[list[dict]] = []
    for pp_rank in range(pp_size):
        layers = LanguageModelProfiler.get_layers_for_rank(
            None,
            global_rank=pp_rank * mp_group,
            n_layers=total_layers,
            pp_size=pp_size,
            tp_size=tp_size,
            cp_size=cp_size,
            ep_size=ep_size,
            num_virtual_pipeline_stages=vpp_size,
            decoder_first_pipeline_num_layers=decoder_first,
            decoder_last_pipeline_num_layers=decoder_last,
        )
        if not layers:
            chunk_timings.append(
                [
                    {"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0}
                    for _ in range(vpp_size)
                ]
            )
            continue

        layers_per_chunk = len(layers) // vpp_size if vpp_size else len(layers)
        if layers_per_chunk == 0:
            chunk_timings.append(
                [
                    {"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0}
                    for _ in range(vpp_size)
                ]
            )
            continue

        rank_chunks = []
        for chunk_idx in range(vpp_size):
            start = chunk_idx * layers_per_chunk
            end = start + layers_per_chunk
            chunk_layers = layers[start:end]
            chunk_entry = {"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0}
            for layer_idx in chunk_layers:
                layer_type = "moe" if layer_type_pattern[layer_idx] else "dense"
                metrics = type_timings.get(layer_type)
                if not metrics:
                    continue
                chunk_entry["fwd"] += metrics["forward"]
                chunk_entry["bwd"] += metrics["backward"]
                chunk_entry["wgrad"] += metrics["wgrad"]
                chunk_entry["activation"] += metrics.get("activation", 0.0)
            rank_chunks.append(chunk_entry)
        chunk_timings.append(rank_chunks)
    _add_io_layer_timings(chunk_timings, layer_results)
    return chunk_timings


def _compute_micro_batches(runtime_cfg, model_parallel_config) -> int:
    global_batch = getattr(runtime_cfg, "global_batch_size", None) or 1
    micro_batch = getattr(runtime_cfg, "micro_batch_size", None) or 1
    data_parallel_size = getattr(runtime_cfg, "data_parallel_size", None) or 1

    denominator = micro_batch * data_parallel_size
    if denominator <= 0:
        return 1
    return max(1, math.ceil(global_batch / denominator))


def _build_scheduler_sim_config(
    training_config, profiling_results, enable_zero_bubble=False
):
    chunk_time_matrix = _build_chunk_time_matrix(training_config, profiling_results)
    assert chunk_time_matrix is not None

    # For zero-bubble scheduling, we need to split backward into B (input grad) and W (weight grad)
    # The zero-bubble scheduler schedules these separately to minimize pipeline bubbles.
    # Typically B and W are roughly equal in duration (each ~50% of total backward).
    if enable_zero_bubble:
        print(
            "[Primus:Performance Projection] Splitting backward time for zero-bubble scheduling:"
        )
        print("  B (input grad) = 50% of backward, W (weight grad) = 50% of backward")
        for rank_chunks in chunk_time_matrix:
            for chunk in rank_chunks:
                total_bwd = chunk.get("bwd", 0.0)
                # Split: bwd becomes input gradient only, wgrad becomes weight gradient
                chunk["bwd"] = total_bwd * 0.5
                chunk["wgrad"] = total_bwd * 0.5

    if chunk_time_matrix:
        print("[Primus:Performance Projection] Per-chunk timings (ms):")
        for rank_idx, rank_chunks in enumerate(chunk_time_matrix):
            for chunk_idx, chunk in enumerate(rank_chunks):
                fwd = chunk.get("fwd", 0.0)
                bwd = chunk.get("bwd", 0.0)
                wgrad = chunk.get("wgrad", 0.0)
                activation = chunk.get("activation", 0.0)
                if enable_zero_bubble:
                    print(
                        f"  Rank {rank_idx:02d} Chunk {chunk_idx:02d} -> "
                        f"fwd={fwd:.2f} ms, bwd(B)={bwd:.2f} ms, wgrad(W)={wgrad:.2f} ms, activation={activation:.2f} GB"
                    )
                else:
                    print(
                        f"  Rank {rank_idx:02d} Chunk {chunk_idx:02d} -> "
                        f"fwd={fwd:.2f} ms, bwd={bwd:.2f} ms, activation={activation:.2f} GB"
                    )

    mp_cfg = training_config.model_parallel_config
    pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1
    vpp_size = getattr(mp_cfg, "virtual_pipeline_model_parallel_size", 1) or 1
    print(f"pp_size: {pp_size}, vpp_size: {vpp_size}")

    micro_batches = _compute_micro_batches(training_config.runtime_config, mp_cfg)

    # Select scheduler based on configuration
    if enable_zero_bubble and vpp_size == 1:
        # Zero-bubble schedule minimizes pipeline bubbles by separating B and W
        scheduler = {
            "name": "zerobubble",
            "class": "primus.core.pipeline_parallel.scheduler.algorithms.zerobubble.ScheduleZeroBubble",
            "pp_size": pp_size,
            "vpp_size": 1,
            "micro_batches": micro_batches,
        }
        print(
            "[Primus:Performance Projection] Using zero-bubble scheduler (enable_zero_bubble=True)"
        )
    elif vpp_size > 1:
        scheduler = {
            "name": "interleaved_1f1b",
            "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
            "pp_size": pp_size,
            "vpp_size": vpp_size,
            "micro_batches": micro_batches,
        }
    else:
        scheduler = {
            "name": "basic_1f1b",
            "class": "primus.core.pipeline_parallel.scheduler.algorithms.basic_1f1b.Schedule1F1B",
            "pp_size": pp_size,
            "vpp_size": 1,
            "micro_batches": micro_batches,
        }

    return {
        "chunk_time_ms": chunk_time_matrix,
        "output_dir": str(Path.cwd() / "pp_simulation_result"),
        "schedulers": [scheduler],
    }


def _report_simulation_results(sim_results, training_config):
    """
    Report simulation results and return the step time.

    Returns:
        float: Step time in ms, or None if no results
    """
    if not sim_results:
        return None

    runtime_config = training_config.runtime_config
    seq_len = getattr(runtime_config, "sequence_length", None)
    micro_batch_size = getattr(runtime_config, "micro_batch_size", None)

    step_time_ms = None
    for sim in sim_results:
        summary = (sim or {}).get("summary") or {}
        step_time_ms = summary.get("step_time_ms")
        micro_batches = summary.get("micro_batches") or 1
        num_gpus = summary.get("pp_size")
        summary.get("rank_totals") or []

        per_rank = sim.get("per_rank") or []
        mp_cfg = training_config.model_parallel_config
        param_mem_cache: Dict[int, float] = {}
        rank_stats = []
        for rank_idx, scheduled_layers in enumerate(per_rank):
            fwd_time = sum(
                end - start
                for start, end in zip(
                    scheduled_layers.get("fwd_start", []),
                    scheduled_layers.get("fwd_end", []),
                )
            )
            bwd_time = sum(
                end - start
                for start, end in zip(
                    scheduled_layers.get("bwd_start", []),
                    scheduled_layers.get("bwd_end", []),
                )
            )
            wgrad_time = sum(
                end - start
                for start, end in zip(
                    scheduled_layers.get("wgrad_start", []),
                    scheduled_layers.get("wgrad_end", []),
                )
            )
            total_layer_time = fwd_time + bwd_time + wgrad_time
            bubble_time = max(0.0, step_time_ms - total_layer_time)
            bubble_ratio = bubble_time / step_time_ms

            activation_trace = scheduled_layers.get("activation_memory_usage") or []
            peak_activation = (
                max(activation_trace)
                if activation_trace
                else scheduled_layers.get("memory", 0.0)
            )

            # Map rank_idx to pipeline rank (rank_idx // vpp_size)
            vpp_size = mp_cfg.virtual_pipeline_model_parallel_size or 1
            pp_rank = rank_idx // vpp_size
            if pp_rank not in param_mem_cache:
                param_mem_cache[pp_rank] = _get_parameter_memory(
                    training_config, pp_rank
                )
            param_mem_gb = param_mem_cache[pp_rank]
            total_peak_gb = peak_activation + param_mem_gb
            rank_stats.append(
                (
                    rank_idx,
                    bubble_time,
                    bubble_ratio,
                    peak_activation,
                    param_mem_gb,
                    total_peak_gb,
                )
            )

        tokens_per_step = seq_len * micro_batch_size * micro_batches
        tokens_per_gpu_sec = tokens_per_step * 1000 / step_time_ms / num_gpus
        scheduler_name = sim.get("name", "unknown")
        print(
            f"Scheduler '{scheduler_name}': {tokens_per_gpu_sec:,.2f} tokens/GPU/s "
            f"(step_time={step_time_ms:.2f}ms, seq_len={seq_len}, "
            f"micro_batch={micro_batch_size}, micro_batches={micro_batches})"
        )
        for rank_info in rank_stats:
            (
                rank_idx,
                bubble_time,
                bubble_ratio,
                peak_activation,
                param_mem_gb,
                total_peak_gb,
            ) = rank_info
            print(
                f"  Rank {rank_idx:02d} bubble: {bubble_time:.2f} ms "
                f"(ratio={bubble_ratio:.2%}), "
                f"activation_peak={peak_activation:.2f} GB, "
                f"param_memory={param_mem_gb:.2f} GB, "
                f"total_peak={total_peak_gb:.2f} GB"
            )

    return step_time_ms


def _run_layer_benchmark(primus_config, unknown_overrides):
    from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

    module_config = primus_config.get_module_config("pre_trainer")
    _limit_layers_for_projection(module_config)
    rescale_info = _rescale_expert_parallelism(module_config)
    training_config = convert_primus_config_to_projection_config(primus_config)

    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print("[Primus:Performance Projection] Initializing MegatronPretrainTrainer...")
    # Disable overlap features and FSDP2 for profiling (they add complexity without benefiting isolated layer benchmarking)
    # FSDP2 uses DTensor which causes issues with benchmarking inputs
    primus_config.get_module_config("pre_trainer").overlap_grad_reduce = False
    primus_config.get_module_config("pre_trainer").overlap_param_gather = False
    primus_config.get_module_config("pre_trainer").use_torch_fsdp2 = False
    print("[Primus:Performance Projection] Config (with profiling overrides):")
    print(
        f"  overlap_grad_reduce: {primus_config.get_module_config('pre_trainer').overlap_grad_reduce}"
    )
    print(
        f"  overlap_param_gather: {primus_config.get_module_config('pre_trainer').overlap_param_gather}"
    )
    print(
        f"  use_torch_fsdp2: {primus_config.get_module_config('pre_trainer').use_torch_fsdp2}"
    )
    trainer = MegatronPretrainTrainer(
        module_name="pre_trainer",
        primus_config=primus_config,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
        extra_args=unknown_overrides,
    )

    print("[Primus:Performance Projection] Initializing Megatron...")
    trainer.init()
    print("[Primus:Performance Projection] Setting up model and optimizer...")
    trainer.setup()

    print("[Primus:Performance Projection] Building model profiler...")
    model_profiler_spec = get_language_model_profiler_spec(training_config)
    model_profiler = build_profiler(model_profiler_spec)

    seq_len = training_config.runtime_config.sequence_length
    batch_size = training_config.runtime_config.micro_batch_size

    print("[Primus:Performance Projection] Benchmarking with:")
    print(f"  Rank: {rank}")
    print(f"  World Size: {world_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    if rescale_info:
        note = (
            f"  NOTE: MoE rescaled -> EP {rescale_info['ep_before']} -> {rescale_info['ep_after']}"
            f" (TP={rescale_info['tp']}, CP={rescale_info['cp']})"
        )
        if rescale_info["num_experts_before"] is not None:
            note += (
                f", num_experts {rescale_info['num_experts_before']}"
                f" -> {rescale_info['num_experts_after']}"
            )
        print(note)

    print("" + "=" * 100)
    print("[Primus:Performance Projection] Starting layer benchmarking...")
    print("=" * 100)

    profiling_results = model_profiler.run_layer_benchmark(
        model=trainer.model,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    return profiling_results


def _run_layer_simulation(primus_config, args):
    """
    Run layer simulation using GEMM + SDPA simulation backends (no GPU required).

    This mirrors :func:`_run_layer_benchmark` but replaces actual GPU kernel
    benchmarks with analytical / model-based simulation.  It does *not*
    instantiate a trainer or model – only the profiler tree is built from the
    ``TrainingConfig``.

    Args:
        primus_config: Primus configuration (will be mutated – layer counts
            are reduced for consistency with the benchmark flow).
        args: CLI arguments (``--gemm-backend``, ``--gpu-arch``).

    Returns:
        dict: Profiling results in the same format as ``_run_layer_benchmark``.
    """
    module_config = primus_config.get_module_config("pre_trainer")
    _limit_layers_for_projection(module_config)
    _rescale_expert_parallelism(module_config)
    training_config = convert_primus_config_to_projection_config(primus_config)

    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    # ---- Create simulation backends ----
    gemm_backend_name = getattr(args, "gemm_backend", None)
    gpu_arch = getattr(args, "gpu_arch", None)
    gpu_clock_mhz = getattr(args, "gpu_clock_mhz", None)

    gemm_backend = get_gemm_simulation_backend(
        backend_name=gemm_backend_name,
        gpu_arch=gpu_arch,
        gpu_clock_mhz=gpu_clock_mhz,
    )
    sdpa_backend = get_sdpa_simulation_backend(
        gpu_arch=gpu_arch, gpu_clock_mhz=gpu_clock_mhz
    )

    # ---- Build profiler tree (no model needed) ----
    if is_rank_0:
        print("[Primus:Performance Projection] Building simulation profiler...")
    model_profiler_spec = get_language_model_profiler_spec(training_config)
    model_profiler = build_profiler(model_profiler_spec)

    # Wire simulation backends into the entire profiler hierarchy
    model_profiler.set_simulation_backends(gemm_backend, sdpa_backend)

    seq_len = training_config.runtime_config.sequence_length
    batch_size = training_config.runtime_config.micro_batch_size

    if is_rank_0:
        print("[Primus:Performance Projection] Simulating with:")
        print(f"  Batch Size: {batch_size}")
        print(f"  Sequence Length: {seq_len}")
        print(f"  GEMM backend: {gemm_backend.name()}")
        print(f"  SDPA backend: {sdpa_backend.name()}")
        print("" + "=" * 100)
        print("[Primus:Performance Projection] Starting layer simulation...")
        print("=" * 100)

    # Run simulation (model=None – no GPU required)
    profiling_results = model_profiler.run_layer_benchmark(
        model=None,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    return profiling_results


def _run_pipeline_simulation_megatron_zb(training_config, profiling_results):
    """
    Run pipeline simulation using the actual Megatron zero-bubble scheduler.

    This uses the same ILP-based scheduler that Megatron uses during actual training,
    which includes bubble-filling with W operations and memory-aware scheduling.

    Args:
        training_config: Training configuration
        profiling_results: Layer profiling results

    Returns:
        float: Step time in ms from pipeline simulation
    """
    from primus.backends.megatron.core.pipeline_parallel.zerobubble.scheduler import zb
    from primus.backends.megatron.core.pipeline_parallel.zerobubble.scheduler.graph import (
        GraphConfig,
    )

    # Build chunk time matrix
    chunk_time_matrix = _build_chunk_time_matrix(training_config, profiling_results)
    if chunk_time_matrix is None:
        return None

    mp_cfg = training_config.model_parallel_config
    pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1
    micro_batches = _compute_micro_batches(training_config.runtime_config, mp_cfg)

    # Extract per-stage costs (F, B, W) from chunk_time_matrix
    # For now, assume single chunk (vpp=1)
    cost_f = []
    cost_b = []
    cost_w = []
    mem_f = []
    mem_b = []
    mem_w = []

    print(
        "[Primus:Performance Projection] Using Megatron zero-bubble scheduler (ILP-based)"
    )
    print(f"  PP size: {pp_size}, Microbatches: {micro_batches}")

    for rank_idx, rank_chunks in enumerate(chunk_time_matrix):
        chunk = rank_chunks[0]  # Assume single chunk
        fwd = chunk.get("fwd", 0.0)
        bwd = chunk.get("bwd", 0.0)

        # Split backward into B and W (50/50 as approximation)
        # The Megatron scheduler expects B and W separately
        b_time = bwd * 0.5
        w_time = bwd * 0.5

        cost_f.append(float(fwd))
        cost_b.append(float(b_time))
        cost_w.append(float(w_time))

        # Memory: GraphConfig requires mem_f + mem_b + mem_w == 0 for each stage
        # F adds activation, B releases half, W releases remaining half
        act_gb = chunk.get("activation", 0.0)
        mem_f.append(float(act_gb))
        mem_b.append(float(-act_gb * 0.5))  # B releases half
        mem_w.append(float(-act_gb * 0.5))  # W releases remaining half

        print(
            f"  Stage {rank_idx}: F={fwd:.2f}ms, B={b_time:.2f}ms, W={w_time:.2f}ms, act={act_gb:.2f}GB"
        )

    # Estimate communication cost (P2P latency)
    # Use a small default value; actual value depends on hardware
    cost_comm = 0.1  # ms, placeholder

    # Create GraphConfig for Megatron ZB scheduler
    config = GraphConfig(
        mem_f=mem_f,
        mem_b=mem_b,
        mem_w=mem_w,
        cost_f=cost_f,
        cost_b=cost_b,
        cost_w=cost_w,
        cost_comm=float(cost_comm),
        n_stages=pp_size,
        n_micro=micro_batches,
    )

    # Run the Megatron ZB scheduler
    print("[Primus:Performance Projection] Running Megatron ZB schedule generation...")

    # Build graph and run initial_solution which explores multiple heuristics
    graph = zb.Graph.build_graph(pp_size, micro_batches, config)
    best_time, order, completion_time = zb.initial_solution(graph, print_result=False)

    step_time_ms = best_time

    # Calculate bubble time
    total_compute_per_mb = (
        sum(cost_f) / pp_size + sum(cost_b) / pp_size + sum(cost_w) / pp_size
    )
    ideal_time = total_compute_per_mb * micro_batches
    bubble_time = step_time_ms - ideal_time
    bubble_ratio = bubble_time / step_time_ms if step_time_ms > 0 else 0

    print("[Primus:Performance Projection] Megatron ZB Schedule Results:")
    print(f"  Step time: {step_time_ms:.2f} ms")
    print(f"  Ideal time (no bubble): {ideal_time:.2f} ms")
    print(f"  Bubble time: {bubble_time:.2f} ms ({bubble_ratio:.1%})")

    return step_time_ms


def _run_pipeline_simulation(
    training_config, profiling_results, enable_zero_bubble=False
):
    """
    Run pipeline simulation and return the step time.

    Args:
        training_config: Training configuration
        profiling_results: Layer profiling results
        enable_zero_bubble: Whether to use zero-bubble scheduling (reduces pipeline bubbles)

    Returns:
        float: Step time in ms from pipeline simulation, or None if simulation failed
    """
    # Use Megatron's actual ZB scheduler for more accurate simulation
    if enable_zero_bubble:
        try:
            return _run_pipeline_simulation_megatron_zb(
                training_config, profiling_results
            )
        except Exception as e:
            print(f"[Primus:Performance Projection] Megatron ZB scheduler failed: {e}")
            print("[Primus:Performance Projection] Falling back to simple simulator...")

    sim_config = _build_scheduler_sim_config(
        training_config, profiling_results, enable_zero_bubble
    )
    if sim_config is None:
        return None
    print("[Primus:Performance Projection] Running pipeline schedule simulator...")
    runner = SchedulerSimulationRunner(sim_config)
    simulation_runs = runner.run()
    step_time_ms = _report_simulation_results(simulation_runs, training_config)
    return step_time_ms


def _get_parameter_memory(training_config, pp_rank: int) -> float:
    profiler_spec = get_language_model_profiler_spec(training_config)
    param_profiler = build_profiler(profiler_spec)
    bytes_per_param = param_profiler.get_num_bytes_per_param()

    mp_cfg = training_config.model_parallel_config
    tp_size = getattr(mp_cfg, "tensor_model_parallel_size", 1) or 1
    cp_size = getattr(mp_cfg, "context_model_parallel_size", 1) or 1
    ep_size = getattr(mp_cfg, "expert_model_parallel_size", 1) or 1
    vpp_size = getattr(mp_cfg, "virtual_pipeline_model_parallel_size", 1) or 1
    pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1

    total_layers = getattr(training_config.model_config, "num_layers", 0) or 0
    mp_group = tp_size * cp_size * ep_size

    if pp_rank < 0 or pp_rank >= pp_size:
        raise ValueError(f"pp_rank {pp_rank} out of range (0-{pp_size-1})")

    layers = LanguageModelProfiler.get_layers_for_rank(
        None,
        global_rank=pp_rank * mp_group,
        n_layers=total_layers,
        pp_size=pp_size,
        tp_size=tp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        num_virtual_pipeline_stages=vpp_size,
    )

    param_profiler.layers = layers
    num_params = param_profiler.estimated_num_params(pp_rank * mp_group)

    return num_params * bytes_per_param / _BYTES_PER_GB


def _run_multinode_projection(
    training_config,
    single_node_time_ms,
    profiling_results,
    args,
    target_nodes: int,
    time_includes_all_microbatches: bool = False,
):
    """
    Run multinode projection to the specified target nodes.

    Args:
        training_config: Configuration object
        single_node_time_ms: Measured single-node time in ms
        profiling_results: Layer profiling results
        args: CLI arguments
        target_nodes: Target number of nodes for projection
        time_includes_all_microbatches: If True, single_node_time_ms already accounts for all microbatches
                                        (e.g., from pipeline simulation). If False, it's per-microbatch time.
    """
    import torch.distributed as dist

    # Only print from rank 0 to avoid duplicate output
    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0

    mp_config = training_config.model_parallel_config

    # Get parallelism config
    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    ep = getattr(mp_config, "expert_model_parallel_size", 1)
    cp = getattr(mp_config, "context_model_parallel_size", 1)
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))

    # Calculate minimum nodes required by parallelism config
    # EP is included in the minimum GPUs calculation (need GPUs to hold experts)
    gpus_required = tp * pp * ep * cp
    min_nodes_required = (gpus_required + gpus_per_node - 1) // gpus_per_node

    # Validate target >= minimum required
    if target_nodes < min_nodes_required:
        raise ValueError(
            f"[Primus:Multinode] ERROR: Cannot project to {target_nodes} nodes."
            f"Minimum required by parallelism config is {min_nodes_required} nodes."
            f"--target-nodes must be >= {min_nodes_required}."
        )

    # Calculate DP for scaling - EXCLUDES EP (DP scaling is independent of EP)
    # EP distributes experts but doesn't affect how many data batches can be processed in parallel
    gpus_for_dp = tp * pp * cp  # EP excluded for DP calculation
    total_gpus_target = target_nodes * gpus_per_node
    dp_target = total_gpus_target // gpus_for_dp

    if is_rank_0:
        print("" + "=" * 100)
        print("Parallelism Configuration")
        print("=" * 100)
        print(f"  TP: {tp}, PP: {pp}, EP: {ep}, CP: {cp}")
        print(f"  GPUs per Node: {gpus_per_node}")
        print(f"  Minimum GPUs Required: {gpus_required}")
        print(f"  Minimum Nodes Required: {min_nodes_required}")
        print(f"  Target Nodes: {target_nodes}")

    # Load hardware config if provided
    hardware_config_dict = None
    if hasattr(args, "hardware_config") and args.hardware_config:
        hardware_config_dict = load_hardware_config(args.hardware_config)
        if is_rank_0:
            print(f"  Using custom hardware config from: {args.hardware_config}")
    else:
        if is_rank_0:
            print(
                "  Using default hardware parameters from custom_hardware_example.yaml"
            )

    # Calculate communication times
    total_comm_time_ms, breakdown, message_info, per_layer_info = (
        calculate_collective_communication_time(
            training_config,
            target_nodes,
            gpus_per_node,
            tp,
            pp,
            ep,
            cp,
            dp_target,
            hardware_config_dict,
        )
    )

    # Benchmarked time is for the minimum node configuration
    benchmarked_time_ms = single_node_time_ms

    # When scaling DP, we need to account for gradient all-reduce
    # Check if gradient all-reduce is overlapped
    overlap_grad_reduce = getattr(mp_config, "overlap_grad_reduce", True)

    # Calculate projected time
    # NOTE: Per-microbatch compute time does NOT change with DP scaling.
    # What changes is the number of microbatches per GPU (handled later).
    #
    # When time_includes_all_microbatches (pipeline simulation), the sim was
    # already run with target_dp microbatch count, so NO further DP scaling
    # is needed — the time already reflects the target configuration.
    #
    # When NOT time_includes_all_microbatches (no pipeline sim), the
    # per-microbatch compute time is constant; total iteration time is
    # computed later as: per_microbatch_time × target_microbatches.
    projected_compute_time_ms = benchmarked_time_ms

    # 2. Handle gradient all-reduce based on overlap setting
    # NOTE: Gradient allreduce happens ONCE per iteration (after the last
    # microbatch), not once per microbatch. We track the exposed (non-overlapped)
    # portion separately and add it as a per-iteration overhead.
    grad_ar_per_iteration_ms = 0.0  # Non-overlapped allreduce time (added once)
    if dp_target > 1:
        # Calculate gradient all-reduce for target
        _, target_breakdown, target_message_info, _ = (
            calculate_collective_communication_time(
                training_config,
                target_nodes,
                gpus_per_node,
                tp,
                pp,
                ep,
                cp,
                dp_target,
                hardware_config_dict,
            )
        )
        target_grad_ar = target_breakdown.get("gradient_allreduce", 0)
        moe_ar_no_overlap = target_message_info.get("moe_ar_no_overlap", False)

        if moe_ar_no_overlap:
            # MoE with EP: all2all sync barriers prevent gradient allreduce
            # from overlapping effectively with backward. Add the full
            # allreduce time as a per-iteration overhead.
            grad_ar_per_iteration_ms = target_grad_ar
        elif overlap_grad_reduce:
            # Overlapped: all-reduce runs concurrently with backward of last microbatch.
            # Only backward (~63% of compute) can overlap.
            # Exposed portion = max(0, allreduce - backward_time)
            backward_time = projected_compute_time_ms * 0.63
            grad_ar_per_iteration_ms = max(0, target_grad_ar - backward_time)
        else:
            # Not overlapped: all-reduce runs sequentially after backward
            grad_ar_per_iteration_ms = target_grad_ar

    # Per-microbatch projected time stays as compute only
    projected_time_ms = projected_compute_time_ms

    # For reporting, get full breakdown for target
    total_comm_time_ms, breakdown, message_info, per_layer_info = (
        calculate_collective_communication_time(
            training_config,
            target_nodes,
            gpus_per_node,
            tp,
            pp,
            ep,
            cp,
            dp_target,
            hardware_config_dict,
        )
    )

    # Add exposed FSDP communication time to projected time
    # (total_comm_time_ms already has overlap accounted for - it's the critical path)
    fsdp_exposed = message_info.get("fsdp_exposed_ms", 0)
    if fsdp_exposed > 0:
        projected_time_ms += fsdp_exposed

    # Calculate values needed for both printing and return
    target_world_size = target_nodes * gpus_per_node
    target_dp_for_microbatch = target_world_size // (tp * pp * cp)

    # Get runtime config for tokens/s calculation
    runtime_config = training_config.runtime_config
    seq_len = getattr(runtime_config, "sequence_length", 4096)
    global_batch = getattr(runtime_config, "global_batch_size", 128)
    micro_batch = getattr(runtime_config, "micro_batch_size", 1)

    # Calculate number of microbatches per GPU for the target configuration
    target_microbatches_per_gpu = (
        global_batch // (micro_batch * target_dp_for_microbatch)
        if target_dp_for_microbatch > 0
        else 1
    )

    # Handle edge case where global_batch is smaller than micro_batch * target_dp
    if target_microbatches_per_gpu == 0:
        if is_rank_0:
            required_batch = micro_batch * target_dp_for_microbatch
            print(
                f"⚠️  WARNING: global_batch_size ({global_batch}) < micro_batch ({micro_batch}) × target_DP ({target_dp_for_microbatch}) = {required_batch}"
            )
            print(
                f"   Consider increasing global_batch_size to at least {required_batch} for {target_nodes} nodes"
            )
            print(
                f"   Using 1 microbatch for projection (effective global_batch = {micro_batch * target_dp_for_microbatch})"
            )
        target_microbatches_per_gpu = 1

    # Estimate optimizer step time (once per iteration, after all microbatches)
    gpu_arch = getattr(args, "gpu_arch", None)
    optimizer_step_ms = _estimate_optimizer_step_ms(
        training_config, dp_target, gpu_arch
    )

    # Build full iteration time:
    #   compute (per-microbatch) × num_microbatches + gradient allreduce + optimizer step
    if time_includes_all_microbatches:
        full_iteration_time_ms = (
            projected_time_ms + grad_ar_per_iteration_ms + optimizer_step_ms
        )
        time_breakdown_str = (
            f"{full_iteration_time_ms:.3f} ms (from pipeline simulation"
        )
        if grad_ar_per_iteration_ms > 0:
            time_breakdown_str += f" + {grad_ar_per_iteration_ms:.1f} ms grad AR"
        time_breakdown_str += f" + {optimizer_step_ms:.1f} ms optimizer)"
    else:
        compute_total = projected_time_ms * target_microbatches_per_gpu
        full_iteration_time_ms = (
            compute_total + grad_ar_per_iteration_ms + optimizer_step_ms
        )
        time_breakdown_str = f"{full_iteration_time_ms:.3f} ms ({target_microbatches_per_gpu} microbatches × {projected_time_ms:.3f} ms"
        if grad_ar_per_iteration_ms > 0:
            time_breakdown_str += f" + {grad_ar_per_iteration_ms:.1f} ms grad AR"
        time_breakdown_str += f" + {optimizer_step_ms:.1f} ms optimizer)"

    # Calculate tokens/s/GPU (tokens processed per second per GPU)
    tokens_per_iter = global_batch * seq_len
    target_tokens_per_sec_per_gpu = (
        tokens_per_iter * 1000 / full_iteration_time_ms / target_world_size
        if full_iteration_time_ms > 0
        else 0
    )

    # Print results (only from rank 0)
    if is_rank_0:
        print("" + "=" * 100)
        print("Multinode Scaling Projection Results")
        print("=" * 100)
        print(f"📊 Parallelism: TP={tp}, PP={pp}, EP={ep}, CP={cp}")

        # Communication Breakdown
        print("📡 Communication Breakdown:")
        for op_name, op_time in breakdown.items():
            if op_time > 0:
                print(f"   {op_name}: {op_time:.3f} ms", end="")
                if (
                    op_name == "gradient_allreduce"
                    and "gradient_allreduce_size_mb" in message_info
                ):
                    moe_no_overlap = message_info.get("moe_ar_no_overlap", False)
                    if moe_no_overlap:
                        detail = " [MoE: NOT overlapped]"
                        expert_ms = message_info.get("expert_ar_time_ms", 0)
                        non_expert_ms = message_info.get("non_expert_ar_time_ms", 0)
                        dp_reps = message_info.get("expert_ar_dp_replicas", 0)
                        detail += f"\n     Expert AR: {expert_ms:.1f} ms (across {dp_reps} nodes)"
                        detail += f" | Non-expert AR: {non_expert_ms:.1f} ms"
                    else:
                        overlapped_flag = message_info.get(
                            "gradient_allreduce_overlapped", False
                        )
                        detail = " [OVERLAPPED]" if overlapped_flag else ""
                    print(
                        f" (message: {message_info['gradient_allreduce_size_mb']:.2f} MB){detail}"
                    )
                elif op_name == "moe_a2a_fwd" and "moe_a2a_size_mb" in message_info:
                    print(
                        f" (message: {message_info['moe_a2a_size_mb']:.2f} MB, {message_info['num_moe_layers']} layers × {message_info['moe_a2a_per_layer_fwd']:.3f} ms/layer)"
                    )
                elif op_name == "moe_a2a_bwd" and "moe_a2a_size_mb" in message_info:
                    print(
                        f" (message: {message_info['moe_a2a_size_mb']:.2f} MB, {message_info['num_moe_layers']} layers × {message_info['moe_a2a_per_layer_fwd']:.3f} ms/layer)"
                    )
                else:
                    print("")
        print(f"   Total Communication (critical path): {total_comm_time_ms:.3f} ms")

        # Target Configuration Summary (at the end for easy visibility)
        print(f"🎯 Target Configuration ({target_nodes} nodes):")
        print(f"   Nodes: {target_nodes}, GPUs: {target_world_size}")
        print(f"   TP={tp}, PP={pp}, EP={ep}, CP={cp}, DP={target_dp_for_microbatch}")
        print(f"   Iteration Time: {time_breakdown_str}")
        print(f"   Tokens/s/GPU: {target_tokens_per_sec_per_gpu:,.0f}")
        print("=" * 100)

    # Return results for final summary
    return {
        "target_nodes": target_nodes,
        "target_gpus": target_world_size,
        "tp": tp,
        "pp": pp,
        "ep": ep,
        "cp": cp,
        "dp": target_dp_for_microbatch,
        "iteration_time_ms": full_iteration_time_ms,
        "tokens_per_sec_per_gpu": target_tokens_per_sec_per_gpu,
    }


def launch_projection_from_cli(args, overrides):
    """
    Entry point for the 'performance_projection' subcommand.

    Benchmarks Megatron transformer layers and aggregates performance metrics.

    If --target-nodes is specified, also runs multinode scaling projection.
    If the parallelism configuration requires multiple nodes, automatically reduces
    to single-node for benchmarking and estimates performance with PP overhead.

    Args:
        args: Command-line arguments
        overrides: Configuration overrides
    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"[Primus:Performance Projection] Config file '{cfg_path}' not found."
        )

    # Load Primus configuration
    primus_config, unknown_overrides = load_primus_config(args, overrides)
    primus_config_original = copy.deepcopy(primus_config)

    # Check if we need to reduce config for single-node benchmarking
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))

    # Get target nodes from CLI flag (--target-nodes)
    target_nodes = getattr(args, "target_nodes", None)

    # Store original parallelism before any modifications
    module_config = primus_config.get_module_config("pre_trainer")
    reduction_info = _calculate_single_node_config(
        copy.deepcopy(module_config), gpus_per_node
    )

    # Calculate minimum nodes required
    min_nodes_required = reduction_info["original_nodes_required"]

    # If target_nodes not specified, default to minimum required
    if target_nodes is None:
        target_nodes = min_nodes_required

    if reduction_info["adjusted"]:
        print("" + "=" * 100)
        print("[Primus:Performance Projection] Multi-node configuration detected")
        print("=" * 100)
        print(f"  Original configuration requires {min_nodes_required} nodes minimum:")
        print(
            f"    TP={reduction_info['original_tp']}, PP={reduction_info['original_pp']}, "
            f"EP={reduction_info['original_ep']}, CP={reduction_info['original_cp']}"
        )
        print("  Reducing to single-node configuration for benchmarking:")
        print(
            f"    TP={reduction_info['original_tp']}, PP={reduction_info['benchmark_pp']}, "
            f"EP={reduction_info['benchmark_ep']}, CP={reduction_info['original_cp']}"
        )

        # Show what was changed
        changes = []
        if reduction_info["original_pp"] != reduction_info["benchmark_pp"]:
            changes.append(
                f"PP {reduction_info['original_pp']} → {reduction_info['benchmark_pp']}"
            )
        if reduction_info["original_ep"] != reduction_info["benchmark_ep"]:
            changes.append(
                f"EP {reduction_info['original_ep']} → {reduction_info['benchmark_ep']}"
            )

        if changes:
            print(f"    ({', '.join(changes)})")

        print("  Will estimate performance by adding PP communication overhead back.")
        print("=" * 100)

        # Apply the reduction to the config used for benchmarking
        primus_config.get_module_config("pre_trainer").pipeline_model_parallel_size = (
            reduction_info["benchmark_pp"]
        )
        primus_config.get_module_config("pre_trainer").expert_model_parallel_size = (
            reduction_info["benchmark_ep"]
        )
        # Also propagate num_experts adjustment so that the profiler sees
        # the correct experts_per_rank (e.g. 128/4=32, not 256/4=64).
        if reduction_info.get("benchmark_num_experts") is not None:
            primus_config.get_module_config("pre_trainer").num_experts = reduction_info[
                "benchmark_num_experts"
            ]

    # Determine profiling mode
    profiling_mode = getattr(args, "profiling_mode", "benchmark")

    if profiling_mode == "simulate":
        # Pure simulation – no GPU / trainer required
        profiling_results = _run_layer_simulation(primus_config, args)
    elif profiling_mode == "both":
        # Run both benchmark and simulation, keep benchmark results for
        # downstream pipeline simulation / multinode projection, but print
        # a side-by-side comparison.
        sim_results = _run_layer_simulation(copy.deepcopy(primus_config), args)
        bench_results = _run_layer_benchmark(primus_config, unknown_overrides)

        is_rank_0 = int(os.getenv("RANK", "0")) == 0
        if is_rank_0:
            print("\n" + "=" * 100)
            print("[Primus:Performance Projection] Benchmark vs Simulation Comparison")
            print("=" * 100)
            for key in bench_results:
                if key in ("embedding", "output"):
                    continue
                bd = bench_results[key]
                sd = sim_results.get(key, {})
                if not isinstance(bd, dict):
                    continue
                lt = bd.get("type", key)
                b_fwd = bd.get("forward_time_ms", 0)
                b_bwd = bd.get("backward_time_ms", 0)
                s_fwd = sd.get("forward_time_ms", 0)
                s_bwd = sd.get("backward_time_ms", 0)
                fwd_err = ((s_fwd - b_fwd) / b_fwd * 100) if b_fwd else 0
                bwd_err = ((s_bwd - b_bwd) / b_bwd * 100) if b_bwd else 0
                print(f"  Layer type: {lt}")
                print(
                    f"    Forward:  bench={b_fwd:.2f} ms  sim={s_fwd:.2f} ms  (err={fwd_err:+.1f}%)"
                )
                print(
                    f"    Backward: bench={b_bwd:.2f} ms  sim={s_bwd:.2f} ms  (err={bwd_err:+.1f}%)"
                )
            print("=" * 100)

        # Use benchmark results for the rest of the pipeline
        profiling_results = bench_results
    else:
        # Default: actual GPU benchmark
        profiling_results = _run_layer_benchmark(primus_config, unknown_overrides)

    # Use original config for projection calculations
    training_config = convert_primus_config_to_projection_config(primus_config_original)

    # Update data_parallel_size based on target_nodes
    # This ensures the pipeline simulation calculates the correct number of microbatches
    # NOTE: For MoE models, EP does NOT reduce DP (experts are distributed but tokens are replicated)
    # DP = world_size / (TP × PP × CP)  [EP is excluded]
    mp_config = training_config.model_parallel_config
    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    cp = getattr(mp_config, "context_parallel_size", 1) or 1
    ep = getattr(mp_config, "expert_model_parallel_size", 1) or 1

    # Calculate DP for the TARGET configuration (what we're projecting to)
    # The pipeline simulator simulates the target config, so it needs target DP for microbatch calculation
    target_world_size = target_nodes * gpus_per_node

    # For MoE models: DP calculation excludes EP since experts are distributed but data is replicated
    target_dp = target_world_size // (tp * pp * cp)

    # Also show benchmark config for reference
    benchmark_world_size = gpus_per_node  # Benchmarking always happens on 1 node
    benchmark_pp = reduction_info.get("benchmark_pp", pp)
    benchmark_ep = reduction_info.get("benchmark_ep", ep)
    benchmark_dp = benchmark_world_size // (tp * benchmark_pp * cp)

    # Only print from rank 0
    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    if is_rank_0:
        print("[Primus:Performance Projection] Configuration Summary:")
        print(
            f"  Benchmark Config: PP={benchmark_pp}, EP={benchmark_ep}, TP={tp}, CP={cp}, DP={benchmark_dp} (1 node)"
        )
        print(
            f"  Target Config: PP={pp}, EP={ep}, TP={tp}, CP={cp}, DP={target_dp} ({target_nodes} nodes)"
        )

    # Use BENCHMARK DP for pipeline simulation to get consistent baseline
    # The multinode projection will then scale from this baseline to target
    global_batch = training_config.runtime_config.global_batch_size
    micro_batch = training_config.runtime_config.micro_batch_size
    # Pipeline simulation must use the TARGET DP for microbatch count because it
    # simulates the target PP stages.  The multinode projection later will NOT
    # re-scale the pipeline time by DP when min_dp == target_dp (which is the
    # common case for configs that already require all target GPUs for their
    # parallelism dims).  Using benchmark_dp here would give 2× too many
    # microbatches when benchmark_dp < target_dp.
    target_microbatches = (
        global_batch // (micro_batch * target_dp) if target_dp > 0 else 1
    )
    target_microbatches = max(1, target_microbatches)
    benchmark_microbatches = global_batch // (micro_batch * benchmark_dp)
    if is_rank_0:
        print(
            f"  Benchmark Microbatches: {benchmark_microbatches} (global_batch={global_batch}, micro_batch={micro_batch}, benchmark_dp={benchmark_dp})"
        )
        print(
            f"  Target Microbatches: {target_microbatches} (global_batch={global_batch}, micro_batch={micro_batch}, target_dp={target_dp})"
        )

    # Set data_parallel_size to target_dp so the pipeline simulation and
    # _compute_micro_batches use the correct microbatch count.
    training_config.runtime_config.data_parallel_size = target_dp

    # If EP was rescaled, adjust profiling_results to add EP overhead BEFORE pipeline simulation
    ep_overhead_applied = False
    if (
        reduction_info["adjusted"]
        and reduction_info["original_ep"] != reduction_info["benchmark_ep"]
    ):
        original_ep = reduction_info["original_ep"]
        benchmark_ep = reduction_info["benchmark_ep"]
        original_num_experts = reduction_info.get("original_num_experts")
        benchmark_num_experts = reduction_info.get("benchmark_num_experts")

        # Load hardware config if provided
        hardware_config_dict = None
        if hasattr(args, "hardware_config") and args.hardware_config:
            hardware_config_dict = load_hardware_config(args.hardware_config)

        # Calculate EP communication overhead per layer (A2A delta)
        fwd_overhead_per_layer, bwd_overhead_per_layer = (
            _estimate_ep_communication_overhead(
                training_config,
                original_ep,
                benchmark_ep,
                hardware_config_dict,
            )
        )

        # EP compute scaling.  Per-GPU routed compute is EP-invariant
        # (A2A redistributes tokens so each GPU always processes
        # batch_tokens × topk total token-expert pairs).  When
        # _rescale_expert_parallelism preserves experts_per_rank the
        # profiled/simulated MLP time already reflects the correct
        # per-rank workload, so ep_mlp_scale == 1.0.
        ep_mlp_scale = _compute_ep_mlp_scale(
            training_config.model_config,
            benchmark_ep,
            original_ep,
            original_num_experts=original_num_experts,
            benchmark_num_experts=benchmark_num_experts,
        )

        if is_rank_0:
            print(
                "[Primus:Performance Projection] Adjusting profiling results for EP scaling:"
            )
            print(f"  EP rescaled: {benchmark_ep} → {original_ep}")
            if original_num_experts is not None and benchmark_num_experts is not None:
                orig_epr = original_num_experts // original_ep
                bench_epr = benchmark_num_experts // benchmark_ep
                print(
                    f"  Experts per rank: benchmark={bench_epr} "
                    f"(E={benchmark_num_experts}, EP={benchmark_ep}), "
                    f"target={orig_epr} "
                    f"(E={original_num_experts}, EP={original_ep})"
                )
            print(f"  MLP time scale factor: {ep_mlp_scale:.3f}")
            if fwd_overhead_per_layer > 0 or bwd_overhead_per_layer > 0:
                print(f"  Adding per-layer All-to-All overhead:")
                print(f"    Forward:  +{fwd_overhead_per_layer:.3f} ms/layer")
                print(f"    Backward: +{bwd_overhead_per_layer:.3f} ms/layer")

        # Adjust MoE layer times in profiling_results using a DELTA approach:
        # new_layer_time = old_layer_time + (mlp_delta) + (a2a_delta)
        # This preserves TP AllReduce, A2A(benchmark_ep), LayerNorm, and
        # other components that are baked into the profiled layer time.
        moe_layers_adjusted = 0
        for layer_idx, layer_data in profiling_results.items():
            if isinstance(layer_data, dict) and layer_data.get("type") == "moe":
                old_fwd = layer_data.get("forward_time_ms", 0)
                old_bwd = layer_data.get("backward_time_ms", 0)

                mlp_info = layer_data.get("mlp", {})
                mlp_fwd = mlp_info.get("forward_time_ms", 0)
                mlp_bwd = mlp_info.get("backward_time_ms", 0)

                # Compute MLP delta (usually 0 when scale == 1.0)
                new_mlp_fwd = mlp_fwd * ep_mlp_scale
                new_mlp_bwd = mlp_bwd * ep_mlp_scale
                mlp_delta_fwd = new_mlp_fwd - mlp_fwd
                mlp_delta_bwd = new_mlp_bwd - mlp_bwd

                # Delta approach: start from the full profiled layer time
                # (which includes TP AR, A2A(benchmark_ep), norms, etc.)
                # and add the MLP compute delta + A2A comm delta.
                new_fwd = old_fwd + mlp_delta_fwd + fwd_overhead_per_layer
                new_bwd = old_bwd + mlp_delta_bwd + bwd_overhead_per_layer

                if is_rank_0 and moe_layers_adjusted == 0:
                    print(f"  MoE layer adjustment (per layer):")
                    print(
                        f"    MLP fwd: {mlp_fwd:.2f} → {new_mlp_fwd:.2f} ms (×{ep_mlp_scale:.3f})"
                    )
                    print(
                        f"    MLP bwd: {mlp_bwd:.2f} → {new_mlp_bwd:.2f} ms (×{ep_mlp_scale:.3f})"
                    )
                    print(f"    A2A fwd delta: +{fwd_overhead_per_layer:.3f} ms")
                    print(f"    A2A bwd delta: +{bwd_overhead_per_layer:.3f} ms")
                    print(f"    Layer fwd: {old_fwd:.2f} → {new_fwd:.2f} ms")
                    print(f"    Layer bwd: {old_bwd:.2f} → {new_bwd:.2f} ms")

                layer_data["forward_time_ms"] = new_fwd
                layer_data["backward_time_ms"] = new_bwd
                # Update MLP sub-component times too
                if mlp_info:
                    mlp_info["forward_time_ms"] = new_mlp_fwd
                    mlp_info["backward_time_ms"] = new_mlp_bwd
                moe_layers_adjusted += 1

        if is_rank_0:
            print(f"  Adjusted {moe_layers_adjusted} MoE layer(s) in profiling results")
        ep_overhead_applied = True

    # Check if zero-bubble scheduling is enabled in the original config
    original_module_config = primus_config_original.get_module_config("pre_trainer")
    enable_zero_bubble = getattr(original_module_config, "enable_zero_bubble", False)

    # Skip pipeline simulation if pp=1 (no pipeline parallelism)
    if pp == 1:
        pipeline_simulation_time_ms = None
        if is_rank_0:
            print("[Primus:Performance Projection] Skipping pipeline simulation (PP=1)")
    else:
        pipeline_simulation_time_ms = _run_pipeline_simulation(
            training_config, profiling_results, enable_zero_bubble
        )

    # Run multinode projection if target_nodes > min_nodes_required (scaling up)
    # or always run to show performance summary
    if target_nodes >= min_nodes_required:
        if is_rank_0:
            print("" + "=" * 100)
            print("[Primus:Performance] Running multinode scaling projection")
            print("=" * 100)

        # Use pipeline simulation time if available, otherwise extract from profiling
        if pipeline_simulation_time_ms is not None:
            if is_rank_0:
                print(
                    f"[Primus:Performance Projection] Using pipeline simulation time: {pipeline_simulation_time_ms:.2f} ms"
                )
            # Pipeline simulation already accounts for PP communication and bubbles
            # No need to add additional PP overhead
            benchmarked_time_ms = pipeline_simulation_time_ms
            if is_rank_0:
                print(
                    f"  (Pipeline simulation already includes PP={reduction_info['original_pp']} effects)"
                )
        else:
            if is_rank_0:
                print(
                    "[Primus:Performance Projection] Pipeline simulation not available, using extrapolated time from profiling"
                )
            measured_time_ms = extract_single_node_time_from_profiling(
                profiling_results, training_config
            )

            # If we reduced PP for benchmarking, estimate the time with PP overhead
            if reduction_info["adjusted"]:
                # Load hardware config if provided
                hardware_config_dict = None
                if hasattr(args, "hardware_config") and args.hardware_config:
                    hardware_config_dict = load_hardware_config(args.hardware_config)

                # Estimate PP overhead for original configuration
                pp_overhead_ms = _estimate_pp_communication_overhead(
                    training_config, reduction_info["original_pp"], hardware_config_dict
                )

                benchmarked_time_ms = measured_time_ms + pp_overhead_ms

                if is_rank_0:
                    print("[Primus:Performance Projection] Time Adjustment:")
                    print(
                        f"  Measured time (PP={reduction_info['benchmark_pp']}): {measured_time_ms:.2f} ms"
                    )
                    print(
                        f"  Estimated PP overhead (PP={reduction_info['original_pp']}): {pp_overhead_ms:.2f} ms"
                    )
                    print(f"  Estimated time: {benchmarked_time_ms:.2f} ms")
            else:
                benchmarked_time_ms = measured_time_ms

            # If EP was rescaled and pipeline simulation wasn't used, add EP overhead here
            # (If pipeline simulation was used, EP overhead was already applied to profiling_results
            # including both compute scaling and A2A comm overhead)
            if (
                not ep_overhead_applied
                and reduction_info["adjusted"]
                and reduction_info["original_ep"] != reduction_info["benchmark_ep"]
            ):
                original_ep = reduction_info["original_ep"]
                benchmark_ep_val = reduction_info["benchmark_ep"]

                # Get the number of MoE layers
                moe_pattern = getattr(
                    training_config.model_config, "moe_layer_pattern", []
                )
                if not moe_pattern:
                    # If no pattern, check if model has MoE layers
                    num_moe_layers = getattr(
                        training_config.model_config, "num_moe_layers", 0
                    )
                else:
                    num_moe_layers = sum(1 for x in moe_pattern if x == 1)

                if num_moe_layers > 0:
                    # Calculate EP communication overhead per layer
                    fwd_overhead_per_layer, bwd_overhead_per_layer = (
                        _estimate_ep_communication_overhead(
                            training_config,
                            original_ep,
                            benchmark_ep_val,
                            hardware_config_dict,
                        )
                    )

                    # Total EP overhead = per-layer overhead * number of MoE layers
                    total_ep_overhead_ms = (
                        fwd_overhead_per_layer + bwd_overhead_per_layer
                    ) * num_moe_layers

                    # EP compute scaling — per-GPU MoE routed compute is
                    # EP-invariant (see _compute_ep_mlp_scale docstring).
                    original_num_experts = reduction_info.get("original_num_experts")
                    benchmark_num_experts = reduction_info.get("benchmark_num_experts")
                    ep_mlp_scale = _compute_ep_mlp_scale(
                        training_config.model_config,
                        benchmark_ep_val,
                        original_ep,
                        original_num_experts=original_num_experts,
                        benchmark_num_experts=benchmark_num_experts,
                    )
                    # Estimate MLP portion of MoE layer time from profiling results
                    mlp_time_reduction = 0.0
                    for layer_idx, layer_data in profiling_results.items():
                        if (
                            isinstance(layer_data, dict)
                            and layer_data.get("type") == "moe"
                        ):
                            mlp_info = layer_data.get("mlp", {})
                            mlp_total = mlp_info.get(
                                "forward_time_ms", 0
                            ) + mlp_info.get("backward_time_ms", 0)
                            mlp_time_reduction = mlp_total * (1 - ep_mlp_scale)
                            break  # All MoE layers have same profiled time

                    total_mlp_reduction_ms = mlp_time_reduction * num_moe_layers

                    if is_rank_0:
                        print(
                            "[Primus:Performance Projection] EP Compute + Communication Adjustment:"
                        )
                        print(f"  EP rescaled: {benchmark_ep_val} → {original_ep}")
                        print(f"  Number of MoE layers: {num_moe_layers}")
                        print(f"  MLP time scale factor: {ep_mlp_scale:.3f}")
                        print(
                            f"  Total MLP compute reduction: -{total_mlp_reduction_ms:.3f} ms"
                        )
                        print(
                            f"  Total A2A comm overhead:     +{total_ep_overhead_ms:.3f} ms"
                        )
                        net_change = total_ep_overhead_ms - total_mlp_reduction_ms
                        print(f"  Net adjustment: {net_change:+.3f} ms")

                    benchmarked_time_ms += total_ep_overhead_ms - total_mlp_reduction_ms
                    if is_rank_0:
                        print(f"  Adjusted time: {benchmarked_time_ms:.3f} ms")

        # Run multinode projection
        # Pass flag indicating whether time already includes all microbatches (from pipeline simulation)
        time_includes_all_microbatches = pipeline_simulation_time_ms is not None
        _run_multinode_projection(
            training_config,
            benchmarked_time_ms,
            profiling_results,
            args,
            target_nodes,
            time_includes_all_microbatches,
        )
