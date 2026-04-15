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
from primus.core.projection.module_profilers.optimizer import OptimizerProfiler
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


def _calculate_min_gpus(tp, pp, ep, cp):
    """Calculate minimum GPUs required by parallelism config.

    For MoE models (EP > 1), CP is folded into EP via MoE Parallel Folding:
    the CP ranks are a subset of the EP ranks, so the minimum GPU count is
    TP × PP × EP.  Constraints: CP ≤ EP and EP % CP == 0.

    For dense models (EP ≤ 1), CP is an independent parallelism axis, so the
    minimum GPU count is TP × PP × CP.

    Note: DP is *not* affected by this folding — EP borrows from the DP
    dimension, so DP = world_size / (TP × PP × CP) in both cases.
    """
    if ep > 1:
        # MoE: CP is folded into EP (MoE Parallel Folding)
        return tp * pp * ep
    else:
        # Dense: CP is an independent axis
        return tp * pp * cp


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

    hw = dict(hardware_config) if hardware_config else {}

    # Setup collective args
    coll_args = get_default_args(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
        hardware_config=hw if hw else None,
    )

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

    # Get model parallel config
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
            msg_scale = (dp_replicas - 1) / dp_replicas
            expert_ar_time_ms = 2 * expert_grad_size * msg_scale / (pod_bw * 1e9) * 1e3

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

        # Use the corrected component-based A2A model
        a2a_per_layer_ms = _estimate_a2a_per_layer_ms(
            training_config, ep, hardware_config
        )

        total_a2a_fwd = a2a_per_layer_ms * num_moe_layers
        total_a2a_bwd = total_a2a_fwd

        breakdown["moe_a2a_fwd"] = total_a2a_fwd
        breakdown["moe_a2a_bwd"] = total_a2a_bwd
        message_info["moe_a2a_size"] = dispatch_size
        message_info["moe_a2a_size_mb"] = dispatch_size / (1024 * 1024)
        message_info["moe_a2a_per_layer_fwd"] = a2a_per_layer_ms
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
            # Use the corrected component-based A2A model
            a2a_per_layer_ms = _estimate_a2a_per_layer_ms(
                training_config, ep, hardware_config
            )
            layer_comm["communications"].append(
                {
                    "type": "MoE All-to-All (fwd+bwd)",
                    "time_ms": a2a_per_layer_ms * 2,  # fwd + bwd
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
            total_fsdp_ag = breakdown.get("fsdp_allgather_fwd", 0)
            total_fsdp_rs = breakdown.get("fsdp_reducescatter_bwd", 0)

            # Overlap factor applied uniformly to all FSDP
            # communication - (AllGather fwd, AllGather recompute, ReduceScatter).
            # This number is based on observed overlap in the benchmarked run of llama3-70b.
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

    # disable pipeline model parallelism for single-node layer benchmarking
    module_config.pipeline_model_parallel_size = 1
    # PP=1 cannot use interleaved schedule (VPP>1)
    if hasattr(module_config, "virtual_pipeline_model_parallel_size"):
        module_config.virtual_pipeline_model_parallel_size = 1
    # Explicit layout is only meaningful with PP>1/VPP mapping.
    if hasattr(module_config, "pipeline_model_parallel_layout"):
        module_config.pipeline_model_parallel_layout = None
    for attr in (
        "num_layers_per_virtual_pipeline_stage",
        "num_virtual_stages_per_pipeline_rank",
    ):
        if hasattr(module_config, attr):
            setattr(module_config, attr, None)


def _rescale_expert_parallelism(module_config):
    """
    Cap expert_model_parallel_size so that EP * TP <= GPUs_per_node and adjust num_experts.

    With MoE Parallel Folding, CP is folded into EP (CP ranks are a subset of
    EP ranks), so the minimum GPUs for a MoE config is EP * TP, not EP * TP * CP.
    """
    expert_mp_size = getattr(module_config, "expert_model_parallel_size", None)
    if expert_mp_size is None or expert_mp_size <= _MAX_EXPERT_PARALLEL_SIZE:
        current_tp = getattr(module_config, "tensor_model_parallel_size", 1) or 1
        current_cp = getattr(module_config, "context_parallel_size", 1) or 1
        if expert_mp_size is None:
            expert_mp_size = 1
        # MoE Parallel Folding: CP is folded into EP, so min GPUs = EP * TP
        if expert_mp_size * current_tp <= _MAX_EXPERT_PARALLEL_SIZE:
            return None

    num_experts = getattr(module_config, "num_experts", None)
    current_tp = getattr(module_config, "tensor_model_parallel_size", 1) or 1
    current_cp = getattr(module_config, "context_parallel_size", 1) or 1
    # MoE Parallel Folding: CP is folded into EP, so only TP contributes to
    # the per-EP-rank GPU cost.
    total_parallel_product = max(1, current_tp)
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


def _calculate_single_node_config(
    original_config, gpus_per_node=8, benchmark_gpus=None
):
    """
    Calculate a reduced parallelism configuration that fits on the benchmark GPU count.

    When ``benchmark_gpus`` is smaller than ``gpus_per_node``, this enables
    *sub-node benchmarking*

    Strategy:
    1. Reduce PP to 1 (easiest to add back communication overhead)
    2. If still doesn't fit, reduce EP (with decomposed A2A benchmarking,
       measured compute stays accurate; only A2A is replaced analytically)
    3. If still doesn't fit, reduce TP (scale compute and add AllReduce overhead)
    4. Keep CP unchanged
    5. Return the adjustment info for later baseline correction

    Args:
        original_config: Original module config
        gpus_per_node: Number of GPUs per node (default 8)
        benchmark_gpus: Number of GPUs available for benchmarking.
                        If None, defaults to gpus_per_node (full node benchmarking).
                        Set to a smaller value for sub-node benchmarking.

    Returns:
        dict with keys:
            'adjusted': bool - whether adjustment was needed
            'original_pp': int - original PP value
            'benchmark_pp': int - PP for benchmarking
            'original_nodes_required': int - original minimum nodes
            'original_tp': int - original TP value
            'benchmark_tp': int - TP for benchmarking
            'original_ep': int - original EP value
            'benchmark_ep': int - EP for benchmarking
            'original_cp': int - original CP value
            'original_num_experts': int/None - original num_experts
            'benchmark_num_experts': int/None - num_experts for benchmarking
            'benchmark_gpus': int - number of GPUs used for benchmarking
    """
    if benchmark_gpus is None:
        benchmark_gpus = gpus_per_node

    tp = getattr(original_config, "tensor_model_parallel_size", 1) or 1
    pp = getattr(original_config, "pipeline_model_parallel_size", 1) or 1
    ep = getattr(original_config, "expert_model_parallel_size", 1) or 1
    cp = getattr(original_config, "context_parallel_size", 1) or 1
    num_experts = getattr(original_config, "num_experts", None)

    gpus_required = _calculate_min_gpus(tp, pp, ep, cp)
    nodes_required = (gpus_required + gpus_per_node - 1) // gpus_per_node

    # If already fits in benchmark_gpus, no adjustment needed - keep all parallelism dimensions unchanged
    if gpus_required <= benchmark_gpus:
        print(
            f"[Primus:Performance Projection] Configuration already fits on {benchmark_gpus} GPUs "
            f"(requires {gpus_required} GPUs: TP={tp}, PP={pp}, EP={ep}, CP={cp}). "
            f"No parallelism reduction needed."
        )
        return {
            "adjusted": False,
            "original_pp": pp,
            "benchmark_pp": pp,
            "original_nodes_required": nodes_required,
            "original_tp": tp,
            "benchmark_tp": tp,
            "original_ep": ep,
            "benchmark_ep": ep,
            "original_cp": cp,
            "original_num_experts": num_experts,
            "benchmark_num_experts": num_experts,
            "benchmark_gpus": benchmark_gpus,
        }

    # Step 1: Reduce PP to 1
    benchmark_pp = 1
    benchmark_tp = tp
    benchmark_ep = ep
    benchmark_num_experts = num_experts
    benchmark_gpus_required = _calculate_min_gpus(
        benchmark_tp, benchmark_pp, benchmark_ep, cp
    )

    # Step 2: If still doesn't fit, reduce EP first (preferred now that we have
    # decomposed A2A benchmarking — the measured compute stays accurate and
    # only the A2A portion is replaced analytically for the target EP)
    if benchmark_gpus_required > benchmark_gpus:
        print(
            f"[Primus:Performance Projection] After reducing PP to 1, "
            f"config still requires {benchmark_gpus_required} GPUs (TP={benchmark_tp}, EP={benchmark_ep}, CP={cp})."
        )
        print(
            f"[Primus:Performance Projection] Reducing EP to fit on {benchmark_gpus} GPUs..."
        )

        # Find maximum EP that fits: TP * PP(=1) * EP * CP <= benchmark_gpus
        max_ep_for_benchmark = benchmark_gpus // max(benchmark_tp * cp, 1)
        if max_ep_for_benchmark < 1:
            max_ep_for_benchmark = 1

        # EP should not exceed original EP
        benchmark_ep = min(benchmark_ep, max_ep_for_benchmark)

        # When benchmark EP >= 2, keep num_experts at the original value so
        # the measured A2A has the correct routing sparsity (topk/num_experts).
        # This changes experts_per_rank, so compute_scale is applied later.
        #
        # When benchmark EP == 1, there is no A2A to measure (analytical is
        # used instead), so we reduce num_experts proportionally to preserve
        # experts_per_rank for accurate compute measurement.
        if num_experts is not None and benchmark_ep >= 2 and benchmark_ep < ep:
            benchmark_num_experts = num_experts
        elif num_experts is not None and benchmark_ep < ep:
            experts_per_rank = math.ceil(num_experts / ep)
            topk = getattr(original_config, "moe_router_topk", 2) or 2
            benchmark_num_experts = max(benchmark_ep * experts_per_rank, topk)
        else:
            benchmark_num_experts = num_experts

        benchmark_gpus_required = _calculate_min_gpus(
            benchmark_tp, benchmark_pp, benchmark_ep, cp
        )

        if benchmark_ep < ep:
            print(
                f"[Primus:Performance Projection] Reduced EP: {ep} → {benchmark_ep} "
                f"to fit on {benchmark_gpus} GPUs."
            )
            print(
                "[Primus:Performance Projection] Note: EP scaling will use decomposed A2A timing — "
                "measured compute is kept, only A2A is replaced analytically."
            )

    # Step 3: If still doesn't fit after EP reduction, reduce TP
    if benchmark_gpus_required > benchmark_gpus:
        print(
            f"[Primus:Performance Projection] After reducing PP to 1 and EP to {benchmark_ep}, "
            f"config still requires {benchmark_gpus_required} GPUs (TP={benchmark_tp}, CP={cp})."
        )
        print(
            f"[Primus:Performance Projection] Reducing TP to fit on {benchmark_gpus} GPUs..."
        )
        print(
            "[Primus:Performance Projection] Note: TP reduction will scale compute by benchmark_tp/target_tp "
            "and add TP AllReduce overhead analytically."
        )

        # Find largest TP that fits: TP * PP(=1) * EP * CP <= benchmark_gpus
        max_tp_for_benchmark = benchmark_gpus // max(benchmark_ep * cp, 1)
        if max_tp_for_benchmark < 1:
            max_tp_for_benchmark = 1

        # TP must be a power of 2 and divide original TP
        benchmark_tp = 1
        for candidate in [max_tp_for_benchmark, max_tp_for_benchmark // 2, 1]:
            if (
                candidate >= 1
                and candidate <= max_tp_for_benchmark
                and tp % candidate == 0
            ):
                benchmark_tp = candidate
                break

        benchmark_gpus_required = _calculate_min_gpus(
            benchmark_tp, benchmark_pp, benchmark_ep, cp
        )

        if benchmark_tp < tp:
            print(
                f"[Primus:Performance Projection] Reduced TP: {tp} → {benchmark_tp} "
                f"to fit on {benchmark_gpus} GPUs."
            )
            print(
                f"[Primus:Performance Projection] Will scale compute by {benchmark_tp}/{tp} = {benchmark_tp/tp:.4f} "
                f"and add TP AllReduce overhead when projecting to TP={tp}."
            )

    # Final validation
    if benchmark_gpus_required > benchmark_gpus:
        raise ValueError(
            f"[Primus:Performance Projection] Cannot reduce config to {benchmark_gpus} GPUs. "
            f"Even with PP=1, TP={benchmark_tp}, EP={benchmark_ep}, "
            f"configuration requires {benchmark_gpus_required} GPUs (CP={cp}). "
            f"Please reduce CP or use more benchmark GPUs."
        )

    # Modify the config for benchmarking
    original_config.pipeline_model_parallel_size = benchmark_pp
    # If benchmarking collapses PP to 1, force VPP/layout off to avoid
    # Megatron interleaved-schedule assertions.
    if benchmark_pp <= 1:
        if hasattr(original_config, "virtual_pipeline_model_parallel_size"):
            original_config.virtual_pipeline_model_parallel_size = 1
        if hasattr(original_config, "pipeline_model_parallel_layout"):
            original_config.pipeline_model_parallel_layout = None
    if benchmark_tp != tp:
        original_config.tensor_model_parallel_size = benchmark_tp

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
        "benchmark_tp": benchmark_tp,
        "original_ep": ep,
        "benchmark_ep": benchmark_ep,
        "original_cp": cp,
        "original_num_experts": num_experts,
        "benchmark_num_experts": benchmark_num_experts,
        "benchmark_gpus": benchmark_gpus,
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
    gpus_required = _calculate_min_gpus(tp, pp_size, ep, cp)
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
    # DP = world_size / (TP × PP × CP) — EP excluded (borrows from DP via folding)
    global_batch_size = runtime_config.global_batch_size
    data_parallel_size = (num_nodes * gpus_per_node) // (tp * pp_size * cp)
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


def _estimate_a2a_per_layer_ms(training_config, ep, hardware_config_dict=None):
    """
    Estimate the analytical All-to-All (dispatch + combine) time per MoE
    layer per direction for a given EP configuration.

    The analytical model uses standard RCCL-style bandwidth parameters.
    When decomposed A2A benchmarking is available, the EP scaling code
    uses *ratio-based* scaling (measured × analytical_ratio) so the
    absolute accuracy of this model does not matter -- only the relative
    scaling between EP sizes needs to be correct.

    This function returns the analytical A2A communication time only.
    Routing overhead (token permutation) is kept separate and not included here.

    Args:
        training_config: Training configuration.
        ep: Expert parallelism size to estimate A2A for.
        hardware_config_dict: Optional hardware config.

    Returns:
        float: Estimated A2A time (dispatch + combine) per layer per
               direction in milliseconds.  Returns 0.0 when ``ep <= 1``.
    """
    if ep <= 1:
        return 0.0

    mp_config = training_config.model_parallel_config
    model_config = training_config.model_config
    runtime_config = training_config.runtime_config

    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    cp = getattr(mp_config, "context_model_parallel_size", 1)

    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    gpus_required = _calculate_min_gpus(tp, pp, ep, cp)
    num_nodes = (gpus_required + gpus_per_node - 1) // gpus_per_node

    hw_overrides = dict(hardware_config_dict) if hardware_config_dict else {}

    coll_args = get_default_args(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
        hardware_config=hw_overrides if hw_overrides else None,
    )

    hidden_size = model_config.hidden_size
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length
    moe_router_topk = getattr(model_config, "moe_router_topk", 2)

    tokens_per_gpu = seq_len * batch_size // max(tp, 1)
    dispatch_size = tokens_per_gpu * hidden_size * moe_router_topk * 2  # BF16

    # Analytical A2A communication time (base model only)
    a2a_dispatch = cm.alltoall(coll_args, dispatch_size, ep, groups=["ep"])
    a2a_combine = cm.alltoall(coll_args, dispatch_size, ep, groups=["ep"])
    return (a2a_dispatch + a2a_combine) / 1000  # ms


def _estimate_ep_communication_overhead(
    training_config, original_ep, benchmark_ep, hardware_config_dict=None
):
    """
    Estimate the additional EP All-to-All communication overhead when scaling
    from benchmark_ep to original_ep.

    This is the legacy delta-based approach used as a fallback when decomposed
    A2A timings are not available (e.g. simulation mode).

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

    a2a_original = _estimate_a2a_per_layer_ms(
        training_config, original_ep, hardware_config_dict
    )
    a2a_benchmark = _estimate_a2a_per_layer_ms(
        training_config, benchmark_ep, hardware_config_dict
    )

    fwd_overhead = a2a_original - a2a_benchmark
    bwd_overhead = fwd_overhead  # Same for backward
    return fwd_overhead, bwd_overhead


def _estimate_tp_scaling(
    training_config,
    profiling_results,
    benchmark_tp,
    target_tp,
    hardware_config_dict=None,
):
    """
    Estimate per-layer time adjustments when TP was reduced for benchmarking.

    When benchmarking with fewer GPUs (e.g. single GPU with TP=1), the per-GPU
    compute is larger because each GPU handles a bigger share of the model.
    Scaling to the target TP involves:

    1. **Compute scaling**: Per-GPU compute decreases by ``benchmark_tp / target_tp``
       because each GPU processes a proportionally smaller tensor slice.
    2. **TP AllReduce delta**: The benchmark includes TP AllReduce for
       ``benchmark_tp`` (zero when benchmark_tp=1).  The target config requires
       TP AllReduce for ``target_tp``.  We add the delta.

    This function modifies ``profiling_results`` **in-place** to reflect the
    target TP configuration.

    Args:
        training_config: Training configuration (for sizes).
        profiling_results: Benchmark results dict (modified in-place).
        benchmark_tp: TP used during benchmarking.
        target_tp: TP for the target deployment.
        hardware_config_dict: Optional hardware config for communication.

    Returns:
        float: Total time adjustment in ms (positive means target is slower
               than benchmark, negative means target is faster — usually
               negative because TP scaling reduces per-GPU compute).
    """
    if benchmark_tp == target_tp:
        return 0.0

    is_rank_0 = int(os.getenv("RANK", "0")) == 0
    tp_compute_scale = benchmark_tp / target_tp  # < 1 means target has more TP

    # Estimate TP AllReduce overhead delta per layer.
    # TP AllReduce size = 2 × batch × seq_len × hidden_size × dtype_bytes
    # (AllReduce after column-parallel GEMM in attention + MLP)
    model_config = training_config.model_config
    runtime_config = training_config.runtime_config
    mp_config = training_config.model_parallel_config

    hidden_size = model_config.hidden_size
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length

    # There are 2 TP AllReduces per transformer layer (attention output + MLP output)
    # Each AllReduce message = batch × seq_len × hidden_size × 2 bytes (BF16)
    ar_msg_size = batch_size * seq_len * hidden_size * 2  # BF16 bytes

    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    pp = mp_config.pipeline_model_parallel_size
    ep = getattr(mp_config, "expert_model_parallel_size", 1) or 1
    cp = getattr(mp_config, "context_parallel_size", 1) or 1

    # ── Analytical AllReduce estimates ──
    analytical_target_ar = 0.0
    if target_tp > 1:
        gpus_required = _calculate_min_gpus(target_tp, pp, ep, cp)
        num_nodes = max(1, (gpus_required + gpus_per_node - 1) // gpus_per_node)
        coll_args_target = get_default_args(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            tp=target_tp,
            pp=pp,
            ep=ep,
            cp=cp,
            hardware_config=hardware_config_dict,
        )
        analytical_target_ar = (
            cm.allreduce(coll_args_target, ar_msg_size, target_tp) / 1000
        )  # ms

    analytical_bench_ar = 0.0
    if benchmark_tp > 1:
        gpus_required_b = _calculate_min_gpus(benchmark_tp, 1, 1, 1)
        num_nodes_b = max(1, (gpus_required_b + gpus_per_node - 1) // gpus_per_node)
        coll_args_bench = get_default_args(
            num_nodes=num_nodes_b,
            gpus_per_node=gpus_per_node,
            tp=benchmark_tp,
            pp=1,
            ep=1,
            cp=1,
            hardware_config=hardware_config_dict,
        )
        analytical_bench_ar = (
            cm.allreduce(coll_args_bench, ar_msg_size, benchmark_tp) / 1000
        )  # ms

    # If measured TP AllReduce data is available from _benchmark_tp_allreduce_on_gpu,
    # anchor to the measured value and scale by the analytical ratio:
    #   target_AR = measured_AR(bench) × [analytical(target) / analytical(bench)]
    # This trusts the analytical model's relative scaling but preserves
    # the measured absolute calibration from real hardware.
    tp_ar_measured = profiling_results.get("_tp_allreduce_benchmark", {})
    measured_bench_data = (
        tp_ar_measured.get(benchmark_tp, {}) if benchmark_tp > 1 else {}
    )
    measured_target_data = tp_ar_measured.get(target_tp, {}) if target_tp > 1 else {}
    measured_bench_ar = (
        measured_bench_data.get("measured_median_ms", 0) if measured_bench_data else 0
    )
    measured_target_ar = (
        measured_target_data.get("measured_median_ms", 0) if measured_target_data else 0
    )

    # Determine final AR per-op values using best available data
    target_ar_per_op = 0.0
    benchmark_ar_per_op = 0.0
    ar_source = "analytical"

    if target_tp > 1 and measured_target_ar > 0:
        # Direct measurement at target TP size available (best case)
        target_ar_per_op = measured_target_ar
        ar_source = "measured (direct)"
    elif target_tp > 1 and measured_bench_ar > 0 and analytical_bench_ar > 0:
        # Ratio-based: anchor to measured benchmark, scale by analytical ratio
        ar_ratio = analytical_target_ar / analytical_bench_ar
        target_ar_per_op = measured_bench_ar * ar_ratio
        ar_source = f"ratio-based (measured×{ar_ratio:.3f})"
    elif target_tp > 1:
        target_ar_per_op = analytical_target_ar
        ar_source = "analytical (no measured data)"

    if benchmark_tp > 1 and measured_bench_ar > 0:
        benchmark_ar_per_op = measured_bench_ar
    elif benchmark_tp > 1:
        benchmark_ar_per_op = analytical_bench_ar

    # 2 AllReduce ops per dense layer (attention + MLP), 2 per MoE layer (attention + MLP)
    # MoE layers may have additional AllReduce for shared experts
    num_ar_ops_per_layer = 2
    ar_delta_per_layer = (target_ar_per_op - benchmark_ar_per_op) * num_ar_ops_per_layer

    if is_rank_0:
        print(
            f"[Primus:Performance Projection] TP Scaling: {benchmark_tp} → {target_tp}"
        )
        print(f"  Compute scale factor: {tp_compute_scale:.4f}")
        print(f"  TP AllReduce source: {ar_source}")
        if measured_bench_ar > 0 or measured_target_ar > 0:
            print(
                f"  Measured AR: bench_tp={benchmark_tp} → {measured_bench_ar:.4f} ms, "
                f"target_tp={target_tp} → {measured_target_ar:.4f} ms"
            )
        print(
            f"  Analytical AR: bench_tp={benchmark_tp} → {analytical_bench_ar:.4f} ms, "
            f"target_tp={target_tp} → {analytical_target_ar:.4f} ms"
        )
        print(
            f"  TP AllReduce per op: benchmark={benchmark_ar_per_op:.4f} ms, target={target_ar_per_op:.4f} ms"
        )
        print(
            f"  TP AllReduce delta per layer ({num_ar_ops_per_layer} ops): {ar_delta_per_layer:.4f} ms"
        )

    # Apply scaling to each layer in profiling_results
    total_adjustment_ms = 0.0
    layers_adjusted = 0
    for layer_idx, layer_data in profiling_results.items():
        if not isinstance(layer_data, dict):
            continue
        if layer_idx in ("embedding", "output"):
            # Embedding/output: scale compute by TP ratio, no AllReduce
            old_fwd = layer_data.get("forward_time_ms", 0)
            old_bwd = layer_data.get("backward_time_ms", 0)
            new_fwd = old_fwd * tp_compute_scale
            new_bwd = old_bwd * tp_compute_scale
            layer_data["forward_time_ms"] = new_fwd
            layer_data["backward_time_ms"] = new_bwd
            total_adjustment_ms += (new_fwd + new_bwd) - (old_fwd + old_bwd)
            continue
        if "forward_time_ms" not in layer_data:
            continue

        old_fwd = layer_data.get("forward_time_ms", 0)
        old_bwd = layer_data.get("backward_time_ms", 0)

        layer_type = layer_data.get("type", "dense")

        if layer_type == "moe":
            # MoE layers: only attention and shared experts scale with TP.
            # Routed experts (GroupedMLP) are sharded by EP, NOT TP, so their
            # compute does not change with TP.  We scale only the attention
            # portion and keep the MLP (dominated by routed experts) unchanged.
            attn_data = layer_data.get("attention", {})
            mlp_data = layer_data.get("mlp", {})

            attn_fwd = (
                attn_data.get("forward_time_ms", 0)
                if isinstance(attn_data, dict)
                else 0
            )
            attn_bwd = (
                attn_data.get("backward_time_ms", 0)
                if isinstance(attn_data, dict)
                else 0
            )
            mlp_fwd = (
                mlp_data.get("forward_time_ms", 0) if isinstance(mlp_data, dict) else 0
            )
            mlp_bwd = (
                mlp_data.get("backward_time_ms", 0) if isinstance(mlp_data, dict) else 0
            )

            # Residual time (layernorm, residual add, etc.)
            residual_fwd = old_fwd - attn_fwd - mlp_fwd
            residual_bwd = old_bwd - attn_bwd - mlp_bwd

            # Scale attention and residual by TP, keep MLP (routed experts) unchanged
            new_attn_fwd = attn_fwd * tp_compute_scale
            new_attn_bwd = attn_bwd * tp_compute_scale
            new_residual_fwd = residual_fwd * tp_compute_scale
            new_residual_bwd = residual_bwd * tp_compute_scale

            new_fwd = new_attn_fwd + mlp_fwd + new_residual_fwd + ar_delta_per_layer / 2
            new_bwd = new_attn_bwd + mlp_bwd + new_residual_bwd + ar_delta_per_layer / 2

            # Update sub-component times
            if isinstance(attn_data, dict):
                attn_data["forward_time_ms"] = new_attn_fwd
                attn_data["backward_time_ms"] = new_attn_bwd
            # MLP sub-component stays unchanged (routed experts dominate)

            if is_rank_0:
                print(
                    f"  MoE layer: attn fwd {attn_fwd:.2f}→{new_attn_fwd:.2f} ms (×{tp_compute_scale:.2f}), "
                    f"MLP fwd {mlp_fwd:.2f} ms (unchanged), "
                    f"total fwd {old_fwd:.2f}→{new_fwd:.2f} ms"
                )
        else:
            # Dense layers: scale entire compute by TP ratio
            new_fwd = (
                old_fwd * tp_compute_scale + ar_delta_per_layer / 2
            )  # half AR delta in fwd
            new_bwd = (
                old_bwd * tp_compute_scale + ar_delta_per_layer / 2
            )  # half AR delta in bwd

            # Also scale sub-component times if available
            for sub_key in ("attention", "mlp"):
                sub_data = layer_data.get(sub_key, {})
                if isinstance(sub_data, dict):
                    if "forward_time_ms" in sub_data:
                        sub_data["forward_time_ms"] = (
                            sub_data["forward_time_ms"] * tp_compute_scale
                        )
                    if "backward_time_ms" in sub_data:
                        sub_data["backward_time_ms"] = (
                            sub_data["backward_time_ms"] * tp_compute_scale
                        )

        layer_data["forward_time_ms"] = new_fwd
        layer_data["backward_time_ms"] = new_bwd

        total_adjustment_ms += (new_fwd + new_bwd) - (old_fwd + old_bwd)
        layers_adjusted += 1

    if is_rank_0:
        print(f"  Adjusted {layers_adjusted} transformer layer(s)")
        print(f"  Total per-microbatch time delta: {total_adjustment_ms:+.3f} ms")

    return total_adjustment_ms


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
    pipeline_layout = getattr(mp_cfg, "pipeline_model_parallel_layout", None)

    mp_group = tp_size * cp_size * ep_size
    chunk_timings: List[list[dict]] = []
    for pp_rank in range(pp_size):
        stage_chunks = LanguageModelProfiler.get_virtual_stage_layers_for_rank(
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
            pipeline_model_parallel_layout=pipeline_layout,
        )
        if not stage_chunks:
            chunk_timings.append(
                [
                    {"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0}
                    for _ in range(vpp_size)
                ]
            )
            continue

        if len(stage_chunks) != vpp_size:
            chunk_timings.append(
                [
                    {"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0}
                    for _ in range(vpp_size)
                ]
            )
            continue

        # Get recomputation settings to account for extra forward pass during backward
        recompute_granularity = getattr(mp_cfg, "recompute_granularity", None)
        recompute_num_layers = getattr(mp_cfg, "recompute_num_layers", 0) or 0

        rank_chunks = []
        for chunk_layers in stage_chunks:
            chunk_entry = {"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0}
            for local_idx, layer_idx in enumerate(chunk_layers):
                layer_type = "moe" if layer_type_pattern[layer_idx] else "dense"
                metrics = type_timings.get(layer_type)
                if not metrics:
                    continue
                fwd_time = metrics["forward"]
                chunk_entry["fwd"] += fwd_time
                chunk_entry["bwd"] += metrics["backward"]
                chunk_entry["wgrad"] += metrics["wgrad"]
                chunk_entry["activation"] += metrics.get("activation", 0.0)

                # Recomputation: with recompute_granularity="full" and block method,
                # the first recompute_num_layers layers per chunk re-run forward
                # during backward, adding an extra forward time to backward.
                if (
                    recompute_granularity == "full"
                    and recompute_num_layers
                    and local_idx < recompute_num_layers
                ):
                    chunk_entry["bwd"] += fwd_time
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
    training_config,
    profiling_results,
    enable_zero_bubble=False,
    scheduler_algorithm="auto",
):
    chunk_time_matrix = _build_chunk_time_matrix(training_config, profiling_results)
    assert chunk_time_matrix is not None

    # For zero-bubble scheduling, we need to split backward into B (input grad) and W (weight grad)
    # The zero-bubble scheduler schedules these separately to minimize pipeline bubbles.
    # Typically B and W are roughly equal in duration (each ~50% of total backward).
    needs_bw_split = enable_zero_bubble or scheduler_algorithm in (
        "zerobubble",
        "zerobubble-heuristic",
        "zbv-formatted",
        "zbv-greedy-half",
        "zbv-greedy-min",
        "all",
    )
    if needs_bw_split:
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
                if needs_bw_split:
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

    # Determine which algorithms to run
    schedulers = []

    if scheduler_algorithm == "all":
        # Run ALL applicable schedulers for comparison
        if vpp_size == 1:
            # ZB basic
            schedulers.append(
                {
                    "name": "zerobubble",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zerobubble.ScheduleZeroBubble",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                }
            )
            # ZB heuristic (tries 8 combinations, picks best)
            cost_f = [chunk_time_matrix[r][0].get("fwd", 0.0) for r in range(pp_size)]
            cost_b = [chunk_time_matrix[r][0].get("bwd", 0.0) for r in range(pp_size)]
            cost_w = [chunk_time_matrix[r][0].get("wgrad", 0.0) for r in range(pp_size)]
            schedulers.append(
                {
                    "name": "zerobubble-heuristic",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zerobubble_heuristic.ScheduleZeroBubbleHeuristic",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                    "cost_f": cost_f,
                    "cost_b": cost_b,
                    "cost_w": cost_w,
                    "cost_comm": 0.1,
                }
            )
        else:
            # VPP > 1: interleaved 1F1B
            schedulers.append(
                {
                    "name": "interleaved_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
                    "pp_size": pp_size,
                    "vpp_size": vpp_size,
                    "micro_batches": micro_batches,
                }
            )
            if vpp_size == 2:
                # ZBV schedulers are designed for VPP=2
                schedulers.append(
                    {
                        "name": "zbv-formatted",
                        "class": "primus.core.pipeline_parallel.scheduler.algorithms.zbv_formatted.ScheduleZBVFormatted",
                        "pp_size": pp_size,
                        "vpp_size": 2,
                        "micro_batches": micro_batches,
                    }
                )
                schedulers.append(
                    {
                        "name": "zbv-greedy-half",
                        "class": "primus.core.pipeline_parallel.scheduler.algorithms.zbv_greedy.ScheduleZBVGreedy",
                        "pp_size": pp_size,
                        "vpp_size": 2,
                        "micro_batches": micro_batches,
                        "memory_config": "half",
                    }
                )
                schedulers.append(
                    {
                        "name": "zbv-greedy-min",
                        "class": "primus.core.pipeline_parallel.scheduler.algorithms.zbv_greedy.ScheduleZBVGreedy",
                        "pp_size": pp_size,
                        "vpp_size": 2,
                        "micro_batches": micro_batches,
                        "memory_config": "min",
                    }
                )
        # NOTE: seaailab-ilp is handled separately in _run_pipeline_simulation (VPP=1 only)
        sched_names = [s["name"] for s in schedulers]
        if vpp_size == 1:
            sched_names.append("seaailab-ilp")
        print(
            f"[Primus:Performance Projection] Compare-all mode: running schedulers: {', '.join(sched_names)}"
        )

    elif scheduler_algorithm == "zerobubble-heuristic":
        if vpp_size > 1:
            print(
                "[WARNING] zerobubble-heuristic requires VPP=1, falling back to interleaved_1f1b"
            )
            schedulers.append(
                {
                    "name": "interleaved_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
                    "pp_size": pp_size,
                    "vpp_size": vpp_size,
                    "micro_batches": micro_batches,
                }
            )
        else:
            cost_f = [chunk_time_matrix[r][0].get("fwd", 0.0) for r in range(pp_size)]
            cost_b = [chunk_time_matrix[r][0].get("bwd", 0.0) for r in range(pp_size)]
            cost_w = [chunk_time_matrix[r][0].get("wgrad", 0.0) for r in range(pp_size)]
            schedulers.append(
                {
                    "name": "zerobubble-heuristic",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zerobubble_heuristic.ScheduleZeroBubbleHeuristic",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                    "cost_f": cost_f,
                    "cost_b": cost_b,
                    "cost_w": cost_w,
                    "cost_comm": 0.1,
                }
            )

    elif scheduler_algorithm == "zerobubble":
        if vpp_size > 1:
            print(
                "[WARNING] zerobubble requires VPP=1, falling back to interleaved_1f1b"
            )
            schedulers.append(
                {
                    "name": "interleaved_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
                    "pp_size": pp_size,
                    "vpp_size": vpp_size,
                    "micro_batches": micro_batches,
                }
            )
        else:
            schedulers.append(
                {
                    "name": "zerobubble",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zerobubble.ScheduleZeroBubble",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                }
            )

    elif scheduler_algorithm == "zbv-formatted":
        if vpp_size != 2:
            print(
                f"[WARNING] zbv-formatted requires VPP=2, but VPP={vpp_size}. Falling back to interleaved_1f1b"
            )
            schedulers.append(
                {
                    "name": "interleaved_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
                    "pp_size": pp_size,
                    "vpp_size": vpp_size,
                    "micro_batches": micro_batches,
                }
            )
        else:
            schedulers.append(
                {
                    "name": "zbv-formatted",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zbv_formatted.ScheduleZBVFormatted",
                    "pp_size": pp_size,
                    "vpp_size": 2,
                    "micro_batches": micro_batches,
                }
            )

    elif scheduler_algorithm in ("zbv-greedy-half", "zbv-greedy-min"):
        mem_cfg = "half" if scheduler_algorithm == "zbv-greedy-half" else "min"
        if vpp_size != 2:
            print(
                f"[WARNING] {scheduler_algorithm} requires VPP=2, but VPP={vpp_size}. Falling back to interleaved_1f1b"
            )
            schedulers.append(
                {
                    "name": "interleaved_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
                    "pp_size": pp_size,
                    "vpp_size": vpp_size,
                    "micro_batches": micro_batches,
                }
            )
        else:
            schedulers.append(
                {
                    "name": scheduler_algorithm,
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zbv_greedy.ScheduleZBVGreedy",
                    "pp_size": pp_size,
                    "vpp_size": 2,
                    "micro_batches": micro_batches,
                    "memory_config": mem_cfg,
                }
            )

    elif scheduler_algorithm == "seaailab-ilp":
        # SeaAILab ILP is handled entirely in _run_pipeline_simulation
        # but we still need a fallback Primus scheduler for the sim config
        if enable_zero_bubble and vpp_size == 1:
            schedulers.append(
                {
                    "name": "zerobubble",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zerobubble.ScheduleZeroBubble",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                }
            )
        elif vpp_size > 1:
            schedulers.append(
                {
                    "name": "interleaved_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
                    "pp_size": pp_size,
                    "vpp_size": vpp_size,
                    "micro_batches": micro_batches,
                }
            )
        else:
            schedulers.append(
                {
                    "name": "basic_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.basic_1f1b.Schedule1F1B",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                }
            )

    else:
        # "auto" — current default behavior
        if enable_zero_bubble and vpp_size == 1:
            schedulers.append(
                {
                    "name": "zerobubble",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.zerobubble.ScheduleZeroBubble",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                }
            )
            print(
                "[Primus:Performance Projection] Using zero-bubble scheduler (enable_zero_bubble=True)"
            )
        elif vpp_size > 1:
            schedulers.append(
                {
                    "name": "interleaved_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.interleaved_1f1b.ScheduleInterleaved1F1B",
                    "pp_size": pp_size,
                    "vpp_size": vpp_size,
                    "micro_batches": micro_batches,
                }
            )
        else:
            schedulers.append(
                {
                    "name": "basic_1f1b",
                    "class": "primus.core.pipeline_parallel.scheduler.algorithms.basic_1f1b.Schedule1F1B",
                    "pp_size": pp_size,
                    "vpp_size": 1,
                    "micro_batches": micro_batches,
                }
            )

    return {
        "chunk_time_ms": chunk_time_matrix,
        "output_dir": str(Path.cwd() / "pp_simulation_result"),
        "schedulers": schedulers,
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

    # Benchmark actual allreduce on GPU for analytical model validation
    tp_ar_results = _benchmark_tp_allreduce_on_gpu(training_config, rank, world_size)
    if tp_ar_results:
        profiling_results["_tp_allreduce_benchmark"] = tp_ar_results

    return profiling_results


def _benchmark_tp_allreduce_on_gpu(training_config, rank, world_size):
    """
    Benchmark actual GPU allreduce operations for TP-relevant message sizes.

    Runs ``torch.distributed.all_reduce`` on real hardware using the same
    message sizes that TP would use (batch × seq_len × hidden_size × 2 bytes
    for BF16).  Tests all power-of-2 group sizes up to ``world_size``.

    This allows direct comparison between the analytical communication model
    (``collective_model.allreduce``) and actual silicon measurements.

    Args:
        training_config: Training configuration (provides hidden_size, batch, seq_len).
        rank: Current process rank.
        world_size: Total number of GPUs available.

    Returns:
        dict: Mapping ``tp_size → {"measured_ms": float, "msg_size_bytes": int}``,
              or empty dict if benchmarking fails.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or world_size < 2:
        return {}

    model_config = training_config.model_config
    runtime_config = training_config.runtime_config
    hidden_size = model_config.hidden_size
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length

    # TP allreduce message: [batch * seq_len, hidden_size] in BF16
    msg_elements = batch_size * seq_len * hidden_size
    msg_size_bytes = msg_elements * 2  # BF16

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    results = {}

    # Test TP group sizes: 2, 4, 8 (powers of 2 up to world_size)
    tp_sizes = [s for s in [2, 4, 8] if s <= world_size]

    num_warmup = 20
    num_iters = 100

    for tp_size in tp_sizes:
        # Only ranks within the first tp_size GPUs participate in timing
        # (all ranks must call into the subgroup though)
        # Create sub-groups of size tp_size across all ranks
        num_groups = world_size // tp_size
        subgroups = []
        my_group = None
        for g in range(num_groups):
            group_ranks = list(range(g * tp_size, (g + 1) * tp_size))
            sg = dist.new_group(group_ranks)
            subgroups.append(sg)
            if rank in group_ranks:
                my_group = sg

        if my_group is None:
            continue

        tensor = torch.randn(msg_elements, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(num_warmup):
            dist.all_reduce(tensor, group=my_group)
        torch.cuda.synchronize(device)

        # Benchmark with CUDA events
        times_ms = []
        for _ in range(num_iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            dist.all_reduce(tensor, group=my_group)
            end_event.record()

            torch.cuda.synchronize(device)
            times_ms.append(start_event.elapsed_time(end_event))

        # Use median to avoid outlier influence
        times_ms.sort()
        median_ms = times_ms[len(times_ms) // 2]
        avg_ms = sum(times_ms) / len(times_ms)
        p10_ms = times_ms[int(len(times_ms) * 0.1)]
        p90_ms = times_ms[int(len(times_ms) * 0.9)]

        results[tp_size] = {
            "measured_median_ms": median_ms,
            "measured_avg_ms": avg_ms,
            "measured_p10_ms": p10_ms,
            "measured_p90_ms": p90_ms,
            "msg_size_bytes": msg_size_bytes,
        }

        del tensor

    # Barrier to sync all ranks before continuing
    dist.barrier()
    torch.cuda.synchronize(device)

    return results


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
    vpp_size = getattr(mp_cfg, "virtual_pipeline_model_parallel_size", 1) or 1
    micro_batches = _compute_micro_batches(training_config.runtime_config, mp_cfg)

    # Extract per-stage costs (F, B, W) from chunk_time_matrix.
    # The Megatron ZB ILP scheduler (zb.py) only supports VPP=1 (single chunk).
    # When VPP > 1, we aggregate all VPP chunks per rank into a single stage
    # so the total per-rank compute is correct. This gives a conservative
    # estimate (VPP interleaving would reduce bubbles further).
    cost_f = []
    cost_b = []
    cost_w = []
    mem_f = []
    mem_b = []
    mem_w = []

    print(
        "[Primus:Performance Projection] Using Megatron zero-bubble scheduler (ILP-based)"
    )
    print(f"  PP size: {pp_size}, VPP size: {vpp_size}, Microbatches: {micro_batches}")
    if vpp_size > 1:
        print(
            f"  NOTE: Aggregating {vpp_size} VPP chunks per rank for ZB scheduler (VPP>1)"
        )

    for rank_idx, rank_chunks in enumerate(chunk_time_matrix):
        # Aggregate ALL VPP chunks for this rank to get correct total compute
        fwd = sum(chunk.get("fwd", 0.0) for chunk in rank_chunks)
        bwd = sum(chunk.get("bwd", 0.0) for chunk in rank_chunks)
        act_gb = sum(chunk.get("activation", 0.0) for chunk in rank_chunks)

        # Split backward into B and W (50/50 as approximation)
        # The Megatron scheduler expects B and W separately
        b_time = bwd * 0.5
        w_time = bwd * 0.5

        cost_f.append(float(fwd))
        cost_b.append(float(b_time))
        cost_w.append(float(w_time))

        # Memory: GraphConfig requires mem_f + mem_b + mem_w == 0 for each stage
        # F adds activation, B releases half, W releases remaining half
        mem_f.append(float(act_gb))
        mem_b.append(float(-act_gb * 0.5))  # B releases half
        mem_w.append(float(-act_gb * 0.5))  # W releases remaining half

        if vpp_size > 1:
            chunk_detail = " + ".join(f"{c.get('fwd', 0):.1f}" for c in rank_chunks)
            print(
                f"  Stage {rank_idx}: F={fwd:.2f}ms ({chunk_detail}), B={b_time:.2f}ms, W={w_time:.2f}ms, act={act_gb:.2f}GB"
            )
        else:
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


def _print_scheduler_comparison(all_results, training_config):
    """Print a comparison table of all scheduler results."""
    runtime_config = training_config.runtime_config
    seq_len = getattr(runtime_config, "sequence_length", None)
    micro_batch_size = getattr(runtime_config, "micro_batch_size", None)

    mp_cfg = training_config.model_parallel_config
    pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1
    micro_batches = _compute_micro_batches(runtime_config, mp_cfg)
    tokens_per_step = (seq_len or 0) * (micro_batch_size or 0) * micro_batches

    print("\n" + "=" * 100)
    print("  PIPELINE SCHEDULER COMPARISON")
    print("=" * 100)
    print(
        f"  {'Scheduler':<35s} {'Step Time (ms)':>15s} {'Tokens/GPU/s':>15s} {'Max Bubble %':>14s}"
    )
    print("  " + "-" * 79)

    best_name = None
    best_time = float("inf")
    for name, info in all_results.items():
        step_time = info.get("step_time_ms")
        bubble_ratio = info.get("max_bubble_ratio", 0.0)
        if step_time is None:
            print(f"  {name:<35s} {'FAILED':>15s} {'—':>15s} {'—':>14s}")
            continue
        tps = tokens_per_step * 1000 / step_time / pp_size if step_time > 0 else 0
        print(f"  {name:<35s} {step_time:>15.2f} {tps:>15,.0f} {bubble_ratio:>13.2%}")
        if step_time < best_time:
            best_time = step_time
            best_name = name

    print("  " + "-" * 79)
    if best_name:
        print(f"  Best scheduler: {best_name} ({best_time:.2f} ms)")
    print("=" * 100 + "\n")


def _run_pipeline_simulation(
    training_config,
    profiling_results,
    enable_zero_bubble=False,
    scheduler_algorithm="auto",
):
    """
    Run pipeline simulation and return the step time.

    Supports multiple scheduler algorithms for comparison.

    Args:
        training_config: Training configuration
        profiling_results: Layer profiling results
        enable_zero_bubble: Whether to use zero-bubble scheduling (reduces pipeline bubbles).
        scheduler_algorithm: Which scheduler(s) to run:
            "auto"               - Default behavior (zerobubble if enabled, else 1f1b)
            "zerobubble"         - Primus basic zero-bubble
            "zerobubble-heuristic" - Primus heuristic (tries 8 combos)
            "seaailab-ilp"       - SeaAILab ILP-based scheduler
            "all"                - Run ALL schedulers and compare

    Returns:
        float: Step time in ms from pipeline simulation, or None if simulation failed
    """
    is_compare_mode = scheduler_algorithm == "all"
    run_seaailab_ilp = scheduler_algorithm in ("seaailab-ilp", "all")

    if is_compare_mode:
        print(
            "[Primus:Performance Projection] Compare-all mode: running multiple pipeline schedulers"
        )
    elif enable_zero_bubble:
        print(
            "[Primus:Performance Projection] Using Primus Pipeline scheduler with zero-bubble"
        )
    else:
        print(
            "[Primus:Performance Projection] Using Primus Pipeline scheduler (zero-bubble disabled)"
        )

    # ── Run Primus schedulers ──
    all_results = {}
    primus_step_time_ms = None

    if scheduler_algorithm != "seaailab-ilp":
        sim_config = _build_scheduler_sim_config(
            training_config, profiling_results, enable_zero_bubble, scheduler_algorithm
        )
        if sim_config is None:
            return None

        print(
            "[Primus:Performance Projection] Running Primus Pipeline schedule simulator..."
        )
        runner = SchedulerSimulationRunner(sim_config)
        simulation_runs = runner.run()

        # Collect results from each Primus scheduler
        for sim in simulation_runs:
            summary = (sim or {}).get("summary") or {}
            step_time = summary.get("step_time_ms")
            per_rank = sim.get("per_rank") or []

            # Compute max bubble ratio
            max_bubble_ratio = 0.0
            if step_time and per_rank:
                for rank_data in per_rank:
                    fwd_time = sum(
                        e - s
                        for s, e in zip(
                            rank_data.get("fwd_start", []), rank_data.get("fwd_end", [])
                        )
                    )
                    bwd_time = sum(
                        e - s
                        for s, e in zip(
                            rank_data.get("bwd_start", []), rank_data.get("bwd_end", [])
                        )
                    )
                    wgrad_time = sum(
                        e - s
                        for s, e in zip(
                            rank_data.get("wgrad_start", []),
                            rank_data.get("wgrad_end", []),
                        )
                    )
                    compute_time = fwd_time + bwd_time + wgrad_time
                    bubble = max(0.0, step_time - compute_time)
                    ratio = bubble / step_time if step_time > 0 else 0.0
                    max_bubble_ratio = max(max_bubble_ratio, ratio)

            all_results[sim.get("name", "unknown")] = {
                "step_time_ms": step_time,
                "max_bubble_ratio": max_bubble_ratio,
            }
            primus_step_time_ms = step_time  # use last Primus scheduler as default

        # Print per-scheduler detailed results
        _report_simulation_results(simulation_runs, training_config)

    # ── Run SeaAILab ILP scheduler (if requested or auto with zero-bubble) ──
    # SeaAILab ILP (zb.py) only supports VPP=1; skip when VPP>1.
    mp_cfg_check = training_config.model_parallel_config
    vpp_size_check = (
        getattr(mp_cfg_check, "virtual_pipeline_model_parallel_size", 1) or 1
    )
    should_run_seaailab = (
        run_seaailab_ilp or (scheduler_algorithm == "auto" and enable_zero_bubble)
    ) and vpp_size_check == 1
    if vpp_size_check > 1 and run_seaailab_ilp:
        print(
            f"[Primus:Performance Projection] Skipping SeaAILab ILP — does not natively support VPP={vpp_size_check}"
        )
    if should_run_seaailab:
        try:
            seaailab_time = _run_pipeline_simulation_megatron_zb(
                training_config, copy.deepcopy(profiling_results)
            )
            if seaailab_time is not None:
                # Compute approximate bubble ratio
                mp_cfg = training_config.model_parallel_config
                pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1
                micro_batches = _compute_micro_batches(
                    training_config.runtime_config, mp_cfg
                )
                chunk_time_matrix = _build_chunk_time_matrix(
                    training_config, profiling_results
                )
                if chunk_time_matrix:
                    avg_compute = 0.0
                    for rank_chunks in chunk_time_matrix:
                        rank_total = sum(
                            c.get("fwd", 0) + c.get("bwd", 0) + c.get("wgrad", 0)
                            for c in rank_chunks
                        )
                        avg_compute += rank_total
                    avg_compute /= pp_size
                    ideal_time = avg_compute * micro_batches
                    bubble_time = seaailab_time - ideal_time
                    seaailab_bubble_ratio = (
                        max(0.0, bubble_time / seaailab_time)
                        if seaailab_time > 0
                        else 0.0
                    )
                else:
                    seaailab_bubble_ratio = 0.0
                all_results["seaailab-ilp"] = {
                    "step_time_ms": seaailab_time,
                    "max_bubble_ratio": seaailab_bubble_ratio,
                }
        except Exception as e:
            print(f"[WARNING] SeaAILab ILP scheduler failed: {e}")
            all_results["seaailab-ilp"] = {
                "step_time_ms": None,
                "max_bubble_ratio": 0.0,
            }

    # ── Print comparison table (if multiple schedulers) ──
    if len(all_results) > 1:
        _print_scheduler_comparison(all_results, training_config)

    # ── Return the best step time ──
    valid_times = [
        r["step_time_ms"]
        for r in all_results.values()
        if r.get("step_time_ms") is not None
    ]
    if valid_times:
        best_time = min(valid_times)
        if is_compare_mode:
            best_name = [
                n for n, r in all_results.items() if r.get("step_time_ms") == best_time
            ][0]
            print(
                f"[Primus:Performance Projection] Using best scheduler '{best_name}' step time: {best_time:.2f} ms"
            )
        return best_time
    elif primus_step_time_ms is not None:
        return primus_step_time_ms
    else:
        return None


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
    pipeline_layout = getattr(mp_cfg, "pipeline_model_parallel_layout", None)

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
        pipeline_model_parallel_layout=pipeline_layout,
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
    benchmark_ep: int = None,
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

    # Calculate minimum nodes required by parallelism config.
    # For MoE (EP > 1): CP is folded into EP via MoE Parallel Folding,
    # so min GPUs = TP × PP × EP.  For dense: min GPUs = TP × PP × CP.
    gpus_required = _calculate_min_gpus(tp, pp, ep, cp)
    min_nodes_required = (gpus_required + gpus_per_node - 1) // gpus_per_node

    # Validate target >= minimum required
    if target_nodes < min_nodes_required:
        raise ValueError(
            f"[Primus:Multinode] ERROR: Cannot project to {target_nodes} nodes."
            f"Minimum required by parallelism config is {min_nodes_required} nodes."
            f"--target-nodes must be >= {min_nodes_required}."
        )

    # Calculate DP for scaling.  EP is excluded from this divisor because, with
    # MoE Parallel Folding, EP borrows from the DP dimension (not from extra
    # GPUs).  Data-loading DP = world_size / (TP × PP × CP) for both dense and
    # MoE models.  Within each EP group the CP ranks share context-parallel
    # attention work while EP/CP ranks provide inner data-parallel streams.
    gpus_for_dp = tp * pp * cp  # EP excluded — it borrows from DP
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

    # ── Override A2A time with measured/ratio-scaled values ──
    # When we have measured A2A, use it directly (if EP unchanged) or scale it by
    # analytical ratio (if EP changed). This is more accurate than pure analytical.
    if ep > 1:
        # Try to get measured A2A from profiling results
        measured_a2a_fwd = None
        for layer_idx, layer_data in profiling_results.items():
            if isinstance(layer_data, dict) and layer_data.get("type") == "moe":
                mlp_info = layer_data.get("mlp", {})
                measured_a2a_fwd = mlp_info.get("a2a_forward_time_ms", 0)
                if measured_a2a_fwd > 0:
                    break

        # If we have measured A2A, use subtract/add approach to get target A2A
        # This keeps other times (like permute) unchanged, only A2A time changes
        if measured_a2a_fwd and measured_a2a_fwd > 0:
            num_moe_layers = message_info.get("num_moe_layers", 0)
            if num_moe_layers > 0:
                analytical_target_a2a = message_info.get("moe_a2a_per_layer_fwd", 0)

                measured_a2a_per_layer = measured_a2a_fwd

                # Check if per-layer adjustment already corrected the A2A
                _a2a_already_adjusted = any(
                    isinstance(ld, dict)
                    and ld.get("type") == "moe"
                    and ld.get("mlp", {}).get("a2a_ep_adjusted", False)
                    for ld in profiling_results.values()
                )

                if _a2a_already_adjusted:
                    # Per-layer MoE adjustment already applied additive A2A
                    # correction and stored the target A2A. Use it directly.
                    target_a2a_per_layer = measured_a2a_per_layer
                    analytical_bench_a2a = None
                    a2a_source = "pre-adjusted by per-layer MoE correction"
                elif benchmark_ep is not None and benchmark_ep != ep and benchmark_ep > 0:
                    analytical_bench_a2a = _estimate_a2a_per_layer_ms(
                        training_config, benchmark_ep, hardware_config_dict
                    )
                    target_a2a_per_layer = (
                        measured_a2a_per_layer
                        - analytical_bench_a2a
                        + analytical_target_a2a
                    )
                    a2a_source = f"measured - analytical_bench({benchmark_ep}) + analytical_target({ep})"
                else:
                    target_a2a_per_layer = measured_a2a_per_layer
                    analytical_bench_a2a = None
                    a2a_source = "measured (benchmark EP not available or unchanged)"

                effective_a2a_per_layer = target_a2a_per_layer

                # When the pipeline simulation was used, A2A is already baked
                # into the per-layer wall-clock times (whether DeepEP overlapped
                # it or not). Showing it again as separate communication would
                # be misleading.  Zero it out in the breakdown so
                # "Total Communication" only reflects truly *additional* overhead.
                if time_includes_all_microbatches:
                    # A2A already inside pipeline sim — remove from breakdown
                    old_a2a_fwd = breakdown.get("moe_a2a_fwd", 0)
                    old_a2a_bwd = breakdown.get("moe_a2a_bwd", 0)
                    breakdown["moe_a2a_fwd"] = 0
                    breakdown["moe_a2a_bwd"] = 0
                    total_comm_time_ms = total_comm_time_ms - old_a2a_fwd - old_a2a_bwd
                    message_info["moe_a2a_per_layer_fwd"] = effective_a2a_per_layer
                    message_info["a2a_in_pipeline_sim"] = True
                else:
                    # No pipeline sim — A2A must be accounted for in the
                    # communication breakdown explicitly.
                    total_a2a_fwd = effective_a2a_per_layer * num_moe_layers
                    total_a2a_bwd = effective_a2a_per_layer * num_moe_layers
                    old_a2a_fwd = breakdown.get("moe_a2a_fwd", 0)
                    old_a2a_bwd = breakdown.get("moe_a2a_bwd", 0)
                    breakdown["moe_a2a_fwd"] = total_a2a_fwd
                    breakdown["moe_a2a_bwd"] = total_a2a_bwd
                    message_info["moe_a2a_per_layer_fwd"] = effective_a2a_per_layer
                    total_comm_time_ms = (
                        total_comm_time_ms
                        - old_a2a_fwd
                        - old_a2a_bwd
                        + total_a2a_fwd
                        + total_a2a_bwd
                    )

                use_deepep = getattr(
                    training_config.model_config, "use_turbo_deepep", False
                )
                if is_rank_0:
                    print("  [INFO] A2A timing (measured/scaled):")
                    print(
                        f"    Analytical target (EP={ep}): {analytical_target_a2a:.3f} ms/layer"
                    )
                    if analytical_bench_a2a is not None:
                        print(
                            f"    Analytical benchmark (EP={benchmark_ep}): {analytical_bench_a2a:.3f} ms/layer"
                        )
                    print(f"    Measured: {measured_a2a_per_layer:.3f} ms/layer")
                    print(
                        f"    Using: {target_a2a_per_layer:.3f} ms/layer ({a2a_source})"
                    )
                    if time_includes_all_microbatches:
                        print(
                            "    → A2A already included in pipeline simulation layer times (not added to comm overhead)"
                        )
                    elif use_deepep:
                        print(
                            "    DeepEP ON: A2A overlap already baked into layer times"
                        )
                    total_a2a_display = effective_a2a_per_layer * num_moe_layers * 2
                    print(
                        f"    Total A2A: {total_a2a_display:.3f} ms ({num_moe_layers} layers)"
                    )

    # Safety net: if the pipeline sim was used but no measured A2A was found
    # (e.g. ep > 1 but profiler didn't decompose A2A), still zero out the
    # analytical A2A in the breakdown — it's already inside the sim.
    if time_includes_all_microbatches and not message_info.get(
        "a2a_in_pipeline_sim", False
    ):
        old_fwd = breakdown.get("moe_a2a_fwd", 0)
        old_bwd = breakdown.get("moe_a2a_bwd", 0)
        if old_fwd > 0 or old_bwd > 0:
            breakdown["moe_a2a_fwd"] = 0
            breakdown["moe_a2a_bwd"] = 0
            total_comm_time_ms -= old_fwd + old_bwd
            message_info["a2a_in_pipeline_sim"] = True

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
    # Only hardware-profile metadata (HBM bandwidth) is needed here, not full
    # GEMM simulation, so require_simulation=False avoids a hard origami
    # dependency when running in benchmark mode.
    gpu_arch = getattr(args, "gpu_arch", None)
    gpu_clock_mhz = getattr(args, "gpu_clock_mhz", None)
    gemm_backend_for_optim = get_gemm_simulation_backend(
        backend_name=None,
        gpu_arch=gpu_arch,
        gpu_clock_mhz=gpu_clock_mhz,
        require_simulation=False,
    )
    optimizer_profiler = OptimizerProfiler(
        config=training_config, gemm_backend=gemm_backend_for_optim
    )
    optimizer_step_ms = optimizer_profiler.estimated_step_time_ms(dp_size=dp_target)

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
        if message_info.get("a2a_in_pipeline_sim", False):
            a2a_per_layer = message_info.get("moe_a2a_per_layer_fwd", 0)
            n_moe = message_info.get("num_moe_layers", 0)
            print(
                f"   moe_a2a: (included in pipeline simulation — {n_moe} layers × {a2a_per_layer:.3f} ms/layer)"
            )
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

    Sub-node benchmarking (--benchmark-gpus):
        When --benchmark-gpus is set lower than GPUS_PER_NODE, the tool reduces
        parallelism (PP, EP, and if necessary TP) to fit the benchmark GPU count,
        benchmarks on those GPUs, and then analytically upscales to the full
        node and multi-node configurations.

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

    # ── Apply projection-specific CLI overrides to the config ──
    # These args are registered in the projection CLI so they don't leak
    # to the Megatron trainer as unknown extra_args.
    module_cfg = primus_config.get_module_config("pre_trainer")
    is_rank_0 = int(os.getenv("RANK", "0")) == 0
    cli_overrides_applied = []

    cli_ep_size = getattr(args, "target_ep_size", None)
    if cli_ep_size is not None:
        module_cfg.expert_model_parallel_size = cli_ep_size
        cli_overrides_applied.append(f"expert_model_parallel_size={cli_ep_size}")

    cli_mbs = getattr(args, "micro_batch_size", None)
    if cli_mbs is not None:
        module_cfg.micro_batch_size = cli_mbs
        cli_overrides_applied.append(f"micro_batch_size={cli_mbs}")

    cli_gbs = getattr(args, "global_batch_size", None)
    if cli_gbs is not None:
        module_cfg.global_batch_size = cli_gbs
        cli_overrides_applied.append(f"global_batch_size={cli_gbs}")

    cli_vpp = getattr(args, "num_virtual_stages_per_pipeline_rank", None)
    if cli_vpp is not None:
        module_cfg.virtual_pipeline_model_parallel_size = cli_vpp
        cli_overrides_applied.append(f"virtual_pipeline_model_parallel_size={cli_vpp}")

    cli_enable_zb = getattr(args, "enable_zero_bubble", False)
    if cli_enable_zb:
        module_cfg.enable_zero_bubble = True
        cli_overrides_applied.append("enable_zero_bubble=True")

    cli_enable_deepep = getattr(args, "enable_deepep", False)
    if cli_enable_deepep:
        if hasattr(module_cfg, "model") and hasattr(module_cfg.model, "moe"):
            module_cfg.model.moe.use_turbo_deepep = True
        else:
            module_cfg.use_turbo_deepep = True
        cli_overrides_applied.append("use_turbo_deepep=True")

    if cli_overrides_applied and is_rank_0:
        print("[Primus:Performance Projection] CLI overrides applied to config:")
        for ov in cli_overrides_applied:
            print(f"  {ov}")

    primus_config_original = copy.deepcopy(primus_config)

    # Check if we need to reduce config for single-node benchmarking
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))

    # Get benchmark GPU count from CLI (sub-node benchmarking support)
    benchmark_gpus = getattr(args, "benchmark_gpus", None)
    if benchmark_gpus is None:
        benchmark_gpus = gpus_per_node

    is_sub_node_benchmark = benchmark_gpus < gpus_per_node

    # Get target nodes from CLI flag (--target-nodes or --target-num-nodes)
    target_nodes = getattr(args, "target_nodes", None)
    if target_nodes is None:
        target_nodes = getattr(args, "target_num_nodes", None)

    # Store original parallelism before any modifications
    module_config = primus_config.get_module_config("pre_trainer")
    reduction_info = _calculate_single_node_config(
        copy.deepcopy(module_config), gpus_per_node, benchmark_gpus=benchmark_gpus
    )

    # Calculate minimum nodes required
    min_nodes_required = reduction_info["original_nodes_required"]

    # If target_nodes not specified, default to minimum required
    if target_nodes is None:
        target_nodes = min_nodes_required

    if reduction_info["adjusted"]:
        benchmark_label = (
            f"{benchmark_gpus}-GPU" if is_sub_node_benchmark else "single-node"
        )
        print("" + "=" * 100)
        if is_sub_node_benchmark:
            print(
                f"[Primus:Performance Projection] Sub-node benchmarking: {benchmark_gpus} GPU(s)"
            )
            gpus_per_tray = benchmark_gpus
            trays_per_node = gpus_per_node // gpus_per_tray if gpus_per_tray > 0 else 1
            print(
                f"  Topology: {gpus_per_tray} GPUs/tray × {trays_per_node} trays/node "
                f"= {gpus_per_node} GPUs/node"
            )
        else:
            print("[Primus:Performance Projection] Multi-node configuration detected")
        print("=" * 100)
        print(f"  Original configuration requires {min_nodes_required} nodes minimum:")
        print(
            f"    TP={reduction_info['original_tp']}, PP={reduction_info['original_pp']}, "
            f"EP={reduction_info['original_ep']}, CP={reduction_info['original_cp']}"
        )
        print(f"  Reducing to {benchmark_label} configuration for benchmarking:")
        print(
            f"    TP={reduction_info['benchmark_tp']}, PP={reduction_info['benchmark_pp']}, "
            f"EP={reduction_info['benchmark_ep']}, CP={reduction_info['original_cp']}"
        )

        # Show what was changed
        changes = []
        if reduction_info["original_tp"] != reduction_info["benchmark_tp"]:
            changes.append(
                f"TP {reduction_info['original_tp']} → {reduction_info['benchmark_tp']}"
            )
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

        print(
            "  Will estimate performance by analytically adding communication overhead back."
        )
        print("=" * 100)

        # Apply the reduction to the config used for benchmarking
        primus_config.get_module_config("pre_trainer").pipeline_model_parallel_size = (
            reduction_info["benchmark_pp"]
        )
        if reduction_info["benchmark_pp"] <= 1:
            if hasattr(
                primus_config.get_module_config("pre_trainer"),
                "virtual_pipeline_model_parallel_size",
            ):
                primus_config.get_module_config(
                    "pre_trainer"
                ).virtual_pipeline_model_parallel_size = 1
            if hasattr(
                primus_config.get_module_config("pre_trainer"),
                "pipeline_model_parallel_layout",
            ):
                primus_config.get_module_config(
                    "pre_trainer"
                ).pipeline_model_parallel_layout = None
        primus_config.get_module_config("pre_trainer").expert_model_parallel_size = (
            reduction_info["benchmark_ep"]
        )
        if reduction_info["benchmark_tp"] != reduction_info["original_tp"]:
            primus_config.get_module_config(
                "pre_trainer"
            ).tensor_model_parallel_size = reduction_info["benchmark_tp"]
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
    # This ensures the pipeline simulation calculates the correct number of microbatches.
    # DP = world_size / (TP × PP × CP) for both dense and MoE.  With MoE Parallel
    # Folding, EP borrows from the DP dimension (CP is folded into EP), so EP
    # does not appear in the DP divisor.
    mp_config = training_config.model_parallel_config
    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    cp = getattr(mp_config, "context_parallel_size", 1) or 1
    ep = getattr(mp_config, "expert_model_parallel_size", 1) or 1

    # Calculate DP for the TARGET configuration (what we're projecting to)
    # The pipeline simulator simulates the target config, so it needs target DP for microbatch calculation
    target_world_size = target_nodes * gpus_per_node

    # DP = world_size / (TP × PP × CP) — EP excluded (borrows from DP via folding)
    target_dp = target_world_size // (tp * pp * cp)

    # Also show benchmark config for reference
    benchmark_pp = reduction_info.get("benchmark_pp", pp)
    benchmark_ep = reduction_info.get("benchmark_ep", ep)
    benchmark_tp = reduction_info.get("benchmark_tp", tp)
    benchmark_world_size = benchmark_gpus
    benchmark_dp = max(1, benchmark_world_size // (benchmark_tp * benchmark_pp * cp))

    # Only print from rank 0
    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    benchmark_label = f"{benchmark_gpus} GPUs" if is_sub_node_benchmark else "1 node"
    if is_rank_0:
        print("[Primus:Training Projection] Configuration Summary:")
        print(
            f"  Benchmark Config: TP={benchmark_tp}, PP={benchmark_pp}, EP={benchmark_ep}, "
            f"CP={cp}, DP={benchmark_dp} ({benchmark_label})"
        )
        print(
            f"  Target Config: TP={tp}, PP={pp}, EP={ep}, CP={cp}, DP={target_dp} ({target_nodes} nodes)"
        )

    # =========================================================================
    # TRAINING MODE — full forward + backward + optimizer + gradient AllReduce
    # =========================================================================

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

    # If TP was reduced for sub-node benchmarking, apply TP scaling BEFORE pipeline simulation
    if reduction_info["adjusted"] and reduction_info.get("benchmark_tp", tp) != tp:
        hardware_config_dict = None
        if hasattr(args, "hardware_config") and args.hardware_config:
            hardware_config_dict = load_hardware_config(args.hardware_config)
        if is_rank_0:
            print(
                "[Primus:Performance Projection] Adjusting profiling results for TP scaling:"
            )
        _estimate_tp_scaling(
            training_config,
            profiling_results,
            benchmark_tp=reduction_info["benchmark_tp"],
            target_tp=tp,
            hardware_config_dict=hardware_config_dict,
        )

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

        # Check if decomposed A2A timings are available from benchmarking.
        # When available, we use the precise approach: strip measured A2A from
        # the MLP time and replace it with the analytical A2A for the target EP.
        # This avoids relying on the absolute accuracy of the analytical model;
        # only the ratio between EP sizes matters.
        has_decomposed_a2a = any(
            isinstance(ld, dict)
            and ld.get("type") == "moe"
            and ld.get("mlp", {}).get("a2a_forward_time_ms", 0) > 0
            for ld in profiling_results.values()
        )

        if has_decomposed_a2a:
            # ── Decomposed A2A approach with ratio-based scaling ──
            # Instead of using the raw analytical A2A for the target EP,
            # we anchor to the measured A2A and scale by the analytical
            # ratio:  target_A2A = measured_A2A × (analytical_target / analytical_bench)
            # This trusts the *relative* scaling of the analytical model
            # but preserves the measured absolute calibration.
            analytical_bench_a2a = _estimate_a2a_per_layer_ms(
                training_config,
                benchmark_ep,
                hardware_config_dict,
            )
            analytical_target_a2a = _estimate_a2a_per_layer_ms(
                training_config,
                original_ep,
                hardware_config_dict,
            )

            if is_rank_0:
                print(
                    "[Primus:Performance Projection] Adjusting profiling results for EP scaling (decomposed A2A):"
                )
                print(f"  EP rescaled: {benchmark_ep} → {original_ep}")
                if (
                    original_num_experts is not None
                    and benchmark_num_experts is not None
                ):
                    orig_epr = original_num_experts // original_ep
                    bench_epr = benchmark_num_experts // benchmark_ep
                    print(
                        f"  Experts per rank: benchmark={bench_epr} "
                        f"(E={benchmark_num_experts}, EP={benchmark_ep}), "
                        f"target={orig_epr} "
                        f"(E={original_num_experts}, EP={original_ep})"
                    )
                print(
                    f"  Analytical A2A: bench EP={benchmark_ep} → {analytical_bench_a2a:.3f} ms, "
                    f"target EP={original_ep} → {analytical_target_a2a:.3f} ms"
                )
                if analytical_bench_a2a > 0:
                    a2a_ratio = analytical_target_a2a / analytical_bench_a2a
                    print(f"  Analytical scaling ratio: {a2a_ratio:.3f}x")
                    # Check if we have measured A2A to compare
                    for layer_idx, layer_data in profiling_results.items():
                        if (
                            isinstance(layer_data, dict)
                            and layer_data.get("type") == "moe"
                        ):
                            mlp_info = layer_data.get("mlp", {})
                            measured_a2a_fwd = mlp_info.get("a2a_forward_time_ms", 0)
                            if measured_a2a_fwd > 0:
                                analytical_vs_measured_ratio = (
                                    analytical_bench_a2a / measured_a2a_fwd
                                )
                                print(
                                    f"  Analytical vs measured A2A (bench EP): {analytical_bench_a2a:.3f} / {measured_a2a_fwd:.3f} = {analytical_vs_measured_ratio:.3f}x"
                                )
                                if (
                                    analytical_vs_measured_ratio > 1.5
                                    or analytical_vs_measured_ratio < 0.67
                                ):
                                    print(
                                        "  [WARNING] Analytical model differs significantly from measured A2A. "
                                        "Projection accuracy may be affected."
                                    )
                            break

            moe_layers_adjusted = 0
            for layer_idx, layer_data in profiling_results.items():
                if isinstance(layer_data, dict) and layer_data.get("type") == "moe":
                    old_fwd = layer_data.get("forward_time_ms", 0)
                    old_bwd = layer_data.get("backward_time_ms", 0)

                    mlp_info = layer_data.get("mlp", {})
                    mlp_fwd = mlp_info.get("forward_time_ms", 0)
                    mlp_bwd = mlp_info.get("backward_time_ms", 0)
                    measured_a2a_fwd = mlp_info.get("a2a_forward_time_ms", 0)
                    measured_a2a_bwd = mlp_info.get("a2a_backward_time_ms", 0)

                    if benchmark_ep == original_ep:
                        # No EP scaling — use measured A2A directly
                        target_a2a_fwd = measured_a2a_fwd
                        target_a2a_bwd = measured_a2a_bwd
                        if is_rank_0 and moe_layers_adjusted == 0:
                            print(
                                f"    [INFO] Benchmark EP ({benchmark_ep}) == target EP ({original_ep}), using measured A2A directly"
                            )

                    # ── Estimate per-expert GEMM time for epr correction ──
                    # The measured "compute" (total MLP minus A2A) includes both
                    # expert MLP GEMMs and large routing/permutation overhead that
                    # does NOT scale with experts_per_rank. Scaling the entire
                    # compute by target_epr/bench_epr drastically underestimates.
                    # Instead, we estimate only the expert MLP GEMM time
                    # analytically and subtract the per-expert delta.
                    model_cfg = training_config.model_config
                    runtime_cfg = training_config.runtime_config
                    _hidden = model_cfg.hidden_size
                    _ffn = getattr(model_cfg, "moe_ffn_hidden_size", None) or \
                           getattr(model_cfg, "ffn_hidden_size", None) or _hidden * 4
                    _topk = getattr(model_cfg, "moe_router_topk", 2) or 2
                    _seq = getattr(runtime_cfg, "sequence_length", 4096)
                    _mbs = getattr(runtime_cfg, "micro_batch_size", 1)

                    bench_epr = (benchmark_num_experts // max(benchmark_ep, 1)
                                 if benchmark_num_experts else 1)
                    target_epr = (original_num_experts // max(original_ep, 1)
                                  if original_num_experts else 1)

                    if benchmark_ep != original_ep and bench_epr != target_epr:
                        # Tokens per expert depends on routing density
                        _tpe_bench = _mbs * _seq * _topk / max(benchmark_num_experts, 1)
                        _tpe_target = _mbs * _seq * _topk / max(original_num_experts, 1)

                        # SwiGLU forward FLOPs per expert: 3 projections × 2 (matmul)
                        GPU_PEAK_TFLOPS = 1300  # MI300X/MI355X BF16 matrix peak
                        _expert_fwd_bench = (
                            bench_epr * 6 * _tpe_bench * _hidden * _ffn
                            / (GPU_PEAK_TFLOPS * 1e9)
                        )
                        _expert_fwd_target = (
                            target_epr * 6 * _tpe_target * _hidden * _ffn
                            / (GPU_PEAK_TFLOPS * 1e9)
                        )
                        expert_correction_fwd = _expert_fwd_bench - _expert_fwd_target
                        expert_correction_bwd = 2.0 * expert_correction_fwd
                    else:
                        expert_correction_fwd = 0.0
                        expert_correction_bwd = 0.0

                    # ── Compute A2A delta (additive, not multiplicative) ──
                    # The measured A2A includes both communication and large
                    # dispatch/combine overhead. Multiplicative ratio-scaling
                    # amplifies that overhead. Additive correction adjusts only
                    # the communication portion via the analytical model.
                    if benchmark_ep != original_ep and measured_a2a_fwd > 0:
                        a2a_delta = analytical_target_a2a - analytical_bench_a2a
                        target_a2a_fwd = measured_a2a_fwd + a2a_delta
                        target_a2a_bwd = measured_a2a_bwd + a2a_delta
                    elif benchmark_ep != original_ep:
                        # No measured A2A (e.g. benchmark EP=1) — use analytical
                        a2a_delta = analytical_target_a2a
                        target_a2a_fwd = analytical_target_a2a
                        target_a2a_bwd = analytical_target_a2a
                        if is_rank_0 and moe_layers_adjusted == 0:
                            print(
                                f"    [INFO] No measured A2A (bench EP={benchmark_ep}), "
                                f"using analytical for target EP={original_ep}: "
                                f"{analytical_target_a2a:.3f} ms/layer"
                            )
                    else:
                        a2a_delta = 0.0

                    # ── Adjust total MLP time ──
                    # Formula: new_mlp = measured_mlp - expert_correction + a2a_delta
                    # This avoids decomposing into compute + A2A (which is unreliable
                    # because measured "compute" is dominated by routing overhead
                    # that doesn't scale with epr).
                    use_deepep = getattr(model_cfg, "use_turbo_deepep", False)

                    if use_deepep and benchmark_ep != original_ep:
                        # With DeepEP overlap, A2A delta is partially hidden.
                        # Compute typically dominates A2A, so overlap = A2A × OE.
                        # Increasing A2A by delta means overlap also grows by delta × OE,
                        # so the effective A2A increase is delta × (1 - OE).
                        DEEPEP_OVERLAP_EFFICIENCY = 0.65
                        effective_a2a_delta = a2a_delta * (1.0 - DEEPEP_OVERLAP_EFFICIENCY)
                        new_mlp_fwd = mlp_fwd - expert_correction_fwd + effective_a2a_delta
                        new_mlp_bwd = mlp_bwd - expert_correction_bwd + effective_a2a_delta
                    else:
                        new_mlp_fwd = mlp_fwd - expert_correction_fwd + a2a_delta
                        new_mlp_bwd = mlp_bwd - expert_correction_bwd + a2a_delta

                    # Safety: new MLP should not be negative
                    new_mlp_fwd = max(new_mlp_fwd, 0.1)
                    new_mlp_bwd = max(new_mlp_bwd, 0.1)

                    # Update layer total: replace old MLP with new MLP
                    new_fwd = (old_fwd - mlp_fwd) + new_mlp_fwd
                    new_bwd = (old_bwd - mlp_bwd) + new_mlp_bwd

                    if is_rank_0 and moe_layers_adjusted == 0:
                        print("  MoE layer adjustment (per layer):")
                        print(
                            f"    MLP fwd: {mlp_fwd:.2f} ms (measured A2A: {measured_a2a_fwd:.2f})"
                        )
                        if abs(expert_correction_fwd) > 0.001:
                            print(
                                f"    Expert GEMM correction (epr {bench_epr} → {target_epr}): "
                                f"-{expert_correction_fwd:.2f} ms "
                                f"(analytical at {GPU_PEAK_TFLOPS} TFLOPS)"
                            )
                        if benchmark_ep != original_ep:
                            print(
                                f"    A2A delta (additive): {a2a_delta:+.3f} ms "
                                f"(analytical {analytical_bench_a2a:.3f} → {analytical_target_a2a:.3f})"
                            )
                        print(f"    Target A2A fwd: {target_a2a_fwd:.2f} ms")
                        print(f"    → New MLP fwd: {new_mlp_fwd:.2f} ms")
                        print(f"    → New MLP bwd: {new_mlp_bwd:.2f} ms")
                        print(f"    Layer fwd: {old_fwd:.2f} → {new_fwd:.2f} ms")
                        print(f"    Layer bwd: {old_bwd:.2f} → {new_bwd:.2f} ms")

                    layer_data["forward_time_ms"] = new_fwd
                    layer_data["backward_time_ms"] = new_bwd
                    if mlp_info:
                        mlp_info["forward_time_ms"] = new_mlp_fwd
                        mlp_info["backward_time_ms"] = new_mlp_bwd
                        mlp_info["a2a_forward_time_ms"] = target_a2a_fwd
                        mlp_info["a2a_backward_time_ms"] = target_a2a_bwd
                        mlp_info["a2a_ep_adjusted"] = True
                    moe_layers_adjusted += 1

        else:
            # ── Fallback: legacy delta approach (simulation mode) ──
            fwd_overhead_per_layer, bwd_overhead_per_layer = (
                _estimate_ep_communication_overhead(
                    training_config,
                    original_ep,
                    benchmark_ep,
                    hardware_config_dict,
                )
            )

            ep_mlp_scale = _compute_ep_mlp_scale(
                training_config.model_config,
                benchmark_ep,
                original_ep,
                original_num_experts=original_num_experts,
                benchmark_num_experts=benchmark_num_experts,
            )

            if is_rank_0:
                print(
                    "[Primus:Performance Projection] Adjusting profiling results for EP scaling (delta approach):"
                )
                print(f"  EP rescaled: {benchmark_ep} → {original_ep}")
                if (
                    original_num_experts is not None
                    and benchmark_num_experts is not None
                ):
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
                    print("  Adding per-layer All-to-All overhead:")
                    print(f"    Forward:  +{fwd_overhead_per_layer:.3f} ms/layer")
                    print(f"    Backward: +{bwd_overhead_per_layer:.3f} ms/layer")

            moe_layers_adjusted = 0
            for layer_idx, layer_data in profiling_results.items():
                if isinstance(layer_data, dict) and layer_data.get("type") == "moe":
                    old_fwd = layer_data.get("forward_time_ms", 0)
                    old_bwd = layer_data.get("backward_time_ms", 0)

                    mlp_info = layer_data.get("mlp", {})
                    mlp_fwd = mlp_info.get("forward_time_ms", 0)
                    mlp_bwd = mlp_info.get("backward_time_ms", 0)

                    new_mlp_fwd = mlp_fwd * ep_mlp_scale
                    new_mlp_bwd = mlp_bwd * ep_mlp_scale
                    mlp_delta_fwd = new_mlp_fwd - mlp_fwd
                    mlp_delta_bwd = new_mlp_bwd - mlp_bwd

                    new_fwd = old_fwd + mlp_delta_fwd + fwd_overhead_per_layer
                    new_bwd = old_bwd + mlp_delta_bwd + bwd_overhead_per_layer

                    if is_rank_0 and moe_layers_adjusted == 0:
                        print("  MoE layer adjustment (per layer):")
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
                    if mlp_info:
                        mlp_info["forward_time_ms"] = new_mlp_fwd
                        mlp_info["backward_time_ms"] = new_mlp_bwd
                    moe_layers_adjusted += 1

            if is_rank_0:
                print(
                    f"  Adjusted {moe_layers_adjusted} MoE layer(s) in profiling results"
                )
        ep_overhead_applied = True

    # ── DeepEP overlap when EP doesn't change ──
    # When DeepEP is ON and EP is unchanged, the benchmark already ran with
    # DeepEP enabled, so the measured wall-clock layer times already include
    # the A2A-compute overlap benefit.  No additional overlap adjustment is
    # needed — applying it again would double-count the savings.
    use_deepep = getattr(training_config.model_config, "use_turbo_deepep", False)
    if use_deepep and not ep_overhead_applied:
        if is_rank_0:
            print(
                "[Primus:Performance Projection] DeepEP ON, EP unchanged: "
                "benchmark times already include A2A overlap — no adjustment needed."
            )

    # Check if zero-bubble scheduling is enabled in the original config
    original_module_config = primus_config_original.get_module_config("pre_trainer")
    enable_zero_bubble = getattr(original_module_config, "enable_zero_bubble", False)

    # Pipeline schedule algorithm from CLI (default: "auto")
    scheduler_algorithm = getattr(args, "pipeline_schedule_algorithm", "auto")
    # When "all" or specific ZB algorithm is requested, auto-enable zero bubble
    if scheduler_algorithm in (
        "all",
        "zerobubble",
        "zerobubble-heuristic",
        "seaailab-ilp",
    ):
        if not enable_zero_bubble:
            enable_zero_bubble = True
            if is_rank_0:
                print(
                    f"[Primus:Performance Projection] Auto-enabling zero-bubble scheduling "
                    f"for --pipeline-schedule-algorithm={scheduler_algorithm}"
                )

    # Use ORIGINAL PP for pipeline simulation decision, not benchmark PP
    # If original PP > 1, we should run pipeline simulation even if we benchmarked with PP=1
    original_pp = reduction_info.get("original_pp", pp)

    # Temporarily set training_config PP to original_pp for pipeline simulation
    # (we'll restore it after if needed)
    original_training_pp = (
        training_config.model_parallel_config.pipeline_model_parallel_size
    )
    training_config.model_parallel_config.pipeline_model_parallel_size = original_pp

    # Skip pipeline simulation only if original PP=1 (no pipeline parallelism)
    if original_pp == 1:
        pipeline_simulation_time_ms = None
        if is_rank_0:
            print(
                "[Primus:Performance Projection] Skipping pipeline simulation (original PP=1)"
            )
    else:
        if is_rank_0 and pp != original_pp:
            print(
                f"[Primus:Performance Projection] Running pipeline simulation with original PP={original_pp} "
                f"(benchmarked with PP={pp})"
            )
        pipeline_simulation_time_ms = _run_pipeline_simulation(
            training_config, profiling_results, enable_zero_bubble, scheduler_algorithm
        )

    # Restore training_config PP to benchmark value for consistency
    training_config.model_parallel_config.pipeline_model_parallel_size = (
        original_training_pp
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
                    num_moe_layers = getattr(
                        training_config.model_config, "num_moe_layers", 0
                    )
                else:
                    num_moe_layers = sum(1 for x in moe_pattern if x == 1)

                if num_moe_layers > 0:
                    # Check if decomposed A2A timings are available
                    has_decomposed_a2a = any(
                        isinstance(ld, dict)
                        and ld.get("type") == "moe"
                        and ld.get("mlp", {}).get("a2a_forward_time_ms", 0) > 0
                        for ld in profiling_results.values()
                    )

                    if has_decomposed_a2a:
                        # ── Decomposed A2A approach with ratio-based scaling ──
                        analytical_bench_a2a = _estimate_a2a_per_layer_ms(
                            training_config,
                            benchmark_ep_val,
                            hardware_config_dict,
                        )
                        analytical_target_a2a = _estimate_a2a_per_layer_ms(
                            training_config,
                            original_ep,
                            hardware_config_dict,
                        )

                        net_change_total = 0.0
                        total_overlap_per_layer = 0.0
                        use_deepep = getattr(
                            training_config.model_config, "use_turbo_deepep", False
                        )
                        for layer_idx, layer_data in profiling_results.items():
                            if (
                                isinstance(layer_data, dict)
                                and layer_data.get("type") == "moe"
                            ):
                                mlp_info = layer_data.get("mlp", {})
                                mlp_fwd = mlp_info.get("forward_time_ms", 0)
                                mlp_bwd = mlp_info.get("backward_time_ms", 0)
                                measured_a2a_fwd = mlp_info.get(
                                    "a2a_forward_time_ms", 0
                                )
                                measured_a2a_bwd = mlp_info.get(
                                    "a2a_backward_time_ms", 0
                                )

                                # Ratio-based target A2A
                                if analytical_bench_a2a > 0 and measured_a2a_fwd > 0:
                                    a2a_ratio = (
                                        analytical_target_a2a / analytical_bench_a2a
                                    )
                                    target_a2a_fwd = measured_a2a_fwd * a2a_ratio
                                    target_a2a_bwd = measured_a2a_bwd * a2a_ratio
                                else:
                                    target_a2a_fwd = analytical_target_a2a
                                    target_a2a_bwd = analytical_target_a2a

                                # Decompose MLP into compute + A2A (same logic as Path A)
                                if use_deepep:
                                    DEEPEP_OVERLAP_EFFICIENCY = 0.65
                                    residual = 1.0 - DEEPEP_OVERLAP_EFFICIENCY
                                    compute_fwd = mlp_fwd - measured_a2a_fwd * residual
                                    compute_bwd = mlp_bwd - measured_a2a_bwd * residual
                                    if compute_fwd < measured_a2a_fwd:
                                        compute_fwd = max(
                                            0, (mlp_fwd - measured_a2a_fwd) / residual
                                        )
                                    if compute_bwd < measured_a2a_bwd:
                                        compute_bwd = max(
                                            0, (mlp_bwd - measured_a2a_bwd) / residual
                                        )
                                    overlap_fwd = (
                                        min(target_a2a_fwd, compute_fwd)
                                        * DEEPEP_OVERLAP_EFFICIENCY
                                    )
                                    overlap_bwd = (
                                        min(target_a2a_bwd, compute_bwd)
                                        * DEEPEP_OVERLAP_EFFICIENCY
                                    )
                                    total_overlap_per_layer = overlap_fwd + overlap_bwd
                                    new_mlp_fwd = (
                                        compute_fwd + target_a2a_fwd - overlap_fwd
                                    )
                                    new_mlp_bwd = (
                                        compute_bwd + target_a2a_bwd - overlap_bwd
                                    )
                                else:
                                    compute_fwd = mlp_fwd - measured_a2a_fwd
                                    compute_bwd = mlp_bwd - measured_a2a_bwd
                                    new_mlp_fwd = compute_fwd + target_a2a_fwd
                                    new_mlp_bwd = compute_bwd + target_a2a_bwd

                                delta_fwd = new_mlp_fwd - mlp_fwd
                                delta_bwd = new_mlp_bwd - mlp_bwd
                                net_change_total = (
                                    delta_fwd + delta_bwd
                                ) * num_moe_layers
                                break  # All MoE layers have same profiled time

                        if is_rank_0:
                            print(
                                "[Primus:Performance Projection] EP Adjustment (decomposed A2A, ratio-based):"
                            )
                            print(f"  EP rescaled: {benchmark_ep_val} → {original_ep}")
                            print(
                                f"  Analytical ratio: {analytical_target_a2a:.3f}/{analytical_bench_a2a:.3f}"
                            )
                            print(f"  Number of MoE layers: {num_moe_layers}")
                            if use_deepep and total_overlap_per_layer > 0:
                                total_overlap = total_overlap_per_layer * num_moe_layers
                                print(
                                    f"  DeepEP overlap benefit: {total_overlap:.3f} ms (hidden behind compute)"
                                )
                            print(f"  Net adjustment: {net_change_total:+.3f} ms")

                        benchmarked_time_ms += net_change_total
                    else:
                        # ── Fallback: legacy delta approach ──
                        fwd_overhead_per_layer, bwd_overhead_per_layer = (
                            _estimate_ep_communication_overhead(
                                training_config,
                                original_ep,
                                benchmark_ep_val,
                                hardware_config_dict,
                            )
                        )
                        total_ep_overhead_ms = (
                            fwd_overhead_per_layer + bwd_overhead_per_layer
                        ) * num_moe_layers

                        original_num_experts = reduction_info.get(
                            "original_num_experts"
                        )
                        benchmark_num_experts = reduction_info.get(
                            "benchmark_num_experts"
                        )
                        ep_mlp_scale = _compute_ep_mlp_scale(
                            training_config.model_config,
                            benchmark_ep_val,
                            original_ep,
                            original_num_experts=original_num_experts,
                            benchmark_num_experts=benchmark_num_experts,
                        )
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
                                break

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

                        benchmarked_time_ms += (
                            total_ep_overhead_ms - total_mlp_reduction_ms
                        )

                    if is_rank_0:
                        print(f"  Adjusted time: {benchmarked_time_ms:.3f} ms")

        # Determine if time already includes all microbatches (from pipeline simulation)
        time_includes_all_microbatches = pipeline_simulation_time_ms is not None

        # ─── Sub-node projection summary ──────────────────────────────────
        # When benchmarking on fewer GPUs than a full node, show intermediate
        # projections at compute-tray and single-node granularities before
        # the final multi-node result.
        if is_sub_node_benchmark and is_rank_0:
            hardware_config_dict = None
            if hasattr(args, "hardware_config") and args.hardware_config:
                hardware_config_dict = load_hardware_config(args.hardware_config)

            gpus_per_tray = benchmark_gpus
            if hardware_config_dict and "gpus_per_tray" in hardware_config_dict:
                gpus_per_tray = hardware_config_dict["gpus_per_tray"]

            trays_per_node = gpus_per_node // gpus_per_tray if gpus_per_tray > 0 else 1

            print("\n" + "=" * 100)
            print("Sub-Node Projection Summary (Intermediate Scales)")
            print("=" * 100)

            # Per-microbatch compute time for the target parallelism
            if time_includes_all_microbatches:
                per_mb_label = "pipeline-simulated (includes all microbatches)"
                per_mb_time = benchmarked_time_ms
            else:
                per_mb_label = "per-microbatch"
                per_mb_time = benchmarked_time_ms

            print(f"  Benchmarked on: {benchmark_gpus} GPUs")
            print(f"  Compute time ({per_mb_label}): {per_mb_time:.3f} ms")
            print("")

            # Compute tray level (if benchmarked on 1 GPU, show tray projection)
            if benchmark_gpus == 1 and gpus_per_tray > 1:
                print(f"  📊 Compute Tray ({gpus_per_tray} GPUs):")
                print(
                    f"     TP={min(tp, gpus_per_tray)}, PP=1, EP={min(ep, gpus_per_tray // min(tp, gpus_per_tray))}"
                )
                print("     Per-microbatch compute: included in node-level projection")
                print("")

            # Node level
            print(
                f"  📊 Full Node ({gpus_per_node} GPUs = {trays_per_node} trays × {gpus_per_tray} GPUs/tray):"
            )
            print(f"     TP={tp}, PP={pp}, EP={ep}, CP={cp}")
            print(f"     Per-microbatch compute: {per_mb_time:.3f} ms")
            print(
                f"     (All intra-node communication modeled at {hardware_config_dict.get('node_bw', 'N/A')} GB/s)"
                if hardware_config_dict
                else "     (Using default intra-node communication model)"
            )
            print("")
            print(f"  📊 Multi-Node ({target_nodes} nodes = {target_nodes * gpus_per_node} GPUs):")
            print("     → See detailed projection below")
            print("=" * 100)

        # Run multinode projection
        benchmark_ep = reduction_info.get(
            "benchmark_ep", reduction_info.get("original_ep", None)
        )
        _run_multinode_projection(
            training_config,
            benchmarked_time_ms,
            profiling_results,
            args,
            target_nodes,
            time_includes_all_microbatches,
            benchmark_ep=benchmark_ep,
        )
