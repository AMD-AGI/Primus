from typing import List, Dict, Optional, Any, Tuple
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import math
import os
from pathlib import Path

import yaml

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.module_profilers.language_model import (
    LanguageModelProfiler,
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.module_profilers.collective_args import get_default_args
from primus.core.projection.module_profilers import collective_model as cm
from primus.core.projection.performance_projection.simulator import (
    SchedulerSimulationRunner,
)
from primus.core.projection.training_config import (
    convert_primus_config_to_projection_config,
)
from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

_MAX_EXPERT_PARALLEL_SIZE = 8
_BYTES_PER_GB = 1024**3


# =============================================================================
# Hardware and Communication Functions (moved from multinode_projection)
# =============================================================================

def load_hardware_config(config_path: str) -> Dict[str, Any]:
    """Load hardware configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('hardware_config', {})


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
    
    # Model parameters
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_layers
    num_experts = model_config.num_experts
    moe_router_topk = model_config.moe_router_topk
    moe_pattern = model_config.moe_pattern
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length
    
    # Count MoE layers
    num_moe_layers = sum(1 for p in moe_pattern if p == 1)
    
    # Calculate per-rank parameters (simplified - should match memory projection)
    total_params = model_config.num_layers * hidden_size * hidden_size * 12  # Rough estimate
    num_params_per_rank = total_params // (tp * pp * ep)
    
    breakdown = {}
    message_info = {}
    per_layer_info = []  # Store per-layer communication details
    
    # 1. Gradient AllReduce (DP group)
    if dp > 1:
        grad_size = num_params_per_rank * 4  # FP32 gradients
        ar_time_dp = cm.allreduce(coll_args, grad_size, dp, groups=['dp'])
        breakdown['gradient_allreduce'] = ar_time_dp / 1000  # Convert to ms
        message_info['gradient_allreduce_size'] = grad_size
        message_info['gradient_allreduce_size_mb'] = grad_size / (1024 * 1024)
    else:
        breakdown['gradient_allreduce'] = 0.0
        message_info['gradient_allreduce_size'] = 0
        message_info['gradient_allreduce_size_mb'] = 0.0
    
    # 2. MoE All-to-All (EP group)
    if ep > 1 and num_moe_layers > 0:
        tokens_per_batch = seq_len * batch_size
        dispatch_size = tokens_per_batch * hidden_size * moe_router_topk * 2  # BF16
        
        a2a_dispatch = cm.alltoall(coll_args, dispatch_size, ep, groups=['ep'])
        a2a_combine = cm.alltoall(coll_args, dispatch_size, ep, groups=['ep'])
        
        total_a2a_fwd = (a2a_dispatch + a2a_combine) * num_moe_layers / 1000  # ms
        total_a2a_bwd = total_a2a_fwd
        
        breakdown['moe_a2a_fwd'] = total_a2a_fwd
        breakdown['moe_a2a_bwd'] = total_a2a_bwd
        message_info['moe_a2a_size'] = dispatch_size
        message_info['moe_a2a_size_mb'] = dispatch_size / (1024 * 1024)
        message_info['moe_a2a_per_layer_fwd'] = (a2a_dispatch + a2a_combine) / 1000
        message_info['num_moe_layers'] = num_moe_layers
    else:
        breakdown['moe_a2a_fwd'] = 0.0
        breakdown['moe_a2a_bwd'] = 0.0
        message_info['moe_a2a_size'] = 0
        message_info['moe_a2a_size_mb'] = 0.0
        message_info['moe_a2a_per_layer_fwd'] = 0.0
        message_info['num_moe_layers'] = 0
    
    # Note: TP AllReduce is already included in the benchmarked run, so we don't add it here
    message_info['num_layers'] = num_layers
    
    # Note: PP P2P communication is NOT calculated here because it's already
    # accounted for in the pipeline scheduler simulator (simulator.py).
    # The simulator handles send/receive synchronization and bubble time.
    
    # Build per-layer communication information
    for layer_idx in range(num_layers):
        layer_comm = {
            'layer_idx': layer_idx,
            'layer_type': 'MoE' if moe_pattern[layer_idx] == 1 else 'Dense',
            'communications': []
        }
        
        # MoE All-to-All (if EP > 1 and this is a MoE layer)
        # Note: TP AllReduce is already included in benchmarked run, so not added here
        if ep > 1 and moe_pattern[layer_idx] == 1:
            layer_comm['communications'].append({
                'type': 'MoE All-to-All (fwd+bwd)',
                'time_ms': (a2a_dispatch + a2a_combine) * 2 / 1000,  # fwd + bwd
                'message_size_mb': dispatch_size / (1024 * 1024),
                'group_size': ep,
            })
        
        per_layer_info.append(layer_comm)
    
    total_comm_time = sum(breakdown.values())
    
    # Check if gradient all-reduce should be overlapped
    mp_config = training_config.model_parallel_config
    overlap_grad_reduce = getattr(mp_config, 'overlap_grad_reduce', True)  # Default to True
    
    # If overlapped, don't add to critical path
    if overlap_grad_reduce and 'gradient_allreduce' in breakdown:
        total_comm_time -= breakdown['gradient_allreduce']
        message_info['gradient_allreduce_overlapped'] = True
    else:
        message_info['gradient_allreduce_overlapped'] = False
    
    return total_comm_time, breakdown, message_info, per_layer_info


def extract_single_node_time_from_profiling(profiling_results: dict, training_config) -> float:
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
    print("\n[Primus:Performance Projection] Extracting timing from benchmark results...")
    print("-" * 100)
    
    model_config = training_config.model_config
    moe_pattern = model_config.moe_pattern  # Full model pattern (e.g., 27 layers)
    
    # Get profiled layer indices
    profiled_layer_indices = sorted([k for k in profiling_results.keys() if isinstance(k, int)])
    print(f"  Profiled layers: {profiled_layer_indices}")
    print(f"  Full model has {len(moe_pattern)} transformer layers")
    
    total_time_ms = 0.0
    
    # Embedding layer
    if "embedding" in profiling_results:
        emb = profiling_results["embedding"]
        emb_time = emb.get("forward_time_ms", 0) + emb.get("backward_time_ms", 0)
        total_time_ms += emb_time
        print(f"  Embedding: {emb_time:.2f} ms")
    
    # Analyze profiled transformer layers
    profiled_dense_times = []
    profiled_moe_times = []
    
    for layer_idx in profiled_layer_indices:
        if layer_idx < len(moe_pattern):
            layer_data = profiling_results[layer_idx]
            layer_time = layer_data.get("forward_time_ms", 0) + layer_data.get("backward_time_ms", 0)
            
            if moe_pattern[layer_idx] == 0:
                profiled_dense_times.append(layer_time)
            else:
                profiled_moe_times.append(layer_time)
    
    # Calculate averages from profiled layers
    avg_dense_time = sum(profiled_dense_times) / len(profiled_dense_times) if profiled_dense_times else 0
    avg_moe_time = sum(profiled_moe_times) / len(profiled_moe_times) if profiled_moe_times else 0
    
    # Count total dense and MoE layers in full model
    num_dense_layers = sum(1 for x in moe_pattern if x == 0)
    num_moe_layers = sum(1 for x in moe_pattern if x == 1)
    
    # Extrapolate to full model
    total_dense_time = avg_dense_time * num_dense_layers
    total_moe_time = avg_moe_time * num_moe_layers
    total_transformer_time = total_dense_time + total_moe_time
    
    total_time_ms += total_transformer_time
    
    # Print detailed breakdown
    if profiled_dense_times:
        print(f"  Dense Layers: {len(profiled_dense_times)} profiled → {num_dense_layers} total")
        print(f"    Avg per layer: {avg_dense_time:.2f} ms")
        print(f"    Total time: {total_dense_time:.2f} ms")
    
    if profiled_moe_times:
        print(f"  MoE Layers: {len(profiled_moe_times)} profiled → {num_moe_layers} total")
        print(f"    Avg per layer: {avg_moe_time:.2f} ms")
        print(f"    Total time: {total_moe_time:.2f} ms")
    
    # Output layer
    if "output" in profiling_results:
        out = profiling_results["output"]
        out_time = out.get("forward_time_ms", 0) + out.get("backward_time_ms", 0)
        total_time_ms += out_time
        print(f"  Output Layer: {out_time:.2f} ms")
    
    print("-" * 100)
    print(f"[Primus:Performance Projection] Extrapolated Baseline Time: {total_time_ms:.2f} ms/iteration")
    print(f"  (Based on {len(profiled_layer_indices)} profiled layers → {len(moe_pattern)} total layers)")
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
    Restrict the transformer stack to at most one dense layer and one MoE layer.
    """
    has_moe = getattr(module_config, "num_experts", None)
    has_moe = has_moe is not None and module_config.num_experts > 0
    original_layers = getattr(module_config, "num_layers", 1) or 1
    original_moe_layout = getattr(module_config, "moe_layer_freq", None)
    dense_layers_present = _has_dense_layers(original_moe_layout)
    max_layers = 2 if has_moe and dense_layers_present else 1
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
    for attr in ("num_layers_per_virtual_pipeline_stage", "num_virtual_stages_per_pipeline_rank"):
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
    
    gpus_required = tp * pp * ep * cp
    nodes_required = (gpus_required + gpus_per_node - 1) // gpus_per_node
    
    # If already fits on 1 node, no adjustment needed
    if nodes_required <= 1:
        return {
            'adjusted': False,
            'original_pp': pp,
            'benchmark_pp': pp,
            'original_nodes_required': nodes_required,
            'original_tp': tp,
            'original_ep': ep,
            'benchmark_ep': ep,
            'original_cp': cp,
        }
    
    # Step 1: Reduce PP to 1
    benchmark_pp = 1
    benchmark_gpus_required = tp * benchmark_pp * ep * cp
    
    # Step 2: If still doesn't fit, rescale EP
    benchmark_ep = ep
    if benchmark_gpus_required > gpus_per_node:
        print(
            f"\n[Primus:Performance Projection] After reducing PP to 1, "
            f"config still requires {benchmark_gpus_required} GPUs (TP={tp}, EP={ep}, CP={cp})."
        )
        print(f"[Primus:Performance Projection] Rescaling EP to fit on {gpus_per_node} GPUs...")
        
        # Rescale EP to fit
        rescale_info = _rescale_expert_parallelism(original_config)
        if rescale_info:
            benchmark_ep = rescale_info['ep_after']
            benchmark_gpus_required = tp * benchmark_pp * benchmark_ep * cp
            
            if benchmark_gpus_required > gpus_per_node:
                raise ValueError(
                    f"[Primus:Performance Projection] Cannot reduce to single node.\n"
                    f"Even with PP=1 and EP={benchmark_ep}, configuration requires {benchmark_gpus_required} GPUs "
                    f"(TP={tp}, EP={benchmark_ep}, CP={cp}).\n"
                    f"Single node has only {gpus_per_node} GPUs.\n"
                    f"Please reduce TP or CP in your configuration."
                )
        else:
            # Rescaling didn't help or wasn't needed
            raise ValueError(
                f"[Primus:Performance Projection] Cannot reduce to single node.\n"
                f"Even with PP=1, configuration requires {benchmark_gpus_required} GPUs "
                f"(TP={tp}, EP={ep}, CP={cp}).\n"
                f"Single node has only {gpus_per_node} GPUs.\n"
                f"Please reduce TP, EP, or CP in your configuration."
            )
    
    # Modify the config
    original_config.pipeline_model_parallel_size = benchmark_pp
    
    # Also disable virtual pipeline stages (already done in _limit_layers_for_projection)
    for attr in ("num_layers_per_virtual_pipeline_stage", "num_virtual_stages_per_pipeline_rank"):
        if hasattr(original_config, attr):
            setattr(original_config, attr, None)
    
    return {
        'adjusted': True,
        'original_pp': pp,
        'benchmark_pp': benchmark_pp,
        'original_nodes_required': nodes_required,
        'original_tp': tp,
        'original_ep': ep,
        'benchmark_ep': benchmark_ep,
        'original_cp': cp,
    }


def _estimate_pp_communication_overhead(training_config, pp_size, hardware_config_dict=None):
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
    ep = getattr(mp_config, 'expert_model_parallel_size', 1)
    cp = getattr(mp_config, 'context_model_parallel_size', 1)
    
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
    total_p2p_time_ms = 2 * (pp_size - 1) * num_microbatches * p2p_time_per_transfer / 1000
    
    return total_p2p_time_ms


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
    hidden_size = model_config.hidden_size
    batch_size = runtime_config.micro_batch_size
    seq_len = runtime_config.sequence_length
    moe_router_topk = getattr(model_config, "moe_router_topk", 2)

    tokens_per_batch = seq_len * batch_size
    dispatch_size = tokens_per_batch * hidden_size * moe_router_topk * 2  # BF16

    # Calculate All-to-All time for original EP (dispatch + combine)
    a2a_dispatch_original = cm.alltoall(coll_args_original, dispatch_size, original_ep, groups=["ep"])
    a2a_combine_original = cm.alltoall(coll_args_original, dispatch_size, original_ep, groups=["ep"])
    a2a_time_original_fwd = (a2a_dispatch_original + a2a_combine_original) / 1000  # ms

    # Calculate All-to-All time for benchmark EP (dispatch + combine)
    a2a_dispatch_benchmark = cm.alltoall(coll_args_benchmark, dispatch_size, benchmark_ep, groups=["ep"])
    a2a_combine_benchmark = cm.alltoall(coll_args_benchmark, dispatch_size, benchmark_ep, groups=["ep"])
    a2a_time_benchmark_fwd = (a2a_dispatch_benchmark + a2a_combine_benchmark) / 1000  # ms

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
        activation = float(result.get("activation_memory_bytes", 0.0) or 0.0) / _BYTES_PER_GB
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
        first_chunk["activation"] += (embedding.get("activation_memory_bytes", 0.0) or 0.0) / _BYTES_PER_GB

    output = profiling_results.get("output")
    if output and chunk_timings[-1]:
        last_chunk = chunk_timings[-1][-1]
        last_chunk["fwd"] += output.get("forward_time_ms", 0.0) or 0.0
        out_bwd = output.get("backward_time_ms", 0.0) or 0.0
        last_chunk["bwd"] += out_bwd
        # wgrad already included in backward, don't add again
        last_chunk["activation"] += (output.get("activation_memory_bytes", 0.0) or 0.0) / _BYTES_PER_GB


def _build_chunk_time_matrix(training_config, layer_results: dict) -> Optional[List[List[dict]]]:
    model_cfg = getattr(training_config, "model_config", None)
    mp_cfg = getattr(training_config, "model_parallel_config", None)
    if model_cfg is None or mp_cfg is None:
        return None

    total_layers = getattr(model_cfg, "num_layers", 0) or 0
    if total_layers <= 0:
        return None

    layer_type_pattern = getattr(model_cfg, "moe_pattern", None)
    if not isinstance(layer_type_pattern, (list, tuple)) or len(layer_type_pattern) != total_layers:
        layer_type_pattern = [0] * total_layers
    type_timings = _extract_layer_type_timings(layer_results)
    if not type_timings:
        return None

    pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1
    vpp_size = getattr(mp_cfg, "virtual_pipeline_model_parallel_size", 1) or 1
    tp_size = getattr(mp_cfg, "tensor_model_parallel_size", 1) or 1
    cp_size = getattr(mp_cfg, "context_model_parallel_size", 1) or 1
    ep_size = getattr(mp_cfg, "expert_model_parallel_size", 1) or 1

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
        )
        if not layers:
            chunk_timings.append(
                [{"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0} for _ in range(vpp_size)]
            )
            continue

        layers_per_chunk = len(layers) // vpp_size if vpp_size else len(layers)
        if layers_per_chunk == 0:
            chunk_timings.append(
                [{"fwd": 0.0, "bwd": 0.0, "wgrad": 0.0, "activation": 0.0} for _ in range(vpp_size)]
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


def _build_scheduler_sim_config(training_config, profiling_results, enable_zero_bubble=False):
    chunk_time_matrix = _build_chunk_time_matrix(training_config, profiling_results)
    assert chunk_time_matrix is not None

    # For zero-bubble scheduling, we need to split backward into B (input grad) and W (weight grad)
    # The zero-bubble scheduler schedules these separately to minimize pipeline bubbles.
    # Typically B and W are roughly equal in duration (each ~50% of total backward).
    if enable_zero_bubble:
        print("\n[Primus:Performance Projection] Splitting backward time for zero-bubble scheduling:")
        print("  B (input grad) = 50% of backward, W (weight grad) = 50% of backward")
        for rank_chunks in chunk_time_matrix:
            for chunk in rank_chunks:
                total_bwd = chunk.get("bwd", 0.0)
                # Split: bwd becomes input gradient only, wgrad becomes weight gradient
                chunk["bwd"] = total_bwd * 0.5
                chunk["wgrad"] = total_bwd * 0.5

    if chunk_time_matrix:
        print("\n[Primus:Performance Projection] Per-chunk timings (ms):")
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
        print(f"[Primus:Performance Projection] Using zero-bubble scheduler (enable_zero_bubble=True)")
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
                    scheduled_layers.get("fwd_start", []), scheduled_layers.get("fwd_end", [])
                )
            )
            bwd_time = sum(
                end - start
                for start, end in zip(
                    scheduled_layers.get("bwd_start", []), scheduled_layers.get("bwd_end", [])
                )
            )
            wgrad_time = sum(
                end - start
                for start, end in zip(
                    scheduled_layers.get("wgrad_start", []), scheduled_layers.get("wgrad_end", [])
                )
            )
            total_layer_time = fwd_time + bwd_time + wgrad_time
            bubble_time = max(0.0, step_time_ms - total_layer_time)
            bubble_ratio = bubble_time / step_time_ms

            activation_trace = scheduled_layers.get("activation_memory_usage") or []
            peak_activation = (
                max(activation_trace) if activation_trace else scheduled_layers.get("memory", 0.0)
            )

            # Map rank_idx to pipeline rank (rank_idx // vpp_size)
            vpp_size = mp_cfg.virtual_pipeline_model_parallel_size or 1
            pp_rank = rank_idx // vpp_size
            if pp_rank not in param_mem_cache:
                param_mem_cache[pp_rank] = _get_parameter_memory(training_config, pp_rank)
            param_mem_gb = param_mem_cache[pp_rank]
            total_peak_gb = peak_activation + param_mem_gb
            rank_stats.append(
                (rank_idx, bubble_time, bubble_ratio, peak_activation, param_mem_gb, total_peak_gb)
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
            rank_idx, bubble_time, bubble_ratio, peak_activation, param_mem_gb, total_peak_gb = rank_info
            print(
                f"  Rank {rank_idx:02d} bubble: {bubble_time:.2f} ms "
                f"(ratio={bubble_ratio:.2%}), "
                f"activation_peak={peak_activation:.2f} GB, "
                f"param_memory={param_mem_gb:.2f} GB, "
                f"total_peak={total_peak_gb:.2f} GB"
            )
    
    return step_time_ms


def _run_layer_benchmark(primus_config, unknown_overrides):
    module_config = primus_config.get_module_config("pre_trainer")
    _limit_layers_for_projection(module_config)
    rescale_info = _rescale_expert_parallelism(module_config)
    training_config = convert_primus_config_to_projection_config(primus_config)

    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print("\n[Primus:Performance Projection] Initializing MegatronPretrainTrainer...")
    print(f"[Primus:Performance Projection] {primus_config}")
    primus_config.get_module_config("pre_trainer").overlap_grad_reduce = False
    primus_config.get_module_config("pre_trainer").overlap_param_gather = False
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

    print("\n[Primus:Performance Projection] Building model profiler...")
    model_profiler_spec = get_language_model_profiler_spec(training_config)
    model_profiler = build_profiler(model_profiler_spec)

    seq_len = training_config.runtime_config.sequence_length
    batch_size = training_config.runtime_config.micro_batch_size

    print(f"\n[Primus:Performance Projection] Benchmarking with:")
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

    print("\n" + "=" * 100)
    print("[Primus:Performance Projection] Starting layer benchmarking...")
    print("=" * 100)

    profiling_results = model_profiler.run_layer_benchmark(
        model=trainer.model,
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
    from primus.backends.megatron.core.pipeline_parallel.zerobubble.scheduler.graph import GraphConfig
    from primus.backends.megatron.core.pipeline_parallel.zerobubble.scheduler import zb
    
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
    
    print("\n[Primus:Performance Projection] Using Megatron zero-bubble scheduler (ILP-based)")
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
        
        print(f"  Stage {rank_idx}: F={fwd:.2f}ms, B={b_time:.2f}ms, W={w_time:.2f}ms, act={act_gb:.2f}GB")
    
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
    print("\n[Primus:Performance Projection] Running Megatron ZB schedule generation...")
    
    # Build graph and run initial_solution which explores multiple heuristics
    graph = zb.Graph.build_graph(pp_size, micro_batches, config)
    best_time, order, completion_time = zb.initial_solution(graph, print_result=False)
    
    step_time_ms = best_time
    
    # Calculate bubble time
    total_compute_per_mb = sum(cost_f) / pp_size + sum(cost_b) / pp_size + sum(cost_w) / pp_size
    ideal_time = total_compute_per_mb * micro_batches
    bubble_time = step_time_ms - ideal_time
    bubble_ratio = bubble_time / step_time_ms if step_time_ms > 0 else 0
    
    print(f"\n[Primus:Performance Projection] Megatron ZB Schedule Results:")
    print(f"  Step time: {step_time_ms:.2f} ms")
    print(f"  Ideal time (no bubble): {ideal_time:.2f} ms")
    print(f"  Bubble time: {bubble_time:.2f} ms ({bubble_ratio:.1%})")
    
    return step_time_ms


def _run_pipeline_simulation(training_config, profiling_results, enable_zero_bubble=False):
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
            return _run_pipeline_simulation_megatron_zb(training_config, profiling_results)
        except Exception as e:
            print(f"\n[Primus:Performance Projection] Megatron ZB scheduler failed: {e}")
            print("[Primus:Performance Projection] Falling back to simple simulator...")
    
    sim_config = _build_scheduler_sim_config(training_config, profiling_results, enable_zero_bubble)
    if sim_config is None:
        return None
    print("\n[Primus:Performance Projection] Running pipeline schedule simulator...")
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


def _run_multinode_projection(training_config, single_node_time_ms, profiling_results, args, target_nodes: int):
    """
    Run multinode projection to the specified target nodes.
    
    Args:
        training_config: Configuration object
        single_node_time_ms: Measured single-node time in ms
        profiling_results: Layer profiling results
        args: CLI arguments
        target_nodes: Target number of nodes for projection
    """
    import torch.distributed as dist
    
    # Only print from rank 0 to avoid duplicate output
    is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
    
    runtime_config = training_config.runtime_config
    mp_config = training_config.model_parallel_config
    
    # Get parallelism config
    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    ep = getattr(mp_config, 'expert_model_parallel_size', 1)
    cp = getattr(mp_config, 'context_model_parallel_size', 1)
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    
    # Calculate minimum nodes required by parallelism config
    # EP is included in the minimum GPUs calculation (need GPUs to hold experts)
    gpus_required = tp * pp * ep * cp
    min_nodes_required = (gpus_required + gpus_per_node - 1) // gpus_per_node
    
    # Validate target >= minimum required
    if target_nodes < min_nodes_required:
        raise ValueError(
            f"\n[Primus:Multinode] ERROR: Cannot project to {target_nodes} nodes.\n"
            f"Minimum required by parallelism config is {min_nodes_required} nodes.\n"
            f"--target-nodes must be >= {min_nodes_required}."
        )
    
    # Calculate DP for scaling - EXCLUDES EP (DP scaling is independent of EP)
    # EP distributes experts but doesn't affect how many data batches can be processed in parallel
    gpus_for_dp = tp * pp * cp  # EP excluded for DP calculation
    total_gpus_target = target_nodes * gpus_per_node
    dp_target = total_gpus_target // gpus_for_dp
    
    # Calculate minimum DP (for baseline)
    min_gpus = min_nodes_required * gpus_per_node
    min_dp = min_gpus // gpus_for_dp
    
    if is_rank_0:
        print("\n" + "=" * 100)
        print("Parallelism Configuration")
        print("=" * 100)
        print(f"  TP: {tp}, PP: {pp}, EP: {ep}, CP: {cp}")
        print(f"  GPUs per Node: {gpus_per_node}")
        print(f"  Minimum GPUs Required: {gpus_required}")
        print(f"  Minimum Nodes Required: {min_nodes_required}")
        print(f"  Target Nodes: {target_nodes}")
    
    # Load hardware config if provided
    hardware_config_dict = None
    if hasattr(args, 'hardware_config') and args.hardware_config:
        hardware_config_dict = load_hardware_config(args.hardware_config)
        if is_rank_0:
            print(f"\n  Using custom hardware config from: {args.hardware_config}")
    else:
        if is_rank_0:
            print(f"\n  Using default hardware parameters from custom_hardware_example.yaml")
    
    # Calculate communication times
    total_comm_time_ms, breakdown, message_info, per_layer_info = calculate_collective_communication_time(
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
    
    # Benchmarked time is for the minimum node configuration
    benchmarked_time_ms = single_node_time_ms
    
    # When scaling DP, we need to account for gradient all-reduce
    # Check if gradient all-reduce is overlapped
    overlap_grad_reduce = getattr(mp_config, 'overlap_grad_reduce', True)
    
    # Calculate projected time
    # 1. Scale compute time with DP
    if dp_target > min_dp:
        projected_compute_time_ms = benchmarked_time_ms * (min_dp / dp_target)
    else:
        projected_compute_time_ms = benchmarked_time_ms
    
    # 2. Add gradient all-reduce if NOT overlapped
    if not overlap_grad_reduce and dp_target > 1:
        # Calculate gradient all-reduce for target
        _, target_breakdown, _, _ = calculate_collective_communication_time(
            training_config, target_nodes, gpus_per_node, tp, pp, ep, cp, dp_target, hardware_config_dict
        )
        target_grad_ar = target_breakdown.get('gradient_allreduce', 0)
        projected_time_ms = projected_compute_time_ms + target_grad_ar
        grad_ar_msg = f"{target_grad_ar:.3f} ms (not overlapped)"
    else:
        # Gradient all-reduce is overlapped, so not in critical path
        projected_time_ms = projected_compute_time_ms
        if dp_target > 1:
            # Still calculate for reporting
            _, target_breakdown, _, _ = calculate_collective_communication_time(
                training_config, target_nodes, gpus_per_node, tp, pp, ep, cp, dp_target, hardware_config_dict
            )
            target_grad_ar = target_breakdown.get('gradient_allreduce', 0)
            grad_ar_msg = f"{target_grad_ar:.3f} ms (overlapped - not in critical path)"
        else:
            target_grad_ar = 0
            grad_ar_msg = "N/A"
    
    # For reporting, get full breakdown for target
    total_comm_time_ms, breakdown, message_info, per_layer_info = calculate_collective_communication_time(
        training_config, target_nodes, gpus_per_node, tp, pp, ep, cp, dp_target, hardware_config_dict
    )
    
    # Calculate speedup
    speedup = benchmarked_time_ms / projected_time_ms if projected_time_ms > 0 else 0
    ideal_speedup = dp_target / min_dp if min_dp > 0 else dp_target
    
    # Print results (only from rank 0)
    if is_rank_0:
        print("\n" + "=" * 100)
        print("Multinode Scaling Projection Results")
        print("=" * 100)
        print(f"\n📊 Parallelism: TP={tp}, PP={pp}, EP={ep}, CP={cp}")
        
        # Calculate DP for microbatch calculation (excludes EP)
        min_world_size = min_nodes_required * gpus_per_node
        target_world_size = target_nodes * gpus_per_node
        min_dp_for_microbatch = min_world_size // (tp * pp * cp)
        target_dp_for_microbatch = target_world_size // (tp * pp * cp)
        
        print(f"\n📦 Minimum Configuration ({min_nodes_required} node{'s' if min_nodes_required > 1 else ''}):")
        print(f"   Nodes: {min_nodes_required}, GPUs: {min_world_size}")
        print(f"   TP={tp}, PP={pp}, EP={ep}, CP={cp}, DP={min_dp_for_microbatch}")
        print(f"   Iteration Time: {benchmarked_time_ms:.3f} ms")
        
        print(f"\n🎯 Target Configuration ({target_nodes} nodes):")
        print(f"   Nodes: {target_nodes}, GPUs: {target_world_size}")
        print(f"   TP={tp}, PP={pp}, EP={ep}, CP={cp}, DP={target_dp_for_microbatch}")
        print(f"   DP Scaling Factor: {target_dp_for_microbatch/min_dp_for_microbatch if min_dp_for_microbatch > 0 else target_dp_for_microbatch:.1f}x")
        
        print(f"\n📡 Communication Breakdown:")
        
        # Show individual comm operations with message sizes
        for op_name, op_time in breakdown.items():
            if op_time > 0:
                # Print operation time
                print(f"   {op_name}: {op_time:.3f} ms", end="")
                
                # Add message size if available
                if op_name == 'gradient_allreduce' and 'gradient_allreduce_size_mb' in message_info:
                    overlapped_str = " [OVERLAPPED]" if message_info.get('gradient_allreduce_overlapped', False) else ""
                    print(f" (message: {message_info['gradient_allreduce_size_mb']:.2f} MB){overlapped_str}")
                elif op_name == 'moe_a2a_fwd' and 'moe_a2a_size_mb' in message_info:
                    print(f" (message: {message_info['moe_a2a_size_mb']:.2f} MB, {message_info['num_moe_layers']} layers × {message_info['moe_a2a_per_layer_fwd']:.3f} ms/layer)")
                elif op_name == 'moe_a2a_bwd' and 'moe_a2a_size_mb' in message_info:
                    print(f" (message: {message_info['moe_a2a_size_mb']:.2f} MB, {message_info['num_moe_layers']} layers × {message_info['moe_a2a_per_layer_fwd']:.3f} ms/layer)")
                else:
                    print()
        
        print(f"\n   Total Communication (critical path): {total_comm_time_ms:.3f} ms")
        
        print(f"\n📊 Performance Summary:")
        print(f"   Base Projection:   {benchmarked_time_ms:.3f} ms ({min_nodes_required} node{'s' if min_nodes_required > 1 else ''}, DP={min_dp_for_microbatch}) [from single-node profiling]")
        print(f"   Target Projection: {projected_time_ms:.3f} ms ({target_nodes} nodes, DP={target_dp_for_microbatch})")
        print(f"   Gradient AllReduce: {grad_ar_msg}")
        
        print("\n" + "=" * 100)


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
        raise FileNotFoundError(f"[Primus:Performance Projection] Config file '{cfg_path}' not found.")

    # Load Primus configuration
    primus_config, unknown_overrides = load_primus_config(args, overrides)
    primus_config_original = copy.deepcopy(primus_config)
    
    # Check if we need to reduce config for single-node benchmarking
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    
    # Get target nodes from CLI flag (--target-nodes)
    target_nodes = getattr(args, 'target_nodes', None)
    
    # Store original parallelism before any modifications
    module_config = primus_config.get_module_config("pre_trainer")
    reduction_info = _calculate_single_node_config(
        copy.deepcopy(module_config), 
        gpus_per_node
    )
    
    # Calculate minimum nodes required
    min_nodes_required = reduction_info['original_nodes_required']
    
    # If target_nodes not specified, default to minimum required
    if target_nodes is None:
        target_nodes = min_nodes_required
    
    if reduction_info['adjusted']:
        print("\n" + "=" * 100)
        print("[Primus:Performance Projection] Multi-node configuration detected")
        print("=" * 100)
        print(f"  Original configuration requires {min_nodes_required} nodes minimum:")
        print(f"    TP={reduction_info['original_tp']}, PP={reduction_info['original_pp']}, "
              f"EP={reduction_info['original_ep']}, CP={reduction_info['original_cp']}")
        print(f"\n  Reducing to single-node configuration for benchmarking:")
        print(f"    TP={reduction_info['original_tp']}, PP={reduction_info['benchmark_pp']}, "
              f"EP={reduction_info['benchmark_ep']}, CP={reduction_info['original_cp']}")
        
        # Show what was changed
        changes = []
        if reduction_info['original_pp'] != reduction_info['benchmark_pp']:
            changes.append(f"PP {reduction_info['original_pp']} → {reduction_info['benchmark_pp']}")
        if reduction_info['original_ep'] != reduction_info['benchmark_ep']:
            changes.append(f"EP {reduction_info['original_ep']} → {reduction_info['benchmark_ep']}")
        
        if changes:
            print(f"    ({', '.join(changes)})")
        
        print(f"\n  Will estimate performance by adding PP communication overhead back.")
        print("=" * 100)
        
        # Apply the reduction to the config used for benchmarking
        primus_config.get_module_config("pre_trainer").pipeline_model_parallel_size = reduction_info['benchmark_pp']
        primus_config.get_module_config("pre_trainer").expert_model_parallel_size = reduction_info['benchmark_ep']

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
    cp = getattr(mp_config, 'context_parallel_size', 1) or 1
    ep = getattr(mp_config, 'expert_model_parallel_size', 1) or 1
    
    # Calculate DP for the TARGET configuration (what we're projecting to)
    # The pipeline simulator simulates the target config, so it needs target DP for microbatch calculation
    target_world_size = target_nodes * gpus_per_node
    
    # For MoE models: DP calculation excludes EP since experts are distributed but data is replicated
    target_dp = target_world_size // (tp * pp * cp)
    
    # Also show benchmark config for reference
    benchmark_world_size = gpus_per_node  # Benchmarking always happens on 1 node
    benchmark_pp = reduction_info.get('benchmark_pp', pp)
    benchmark_ep = reduction_info.get('benchmark_ep', ep)
    benchmark_dp = benchmark_world_size // (tp * benchmark_pp * cp)
    
    print(f"\n[Primus:Performance Projection] Configuration Summary:")
    print(f"  Benchmark Config: PP={benchmark_pp}, EP={benchmark_ep}, TP={tp}, CP={cp}, DP={benchmark_dp} (1 node)")
    print(f"  Target Config: PP={pp}, EP={ep}, TP={tp}, CP={cp}, DP={target_dp} ({target_nodes} nodes)")
    print(f"    Note: EP does not reduce DP for microbatch calculation")
    
    # Use BENCHMARK DP for pipeline simulation to get consistent baseline
    # The multinode projection will then scale from this baseline to target
    global_batch = training_config.runtime_config.global_batch_size
    micro_batch = training_config.runtime_config.micro_batch_size
    benchmark_microbatches = global_batch // (micro_batch * benchmark_dp)
    print(f"  Benchmark Microbatches: {benchmark_microbatches} (global_batch={global_batch}, micro_batch={micro_batch}, benchmark_dp={benchmark_dp})")
    
    training_config.runtime_config.data_parallel_size = benchmark_dp


    # If EP was rescaled, adjust profiling_results to add EP overhead BEFORE pipeline simulation
    ep_overhead_applied = False
    if reduction_info["adjusted"] and reduction_info["original_ep"] != reduction_info["benchmark_ep"]:
        # Load hardware config if provided
        hardware_config_dict = None
        if hasattr(args, "hardware_config") and args.hardware_config:
            hardware_config_dict = load_hardware_config(args.hardware_config)

        # Calculate EP communication overhead per layer
        fwd_overhead_per_layer, bwd_overhead_per_layer = _estimate_ep_communication_overhead(
            training_config,
            reduction_info["original_ep"],
            reduction_info["benchmark_ep"],
            hardware_config_dict,
        )

        if fwd_overhead_per_layer > 0 or bwd_overhead_per_layer > 0:
            print(f"\n[Primus:Performance Projection] Adjusting profiling results for EP scaling:")
            print(f"  EP rescaled: {reduction_info['benchmark_ep']} → {reduction_info['original_ep']}")
            print(f"  Adding per-layer All-to-All overhead to MoE layers:")
            print(f"    Forward:  +{fwd_overhead_per_layer:.3f} ms/layer")
            print(f"    Backward: +{bwd_overhead_per_layer:.3f} ms/layer")

            # Adjust MoE layer times in profiling_results
            moe_layers_adjusted = 0
            for layer_idx, layer_data in profiling_results.items():
                if isinstance(layer_data, dict) and layer_data.get("type") == "moe":
                    layer_data["forward_time_ms"] = layer_data.get("forward_time_ms", 0) + fwd_overhead_per_layer
                    layer_data["backward_time_ms"] = layer_data.get("backward_time_ms", 0) + bwd_overhead_per_layer
                    moe_layers_adjusted += 1

            print(f"  Adjusted {moe_layers_adjusted} MoE layer(s) in profiling results")
            ep_overhead_applied = True

    # Check if zero-bubble scheduling is enabled in the original config
    original_module_config = primus_config_original.get_module_config("pre_trainer")
    enable_zero_bubble = getattr(original_module_config, "enable_zero_bubble", False)
    
    pipeline_simulation_time_ms = _run_pipeline_simulation(training_config, profiling_results, enable_zero_bubble)
    
    # Run multinode projection if target_nodes > min_nodes_required (scaling up)
    # or always run to show performance summary
    if target_nodes >= min_nodes_required:
        print("\n" + "=" * 100)
        print("[Primus:Performance] Running multinode scaling projection")
        print("=" * 100)
        
        # Use pipeline simulation time if available, otherwise extract from profiling
        if pipeline_simulation_time_ms is not None:
            print(f"\n[Primus:Performance Projection] Using pipeline simulation time: {pipeline_simulation_time_ms:.2f} ms")
            # Pipeline simulation already accounts for PP communication and bubbles
            # No need to add additional PP overhead
            benchmarked_time_ms = pipeline_simulation_time_ms
            print(f"  (Pipeline simulation already includes PP={reduction_info['original_pp']} effects)")
        else:
            print(f"\n[Primus:Performance Projection] Pipeline simulation not available, using extrapolated time from profiling")
            measured_time_ms = extract_single_node_time_from_profiling(profiling_results, training_config)
            
            # If we reduced PP for benchmarking, estimate the time with PP overhead
            if reduction_info['adjusted']:
                # Load hardware config if provided
                hardware_config_dict = None
                if hasattr(args, 'hardware_config') and args.hardware_config:
                    hardware_config_dict = load_hardware_config(args.hardware_config)
                
                # Estimate PP overhead for original configuration
                pp_overhead_ms = _estimate_pp_communication_overhead(
                    training_config, 
                    reduction_info['original_pp'],
                    hardware_config_dict
                )
                
                benchmarked_time_ms = measured_time_ms + pp_overhead_ms
                
                print(f"\n[Primus:Performance Projection] Time Adjustment:")
                print(f"  Measured time (PP={reduction_info['benchmark_pp']}): {measured_time_ms:.2f} ms")
                print(f"  Estimated PP overhead (PP={reduction_info['original_pp']}): {pp_overhead_ms:.2f} ms")
                print(f"  Estimated time: {benchmarked_time_ms:.2f} ms")
            else:
                benchmarked_time_ms = measured_time_ms

            # If EP was rescaled and pipeline simulation wasn't used, add EP overhead here
            # (If pipeline simulation was used, EP overhead was already applied to profiling_results)
            if not ep_overhead_applied and reduction_info["adjusted"] and reduction_info["original_ep"] != reduction_info["benchmark_ep"]:
                # Get the number of MoE layers
                moe_pattern = getattr(training_config.model_config, "moe_layer_pattern", [])
                if not moe_pattern:
                    # If no pattern, check if model has MoE layers
                    num_moe_layers = getattr(training_config.model_config, "num_moe_layers", 0)
                else:
                    num_moe_layers = sum(1 for x in moe_pattern if x == 1)

                if num_moe_layers > 0:
                    # Calculate EP communication overhead per layer
                    fwd_overhead_per_layer, bwd_overhead_per_layer = _estimate_ep_communication_overhead(
                        training_config,
                        reduction_info["original_ep"],
                        reduction_info["benchmark_ep"],
                        hardware_config_dict,
                    )

                    # Total EP overhead = per-layer overhead * number of MoE layers
                    total_ep_overhead_ms = (fwd_overhead_per_layer + bwd_overhead_per_layer) * num_moe_layers

                    print(f"\n[Primus:Performance Projection] EP Communication Adjustment:")
                    print(f"  EP rescaled: {reduction_info['benchmark_ep']} → {reduction_info['original_ep']}")
                    print(f"  Number of MoE layers: {num_moe_layers}")
                    print(f"  Additional All-to-All overhead per MoE layer:")
                    print(f"    Forward:  {fwd_overhead_per_layer:.3f} ms")
                    print(f"    Backward: {bwd_overhead_per_layer:.3f} ms")
                    print(f"  Total EP overhead: {total_ep_overhead_ms:.3f} ms")

                    benchmarked_time_ms += total_ep_overhead_ms
                    print(f"  Adjusted time: {benchmarked_time_ms:.3f} ms")

        # Run multinode projection
        _run_multinode_projection(
            training_config, 
            benchmarked_time_ms,
            profiling_results,
            args,
            target_nodes
        )

