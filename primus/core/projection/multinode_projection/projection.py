###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import copy
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.training_config import convert_primus_config_to_projection_config
from primus.core.projection.module_profilers.collective_args import get_default_args
from primus.core.projection.module_profilers import collective_model as cm


def parse_parallelism_strategies(strategies_str: str) -> List[Dict[str, Any]]:
    """
    Parse parallelism strategies from string or config.
    
    Format: "tp=2,pp=1,ep=8,cp=1,dp=-1;tp=1,pp=2,ep=8,cp=1,dp=-1"
    dp=-1 means auto-calculate based on available GPUs
    """
    if not strategies_str:
        return []
    
    strategies = []
    for strategy_str in strategies_str.split(';'):
        strategy = {}
        for param in strategy_str.split(','):
            key, value = param.split('=')
            strategy[key.strip()] = int(value.strip())
        strategies.append(strategy)
    
    return strategies


def get_parallelism_strategies_from_config(training_config) -> List[Dict[str, Any]]:
    """
    Get parallelism strategies from the training config.
    Returns a list with single strategy based on config values.
    """
    mp_config = training_config.model_parallel_config
    
    return [{
        'tp': mp_config.tensor_model_parallel_size,
        'pp': mp_config.pipeline_model_parallel_size,
        'ep': mp_config.expert_model_parallel_size,
        'cp': mp_config.context_model_parallel_size,
        'dp': -1  # Auto-calculate
    }]


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
    
    # 3. Tensor Parallel AllReduce (TP group)
    if tp > 1:
        tp_ar_size = batch_size * seq_len * hidden_size * 2  # BF16
        tp_ar_time = cm.allreduce(coll_args, tp_ar_size, tp, groups=['tp'])
        total_tp_time = tp_ar_time * 2 * num_layers / 1000  # 2 per layer, convert to ms
        breakdown['tp_allreduce'] = total_tp_time
        message_info['tp_allreduce_size'] = tp_ar_size
        message_info['tp_allreduce_size_mb'] = tp_ar_size / (1024 * 1024)
        message_info['tp_allreduce_per_layer'] = tp_ar_time * 2 / 1000
        message_info['num_layers'] = num_layers
    else:
        breakdown['tp_allreduce'] = 0.0
        message_info['tp_allreduce_size'] = 0
        message_info['tp_allreduce_size_mb'] = 0.0
        message_info['tp_allreduce_per_layer'] = 0.0
        message_info['num_layers'] = num_layers
    
    # 4. Pipeline Parallel P2P (PP group)
    if pp > 1:
        p2p_size = batch_size * seq_len * hidden_size * 2  # BF16
        p2p_time = cm.sendrecv(coll_args, p2p_size)
        num_microbatches = runtime_config.global_batch_size // (batch_size * dp)
        total_pp_time = p2p_time * num_microbatches * 2 / 1000  # fwd + bwd, convert to ms
        breakdown['pp_p2p'] = total_pp_time
        message_info['pp_p2p_size'] = p2p_size
        message_info['pp_p2p_size_mb'] = p2p_size / (1024 * 1024)
        message_info['num_microbatches'] = num_microbatches
        message_info['pp_p2p_per_microbatch'] = p2p_time * 2 / 1000
    else:
        breakdown['pp_p2p'] = 0.0
        message_info['pp_p2p_size'] = 0
        message_info['pp_p2p_size_mb'] = 0.0
    
    # Build per-layer communication information
    for layer_idx in range(num_layers):
        layer_comm = {
            'layer_idx': layer_idx,
            'layer_type': 'MoE' if moe_pattern[layer_idx] == 1 else 'Dense',
            'communications': []
        }
        
        # TP AllReduce (if TP > 1, happens in every layer)
        if tp > 1:
            layer_comm['communications'].append({
                'type': 'TP AllReduce',
                'time_ms': tp_ar_time * 2 / 1000,  # 2 per layer
                'message_size_mb': tp_ar_size / (1024 * 1024),
                'group_size': tp,
            })
        
        # MoE All-to-All (if EP > 1 and this is a MoE layer)
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
    print("\n[Primus:Multinode] Extracting timing from benchmark results...")
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
    print(f"[Primus:Multinode] Extrapolated Baseline Time: {total_time_ms:.2f} ms/iteration")
    print(f"  (Based on {len(profiled_layer_indices)} profiled layers → {len(moe_pattern)} total layers)")
    print("=" * 100)
    
    return total_time_ms


def launch_projection_from_cli(args, overrides):
    """
    Entry point for 'projection multinode' subcommand.
    
    DEFAULT BEHAVIOR: Automatically runs performance projection to benchmark and get single-node time.
    OPTIONAL: Provide --single-node-time to skip benchmarking and use provided value.
    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Projection] Config file '{cfg_path}' not found.")
    
    # Load configuration first
    primus_config, unknown_overrides = load_primus_config(args, overrides)
    primus_config_original = copy.deepcopy(primus_config)
    training_config = convert_primus_config_to_projection_config(primus_config_original)
    
    # Determine single-node time
    if hasattr(args, 'single_node_time') and args.single_node_time is not None:
        # User provided manual time - skip benchmarking
        single_node_time_ms = args.single_node_time
        print(f"\n[Primus:Multinode] Using Provided Single-Node Time: {single_node_time_ms:.2f} ms/iteration")
    else:
        # DEFAULT: Automatically benchmark using performance projection's _run_layer_benchmark
        print("\n" + "=" * 100)
        print("[Primus:Multinode] No --single-node-time provided.")
        print("[Primus:Multinode] Running performance projection to benchmark model automatically...")
        print("[Primus:Multinode] Benchmarking will run on SINGLE NODE to measure baseline performance.")
        print("[Primus:Multinode] Then collective model will project to multiple nodes.")
        print("[Primus:Multinode] This requires GPU access.")
        print("=" * 100)
        
        # Import and run the same benchmark function as performance projection
        from primus.core.projection.performance_projection.projection import _run_layer_benchmark
        
        print("\n[Primus:Multinode] Starting single-node layer benchmarking...")
        profiling_results = _run_layer_benchmark(primus_config, unknown_overrides)
        
        # Extract single-node time from profiling results
        single_node_time_ms = extract_single_node_time_from_profiling(profiling_results, training_config)
    
    # Load hardware config if provided
    hardware_config = None
    if hasattr(args, 'hardware_config') and args.hardware_config:
        hardware_config = load_hardware_config(args.hardware_config)
    
    # Get parallelism from config
    mp_config = training_config.model_parallel_config
    model_config = training_config.model_config
    runtime_config = training_config.runtime_config
    
    # Get environment variables for multinode projection
    # Use PROJECTION_NNODES for target if set (allows runner to use different NNODES for execution)
    # Otherwise use NNODES for both execution and projection target
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    num_nodes_target = int(os.getenv("PROJECTION_NNODES", os.getenv("NNODES", "4")))
    actual_execution_nnodes = int(os.getenv("NNODES", "1"))
    
    # Get parallelism from config (avoid duplicate)
    tp_base = mp_config.tensor_model_parallel_size
    pp_base = mp_config.pipeline_model_parallel_size
    ep_base = mp_config.expert_model_parallel_size
    cp_base = mp_config.context_model_parallel_size
    
    # Calculate baseline nodes required by parallelism configuration
    gpus_required = tp_base * pp_base * ep_base * cp_base
    baseline_nodes = (gpus_required + gpus_per_node - 1) // gpus_per_node  # Ceiling division
    
    # Validate: can't project to fewer nodes than baseline
    if num_nodes_target < baseline_nodes:
        raise ValueError(
            f"[Primus:Multinode] ERROR: Cannot project to {num_nodes_target} nodes.\n"
            f"  Parallelism configuration requires at least {baseline_nodes} nodes:\n"
            f"    TP={tp_base} × PP={pp_base} × EP={ep_base} × CP={cp_base} = {gpus_required} GPUs\n"
            f"    {gpus_required} GPUs / {gpus_per_node} GPUs/node = {baseline_nodes} nodes minimum\n"
            f"  Please either:\n"
            f"    - Increase NNODES/PROJECTION_NNODES to at least {baseline_nodes}\n"
            f"    - Reduce parallelism in config (TP/PP/EP/CP)"
        )
    
    # Warn if auto-benchmarking but runner launched with wrong NNODES
    if not (hasattr(args, 'single_node_time') and args.single_node_time is not None):
        # Auto-benchmarking mode
        if actual_execution_nnodes != baseline_nodes:
            print(f"\n⚠️  WARNING: Runner was launched with NNODES={actual_execution_nnodes}")
            print(f"    But baseline configuration requires {baseline_nodes} nodes for benchmarking")
            print(f"    Projection target: {num_nodes_target} nodes")
            print(f"\n    For correct usage:")
            print(f"      NNODES={baseline_nodes} PROJECTION_NNODES={num_nodes_target} bash runner/primus-cli direct ...")
            print(f"    Or use --single-node-time to skip GPU benchmarking\n")
            
            if actual_execution_nnodes < baseline_nodes:
                raise ValueError(
                    f"[Primus:Multinode] Cannot benchmark with NNODES={actual_execution_nnodes} < baseline={baseline_nodes}. "
                    f"Set NNODES={baseline_nodes} for runner execution."
                )
    
    
    print(f"\n[Primus:Multinode] Parallelism Configuration Analysis:")
    print(f"  TP={tp_base}, PP={pp_base}, EP={ep_base}, CP={cp_base}")
    print(f"  Total GPUs required: {gpus_required}")
    print(f"  Baseline nodes (for benchmarking): {baseline_nodes} nodes")
    print(f"  Target nodes (for projection): {num_nodes_target} nodes")
    
    if baseline_nodes > 1:
        print(f"\n[Primus:Multinode] NOTE: Parallelism config requires {baseline_nodes} nodes minimum")
        print(f"  Benchmarking will be performed on {baseline_nodes} nodes (baseline)")
        print(f"  Projection will scale from {baseline_nodes} nodes → {num_nodes_target} nodes")
    else:
        print(f"\n[Primus:Multinode] Benchmarking will be performed on 1 node")
        print(f"  Projection will scale from 1 node → {num_nodes_target} nodes")
    
    
    # Get parallelism strategies from config (default behavior)
    print("\n[Primus:Multinode] Reading parallelism strategies from config...")
    strategies = get_parallelism_strategies_from_config(training_config)
    
    if not strategies:
        # Fallback: use config values as single strategy
        strategies = [{
            'tp': tp_base,
            'pp': pp_base,
            'ep': ep_base,
            'cp': cp_base,
            'dp': -1  # Auto
        }]
    
    print(f"[Primus:Multinode] Found {len(strategies)} parallelism strateg{'y' if len(strategies) == 1 else 'ies'} to evaluate")
    
    print("\n" + "=" * 100)
    print("[Primus:Projection] Multinode Scaling Projection")
    print("=" * 100)
    print(f"\nModel: {model_config.num_layers} layers, Hidden Size: {model_config.hidden_size}")
    print(f"Batch Size: {runtime_config.micro_batch_size}, Sequence Length: {runtime_config.sequence_length}")
    print(f"\nBaseline Cluster: {baseline_nodes} nodes × {gpus_per_node} GPUs = {baseline_nodes * gpus_per_node} total GPUs")
    print(f"Target Cluster: {num_nodes_target} nodes × {gpus_per_node} GPUs = {num_nodes_target * gpus_per_node} total GPUs")
    print(f"Measured Baseline Time: {single_node_time_ms:.2f} ms/iteration (on {baseline_nodes} node{'s' if baseline_nodes > 1 else ''})")
    
    if hardware_config:
        print(f"\nUsing Custom Hardware Configuration:")
        for key, value in hardware_config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 100)
    print("Parallelism Strategy Analysis")
    print("=" * 100)
    
    results = []
    
    for idx, strategy in enumerate(strategies, 1):
        tp = strategy.get('tp', tp_base)
        pp = strategy.get('pp', pp_base)
        ep = strategy.get('ep', ep_base)
        cp = strategy.get('cp', cp_base)
        dp = strategy.get('dp', -1)
        
        # Calculate DP for target cluster
        total_gpus_target = num_nodes_target * gpus_per_node
        if dp == -1:
            dp = total_gpus_target // (tp * pp * ep * cp)
        
        # Calculate baseline DP (for reference)
        total_gpus_baseline = baseline_nodes * gpus_per_node
        dp_baseline = total_gpus_baseline // (tp * pp * ep * cp)
        
        print(f"\n{'─' * 100}")
        print(f"Strategy #{idx}: TP={tp}, PP={pp}, EP={ep}, CP={cp}")
        print(f"  Baseline: DP={dp_baseline} ({baseline_nodes} nodes × {gpus_per_node} GPUs)")
        print(f"  Target:   DP={dp} ({num_nodes_target} nodes × {gpus_per_node} GPUs)")
        print(f"{'─' * 100}")
        
        # Calculate communication time for target configuration
        comm_time_ms, breakdown, message_info, per_layer_info = calculate_collective_communication_time(
            training_config,
            num_nodes_target,
            gpus_per_node,
            tp, pp, ep, cp, dp,
            hardware_config,
        )
        
        # Projected total time
        # Baseline time is measured on baseline_nodes, compute doesn't scale beyond baseline
        projected_total_time = single_node_time_ms + comm_time_ms
        
        # Calculate speedup relative to baseline
        speedup = single_node_time_ms / projected_total_time if projected_total_time > 0 else 0
        ideal_speedup = dp / dp_baseline if dp_baseline > 0 else dp
        
        print(f"\nCommunication Breakdown:")
        if breakdown['gradient_allreduce'] > 0:
            overlapped_str = " [OVERLAPPED]" if message_info.get('gradient_allreduce_overlapped', False) else ""
            print(f"  Gradient AllReduce (DP={dp}): {breakdown['gradient_allreduce']:.3f} ms (message: {message_info['gradient_allreduce_size_mb']:.2f} MB){overlapped_str}")
        if breakdown['moe_a2a_fwd'] > 0:
            print(f"  MoE All-to-All (forward): {breakdown['moe_a2a_fwd']:.3f} ms (message: {message_info['moe_a2a_size_mb']:.2f} MB, {message_info['num_moe_layers']} layers × {message_info['moe_a2a_per_layer_fwd']:.3f} ms/layer)")
            print(f"  MoE All-to-All (backward): {breakdown['moe_a2a_bwd']:.3f} ms (message: {message_info['moe_a2a_size_mb']:.2f} MB, {message_info['num_moe_layers']} layers × {message_info['moe_a2a_per_layer_fwd']:.3f} ms/layer)")
        if breakdown['tp_allreduce'] > 0:
            print(f"  TP AllReduce: {breakdown['tp_allreduce']:.3f} ms (message: {message_info['tp_allreduce_size_mb']:.2f} MB, {message_info['num_layers']} layers × {message_info['tp_allreduce_per_layer']:.3f} ms/layer)")
        if breakdown['pp_p2p'] > 0:
            print(f"  PP P2P: {breakdown['pp_p2p']:.3f} ms (message: {message_info['pp_p2p_size_mb']:.2f} MB, {message_info.get('num_microbatches', 'N/A')} microbatches × {message_info.get('pp_p2p_per_microbatch', 0):.3f} ms/microbatch)")
        
        print(f"\n  Total Communication Time: {comm_time_ms:.3f} ms")
        
        print(f"\n📊 Performance Summary:")
        print(f"  Baseline:       {single_node_time_ms:.3f} ms ({baseline_nodes} node{'s' if baseline_nodes > 1 else ''}, DP={dp_baseline})")
        print(f"  Projected:      {projected_total_time:.3f} ms ({num_nodes_target} nodes, DP={dp})")
        print(f"  Communication:  {comm_time_ms:.3f} ms ({(comm_time_ms/projected_total_time*100):.1f}% of projected time)")
        
        print(f"\n⚡ Speedup vs Baseline: {speedup:.2f}x")
        print(f"  Ideal speedup (no communication overhead): {ideal_speedup:.1f}x")
        
        results.append({
            'strategy': f"TP={tp},PP={pp},EP={ep},CP={cp},DP={dp}",
            'comm_time': comm_time_ms,
            'total_time': projected_total_time,
            'speedup': speedup,
            'ideal_speedup': ideal_speedup,
        })
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 100)
        print("Strategy Comparison Summary")
        print("=" * 100)
        print(f"\n{'Strategy':<30} {'Comm (ms)':<12} {'Total (ms)':<12} {'Speedup':<12} {'Ideal Speedup':<15}")
        print("─" * 100)
        for r in results:
            print(f"{r['strategy']:<30} {r['comm_time']:<12.3f} {r['total_time']:<12.3f} {r['speedup']:<11.2f}x {r['ideal_speedup']:<14.1f}x")
        
        # Find best strategy
        best = min(results, key=lambda x: x['total_time'])
        print(f"\n🏆 Best Strategy: {best['strategy']}")
        print(f"   Projected Time: {best['total_time']:.3f} ms/iteration")
        print(f"   Speedup: {best['speedup']:.2f}x (ideal: {best['ideal_speedup']:.1f}x)")
    
    print("\n" + "=" * 100)
    print("Projection Complete")
    print("=" * 100)
