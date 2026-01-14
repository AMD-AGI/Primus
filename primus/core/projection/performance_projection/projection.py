from typing import List, Dict,  Optional
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import math
import os
from pathlib import Path

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.module_profilers.language_model import (
    LanguageModelProfiler,
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.performance_projection.simulator import (
    SchedulerSimulationRunner,
)
from primus.core.projection.training_config import (
    convert_primus_config_to_projection_config,
)
from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

_MAX_EXPERT_PARALLEL_SIZE = 8
_BYTES_PER_GB = 1024**3


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
    from primus.core.projection.multinode_projection.projection import get_default_args
    from primus.core.projection.module_profilers import collective_model as cm
    
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
            # weight grad measured together with backward in our approximation
            "wgrad": backward,
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
        first_chunk["wgrad"] += emb_bwd
        first_chunk["activation"] += (embedding.get("activation_memory_bytes", 0.0) or 0.0) / _BYTES_PER_GB

    output = profiling_results.get("output")
    if output and chunk_timings[-1]:
        last_chunk = chunk_timings[-1][-1]
        last_chunk["fwd"] += output.get("forward_time_ms", 0.0) or 0.0
        out_bwd = output.get("backward_time_ms", 0.0) or 0.0
        last_chunk["bwd"] += out_bwd
        last_chunk["wgrad"] += out_bwd
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
    expert_model_parallel_size = getattr(model_parallel_config, "expert_model_parallel_size", None) or 1

    denominator = micro_batch * data_parallel_size * expert_model_parallel_size
    if denominator <= 0:
        return 1
    return max(1, math.ceil(global_batch / denominator))


def _build_scheduler_sim_config(training_config, profiling_results):
    chunk_time_matrix = _build_chunk_time_matrix(training_config, profiling_results)
    assert chunk_time_matrix is not None

    if chunk_time_matrix:
        print("\n[Primus:Performance Projection] Per-chunk timings (ms):")
        for rank_idx, rank_chunks in enumerate(chunk_time_matrix):
            for chunk_idx, chunk in enumerate(rank_chunks):
                fwd = chunk.get("fwd", 0.0)
                bwd = chunk.get("bwd", 0.0)
                activation = chunk.get("activation", 0.0)
                print(
                    f"  Rank {rank_idx:02d} Chunk {chunk_idx:02d} -> "
                    f"fwd={fwd:.2f} ms, bwd={bwd:.2f} ms, activation={activation:.2f} GB"
                )

    mp_cfg = training_config.model_parallel_config
    pp_size = getattr(mp_cfg, "pipeline_model_parallel_size", 1) or 1
    vpp_size = getattr(mp_cfg, "virtual_pipeline_model_parallel_size", 1) or 1
    print(f"pp_size: {pp_size}, vpp_size: {vpp_size}")

    micro_batches = _compute_micro_batches(training_config.runtime_config, mp_cfg)

    if vpp_size > 1:
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


def _run_pipeline_simulation(training_config, profiling_results):
    """
    Run pipeline simulation and return the step time.
    
    Returns:
        float: Step time in ms from pipeline simulation, or None if simulation failed
    """
    sim_config = _build_scheduler_sim_config(training_config, profiling_results)
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


def _run_multinode_projection(training_config, single_node_time_ms, profiling_results, args):
    """
    Run multinode projection using PROJECTION_NNODES as the target.
    
    Args:
        training_config: Configuration object
        single_node_time_ms: Measured single-node time in ms
        profiling_results: Layer profiling results
        args: CLI arguments
    """
    from primus.core.projection.multinode_projection.projection import (
        load_hardware_config,
        calculate_collective_communication_time,
    )
    
    runtime_config = training_config.runtime_config
    mp_config = training_config.model_parallel_config
    
    # Get baseline calculation
    tp = mp_config.tensor_model_parallel_size
    pp = mp_config.pipeline_model_parallel_size
    ep = getattr(mp_config, 'expert_model_parallel_size', 1)
    cp = getattr(mp_config, 'context_model_parallel_size', 1)
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    
    # Calculate minimum nodes required by parallelism config
    gpus_required = tp * pp * ep * cp
    min_nodes_required = (gpus_required + gpus_per_node - 1) // gpus_per_node
    
    # Baseline is always the minimum required by the original parallelism config
    # (even if we benchmarked on 1 node with reduced PP)
    baseline_nodes = min_nodes_required
    
    # Get target nodes from PROJECTION_NNODES
    projection_nnodes_str = os.getenv("PROJECTION_NNODES")
    if not projection_nnodes_str:
        print("\n[Primus:Multinode] ERROR: PROJECTION_NNODES not set")
        return
    
    num_nodes_target = int(projection_nnodes_str)
    
    # Validate target >= baseline
    if num_nodes_target < baseline_nodes:
        raise ValueError(
            f"\n[Primus:Multinode] ERROR: Cannot project to {num_nodes_target} nodes.\n"
            f"Baseline is {baseline_nodes} nodes (minimum required by parallelism config).\n"
            f"PROJECTION_NNODES must be >= {baseline_nodes}."
        )
    
    # Calculate target DP
    total_gpus_target = num_nodes_target * gpus_per_node
    dp_target = total_gpus_target // gpus_required
    
    print("\n" + "=" * 100)
    print("Parallelism Configuration Analysis")
    print("=" * 100)
    print(f"  TP: {tp}, PP: {pp}, EP: {ep}, CP: {cp}")
    print(f"  GPUs per Node: {gpus_per_node}")
    print(f"  Minimum GPUs Required: {gpus_required}")
    print(f"  Minimum Nodes Required: {min_nodes_required}")
    print(f"  Baseline Nodes: {baseline_nodes}")
    print(f"  Target Nodes (PROJECTION_NNODES): {num_nodes_target}")
    
    # Load hardware config if provided
    hardware_config_dict = None
    if hasattr(args, 'hardware_config') and args.hardware_config:
        hardware_config_dict = load_hardware_config(args.hardware_config)
    
    # Calculate communication times
    total_comm_time_ms, breakdown, message_info, per_layer_info = calculate_collective_communication_time(
        training_config,
        num_nodes_target,
        gpus_per_node,
        tp,
        pp,
        ep,
        cp,
        dp_target,
        hardware_config_dict,
    )
    
    # Calculate baseline DP (for minimum nodes required)
    baseline_gpus = baseline_nodes * gpus_per_node
    baseline_dp = baseline_gpus // gpus_required
    
    # Baseline time from benchmarking includes:
    # - Compute (forward + backward)
    # - TP AllReduce (if TP > 1) - this is in the critical path during compute
    # - MoE All-to-All (if EP > 1) - this is in the critical path during compute
    # But does NOT include:
    # - Gradient AllReduce across DP (because overlap_grad_reduce was disabled during benchmark)
    baseline_time_ms = single_node_time_ms
    
    # When scaling DP, we need to account for gradient all-reduce
    # Check if gradient all-reduce is overlapped
    overlap_grad_reduce = getattr(mp_config, 'overlap_grad_reduce', True)
    
    # Calculate projected time
    # 1. Scale compute time with DP
    if dp_target > baseline_dp:
        projected_compute_time_ms = baseline_time_ms * (baseline_dp / dp_target)
    else:
        projected_compute_time_ms = baseline_time_ms
    
    # 2. Add gradient all-reduce if NOT overlapped
    if not overlap_grad_reduce and dp_target > 1:
        # Calculate gradient all-reduce for target
        _, target_breakdown, _, _ = calculate_collective_communication_time(
            training_config, num_nodes_target, gpus_per_node, tp, pp, ep, cp, dp_target, hardware_config_dict
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
                training_config, num_nodes_target, gpus_per_node, tp, pp, ep, cp, dp_target, hardware_config_dict
            )
            target_grad_ar = target_breakdown.get('gradient_allreduce', 0)
            grad_ar_msg = f"{target_grad_ar:.3f} ms (overlapped - not in critical path)"
        else:
            target_grad_ar = 0
            grad_ar_msg = "N/A (DP=1)"
    
    # For reporting, get full breakdown for target
    total_comm_time_ms, breakdown, message_info, per_layer_info = calculate_collective_communication_time(
        training_config, num_nodes_target, gpus_per_node, tp, pp, ep, cp, dp_target, hardware_config_dict
    )
    
    # Calculate speedup vs baseline
    speedup = baseline_time_ms / projected_time_ms if projected_time_ms > 0 else 0
    ideal_speedup = dp_target / baseline_dp if baseline_dp > 0 else dp_target
    
    # Print results
    print("\n" + "=" * 100)
    print("Multinode Scaling Projection Results")
    print("=" * 100)
    print(f"\n📊 Parallelism: TP={tp}, PP={pp}, EP={ep}, CP={cp}")
    
    print(f"\n📦 Baseline Configuration ({baseline_nodes} node{'s' if baseline_nodes > 1 else ''}):")
    print(f"   Nodes: {baseline_nodes}")
    print(f"   Data Parallelism (DP): {baseline_dp}")
    print(f"   Iteration Time: {baseline_time_ms:.3f} ms")
    
    print(f"\n🎯 Target Configuration ({num_nodes_target} nodes):")
    print(f"   Nodes: {num_nodes_target}")
    print(f"   Data Parallelism (DP): {dp_target}")
    print(f"   DP Scaling Factor: {dp_target/baseline_dp if baseline_dp > 0 else dp_target:.1f}x")
    
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
            elif op_name == 'tp_allreduce' and 'tp_allreduce_size_mb' in message_info:
                print(f" (message: {message_info['tp_allreduce_size_mb']:.2f} MB, {message_info['num_layers']} layers × {message_info['tp_allreduce_per_layer']:.3f} ms/layer)")
            elif op_name == 'pp_p2p' and 'pp_p2p_size_mb' in message_info:
                print(f" (message: {message_info['pp_p2p_size_mb']:.2f} MB, {message_info.get('num_microbatches', 'N/A')} microbatches × {message_info.get('pp_p2p_per_microbatch', 0):.3f} ms/microbatch)")
            else:
                print()
    
    print(f"\n   Total Communication (critical path): {total_comm_time_ms:.3f} ms")
    
    print(f"\n📊 Performance Summary:")
    print(f"   Baseline:       {baseline_time_ms:.3f} ms ({baseline_nodes} node{'s' if baseline_nodes > 1 else ''}, DP={baseline_dp})")
    print(f"   Projected:      {projected_time_ms:.3f} ms ({num_nodes_target} nodes, DP={dp_target})")
    print(f"   Gradient AllReduce: {grad_ar_msg}")
    
    print("\n" + "=" * 100)


def launch_projection_from_cli(args, overrides):
    """
    Entry point for the 'performance_projection' subcommand.

    Benchmarks Megatron transformer layers and aggregates performance metrics.
    
    If PROJECTION_NNODES is set, also runs multinode scaling projection.
    If the baseline configuration requires multiple nodes, automatically reduces
    to single-node for benchmarking and estimates the baseline with PP overhead.

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
    projection_nnodes = os.getenv("PROJECTION_NNODES")
    
    # Store original parallelism before any modifications
    module_config = primus_config.get_module_config("pre_trainer")
    reduction_info = _calculate_single_node_config(
        copy.deepcopy(module_config), 
        gpus_per_node
    )
    
    if reduction_info['adjusted']:
        print("\n" + "=" * 100)
        print("[Primus:Performance Projection] Multi-node baseline detected")
        print("=" * 100)
        print(f"  Original configuration requires {reduction_info['original_nodes_required']} nodes minimum:")
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
        
        print(f"\n  Will estimate baseline by adding PP communication overhead back.")
        print("=" * 100)
        
        # Apply the reduction to the config used for benchmarking
        primus_config.get_module_config("pre_trainer").pipeline_model_parallel_size = reduction_info['benchmark_pp']
        primus_config.get_module_config("pre_trainer").expert_model_parallel_size = reduction_info['benchmark_ep']

    profiling_results = _run_layer_benchmark(primus_config, unknown_overrides)

    # Use original config for projection calculations
    training_config = convert_primus_config_to_projection_config(primus_config_original)
    pipeline_simulation_time_ms = _run_pipeline_simulation(training_config, profiling_results)
    
    # If PROJECTION_NNODES is set, also run multinode projection
    if projection_nnodes:
        print("\n" + "=" * 100)
        print("[Primus:Performance] PROJECTION_NNODES detected - running multinode scaling projection")
        print("=" * 100)
        
        # Import multinode projection functions
        from primus.core.projection.multinode_projection.projection import (
            extract_single_node_time_from_profiling,
        )
        
        # Use pipeline simulation time if available, otherwise extract from profiling
        if pipeline_simulation_time_ms is not None:
            print(f"\n[Primus:Performance Projection] Using pipeline simulation time: {pipeline_simulation_time_ms:.2f} ms")
            # Pipeline simulation already accounts for PP communication and bubbles
            # No need to add additional PP overhead
            baseline_time_ms = pipeline_simulation_time_ms
            print(f"  (Pipeline simulation already includes PP={reduction_info['original_pp']} effects)")
        else:
            print(f"\n[Primus:Performance Projection] Pipeline simulation not available, using extrapolated time from profiling")
            measured_time_ms = extract_single_node_time_from_profiling(profiling_results, training_config)
            
            # If we reduced PP for benchmarking, estimate the baseline with PP overhead
            if reduction_info['adjusted']:
                # Load hardware config if provided
                hardware_config_dict = None
                if hasattr(args, 'hardware_config') and args.hardware_config:
                    from primus.core.projection.multinode_projection.projection import load_hardware_config
                    hardware_config_dict = load_hardware_config(args.hardware_config)
                
                # Estimate PP overhead for original configuration
                pp_overhead_ms = _estimate_pp_communication_overhead(
                    training_config, 
                    reduction_info['original_pp'],
                    hardware_config_dict
                )
                
                baseline_time_ms = measured_time_ms + pp_overhead_ms
                
                print(f"\n[Primus:Performance Projection] Baseline Time Adjustment:")
                print(f"  Measured time (PP={reduction_info['benchmark_pp']}): {measured_time_ms:.2f} ms")
                print(f"  Estimated PP overhead (PP={reduction_info['original_pp']}): {pp_overhead_ms:.2f} ms")
                print(f"  Estimated baseline time: {baseline_time_ms:.2f} ms")
                print(f"  (Baseline = {reduction_info['original_nodes_required']} nodes with full parallelism config)")
            else:
                baseline_time_ms = measured_time_ms
        
        # Run multinode projection with the adjusted baseline
        _run_multinode_projection(
            training_config, 
            baseline_time_ms,
            profiling_results,
            args
        )

