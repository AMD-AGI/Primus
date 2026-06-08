###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Memory projection — *simulate* mode.

This is the original analytical memory projection: it walks the model
profiler tree and computes per-rank parameter / activation / optimizer
memory purely from the training config, with no GPU benchmark.  It is
kept as the default mode for back-compat (``primus projection memory``
without ``--memory-mode`` invokes this path).

For OOM-accurate projection that anchors on a measured bench peak, see
:mod:`primus.core.projection.memory_projection.benchmark`.
"""

import os
from pathlib import Path
from typing import Optional

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.config_validation import assert_recompute_pipeline_compat
from primus.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.training_config import (
    convert_primus_config_to_projection_config,
)


def print_profiler_hierarchy(profiler, batch_size, seq_len, rank=None, name="root", depth=0, visited=None):
    """
    Recursively print the profiler hierarchy with num_params and activation_memory for each component.

    Args:
        profiler: The profiler instance to print
        batch_size: Batch size for activation memory calculation
        seq_len: Sequence length for activation memory calculation
        rank: Rank for parameter calculation (if None, calculates total parameters)
        name: Name of the current profiler component
        depth: Current depth in the hierarchy (for indentation)
        visited: Set of visited profiler IDs to avoid infinite recursion
    """
    if visited is None:
        visited = set()

    # Avoid infinite recursion if profilers reference each other
    profiler_id = id(profiler)
    if profiler_id in visited:
        return
    visited.add(profiler_id)

    indent = "  " * depth

    # Calculate metrics for this profiler
    try:
        if depth == 0:
            # Only output the total number of parameters for the entire model for depth 0.
            num_params = profiler.estimated_num_params(rank=None)
            print(f"{indent}  Total Number of Parameters: {num_params / 1e9:.6f} Billion ({num_params:,})")
        else:
            num_params = profiler.estimated_num_params(rank=rank)
            activation_mem = profiler.estimated_activation_memory(batch_size, seq_len)
            print(f"{indent}[{name}]")
            print(f"{indent}  Params: {num_params / 1e9:.6f} Billion ({num_params:,})")
            print(f"{indent}  Activation Memory: {activation_mem / 1024 / 1024 / 1024:.4f} GB")

        # Recursively process sub_profilers if they exist
        if hasattr(profiler, "sub_profilers") and profiler.sub_profilers:
            for sub_name, sub_profiler in profiler.sub_profilers.items():
                if sub_profiler is not None:
                    print()  # Add spacing between components
                    print_profiler_hierarchy(
                        sub_profiler,
                        batch_size,
                        seq_len,
                        rank,
                        sub_name,
                        depth + 1,
                        visited,
                    )
    except Exception as e:
        print(f"{indent}[{name}] - Error calculating metrics: {e}")


def project_from_config(
    primus_config,
    *,
    rank: Optional[int] = None,
    verbose: bool = True,
    pipeline_schedule_algorithm: Optional[str] = None,
):
    """Run analytical-only memory projection against a loaded primus_config.

    Returns a dict with the keyed totals so the ``both`` dispatcher (and
    callers wanting comparison) can grab structured numbers.  When
    ``verbose=True`` the historical hierarchy + summary print-out is
    emitted.
    """
    training_config = convert_primus_config_to_projection_config(primus_config)
    assert_recompute_pipeline_compat(
        training_config,
        primus_config=primus_config,
        pipeline_schedule_algorithm=pipeline_schedule_algorithm,
    )
    model_profiler_spec = get_language_model_profiler_spec(training_config)
    model_profiler = build_profiler(model_profiler_spec)

    seq_len = training_config.runtime_config.sequence_length
    batch_size = training_config.runtime_config.micro_batch_size
    eff_rank = int(os.getenv("RANK", "0")) if rank is None else int(rank)

    if verbose:
        print("\n" + "=" * 100)
        print(f"[Primus:Projection] Component-wise Profiling Results (Rank {eff_rank}):")
        print("=" * 100)
        print("")
        print_profiler_hierarchy(
            model_profiler,
            batch_size,
            seq_len,
            rank=eff_rank,
            name="LanguageModelProfiler",
            depth=0,
        )

    num_params = model_profiler.estimated_num_params(rank=eff_rank)
    activation_memory = model_profiler.estimated_activation_memory(batch_size, seq_len)
    num_bytes_per_param = model_profiler.get_num_bytes_per_param()
    param_optimizer_bytes = int(num_params * num_bytes_per_param)
    total_bytes = int(param_optimizer_bytes + activation_memory)

    if verbose:
        print("")
        print("=" * 100)
        print(f"[Primus:Projection] Memory Projection Summary on Rank {eff_rank}:")
        print(f"  Params: {num_params / 1e9:.6f} Billion ({num_params:,})")
        print(f"  Param+Optimizer Memory: " f"{param_optimizer_bytes / 1024 / 1024 / 1024:.4f} GB")
        print(
            f"  Activation Memory (per batch size {batch_size}, seq len {seq_len}): "
            f"{activation_memory / 1024 / 1024 / 1024:.4f} GB"
        )
        print(f"  Projected Total Memory: " f"{total_bytes / 1024 / 1024 / 1024:.4f} GB")
        print("=" * 100)

    return {
        "rank": eff_rank,
        "num_params": int(num_params),
        "num_bytes_per_param": float(num_bytes_per_param),
        "param_optimizer_bytes": param_optimizer_bytes,
        "activation_bytes": int(activation_memory),
        "total_bytes": total_bytes,
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
    }


def launch_projection_from_cli(args, overrides):
    """Entry point for ``projection memory --memory-mode simulate``."""
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Projection] Config file '{cfg_path}' not found.")

    primus_config, _unknown_overrides = load_primus_config(args, overrides or [])
    return project_from_config(
        primus_config,
        verbose=True,
        pipeline_schedule_algorithm=getattr(args, "pipeline_schedule_algorithm", None),
    )
