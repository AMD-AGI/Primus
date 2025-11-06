import argparse
import os
import sys
from pathlib import Path
 
from primus.core.launcher.parser import PrimusParser
from primus.core.projection.training_config import convert_primus_config_to_projection_config
from primus.core.projection.module_profilers.language_model import build_profiler, get_language_model_profiler_spec


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
        if hasattr(profiler, 'sub_profilers') and profiler.sub_profilers:
            for sub_name, sub_profiler in profiler.sub_profilers.items():
                if sub_profiler is not None:
                    print()  # Add spacing between components
                    print_profiler_hierarchy(sub_profiler, batch_size, seq_len, rank, sub_name, depth + 1, visited)
    except Exception as e:
        print(f"{indent}[{name}] - Error calculating metrics: {e}")


def launch_projection_from_cli(args, overrides):
    """
    Entry point for the 'projection' subcommand.

    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Projection] Config file '{cfg_path}' not found.")

    config_parser = PrimusParser()
    primus_config = config_parser.parse(args)
    training_config = convert_primus_config_to_projection_config(primus_config)
    print(training_config)

    model_profiler_spec = get_language_model_profiler_spec(training_config)
    model_profiler = build_profiler(model_profiler_spec)

    seq_len = training_config.runtime_config.sequence_length
    batch_size = training_config.runtime_config.micro_batch_size
    rank = int(os.getenv('RANK', '0'))
    
    # Print recursive profiler hierarchy with detailed breakdown
    print("\n" + "=" * 100)
    print(f"[Primus:Projection] Component-wise Profiling Results (Rank {rank}):")
    print("=" * 100)
    print()
    
    # Print the complete hierarchy recursively
    print_profiler_hierarchy(model_profiler, batch_size, seq_len, rank=rank, name="LanguageModelProfiler", depth=0)
    
    # Get overall totals from the model profiler for this rank
    num_params = model_profiler.estimated_num_params(rank=rank)
    activation_memory = model_profiler.estimated_activation_memory(batch_size, seq_len)
    num_bytes_per_param = model_profiler.get_num_bytes_per_param()
    print()
    print("=" * 100)
    print(f"[Primus:Projection] Memory Projection Summary on Rank {rank}:")
    print(f"  Params: {num_params / 1e9:.6f} Billion ({num_params:,})")
    print(f"  Param+Optimizer Memory: {num_params * num_bytes_per_param / 1024 / 1024 / 1024:.4f} GB")
    print(f"  Activation Memory (per batch size {batch_size}, seq len {seq_len}): "
          f"{activation_memory / 1024 / 1024 / 1024:.4f} GB")
    print(f"  Projected Total Memory: "
          f"{(num_params * num_bytes_per_param + activation_memory) / 1024 / 1024 / 1024:.4f} GB")
    print("=" * 100)