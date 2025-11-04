import argparse
import os
import sys
from pathlib import Path
 
from primus.core.launcher.parser import PrimusParser
from primus.core.projection.training_config import convert_primus_config_to_projection_config
from primus.core.projection.module_profilers.language_model import build_profiler, get_language_model_profiler_spec


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
    num_params = model_profiler.estimated_num_params()
    activation_memory = model_profiler.estimated_activation_memory(batch_size, seq_len)

    print("\n[Primus:Projection] Model Profiling Results:")
    print(f"  Estimated Number of Parameters: {num_params / 1e9} Billion")
    print(f"  Estimated Parameter Memory: {num_params * 2 / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Estimated Activation Memory (per batch size {batch_size}, seq len {seq_len}): "
          f"{activation_memory / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Estimated Total Memory: "
          f"{(num_params * 2 + activation_memory) / 1024 / 1024 / 1024:.2f} GB")