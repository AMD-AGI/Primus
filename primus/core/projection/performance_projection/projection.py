###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from pathlib import Path

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.training_config import (
    convert_primus_config_to_projection_config,
)
from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer


def launch_projection_from_cli(args, overrides):
    """
    Entry point for the 'performance_projection' subcommand.

    Benchmarks Megatron transformer layers and aggregates performance metrics.

    Args:
        args: Command-line arguments
        overrides: Configuration overrides
    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Performance Projection] Config file '{cfg_path}' not found.")

    # Load Primus configuration
    primus_config, unknown_overrides = load_primus_config(args, overrides)
    training_config = convert_primus_config_to_projection_config(primus_config)

    print("\n" + "=" * 100)
    print("[Primus:Performance Projection] Configuration:")
    print("=" * 100)
    print(training_config)
    print("=" * 100)

    # Get distributed environment variables
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Initialize MegatronPretrainTrainer
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

    # Initialize Megatron
    print("[Primus:Performance Projection] Initializing Megatron...")
    trainer.init()

    # Setup model and optimizer
    print("[Primus:Performance Projection] Setting up model and optimizer...")
    trainer.setup()

    print(f"\n[Primus:Performance Projection] Model setup complete:")
    print(f"  Model type: {type(trainer.model)}")
    print(f"  Number of model chunks: {len(trainer.model) if isinstance(trainer.model, list) else 1}")

    # Build the model profiler for comparison
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
    print(f"  Layers on this rank: {len(model_profiler.layers)}")

    # Run layer benchmarking
    print("\n" + "=" * 100)
    print("[Primus:Performance Projection] Starting layer benchmarking...")
    print("=" * 100)

    model_profiler.run_layer_benchmark(
        model=trainer.model,
        batch_size=batch_size,
        seq_len=seq_len,
    )
