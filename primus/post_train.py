###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Post-training entry point for Primus CLI.

This module handles post-training workflows (fine-tuning, alignment, etc.)
by routing to the appropriate backend based on the YAML configuration.

Currently supported backends:
- megatron-bridge: For post-training with Megatron-LM backend
"""

import argparse
import os
import sys
from pathlib import Path

from primus.core.utils import yaml_utils


def setup_megatron_bridge_path(backend_path=None, verbose: bool = True):
    """
    Setup Python path for Megatron-Bridge backend.

    Priority order:
    1. --backend-path from CLI
    2. MEGATRON_BRIDGE_PATH from environment
    3. Source tree fallback: <primus>/../../third_party/Megatron-Bridge

    Returns:
        str: The first valid backend path inserted into sys.path.
    """
    candidate_paths = []

    # 1) From CLI
    if backend_path:
        if isinstance(backend_path, str):
            backend_path = [backend_path]
        candidate_paths.extend(backend_path)

    # 2) From environment variable
    env_path = os.getenv("MEGATRON_BRIDGE_PATH")
    if env_path:
        candidate_paths.append(env_path)

    # 3) Fallback to source tree under third_party
    default_path = Path(__file__).resolve().parent.parent / "third_party" / "megatron-bridge"
    candidate_paths.append(default_path)

    # Normalize & deduplicate
    candidate_paths = list(dict.fromkeys(os.path.normpath(os.path.abspath(p)) for p in candidate_paths))

    # Insert the first existing path into sys.path
    for path in candidate_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
                if verbose:
                    print(f"[Primus] sys.path.insert: {path}")
            return path  # Return the first valid path

    # None of the candidate paths exist
    raise FileNotFoundError(
        f"[Primus] Megatron-Bridge path not found. " f"Tried paths: {candidate_paths}"
    )


def extract_megatron_bridge_config(yaml_config_path: str) -> dict:
    """
    Extract Megatron-Bridge specific configuration from the YAML file.

    Expected YAML structure:
    ```yaml
    post_train:
      backend: megatron-bridge
      recipe_script: examples/recipes/qwen/finetune_qwen3_next_80b_a3b.py
      pretrained_checkpoint: /path/to/megatron_ckpt
      config_file: conf/qwen3_next_80b_a3b_pretrain_override_example.yaml
      nproc_per_node: 8
    ```

    Returns:
        dict: Configuration dictionary with keys:
            - backend: Backend framework name
            - recipe_script: Path to the recipe/finetune script
            - pretrained_checkpoint: Path to pretrained checkpoint
            - config_file: Path to backend-specific config file
            - nproc_per_node: Number of processes per node
    """
    config = yaml_utils.parse_yaml_to_namespace(yaml_config_path)

    # Extract post_train configuration
    if not hasattr(config, "post_train"):
        raise ValueError(
            f"[Primus] YAML config '{yaml_config_path}' must contain 'post_train' section "
            "with backend-specific configuration."
        )

    post_train_cfg = config.post_train

    # Validate required fields
    required_fields = ["backend", "recipe_script"]
    for field in required_fields:
        if not hasattr(post_train_cfg, field):
            raise ValueError(
                f"[Primus] post_train config missing required field: '{field}'"
            )

    # Convert namespace to dict for easier handling
    cfg_dict = {
        "backend": post_train_cfg.backend,
        "recipe_script": post_train_cfg.recipe_script,
        "pretrained_checkpoint": getattr(post_train_cfg, "pretrained_checkpoint", None),
        "config_file": getattr(post_train_cfg, "config_file", None),
        "nproc_per_node": getattr(post_train_cfg, "nproc_per_node", 8),
    }

    return cfg_dict


def launch_megatron_bridge_posttrain(yaml_config_path: str, backend_path=None):
    """
    Launch Megatron-Bridge post-training directly.

    This function extracts the configuration from the YAML file and executes
    the Megatron-Bridge recipe script using torchrun, without modifying Primus
    core logic.

    Args:
        yaml_config_path: Path to the Primus YAML config file
        backend_path: Optional path to Megatron-Bridge installation
    """
    # Check if we're already running under torchrun (via primus-cli direct)
    # If so, only execute on rank 0 to avoid nested torchrun calls
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))

    if rank > 0 or local_rank > 0:
        # We're in a distributed context but not rank 0, exit silently
        print(f"[Primus] Rank {rank}/LocalRank {local_rank}: Skipping post_train execution (only rank 0 executes)")
        sys.exit(0)

    if rank == 0 or local_rank == 0:
        # We're already in torchrun context, just notify user
        print(f"[Primus] Detected torchrun context (RANK={rank}, LOCAL_RANK={local_rank})")
        print(f"[Primus] Will execute Megatron-Bridge recipe from rank 0 only")

    # Extract configuration from YAML
    bridge_cfg = extract_megatron_bridge_config(yaml_config_path)

    # Setup Megatron-Bridge path
    bridge_path = setup_megatron_bridge_path(backend_path=backend_path, verbose=True)

    # Build the torchrun command
    nproc = bridge_cfg["nproc_per_node"]
    recipe_script = bridge_cfg["recipe_script"]

    # Resolve recipe script path relative to Megatron-Bridge
    recipe_full_path = os.path.join(bridge_path, recipe_script)
    if not os.path.exists(recipe_full_path):
        raise FileNotFoundError(
            f"[Primus] Recipe script not found: {recipe_full_path}\n"
            f"Expected at: {bridge_path}/{recipe_script}"
        )

    # Build command arguments
    # Note: First arg must be the executable name for argv[0]
    cmd_args = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        recipe_full_path,
    ]

    # Add optional arguments
    if bridge_cfg["pretrained_checkpoint"]:
        cmd_args.extend(["--pretrained-checkpoint", bridge_cfg["pretrained_checkpoint"]])

    if bridge_cfg["config_file"]:
        # Resolve config file path relative to Megatron-Bridge
        config_full_path = os.path.join(bridge_path, bridge_cfg["config_file"])
        cmd_args.extend(["--config-file", config_full_path])

    # Print the command for transparency
    print("\n" + "="*80)
    print("[Primus] Launching Megatron-Bridge post-training")
    print("="*80)
    print(f"Backend path: {bridge_path}")
    print(f"Recipe script: {recipe_script}")
    print(f"Command: {' '.join(cmd_args)}")
    print("="*80 + "\n")

    # Execute using subprocess instead of execvp to avoid process replacement issues
    import subprocess
    result = subprocess.run(cmd_args, env=os.environ.copy())
    sys.exit(result.returncode)


def launch_posttrain_from_cli(args, overrides):
    """
    Entry point for post-training from CLI.

    Steps:
        1. Load the experiment YAML config
        2. Determine the backend from config
        3. Route to appropriate backend launcher

    Args:
        args: Parsed command-line arguments
        overrides: List of override arguments
    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:PostTrain] Config file '{cfg_path}' not found.")

    # Extract backend from config
    bridge_cfg = extract_megatron_bridge_config(str(cfg_path))
    backend = bridge_cfg["backend"]

    if backend == "megatron-bridge":
        launch_megatron_bridge_posttrain(
            yaml_config_path=str(cfg_path),
            backend_path=args.backend_path
        )
    else:
        raise NotImplementedError(
            f"[Primus] Post-training backend '{backend}' not supported. "
            "Currently supported: megatron-bridge"
        )


if __name__ == "__main__":
    from primus.core.launcher.parser import add_pretrain_parser

    parser = argparse.ArgumentParser(description="post_train")
    add_pretrain_parser(parser)

    args, unknown_args = parser.parse_known_args()

    launch_posttrain_from_cli(args, unknown_args)
