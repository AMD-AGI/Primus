#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Convert HuggingFace checkpoint to Megatron format using Megatron-Bridge.

This hook runs before Megatron SFT training to prepare pretrained checkpoints.
Uses Megatron-Bridge's AutoBridge for HF → Megatron conversion.

The workflow is simple:
1. Convert HF checkpoint to Megatron torch_dist format using AutoBridge
2. Set pretrained_checkpoint path in config
3. Megatron-LM's finetune mode handles the rest automatically
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add primus to path
PRIMUS_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PRIMUS_ROOT))

from primus.core.config.primus_config import load_primus_config, get_module_config


def _is_rank_0() -> bool:
    """Check if current process is rank 0."""
    return int(os.environ.get("RANK", os.environ.get("NODE_RANK", 0))) == 0


def log_info(msg: str):
    """Log info message (only on rank 0)."""
    if _is_rank_0():
        print(f"[INFO] {msg}")


def log_error(msg: str):
    """Log error message (all ranks)."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def log_success(msg: str):
    """Log success message (only on rank 0)."""
    if _is_rank_0():
        print(f"[OK] {msg}")


def get_checkpoint_config(config_file: str) -> tuple[str | None, str | None]:
    """
    Extract hf_path and pretrained_checkpoint path from config file.

    Returns:
        tuple: (hf_path, pretrained_checkpoint)
            - hf_path: HuggingFace model path to convert
            - pretrained_checkpoint: Existing Megatron checkpoint path (if configured)
    """
    cfg = load_primus_config(Path(config_file), None)

    hf_path = None
    pretrained_checkpoint = None

    # Try different module names that might contain the paths
    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is not None:
            # Check for hf_path in params or directly in module
            if hasattr(module, "params") and hasattr(module.params, "tokenizer_model"):
                hf_path = module.params.tokenizer_model
            elif hasattr(module, "tokenizer_model"):
                hf_path = module.tokenizer_model

            # Check for pretrained_checkpoint
            if hasattr(module, "params") and hasattr(module.params, "pretrained_checkpoint"):
                pretrained_checkpoint = module.params.pretrained_checkpoint
            elif hasattr(module, "pretrained_checkpoint"):
                pretrained_checkpoint = module.pretrained_checkpoint

            break

    return hf_path, pretrained_checkpoint


def convert_checkpoint(hf_path: str, megatron_path: str):
    """
    Convert HuggingFace checkpoint to Megatron torch_dist format.
    
    Uses Megatron-Bridge's AutoBridge which supports a wide range of models:
    - LLaMA family (2, 3, 3.1, 3.2, Nemotron)
    - Qwen family (2, 2.5, 3, 3-Next, MoE)
    - Gemma family (2, 3, 3 VL)
    - DeepSeek family (V2, V3)
    - Mistral, Moonlight, Nemotron-H, GLM-4.5
    - MoE models (Qwen 3 MoE, DeepSeek V2/V3, OLMoE)
    - Vision-Language models (Gemma 3 VL, Qwen 2.5/3 VL, Nemotron Nano V2 VL)
    
    The converted checkpoint is in torch_dist format and can be directly loaded
    by Megatron-LM's finetune mode without any post-processing.
    """
    # Add Megatron-Bridge to path
    bridge_root = os.environ.get("MEGATRON_BRIDGE_PATH")
    if bridge_root:
        bridge_root = Path(bridge_root)
        log_info(f"Using MEGATRON_BRIDGE_PATH: {bridge_root}")
    else:
        bridge_root = PRIMUS_ROOT / "third_party" / "Megatron-Bridge"

    bridge_path = bridge_root / "src"
    bridge_megatron_path = bridge_root / "3rdparty" / "Megatron-LM"
    
    # Add Bridge paths to sys.path
    sys.path.insert(0, str(bridge_path))
    sys.path.insert(0, str(bridge_megatron_path))

    log_info(f"Megatron-Bridge path: {bridge_path}")
    log_info(f"Using Megatron-LM from: {bridge_megatron_path}")

    from megatron.bridge import AutoBridge

    log_info(f"Converting HF → Megatron checkpoint...")
    log_info(f"  Source: {hf_path}")
    log_info(f"  Target: {megatron_path}")

    # Convert using AutoBridge - creates torch_dist format checkpoint
    AutoBridge.import_ckpt(
        hf_model_id=hf_path,
        megatron_path=megatron_path,
        trust_remote_code=True,
    )

    log_success("Checkpoint conversion completed")


def wait_for_conversion(done_file: Path, lock_file: Path, timeout: int = 600):
    """Wait for rank 0 to complete checkpoint conversion."""
    elapsed = 0
    while not done_file.exists() and elapsed < timeout:
        if not lock_file.exists() and not done_file.exists():
            time.sleep(2)
        else:
            time.sleep(5)
        elapsed += 5

    if not done_file.exists():
        raise TimeoutError("Timeout waiting for checkpoint conversion")


def main():
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Megatron format")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args, _ = parser.parse_known_args()

    # Get config file path
    config_file = args.config
    if not os.path.isabs(config_file):
        config_file = str(PRIMUS_ROOT / config_file)

    log_info("Preparing Megatron SFT checkpoint...")

    # Extract hf_path and pretrained_checkpoint from config
    hf_path, pretrained_checkpoint = get_checkpoint_config(config_file)

    # If pretrained_checkpoint is already configured, skip conversion
    if pretrained_checkpoint:
        log_info(f"Pretrained checkpoint already configured: {pretrained_checkpoint}, skipping conversion")
        return

    # If no hf_path, nothing to convert
    if not hf_path:
        log_info("No hf_path found in config, assuming checkpoint already exists")
        return

    # Set paths
    data_path = Path(os.environ.get("DATA_PATH", PRIMUS_ROOT / "data"))
    megatron_path = data_path / "megatron_checkpoints" / Path(hf_path).name

    log_info(f"HF Model: {hf_path}")
    log_info(f"Megatron Path: {megatron_path}")

    # Check if Megatron checkpoint already exists
    if megatron_path.exists():
        log_info(f"Megatron checkpoint already exists at {megatron_path}, skipping conversion")
        print(f"extra.pretrained_checkpoint={megatron_path}")
        print(f"extra.finetune=true")
        return

    # Convert checkpoint (only on rank 0, others wait)
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", 0)))
    lock_file = Path(f"{megatron_path}.converting.lock")
    done_file = Path(f"{megatron_path}.done")

    if node_rank == 0:
        # Rank 0: perform the conversion
        log_info("Converting HF checkpoint to Megatron format using Megatron-Bridge...")
        megatron_path.parent.mkdir(parents=True, exist_ok=True)

        # Create lock file
        lock_file.touch()

        try:
            convert_checkpoint(hf_path, str(megatron_path))
            
            # Fix metadata and directory structure for converted checkpoints
            # Megatron-Bridge creates iter_0000000 directory with iteration=0 metadata
            # But Megatron-LM requires:
            #   - iteration > 0 OR metadata = "release"
            #   - If metadata = "release", checkpoint must be in "release/" directory
            
            metadata_file = megatron_path / "latest_checkpointed_iteration.txt"
            iter_dir = megatron_path / "iter_0000000"
            release_dir = megatron_path / "release"
            
            if metadata_file.exists() and iter_dir.exists():
                with open(metadata_file, 'r') as f:
                    content = f.read().strip()
                
                if content == "0":
                    log_info("Fixing HuggingFace converted checkpoint structure:")
                    
                    # Step 1: Update metadata file
                    log_info("  1. Changing metadata from '0' to 'release'")
                    with open(metadata_file, 'w') as f:
                        f.write("release")
                    
                    # Step 2: Rename directory to match
                    if not release_dir.exists():
                        log_info("  2. Renaming 'iter_0000000' -> 'release'")
                        iter_dir.rename(release_dir)
                    
                    log_success("Checkpoint structure fixed for Megatron-LM compatibility")
            
            done_file.touch()
            log_success(f"Checkpoint prepared at {megatron_path}")
        finally:
            lock_file.unlink(missing_ok=True)
    else:
        # Other ranks: wait for rank 0 to complete
        log_info(f"[RANK {node_rank}] Waiting for rank 0 to complete checkpoint conversion...")
        wait_for_conversion(done_file, lock_file)
        log_success(f"[RANK {node_rank}] Checkpoint ready at {megatron_path}")

    # Output the checkpoint path for the main training process
    # Use pretrained_checkpoint + finetune=true (same as Megatron-Bridge workflow)
    print(f"extra.pretrained_checkpoint={megatron_path}")
    print(f"extra.finetune=true")


if __name__ == "__main__":
    main()
