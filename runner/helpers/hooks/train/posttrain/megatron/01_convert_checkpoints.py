#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Convert HuggingFace checkpoints for native Megatron SFT using Megatron-Bridge.

This hook runs before Megatron SFT training to prepare pretrained checkpoints.
It calls ``AutoBridge.import_ckpt()`` directly and then normalizes the produced
checkpoint layout so native Megatron-LM finetune loading can consume it.
"""

import os
import sys
import time
import argparse
from contextlib import contextmanager
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


def _first_config_value(obj, *names: str) -> str | None:
    """Return the first non-empty attribute value found on an object."""
    if obj is None:
        return None

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value not in (None, ""):
                return value
    return None


def get_checkpoint_config(config_file: str) -> tuple[str | None, str | None]:
    """
    Extract HF source path and any already-configured checkpoint path.

    Returns:
        tuple: (hf_path, checkpoint_path)
            - hf_path: HuggingFace model path to convert
            - checkpoint_path: Existing Megatron checkpoint path (if configured)
    """
    cfg = load_primus_config(Path(config_file), None)

    hf_path = None
    checkpoint_path = None

    # Try different module names that might contain the paths
    for module_name in ["sft_trainer", "post_trainer"]:
        module = get_module_config(cfg, module_name)
        if module is not None:
            params = getattr(module, "params", None)

            # Old Megatron-Bridge SFT configs used hf_path, while native Megatron
            # configs typically reuse tokenizer_model as the HF source identifier.
            hf_path = _first_config_value(params, "hf_path", "tokenizer_model") or _first_config_value(
                module, "hf_path", "tokenizer_model"
            )

            # If the user already configured a Megatron-format checkpoint via either
            # pretrained_checkpoint or load, the conversion hook should be skipped.
            checkpoint_path = _first_config_value(
                params, "pretrained_checkpoint", "load"
            ) or _first_config_value(module, "pretrained_checkpoint", "load")

            break

    return hf_path, checkpoint_path


def _resolve_bridge_paths() -> tuple[Path, Path]:
    """Resolve Megatron-Bridge source paths for direct AutoBridge import."""
    bridge_root = os.environ.get("MEGATRON_BRIDGE_PATH")
    if bridge_root:
        bridge_root = Path(bridge_root)
        log_info(f"Using MEGATRON_BRIDGE_PATH: {bridge_root}")
    else:
        bridge_root = PRIMUS_ROOT / "third_party" / "Megatron-Bridge"

    bridge_path = bridge_root / "src"
    bridge_megatron_path = bridge_root / "3rdparty" / "Megatron-LM"
    return bridge_path, bridge_megatron_path


@contextmanager
def _prepend_sys_path(*paths: Path):
    """Temporarily prepend import paths needed by Megatron-Bridge."""
    original_sys_path = list(sys.path)
    for path in reversed([str(path) for path in paths if path]):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original_sys_path


def convert_checkpoint(hf_path: str, megatron_path: str):
    """
    Convert HuggingFace checkpoint to Megatron torch_dist format.
    """
    bridge_path, bridge_megatron_path = _resolve_bridge_paths()
    log_info(f"Megatron-Bridge path: {bridge_path}")
    log_info(f"Using Megatron-LM from: {bridge_megatron_path}")

    log_info(f"Converting HF → Megatron checkpoint...")
    log_info(f"  Source: {hf_path}")
    log_info(f"  Target: {megatron_path}")

    with _prepend_sys_path(bridge_path, bridge_megatron_path):
        from megatron.bridge import AutoBridge

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


def fix_common_pt_for_megatron_lm(checkpoint_dir: Path):
    """
    Fix common.pt to include 'args' for Megatron-LM compatibility.
    
    Megatron-LM expects 'args' in common.pt's state_dict for loading torch_dist
    checkpoints. HuggingFace converted checkpoints are always TP=1, PP=1.
    """
    import torch
    from types import SimpleNamespace
    
    common_pt = checkpoint_dir / "common.pt"
    
    log_info(f"  3. Adding 'args' to common.pt for Megatron-LM compatibility")
    
    # Load existing common.pt
    state_dict = torch.load(common_pt, map_location='cpu')
    
    # Check if args already exists
    if 'args' in state_dict:
        log_info("     'args' already exists in common.pt, skipping")
        return
    
    # Create args namespace with default values for HuggingFace converted checkpoints
    # HF models are single-device, so TP=1, PP=1
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=1,
        data_parallel_size=1,
        no_save_rng=True,
        no_save_optim=True,
        ckpt_fully_parallel_save=False,
    )
    
    # Add args to state_dict and save
    state_dict['args'] = args
    torch.save(state_dict, common_pt)
    log_success("     Successfully added 'args' to common.pt")


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
            
            # Step 3: Add 'args' to common.pt for Megatron-LM compatibility
            # Megatron-Bridge saves config to run_config.yaml, but Megatron-LM expects 'args' in common.pt
            fix_common_pt_for_megatron_lm(release_dir if release_dir.exists() else iter_dir)
            
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
