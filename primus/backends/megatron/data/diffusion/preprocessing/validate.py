# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Energon dataset validation for Primus diffusion pipelines.

Validates that a prepared Energon WebDataset is structurally correct and
loadable by the training code. Replaces the broken ``energon info`` CLI
tool which does not support CrudeWebdataset format.

Standalone usage:
    python -m primus.backends.megatron.data.diffusion.preprocessing.validate /path/to/dataset

Programmatic usage:
    from primus.backends.megatron.data.diffusion.preprocessing.validate import (
        validate_energon_dataset,
    )
    validate_energon_dataset('/path/to/dataset', encoding='preencoded')
"""

import json
import logging
import tarfile
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
import yaml

logger = logging.getLogger(__name__)

EXPECTED_KEYS_PREENCODED = {"latents.pth", "prompt_embeds.pth", "pooled_prompt_embeds.pth"}
EXPECTED_KEYS_PREENCODED_NUMPY = {"t5.bytes", "clip.bytes", "mean.bytes", "logvar.bytes"}
EXPECTED_KEYS_RAW_IMAGE = {"jpg", "jpeg", "png", "webp"}


def _check_metadata(output_path: Path) -> Optional[Dict]:
    """Check that .nv-meta files exist and are valid. Returns info dict or None."""
    meta_dir = output_path / ".nv-meta"

    info_path = meta_dir / ".info.json"
    if not info_path.exists():
        info_path_yaml = meta_dir / ".info.yaml"
        if info_path_yaml.exists():
            info_path = info_path_yaml
        else:
            logger.error(f"Missing metadata: neither .info.json nor .info.yaml in {meta_dir}")
            return None

    try:
        with open(info_path) as f:
            if info_path.suffix == ".json":
                info = json.load(f)
            else:
                info = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse {info_path}: {e}")
        return None

    if "shard_counts" not in info:
        logger.error(f"Missing 'shard_counts' in {info_path}")
        return None

    split_path = meta_dir / "split.yaml"
    if not split_path.exists():
        split_path = meta_dir / "split.json"
    if not split_path.exists():
        logger.error(f"Missing split config in {meta_dir}")
        return None

    try:
        with open(split_path) as f:
            splits = yaml.safe_load(f) if split_path.suffix == ".yaml" else json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse {split_path}: {e}")
        return None

    train_parts = splits.get("split_parts", {}).get("train", [])
    if not train_parts:
        logger.warning("Train split is empty")

    dataset_yaml = meta_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.error(f"Missing {dataset_yaml}")
        return None

    missing_files: List[str] = []
    for shard_name in info["shard_counts"]:
        if not (output_path / shard_name).exists():
            missing_files.append(shard_name)
        idx_name = shard_name + ".idx"
        if not (output_path / idx_name).exists():
            missing_files.append(idx_name)

    if missing_files:
        logger.error(f"Missing files on disk: {missing_files[:10]}")
        return None

    info["_splits"] = splits
    return info


def _check_sample_counts(output_path: Path, info: Dict) -> bool:
    """Spot-check that the first shard's entry count matches .info.json."""
    shard_counts = info["shard_counts"]
    first_shard = next(iter(shard_counts))
    expected = shard_counts[first_shard]

    tar_path = output_path / first_shard
    try:
        with tarfile.open(str(tar_path), "r") as tar:
            members = tar.getmembers()
            sample_keys = set()
            for m in members:
                key = m.name.split(".", 1)[0]
                sample_keys.add(key)
            actual = len(sample_keys)
    except Exception as e:
        logger.error(f"Failed to read {tar_path}: {e}")
        return False

    if actual != expected:
        logger.error(
            f"Sample count mismatch in {first_shard}: " f".info.json says {expected}, tar contains {actual}"
        )
        return False

    return True


def _check_sample_load(
    output_path: Path,
    encoding: str,
) -> Optional[Dict[str, str]]:
    """
    Load one sample through energon's Python API (same path as training).
    Returns a dict of {key: description} for the sample, or None on failure.
    """
    try:
        from megatron.energon import StandardWebdatasetFactory
        from megatron.energon.dataset_config import load_config
        from megatron.energon.epathlib import EPath
        from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
        from megatron.energon.loader import get_loader
        from megatron.energon.worker import WorkerConfig
    except ImportError as e:
        logger.warning(f"Cannot import megatron.energon for sample validation: {e}")
        return None

    try:
        ds_path = EPath(str(output_path))
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        dataset = load_config(
            ds_path / MAIN_FOLDER_NAME / "dataset.yaml",
            default_kwargs=dict(
                path=ds_path,
                split_part="train",
                training=False,
                worker_config=worker_config,
            ),
            default_type=StandardWebdatasetFactory,
        )
        sample = next(iter(get_loader(dataset.build())))
    except Exception as e:
        logger.error(f"Failed to load sample via energon API: {e}")
        return None

    data_keys = [k for k in sample.keys() if not k.startswith("_")]

    if encoding == "preencoded":
        missing = EXPECTED_KEYS_PREENCODED - set(data_keys)
        if missing:
            logger.error(f"Sample missing expected keys for preencoded data: {missing}")
            return None
    elif encoding == "preencoded_numpy":
        missing = EXPECTED_KEYS_PREENCODED_NUMPY - set(data_keys)
        if missing:
            logger.error(f"Sample missing expected keys for preencoded_numpy data: {missing}")
            return None
    elif encoding == "raw":
        has_image = any(k in EXPECTED_KEYS_RAW_IMAGE for k in data_keys)
        has_text = "txt" in data_keys
        if not has_image or not has_text:
            logger.error(f"Raw sample missing image or text key. Got: {data_keys}")
            return None

    descriptions = {}
    for key in sorted(data_keys):
        val = sample[key]
        if isinstance(val, torch.Tensor):
            descriptions[key] = f"shape={tuple(val.shape)}, dtype={val.dtype}"
        elif isinstance(val, bytes):
            descriptions[key] = f"bytes, len={len(val)}"
        else:
            descriptions[key] = f"{type(val).__name__}"

    return descriptions


def validate_energon_dataset(
    output_dir: str,
    encoding: Literal["preencoded", "preencoded_numpy", "raw"] = "preencoded",
) -> bool:
    """
    Validate a prepared Energon WebDataset.

    Performs four checks:
      1. Metadata files (.info.json, split.yaml, dataset.yaml) are present and valid
      2. Sample count in .info.json matches actual tar contents (spot-check)
      3. One sample loads successfully through energon's Python API
      4. Prints a structured summary of the dataset

    Args:
        output_dir: Path to the dataset directory
        encoding: Expected encoding ('preencoded', 'preencoded_numpy', or 'raw')

    Returns:
        True if all checks pass, False otherwise.
    """
    output_path = Path(output_dir)
    passed = True

    logger.info("Validating dataset...")

    # --- Check 1: metadata ---
    info = _check_metadata(output_path)
    if info is None:
        logger.error("  Metadata check FAILED")
        return False
    logger.info("  Metadata check passed")

    # --- Check 2: sample counts ---
    if not _check_sample_counts(output_path, info):
        logger.error("  Sample count check FAILED")
        passed = False
    else:
        logger.info("  Sample count check passed")

    # --- Check 3: load one sample via energon API ---
    descriptions = _check_sample_load(output_path, encoding)
    if descriptions is None:
        logger.error("  Sample load check FAILED")
        passed = False
    else:
        logger.info("  Sample load check passed")

    # --- Check 4: summary ---
    shard_counts = info["shard_counts"]
    total_samples = sum(shard_counts.values())
    num_shards = len(shard_counts)

    splits = info.get("_splits", {}).get("split_parts", {})
    split_summary = ", ".join(
        (
            f"{name}={len(shards)}"
            if isinstance(shards, list) and not any("{" in s for s in shards)
            else f"{name}={'non-empty' if shards else 'empty'}"
        )
        for name, shards in splits.items()
    )

    total_bytes = sum(f.stat().st_size for f in output_path.glob("**/*.tar"))
    size_gb = total_bytes / (1024**3)

    logger.info("  " + "-" * 40)
    logger.info(f"  Encoding:       {encoding}")
    logger.info(f"  Total samples:  {total_samples} across {num_shards} shard(s)")
    logger.info(f"  Splits:         {split_summary}")
    if descriptions:
        logger.info("  Spot check:")
        for key, desc in descriptions.items():
            logger.info(f"    {key}: {desc}")
    logger.info(f"  Dataset size:   {size_gb:.2f} GB")

    if passed:
        logger.info("✓ Dataset verified and ready for training")
    else:
        logger.warning("Dataset has issues — see errors above")

    return passed


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Validate a Primus Energon WebDataset",
    )
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument(
        "--encoding",
        choices=["preencoded", "preencoded_numpy", "raw"],
        default="preencoded",
        help="Expected encoding type (default: preencoded)",
    )
    args = parser.parse_args()

    ok = validate_energon_dataset(args.dataset_path, encoding=args.encoding)
    sys.exit(0 if ok else 1)
