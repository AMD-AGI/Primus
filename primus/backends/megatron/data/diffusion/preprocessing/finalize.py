# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Automatic Energon dataset finalization for Primus diffusion pipelines.

This module automates the post-processing steps required after data preprocessing:
1. Create .nv-meta/dataset.yaml with correct sample type configuration
2. Run ``energon prepare`` non-interactively to index the dataset
3. Verify the dataset is ready for training

Supports two indexing strategies:
- **Ratio-based** (default): splits shards by train/val/test ratio.
  Works with both the programmatic ``BaseWebdatasetFactory`` API and
  the ``energon prepare`` subprocess fallback.
- **Pattern-based**: assigns shards to splits by regex patterns on
  relative paths (e.g. ``"train/.*"``).  Requires the programmatic
  API (``megatron-energon >= 7.x``).

Usage:
    from primus.backends.megatron.data.diffusion.preprocessing.finalize import finalize_energon_dataset

    # Ratio-based (original behavior)
    finalize_energon_dataset(
        output_dir='/workspace/Primus/data/encoded_pokemon',
        train_split=1.0,
        encoding='preencoded',
    )

    # Pattern-based (MLPerf layout with separate train/ and val/ dirs)
    finalize_energon_dataset(
        output_dir='/workspace/Primus/data/mlperf_flux1',
        encoding='preencoded_numpy',
        split_parts_patterns=[("train", "train/.*"), ("val", "val/.*")],
    )
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

_ENCODING_TYPE = Literal["preencoded", "preencoded_numpy", "raw"]


def _write_dataset_yaml(output_path: Path, encoding: str) -> Path:
    """Write .nv-meta/dataset.yaml with the correct subflavor."""
    meta_dir = output_path / ".nv-meta"
    meta_dir.mkdir(exist_ok=True)

    dataset_yaml_path = meta_dir / "dataset.yaml"
    dataset_yaml_path.write_text(
        f"__module__: megatron.energon\n"
        f"__class__: CrudeWebdataset\n"
        f"subflavors:\n"
        f"  encoding: {encoding}\n"
    )
    logger.info(f"✓ Created {dataset_yaml_path}")
    return dataset_yaml_path


def _prepare_programmatic(
    output_path: Path,
    num_workers: int,
    train_split: float,
    split_parts_patterns: Optional[List[Tuple[str, str]]],
) -> None:
    """Index dataset using BaseWebdatasetFactory.prepare_dataset()."""
    from megatron.energon import BaseWebdatasetFactory

    shard_paths = sorted(str(p.relative_to(output_path)) for p in output_path.glob("**/*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No .tar shards found under {output_path}")

    kwargs: dict = {
        "parent_path": str(output_path),
        "paths": shard_paths,
        "workers": num_workers,
    }

    if split_parts_patterns:
        kwargs["split_parts_patterns"] = split_parts_patterns
    else:
        val_split = (1.0 - train_split) / 2
        test_split = (1.0 - train_split) / 2
        kwargs["split_parts_ratio"] = [
            ("train", train_split),
            ("val", val_split),
            ("test", test_split),
        ]

    logger.info(f"  Indexing with BaseWebdatasetFactory ({len(shard_paths)} shards)")
    BaseWebdatasetFactory.prepare_dataset(**kwargs)
    logger.info("✓ Energon indexing complete (programmatic API)")


def _prepare_subprocess(
    output_path: Path,
    num_workers: int,
    train_split: float,
    split_parts_patterns: Optional[List[Tuple[str, str]]],
) -> None:
    """Fallback: index dataset using ``energon prepare`` subprocess.

    NOTE: The subprocess path only supports ratio-based splitting.
    Pattern-based ``split_parts_patterns`` requires the programmatic API.
    """
    if split_parts_patterns:
        raise RuntimeError(
            "Pattern-based split_parts_patterns requires "
            "megatron.energon.BaseWebdatasetFactory (programmatic API). "
            "Install megatron-energon >= 7.x or use ratio-based splitting."
        )

    val_split = (1.0 - train_split) / 2
    test_split = (1.0 - train_split) / 2
    split_input = f"{train_split}, {val_split}, {test_split}\nn\n"

    logger.info(f"  Split ratios: train={train_split:.2f}, " f"val={val_split:.2f}, test={test_split:.2f}")
    logger.info("  Running energon prepare subprocess...")

    try:
        process = subprocess.Popen(
            [
                "energon",
                "prepare",
                str(output_path),
                "--num-workers",
                str(num_workers),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        stdout, _ = process.communicate(input=split_input, timeout=300)

        if process.returncode != 0:
            logger.error("energon prepare failed:")
            logger.error(stdout)
            raise RuntimeError(f"energon prepare exited with code {process.returncode}")

        for line in stdout.splitlines():
            if any(skip in line for skip in ["libibverbs", "Warning:", "UserWarning"]):
                continue
            if any(key in line for key in ["Indexing", "Done", "Found", "samples", "shards"]):
                logger.info(f"  {line.strip()}")

        logger.info("✓ Energon indexing complete (subprocess)")

    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(
            "energon prepare timed out after 5 minutes. "
            "This may indicate a problem with the dataset or system resources."
        )
    except FileNotFoundError:
        raise RuntimeError(
            "energon command not found. Make sure megatron-energon is installed:\n"
            "  pip install megatron-energon"
        )


def finalize_energon_dataset(
    output_dir: str,
    train_split: float = 1.0,
    encoding: _ENCODING_TYPE = "preencoded",
    num_workers: int = 8,
    split_parts_patterns: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Finalize Energon WebDataset by creating dataset.yaml and running energon prepare.

    This automates the manual steps typically required after data preprocessing:
    1. Create .nv-meta/dataset.yaml with correct subflavor configuration
    2. Run energon indexing (programmatic API with subprocess fallback)
    3. Verify the dataset is ready for training

    Args:
        output_dir: Path to dataset directory containing tar files.
        train_split: Fraction for training (rest split evenly between
            val/test). Ignored when split_parts_patterns is provided.
        encoding: Dataset encoding mode.
        num_workers: Number of workers for energon prepare (default: 8).
        split_parts_patterns: List of (split_name, pattern) tuples for
            pattern-based splitting, e.g.
            ``[("train", "train/.*"), ("val", "val/.*")]``.
            When provided, train_split is ignored.

    Raises:
        RuntimeError: If energon prepare fails or energon command not found
        FileNotFoundError: If output_dir doesn't exist or has no tar files
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    tar_files = list(output_path.glob("**/*.tar"))
    if not tar_files:
        raise FileNotFoundError(
            f"No .tar files found in {output_dir}. " "Make sure data preprocessing completed successfully."
        )

    logger.info("=" * 80)
    logger.info("Finalizing Energon dataset for training...")
    logger.info("=" * 80)
    logger.info(f"Found {len(tar_files)} shard(s) to index")

    _write_dataset_yaml(output_path, encoding)

    logger.info("Running energon prepare (this may take a few minutes)...")

    try:
        _prepare_programmatic(output_path, num_workers, train_split, split_parts_patterns)
    except ImportError:
        logger.info("BaseWebdatasetFactory not available, falling back to subprocess")
        _prepare_subprocess(output_path, num_workers, train_split, split_parts_patterns)

    from .validate import validate_energon_dataset

    validate_energon_dataset(output_dir, encoding=encoding)

    logger.info("=" * 80)
    logger.info(f"✓ Dataset finalized: {output_dir}")
    logger.info(f"  To use in training, set: dataset_path: {output_dir}")
    logger.info("=" * 80)
