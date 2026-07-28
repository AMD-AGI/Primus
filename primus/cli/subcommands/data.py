###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Dataset preparation subcommands for Megatron backend diffusion training.

Provides tools to prepare datasets in Energon WebDataset format specifically
for diffusion models using the Megatron backend.

Commands:
    primus data diffusion-raw      - Prepare raw Energon WebDataset
    primus data diffusion-encoded  - Prepare pre-encoded Energon WebDataset
    primus data diffusion-ingest   - Stream pre-encoded Arrow data into WebDataset

Supports torch.distributed for multi-GPU processing:
    torchrun --nproc_per_node=8 primus data diffusion-raw ...
"""

import argparse
import logging

from primus.backends.megatron.data.diffusion.preprocessing.auth import (
    HFAuthError,
    setup_hf_authentication,
)
from primus.backends.megatron.data.diffusion.preprocessing.pipelines.encoded import (
    EncodedDatasetPipeline,
)
from primus.backends.megatron.data.diffusion.preprocessing.pipelines.raw import (
    RawDatasetPipeline,
)
from primus.tools.utils import finalize_distributed, init_distributed

logger = logging.getLogger(__name__)


def _add_common_args(parser):
    """Add arguments common to all data preparation commands."""
    # Source configuration
    source_group = parser.add_argument_group("Source Configuration")
    source_group.add_argument(
        "--source-type",
        required=False,
        choices=["directory", "huggingface", "webdataset"],
        help="Type of input data source (required, can be set via --config)",
    )
    source_group.add_argument("--input-dir", type=str, help="Input directory (for directory source)")
    source_group.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name (for HF source)")
    source_group.add_argument(
        "--hf-split", type=str, default="train", help="HuggingFace dataset split (default: train)"
    )
    source_group.add_argument(
        "--hf-data-files",
        type=str,
        default=None,
        help='Specific files/paths within HF dataset (e.g., "data_1024_10K/*.tar")',
    )
    source_group.add_argument("--input-path", type=str, help="WebDataset path/glob (for webdataset source)")

    # Output configuration (Energon format)
    output_group = parser.add_argument_group("Output Configuration (Energon WebDataset)")
    output_group.add_argument(
        "--output-dir",
        required=False,
        help="Output directory for Energon WebDataset (required, can be set via --config)",
    )
    output_group.add_argument(
        "--shard-size", type=int, default=1000, help="Samples per shard (default: 1000)"
    )
    output_group.add_argument(
        "--max-samples", type=int, default=None, help="Maximum samples to process (default: all)"
    )
    output_group.add_argument("--compress", action="store_true", help="Compress tar files with gzip")

    # Image preprocessing
    image_group = parser.add_argument_group("Image Preprocessing")
    image_group.add_argument(
        "--variable-size",
        action="store_true",
        help="Resize images to the nearest multiple of 16 instead of fixed size",
    )
    image_group.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Resize images to this size (default: 1024) when variable_size is disabled",
    )
    image_group.add_argument(
        "--center-crop", action="store_false", help="Disable center cropping of images before resize"
    )
    image_group.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Maximum image dimension for variable size mode (default: 1024)",
    )

    # HuggingFace Authentication
    auth_group = parser.add_argument_group("HuggingFace Authentication")
    auth_group.add_argument(
        "--hf-token-file",
        type=str,
        default=None,
        help=(
            "Path to file containing HuggingFace token. "
            "File must have secure permissions (600 or 400). "
            "If not provided, falls back to HF_TOKEN env variable "
            "or HF CLI login (~/.cache/huggingface/token)."
        ),
    )

    return parser


def _validate_source_args(args):
    """Validate that required source arguments are provided."""
    if args.source_type == "directory" and not args.input_dir:
        raise ValueError("--input-dir required for directory source")
    if args.source_type == "huggingface" and not args.hf_dataset:
        raise ValueError("--hf-dataset required for huggingface source")
    if args.source_type == "webdataset" and not args.input_path:
        raise ValueError("--input-path required for webdataset source")


def _prepare_raw(args):
    """Prepare raw Energon WebDataset for on-the-fly encoding during training."""
    # Initialize torch.distributed if launched with torchrun
    init_distributed()

    try:
        _validate_source_args(args)

        # Setup HF authentication
        try:
            setup_hf_authentication(token_file=getattr(args, "hf_token_file", None))
        except HFAuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise

        logger.info("=" * 80)
        logger.info("Megatron Diffusion: Raw Energon WebDataset Preparation")
        logger.info("=" * 80)

        pipeline = RawDatasetPipeline(
            source_type=args.source_type,
            output_dir=args.output_dir,
            variable_size=args.variable_size,
            image_size=args.image_size,
            center_crop=args.center_crop,
            max_size=args.max_size,
            image_format=getattr(args, "image_format", "JPEG"),
            image_quality=getattr(args, "image_quality", 95),
            shard_size=args.shard_size,
            max_samples=args.max_samples,
            compress=args.compress,
        )

        source_kwargs = {}
        if args.source_type == "directory":
            source_kwargs["input_dir"] = args.input_dir
        elif args.source_type == "huggingface":
            source_kwargs["hf_dataset"] = args.hf_dataset
            source_kwargs["hf_split"] = args.hf_split
            source_kwargs["hf_data_files"] = getattr(args, "hf_data_files", None)
            # Add data format configuration
            source_kwargs["image_key"] = getattr(args, "image_key", None)
            source_kwargs["caption_key"] = getattr(args, "caption_key", None)
            source_kwargs["image_keys"] = getattr(args, "image_keys", None)
            source_kwargs["caption_keys"] = getattr(args, "caption_keys", None)
        elif args.source_type == "webdataset":
            source_kwargs["input_path"] = args.input_path

        results = pipeline.run(**source_kwargs)

        logger.info("=" * 80)
        logger.info("✓ Raw Energon WebDataset preparation complete!")
        logger.info(f"  Samples processed: {results['samples_processed']}")
        logger.info(f"  Samples skipped: {results['samples_skipped']}")
        logger.info(f"  Shards created: {results['shards_written']}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info(f"  Format: Raw Energon WebDataset (images + captions)")
        logger.info(f"  Note: Encoding will be done on-the-fly during training")
        logger.info("=" * 80)

        # Finalize dataset unless --no-finalize was passed
        if not getattr(args, "no_finalize", False):
            # Import distributed utilities
            import torch.distributed as dist

            # Wait for all ranks to complete data preparation
            if dist.is_initialized():
                logger.info("Waiting for all ranks to complete...")
                dist.barrier()

            # Only rank 0 runs finalization
            should_finalize = not dist.is_initialized() or dist.get_rank() == 0

            if should_finalize:
                from primus.backends.megatron.data.diffusion.preprocessing.finalize import (
                    finalize_energon_dataset,
                )

                try:
                    finalize_energon_dataset(
                        output_dir=args.output_dir,
                        train_split=getattr(args, "train_split", 1.0),
                        encoding="raw",
                        num_workers=8,
                    )
                except Exception as e:
                    logger.error(f"Finalization failed: {e}")
                    raise

            # Wait again so all ranks exit together
            if dist.is_initialized():
                dist.barrier()

    finally:
        # Clean up distributed
        finalize_distributed()


def _flatten_preprocessing_config(config_dict: dict) -> dict:
    """
    Flatten nested YAML config to match CLI argument structure.

    Transforms hierarchical YAML structure into flat dict matching argparse namespace.

    Args:
        config_dict: Nested config from YAML file

    Returns:
        Flattened dict with CLI argument names as keys

    Example:
        >>> config = {
        ...     'source': {'type': 'huggingface', 'hf_dataset': 'pokemon'},
        ...     'model': {'batch_size': 8}
        ... }
        >>> flat = _flatten_preprocessing_config(config)
        >>> flat['source_type'], flat['batch_size']
        ('huggingface', 8)
    """
    flat = {}

    # Source configuration
    if "source" in config_dict:
        source = config_dict["source"]
        flat["source_type"] = source.get("type")
        flat["hf_dataset"] = source.get("hf_dataset")
        flat["hf_split"] = source.get("hf_split", "train")
        flat["hf_data_files"] = source.get("hf_data_files")
        flat["input_dir"] = source.get("input_dir")
        flat["input_path"] = source.get("input_path")

    # Data format configuration (field mappings for image/caption extraction)
    if "data_format" in config_dict:
        data_format = config_dict["data_format"]
        flat["image_key"] = data_format.get("image_key")
        flat["caption_key"] = data_format.get("caption_key")
        flat["image_keys"] = data_format.get("image_keys")
        flat["caption_keys"] = data_format.get("caption_keys")

    # Output configuration
    if "output" in config_dict:
        output = config_dict["output"]
        flat["output_dir"] = output.get("output_dir")
        flat["shard_size"] = output.get("shard_size", 1000)
        flat["max_samples"] = output.get("max_samples")
        flat["compress"] = output.get("compress", False)

    # Model configuration
    if "model" in config_dict:
        model = config_dict["model"]
        flat["model_path"] = model.get("model_path", "black-forest-labs/FLUX.1-dev")
        flat["vae_path"] = model.get("vae_path")
        flat["t5_path"] = model.get("t5_path")
        flat["clip_path"] = model.get("clip_path")
        flat["precision"] = model.get("precision", "bf16")
        flat["device"] = model.get("device", "cuda")
        flat["batch_size"] = model.get("batch_size", 8)
        flat["t5_max_length"] = model.get("t5_max_length", 512)
        flat["vae_latent_mode"] = model.get("vae_latent_mode", "presampled")

    # Image preprocessing
    if "image" in config_dict:
        image = config_dict["image"]
        flat["image_size"] = image.get("image_size", 1024)
        flat["variable_size"] = image.get("variable_size", False)
        flat["center_crop"] = image.get("center_crop", True)
        flat["max_size"] = image.get("max_size", 1024)

    # Authentication
    if "auth" in config_dict:
        auth = config_dict["auth"]
        flat["hf_token_file"] = auth.get("hf_token_file")

    return flat


def _get_encoded_parser_defaults() -> dict:
    """
    Get default values from encoded parser for override detection.

    Returns dict of argument name -> default value to detect which CLI
    arguments were explicitly set vs using defaults.
    """
    return {
        "config": None,
        "source_type": None,
        "hf_dataset": None,
        "hf_split": "train",
        "hf_data_files": None,
        "input_dir": None,
        "input_path": None,
        "image_key": None,
        "caption_key": None,
        "image_keys": None,
        "caption_keys": None,
        "output_dir": None,
        "shard_size": 1000,
        "max_samples": None,
        "compress": False,
        "model_path": "black-forest-labs/FLUX.1-dev",
        "vae_path": None,
        "t5_path": None,
        "clip_path": None,
        "precision": "bf16",
        "device": "cuda",
        "batch_size": 8,
        "t5_max_length": 512,
        "image_size": 1024,
        "variable_size": False,
        "center_crop": True,
        "max_size": 1024,
        "hf_token_file": None,
        "vae_latent_mode": "presampled",
    }


def _load_config_with_cli_overrides(args: "argparse.Namespace") -> "argparse.Namespace":
    """
    Load YAML config and merge with CLI arguments.

    Priority order (highest to lowest):
    1. Explicitly provided CLI arguments
    2. YAML config values
    3. CLI default values

    Args:
        args: Parsed CLI arguments

    Returns:
        Merged namespace with final configuration

    Example:
        >>> # With config file specifying batch_size: 8
        >>> # And CLI arg --batch-size 16
        >>> # Result: batch_size = 16 (CLI wins)
    """
    import argparse

    # If no config file, return args as-is
    if not getattr(args, "config", None):
        return args

    from primus.core.utils import yaml_utils

    logger.info(f"Loading config from: {args.config}")

    # Load and flatten YAML config
    config_dict = yaml_utils.parse_yaml(args.config)
    flat_config = _flatten_preprocessing_config(config_dict)

    # Start with config values
    merged_dict = flat_config.copy()

    # Override with explicitly provided CLI arguments
    parser_defaults = _get_encoded_parser_defaults()
    cli_args = vars(args)

    for key, cli_value in cli_args.items():
        if key == "config":
            # Keep config path for reference
            merged_dict["config"] = cli_value
            continue

        # If CLI value differs from default, it was explicitly set
        default_value = parser_defaults.get(key)
        if cli_value != default_value:
            merged_dict[key] = cli_value
            logger.debug(f"CLI override: {key} = {cli_value}")

    logger.info("Configuration merged (CLI args override YAML)")
    return argparse.Namespace(**merged_dict)


def _validate_preprocessing_config(args: "argparse.Namespace") -> None:
    """
    Validate preprocessing configuration after merging.

    Ensures all required parameters are present regardless of source
    (CLI, YAML, or both).

    Args:
        args: Merged configuration namespace

    Raises:
        ValueError: If required arguments are missing or invalid

    Example:
        >>> args = argparse.Namespace(source_type='huggingface', output_dir=None)
        >>> _validate_preprocessing_config(args)  # Raises ValueError
    """
    # Check required arguments
    required = ["source_type", "output_dir"]
    missing = [arg for arg in required if not getattr(args, arg, None)]

    if missing:
        raise ValueError(
            f"Missing required arguments: {', '.join(missing)}. "
            f"Provide via --config YAML or CLI arguments."
        )

    # Validate source-specific requirements
    if args.source_type == "huggingface":
        if not getattr(args, "hf_dataset", None):
            raise ValueError(
                "Missing --hf-dataset for source-type 'huggingface'. "
                "Specify in config file (source.hf_dataset) or via CLI."
            )
    elif args.source_type == "directory":
        if not getattr(args, "input_dir", None):
            raise ValueError(
                "Missing --input-dir for source-type 'directory'. "
                "Specify in config file (source.input_dir) or via CLI."
            )
    elif args.source_type == "webdataset":
        if not getattr(args, "input_path", None):
            raise ValueError(
                "Missing --input-path for source-type 'webdataset'. "
                "Specify in config file (source.input_path) or via CLI."
            )
    else:
        raise ValueError(
            f"Invalid source-type: {args.source_type}. " f"Must be one of: huggingface, directory, webdataset"
        )


def _prepare_encoded(args):
    """Prepare pre-encoded Energon WebDataset with VAE/T5/CLIP for fast training."""
    # Load and merge config if provided
    args = _load_config_with_cli_overrides(args)

    # Validate merged configuration
    _validate_preprocessing_config(args)

    # Initialize torch.distributed if launched with torchrun
    init_distributed()

    try:
        _validate_source_args(args)

        # Setup HF authentication
        try:
            setup_hf_authentication(token_file=getattr(args, "hf_token_file", None))
        except HFAuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise

        logger.info("=" * 80)
        logger.info("Megatron Diffusion: Pre-encoded Energon WebDataset Preparation")
        logger.info("=" * 80)

        pipeline = EncodedDatasetPipeline(
            source_type=args.source_type,
            output_dir=args.output_dir,
            model_path=args.model_path,
            vae_path=args.vae_path,
            t5_path=args.t5_path,
            clip_path=args.clip_path,
            precision=args.precision,
            device=args.device,
            batch_size=args.batch_size,
            t5_max_length=args.t5_max_length,
            variable_size=args.variable_size,
            image_size=args.image_size,
            center_crop=args.center_crop,
            max_size=args.max_size,
            shard_size=args.shard_size,
            max_samples=args.max_samples,
            compress=args.compress,
            vae_latent_mode=getattr(args, "vae_latent_mode", "presampled"),
        )

        source_kwargs = {}
        if args.source_type == "directory":
            source_kwargs["input_dir"] = args.input_dir
        elif args.source_type == "huggingface":
            source_kwargs["hf_dataset"] = args.hf_dataset
            source_kwargs["hf_split"] = args.hf_split
            source_kwargs["hf_data_files"] = getattr(args, "hf_data_files", None)
            # Add data format configuration
            source_kwargs["image_key"] = getattr(args, "image_key", None)
            source_kwargs["caption_key"] = getattr(args, "caption_key", None)
            source_kwargs["image_keys"] = getattr(args, "image_keys", None)
            source_kwargs["caption_keys"] = getattr(args, "caption_keys", None)

        elif args.source_type == "webdataset":
            source_kwargs["input_path"] = args.input_path

        results = pipeline.run(**source_kwargs)

        logger.info("=" * 80)
        logger.info("✓ Pre-encoded Energon WebDataset preparation complete!")
        logger.info(f"  Samples processed: {results['samples_processed']}")
        logger.info(f"  Samples skipped: {results['samples_skipped']}")
        logger.info(f"  Shards created: {results['shards_written']}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info(f"  Format: Pre-encoded Energon WebDataset (VAE/T5/CLIP tensors)")
        logger.info(f"  Note: Training will be faster (no on-the-fly encoding)")
        logger.info("=" * 80)

        # Finalize dataset unless --no-finalize was passed
        if not getattr(args, "no_finalize", False):
            # Import distributed utilities
            import torch.distributed as dist

            # Wait for all ranks to complete data preparation
            if dist.is_initialized():
                logger.info("Waiting for all ranks to complete...")
                dist.barrier()

            # Only rank 0 runs finalization
            should_finalize = not dist.is_initialized() or dist.get_rank() == 0

            if should_finalize:
                from primus.backends.megatron.data.diffusion.preprocessing.finalize import (
                    finalize_energon_dataset,
                )

                try:
                    finalize_energon_dataset(
                        output_dir=args.output_dir,
                        train_split=getattr(args, "train_split", 1.0),
                        encoding="preencoded",
                        num_workers=8,
                    )
                except Exception as e:
                    logger.error(f"Finalization failed: {e}")
                    raise

            # Wait again so all ranks exit together
            if dist.is_initialized():
                dist.barrier()

    finally:
        # Clean up distributed
        finalize_distributed()


def _load_ingest_config(args):
    """Load ingest YAML config and merge with CLI overrides.

    Returns a flat dict consumed by _prepare_ingest.
    """
    from primus.core.utils import yaml_utils

    config_path = args.config
    if not config_path:
        raise ValueError("--config is required for diffusion-ingest")

    raw = yaml_utils.parse_yaml(config_path)

    config = {
        "datasets": raw.get("datasets", []),
        "output_dir": raw.get("output", {}).get("output_dir"),
        "max_workers": raw.get("pipeline", {}).get("max_workers", 4),
        "prefetch_depth": raw.get("pipeline", {}).get("prefetch_depth", 6),
        "max_files": raw.get("pipeline", {}).get("max_files"),
        "no_finalize": False,
    }

    if raw.get("empty_encodings"):
        config["empty_encodings"] = raw["empty_encodings"]

    # CLI overrides
    if getattr(args, "output_dir", None):
        config["output_dir"] = args.output_dir
    if getattr(args, "input_dir", None):
        config["input_dir"] = args.input_dir
    if getattr(args, "max_workers", None) is not None:
        config["max_workers"] = args.max_workers
    if getattr(args, "prefetch_depth", None) is not None:
        config["prefetch_depth"] = args.prefetch_depth
    if getattr(args, "max_files", None) is not None:
        config["max_files"] = args.max_files
    if getattr(args, "no_finalize", False):
        config["no_finalize"] = True

    if not config.get("output_dir"):
        raise ValueError("output_dir is required. Set in config (output.output_dir) " "or via --output-dir.")

    # Default input_dir
    if "input_dir" not in config:
        from pathlib import Path

        config["input_dir"] = str(Path(config["output_dir"]) / "_arrow_tmp")

    return config


def _download_empty_encodings(config):
    """Download empty_encodings files (small .npy files for CFG dropout)."""
    from pathlib import Path

    from primus.backends.megatron.data.diffusion.preprocessing.download import (
        download_with_backoff,
        fetch_manifest,
    )

    enc_cfg = config["empty_encodings"]
    output_dir = Path(config["output_dir"]) / enc_cfg.get("output_subdir", "empty_encodings")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url, entries = fetch_manifest(enc_cfg["manifest_url"])

    for md5, fname in entries:
        dest = output_dir / fname
        if dest.exists():
            logger.info(f"  [skip] {fname} (exists)")
            continue
        file_url = f"{base_url}/{fname}"
        logger.info(f"  [download] {fname}")
        download_with_backoff(file_url, dest, expected_md5=md5)

    logger.info(f"Empty encodings downloaded to {output_dir}")


def _prepare_ingest(args):
    """Download and convert pre-encoded Arrow data into Energon WebDataset."""
    from primus.backends.megatron.data.diffusion.preprocessing.pipelines.ingest import (
        StreamingIngestPipeline,
    )

    config = _load_ingest_config(args)

    logger.info("=" * 80)
    logger.info("Megatron Diffusion: Streaming Ingest Pipeline")
    logger.info("=" * 80)

    total_failed = 0
    for ds in config["datasets"]:
        logger.info(f"Processing dataset: {ds['name']} (split: {ds['split_name']})")
        pipeline = StreamingIngestPipeline(
            manifest_url=ds["manifest_url"],
            input_dir=config["input_dir"],
            output_dir=config["output_dir"],
            split_name=ds["split_name"],
            max_files=config.get("max_files"),
            max_workers=config["max_workers"],
            prefetch_depth=config["prefetch_depth"],
        )
        result = pipeline.run()
        total_failed += result.get("files_failed", 0)

    if "empty_encodings" in config:
        logger.info("Downloading empty encodings...")
        _download_empty_encodings(config)

    if not config.get("no_finalize"):
        from primus.backends.megatron.data.diffusion.preprocessing.finalize import (
            finalize_energon_dataset,
        )

        try:
            finalize_energon_dataset(
                output_dir=config["output_dir"],
                encoding="preencoded_numpy",
                split_parts_patterns=[("train", "train/.*"), ("val", "val/.*")],
            )
        except Exception as e:
            logger.error(f"Finalization failed: {e}")
            raise

    logger.info("=" * 80)
    logger.info("Ingest complete!")
    logger.info(f"  Output: {config['output_dir']}")
    if total_failed:
        logger.warning(
            f"  {total_failed} file(s) failed across all datasets. "
            f"Check failed_files.json in each split directory."
        )
    logger.info("=" * 80)


def register_subcommand(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """
    Register 'primus data' subcommand for Megatron diffusion dataset preparation.

    Provides dataset preparation utilities specifically for diffusion models
    using the Megatron backend, outputting Energon WebDataset format.
    """
    parser = subparsers.add_parser(
        "data",
        help="Dataset preparation tools (Megatron diffusion, Energon format)",
        description=(
            "Data preparation utilities for Megatron-backend diffusion models.\n"
            "Creates datasets in Energon WebDataset format.\n\n"
            "Supported modes:\n"
            "  - diffusion-raw: Raw WebDataset (smaller, on-the-fly encoding)\n"
            "  - diffusion-encoded: Pre-encoded WebDataset (larger, faster training)\n"
            "  - diffusion-ingest: Stream pre-encoded Arrow data into WebDataset\n\n"
            "Multi-GPU support via torch.distributed:\n"
            "  torchrun --nproc_per_node=8 primus data diffusion-raw ...\n"
            "  torchrun --nproc_per_node=8 primus data diffusion-encoded ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data_subparsers = parser.add_subparsers(dest="data_command", required=True, help="Data preparation mode")

    # ===== primus data diffusion-raw =====
    raw_parser = data_subparsers.add_parser(
        "diffusion-raw",
        help="Prepare raw Energon WebDataset (Megatron diffusion: images + captions)",
        description=(
            "Prepare raw Energon WebDataset for Megatron diffusion training.\n\n"
            "Creates smaller datasets with raw images and captions.\n"
            "Encoding (VAE/T5/CLIP) happens on-the-fly during training.\n\n"
            "Output format: Energon WebDataset (tar shards)\n\n"
            "Use this when:\n"
            "  - Storage is limited\n"
            "  - Experimenting with different encoders\n"
            "  - Rapid prototyping\n\n"
            "Example:\n"
            "  primus data diffusion-raw \\\n"
            "    --source-type huggingface \\\n"
            "    --hf-dataset diffusers/pokemon-gpt4-captions \\\n"
            "    --output-dir /data/raw_pokemon \\\n"
            "    --image-size 1024 --center-crop\n\n"
            "Multi-GPU:\n"
            "  torchrun --nproc_per_node=8 primus data diffusion-raw ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(raw_parser)

    # Raw-specific arguments
    raw_specific = raw_parser.add_argument_group("Raw Dataset Options")
    raw_specific.add_argument(
        "--image-format",
        choices=["JPEG", "PNG", "WEBP"],
        default="JPEG",
        help="Output image format (default: JPEG)",
    )
    raw_specific.add_argument(
        "--image-quality", type=int, default=95, help="JPEG/WEBP quality 1-100 (default: 95)"
    )

    # Finalization arguments
    raw_finalize = raw_parser.add_argument_group("Dataset Finalization")
    raw_finalize.add_argument(
        "--no-finalize",
        action="store_true",
        dest="no_finalize",
        help="Skip automatic dataset.yaml creation and energon prepare (default: finalize is ON)",
    )
    raw_finalize.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Training split ratio for finalization (default: 1.0 = 100%% train, 0%% val/test)",
    )

    raw_parser.set_defaults(func=lambda args, unknown: _prepare_raw(args))

    # ===== primus data diffusion-encoded =====
    encoded_parser = data_subparsers.add_parser(
        "diffusion-encoded",
        help="Prepare pre-encoded Energon WebDataset (Megatron diffusion: VAE/T5/CLIP)",
        description=(
            "Prepare pre-encoded Energon WebDataset for Megatron diffusion training.\n\n"
            "Pre-encodes images with VAE and text with T5/CLIP, creating larger\n"
            "datasets that enable faster training (no encoding overhead).\n\n"
            "Output format: Energon WebDataset with pre-encoded tensors\n\n"
            "Use this when:\n"
            "  - Training on production datasets\n"
            "  - Maximum training speed is needed\n"
            "  - Storage space is available\n\n"
            "Example:\n"
            "  primus data diffusion-encoded \\\n"
            "    --source-type huggingface \\\n"
            "    --hf-dataset diffusers/pokemon-gpt4-captions \\\n"
            "    --output-dir /data/encoded_pokemon \\\n"
            "    --model-path black-forest-labs/FLUX.1-dev \\\n"
            "    --batch-size 8 --precision bf16\n\n"
            "Config file:\n"
            "  primus data diffusion-encoded --config preprocessing.yaml\n\n"
            "Multi-GPU:\n"
            "  torchrun --nproc_per_node=8 primus data diffusion-encoded ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file support (optional, CLI args override config values)
    encoded_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional, CLI args override config values)",
    )

    _add_common_args(encoded_parser)

    # Encoded-specific arguments
    model_group = encoded_parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-path",
        default="black-forest-labs/FLUX.1-dev",
        help="Pretrained model path (HF or local). Default: black-forest-labs/FLUX.1-dev "
        "(requires HF authentication -- see --hf-token-file)",
    )
    model_group.add_argument("--vae-path", default=None, help="Custom VAE path (overrides --model-path)")
    model_group.add_argument("--t5-path", default=None, help="Custom T5 path (overrides --model-path)")
    model_group.add_argument("--clip-path", default=None, help="Custom CLIP path (overrides --model-path)")
    model_group.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model precision (default: bf16)",
    )
    model_group.add_argument("--device", default="cuda", help="Device for encoding (default: cuda)")
    model_group.add_argument("--batch-size", type=int, default=8, help="Encoding batch size (default: 8)")
    model_group.add_argument(
        "--t5-max-length",
        type=int,
        default=512,
        help="T5 max sequence length (default: 512, use 256 for FLUX.1-schnell)",
    )
    model_group.add_argument(
        "--vae-latent-mode",
        choices=["presampled", "resample"],
        default="presampled",
        help=(
            "VAE latent storage mode (default: presampled). "
            "'presampled' stores a single sampled latent per image. "
            "'resample' additionally stores mean and logvar so the training "
            "loop can re-draw latents via reparameterization at every step."
        ),
    )

    # Finalization arguments
    encoded_finalize = encoded_parser.add_argument_group("Dataset Finalization")
    encoded_finalize.add_argument(
        "--no-finalize",
        action="store_true",
        dest="no_finalize",
        help="Skip automatic dataset.yaml creation and energon prepare (default: finalize is ON)",
    )
    encoded_finalize.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Training split ratio for finalization (default: 1.0 = 100%% train, 0%% val/test)",
    )

    encoded_parser.set_defaults(func=lambda args, unknown: _prepare_encoded(args))

    # ===== primus data diffusion-ingest =====
    ingest_parser = data_subparsers.add_parser(
        "diffusion-ingest",
        help="Stream pre-encoded Arrow data into Energon WebDataset",
        description=(
            "Stream pre-encoded Arrow data into Energon WebDataset.\n\n"
            "Downloads Apache Arrow IPC files from MLCommons R2 (or similar),\n"
            "converts them to WebDataset tar shards in a single pass using a\n"
            "producer-consumer prefetch architecture. Minimizes disk usage by\n"
            "deleting each Arrow file after conversion.\n\n"
            "Requires a YAML config file specifying datasets and parameters.\n\n"
            "Example:\n"
            "  primus data diffusion-ingest \\\n"
            "    --config primus/configs/data/megatron/diffusion/"
            "preprocessing/mlperf_flux1.yaml\n\n"
            "Override output:\n"
            "  primus data diffusion-ingest \\\n"
            "    --config .../mlperf_flux1.yaml \\\n"
            "    --output-dir /custom/path --max-files 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ingest_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with datasets list and pipeline settings",
    )

    ingest_output = ingest_parser.add_argument_group("Output Configuration")
    ingest_output.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config output.output_dir)",
    )
    ingest_output.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Temp directory for Arrow downloads (default: <output-dir>/_arrow_tmp)",
    )

    ingest_pipeline = ingest_parser.add_argument_group("Pipeline Configuration")
    ingest_pipeline.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit Arrow files per dataset (for testing)",
    )
    ingest_pipeline.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Concurrent download threads (overrides config, default: 4)",
    )
    ingest_pipeline.add_argument(
        "--prefetch-depth",
        type=int,
        default=None,
        help="Max Arrow files buffered on disk (overrides config, default: 6)",
    )

    ingest_finalize = ingest_parser.add_argument_group("Dataset Finalization")
    ingest_finalize.add_argument(
        "--no-finalize",
        action="store_true",
        dest="no_finalize",
        help="Skip automatic Energon finalization (default: finalize is ON)",
    )

    ingest_parser.set_defaults(func=lambda args, unknown: _prepare_ingest(args))

    # Set default for parent parser (required by CLI framework, but never called due to required=True on subparsers)
    parser.set_defaults(func=lambda args, unknown: None)

    return parser
