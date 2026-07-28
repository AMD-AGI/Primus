# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Raw dataset preparation pipeline for Megatron diffusion models.

Creates raw Energon WebDatasets with images and captions for on-the-fly
encoding during training.
"""

import logging
from pathlib import Path
from typing import Optional

from ..utils import (
    encode_image_to_bytes,
    get_distributed_info,
    preprocess_image,
    save_to_webdataset,
)

logger = logging.getLogger(__name__)


from .base import DatasetPipeline


class RawDatasetPipeline(DatasetPipeline):
    """
    Pipeline for creating raw Energon WebDataset with images and captions.

    This pipeline creates smaller datasets where encoding (VAE/T5/CLIP) happens
    on-the-fly during training. Suitable for experimentation and when storage
    is limited.

    Args:
        source_type: Type of input source ('directory', 'huggingface', 'webdataset')
        output_dir: Output directory for Energon WebDataset
        image_size: Target image size (default: 1024)
        center_crop: Whether to center crop images (default: True)
        image_format: Output image format (default: 'JPEG')
        image_quality: JPEG/WEBP quality 1-100 (default: 95)
        shard_size: Samples per shard (default: 1000)
        max_samples: Maximum samples to process (default: None for all)
        compress: Whether to compress tar files with gzip (default: False)
        hf_token_file: Path to HuggingFace token file (default: None)
        variable_size: If True, resize to nearest multiple of 16 instead of fixed size (default: False)
        max_size: Maximum dimension when variable_size is True (default: 1024)

    Example:
        >>> pipeline = RawDatasetPipeline(
        ...     source_type='huggingface',
        ...     output_dir='/data/raw_pokemon',
        ...     image_size=1024,
        ...     center_crop=True,
        ... )
        >>> results = pipeline.run(
        ...     hf_dataset='diffusers/pokemon-gpt4-captions',
        ...     hf_split='train'
        ... )
        >>> print(f"Processed {results['samples_processed']} samples")
    """

    def __init__(
        self,
        source_type: str,
        output_dir: str,
        variable_size: bool = False,
        image_size: int = 1024,
        center_crop: bool = True,
        max_size: int = 1024,
        image_format: str = "JPEG",
        image_quality: int = 95,
        shard_size: int = 1000,
        max_samples: Optional[int] = None,
        compress: bool = False,
        hf_token_file: Optional[str] = None,
    ):
        self.source_type = source_type
        self.output_dir = Path(output_dir)
        self.variable_size = variable_size
        self.image_size = image_size
        self.center_crop = center_crop
        self.max_size = max_size
        self.image_format = image_format
        self.image_quality = image_quality
        self.shard_size = shard_size
        self.max_samples = max_samples
        self.compress = compress

        # Setup HF authentication if token file provided
        if hf_token_file:
            from ..auth import HFAuthError, setup_hf_authentication

            try:
                setup_hf_authentication(token_file=hf_token_file)
            except HFAuthError as e:
                raise ValueError(f"HuggingFace authentication failed: {e}")

        # Get distributed info
        self.rank, self.world_size = get_distributed_info()

        # Determine output format extension
        self.format_ext = {
            "JPEG": "jpg",
            "PNG": "png",
            "WEBP": "webp",
        }[self.image_format]

        logger.info(f"Initialized RawDatasetPipeline (rank {self.rank}/{self.world_size})")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Image preprocessing: size={self.image_size}, crop={self.center_crop}")

    def process_sample(self, item):
        """
        Process a single sample.

        Args:
            item: Sample dict with 'image' (PIL Image) and 'caption' (str)

        Returns:
            Sample dict with image bytes and caption text
        """
        # Preprocess image
        image = preprocess_image(
            item["image"],
            variable_size=self.variable_size,
            size=self.image_size,
            center_crop=self.center_crop,
            max_size=self.max_size,
        )

        # Encode to bytes
        image_bytes = encode_image_to_bytes(image, format=self.image_format, quality=self.image_quality)

        # Create sample with standard keys
        return {
            self.format_ext: image_bytes,
            "txt": item["caption"],
        }

    def run(self, **source_kwargs):
        """
        Execute the raw dataset preparation pipeline.

        Args:
            **source_kwargs: Source-specific arguments (passed to load_data)

        Returns:
            Dictionary with processing statistics:
                - samples_processed: Number of samples successfully processed
                - samples_skipped: Number of samples that failed
                - shards_written: Number of output shards created
        """
        logger.info(f"Starting raw dataset preparation (rank {self.rank}/{self.world_size})")
        logger.info(f"Source: {self.source_type}")

        # Load data as a lazy iterator (never materialized in full)
        data_iter = self.load_data(**source_kwargs)

        if self.world_size > 1:
            logger.info(
                f"Distributed mode: rank {self.rank}/{self.world_size} "
                f"processing every {self.world_size}th item (round-robin by load index)"
            )

        # Process and accumulate samples
        samples = []
        samples_processed = 0
        samples_skipped = 0
        shards_written = 0

        # Stream items and assign them to ranks round-robin by global load
        # index, so the full dataset is never held in memory. max_samples caps
        # the GLOBAL index (total across ranks, not per-rank), preserving the
        # prior semantics; the rank split changes from contiguous ranges to
        # round-robin (same overall dataset, more balanced load). The index is
        # assigned at load time, before process_sample, so the rank assignment
        # stays deterministic even when samples are skipped.
        for global_idx, item in enumerate(data_iter):
            # Truthy check (not `is not None`) mirrors the original semantics
            # where max_samples=0 / None meant "no limit".
            if self.max_samples and global_idx >= self.max_samples:
                break
            if global_idx % self.world_size != self.rank:
                continue
            try:
                sample = self.process_sample(item)
                samples.append(sample)
                samples_processed += 1

                # Log progress
                if samples_processed % 100 == 0:
                    logger.info(f"Processed {samples_processed} samples (skipped {samples_skipped})")

                # Save shard when full
                if len(samples) >= self.shard_size:
                    # Use distributed-aware shard naming to prevent conflicts
                    shard_offset = shards_written * self.world_size + self.rank
                    if self.world_size > 1 and shards_written == 0:
                        logger.info(f"Rank {self.rank}: First shard will be numbered {shard_offset:06d}.tar")

                    num_shards = save_to_webdataset(
                        samples,
                        str(self.output_dir),
                        self.shard_size,
                        shard_offset=shard_offset,
                        compress=self.compress,
                    )
                    shards_written += num_shards
                    samples = []

            except Exception as e:
                logger.warning(f"Failed to process sample {global_idx}: {e}")
                samples_skipped += 1
                continue

        # Save remaining samples
        if samples:
            shard_offset = shards_written * self.world_size + self.rank
            num_shards = save_to_webdataset(
                samples,
                str(self.output_dir),
                self.shard_size,
                shard_offset=shard_offset,
                compress=self.compress,
            )
            shards_written += num_shards

        return {
            "samples_processed": samples_processed,
            "samples_skipped": samples_skipped,
            "shards_written": shards_written,
        }
