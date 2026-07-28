# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Pre-encoded dataset preparation pipeline for Megatron diffusion models.

Creates pre-encoded Energon WebDatasets with VAE/T5/CLIP encoded tensors
for faster training (no on-the-fly encoding overhead).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from primus.backends.megatron.data.diffusion.encoders import get_encoder
from primus.backends.megatron.data.diffusion.encoders.config import (
    CLIPLConfig,
    T5XXLConfig,
    VAEConfig,
)

from ..utils import (
    get_distributed_info,
    preprocess_image,
    save_to_webdataset,
    split_work_for_rank,
)

logger = logging.getLogger(__name__)


def set_reproducibility(seed: int = 42) -> None:
    """Set global seeds and cuDNN flags for reproducible encoding.

    Called explicitly from the pipeline entry point rather than at import time,
    so importing this module does not mutate global torch/cuDNN state for
    unrelated callers.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from .base import DatasetPipeline


class EncodedDatasetPipeline(DatasetPipeline):
    """
    Pipeline for creating pre-encoded Energon WebDataset with VAE/T5/CLIP tensors.

    This pipeline pre-encodes images with VAE and text with T5/CLIP, creating larger
    datasets that enable faster training (no encoding overhead).

    Args:
        source_type: Type of input source ('directory', 'huggingface', 'webdataset')
        output_dir: Output directory for Energon WebDataset
        model_path: Pretrained model path (HF or local, default: black-forest-labs/FLUX.1-dev)
        vae_path: Custom VAE path (overrides model_path)
        t5_path: Custom T5 path (overrides model_path)
        clip_path: Custom CLIP path (overrides model_path)
        precision: Model precision ('bf16', 'fp16', 'fp32', default: 'bf16')
        device: Device for encoding (default: 'cuda')
        batch_size: Encoding batch size (default: 8)
        t5_max_length: T5 max sequence length (default: 512, use 256 for schnell)
        image_size: Target image size (default: 1024)
        variable_size: If True, resize to nearest multiple of 16 instead of fixed size (default: False)
        center_crop: Whether to center crop images (default: True)
        max_size: Maximum dimension when variable_size is True (default: 1024)
        shard_size: Samples per shard (default: 1000)
        max_samples: Maximum samples to process (default: None for all)
        compress: Whether to compress tar files with gzip (default: False)
        hf_token_file: Path to HuggingFace token file (default: None)
        vae_latent_mode: 'presampled' stores only latents; 'resample' additionally
            stores mean and logvar for training-time reparameterization (default: 'presampled')

    Example:
        >>> pipeline = EncodedDatasetPipeline(
        ...     source_type='huggingface',
        ...     output_dir='/data/encoded_pokemon',
        ...     model_path='black-forest-labs/FLUX.1-dev',
        ...     batch_size=8,
        ...     image_size=1024,
        ... )
        >>> results = pipeline.run(
        ...     hf_dataset='diffusers/pokemon-gpt4-captions',
        ...     hf_split='train'
        ... )
        >>> print(f"Processed {results['samples_processed']} samples")
    """

    # Encoded pipeline materializes the full HF dataset (non-streaming) so the
    # distributed loader can split by total count.
    HF_STREAMING: bool = False

    def __init__(
        self,
        source_type: str,
        output_dir: str,
        model_path: str = "black-forest-labs/FLUX.1-dev",
        vae_path: Optional[str] = None,
        t5_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        precision: str = "bf16",
        device: str = "cuda",
        batch_size: int = 8,
        t5_max_length: int = 512,
        variable_size: bool = False,
        image_size: int = 1024,
        center_crop: bool = True,
        max_size: int = 1024,
        shard_size: int = 1000,
        max_samples: Optional[int] = None,
        compress: bool = False,
        hf_token_file: Optional[str] = None,
        vae_latent_mode: str = "presampled",
    ):
        if vae_latent_mode not in ("presampled", "resample"):
            raise ValueError(f"vae_latent_mode must be 'presampled' or 'resample', got '{vae_latent_mode}'")

        self.source_type = source_type
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.vae_path = vae_path
        self.t5_path = t5_path
        self.clip_path = clip_path
        self.precision = precision
        self.device = device
        self.batch_size = batch_size
        self.t5_max_length = t5_max_length
        self.variable_size = variable_size
        self.image_size = image_size
        self.center_crop = center_crop
        self.max_size = max_size
        self.shard_size = shard_size
        self.max_samples = max_samples
        self.compress = compress
        self.vae_latent_mode = vae_latent_mode

        # Setup HF authentication if token file provided
        if hf_token_file:
            from ..auth import HFAuthError, setup_hf_authentication

            try:
                setup_hf_authentication(token_file=hf_token_file)
            except HFAuthError as e:
                raise ValueError(f"HuggingFace authentication failed: {e}")

        # Get distributed info
        self.rank, self.world_size = get_distributed_info()

        logger.info(f"Initialized EncodedDatasetPipeline (rank {self.rank}/{self.world_size})")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Precision: {self.precision}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"T5 max sequence length: {self.t5_max_length}")
        logger.info(f"VAE latent mode: {self.vae_latent_mode}")

        # Load encoders
        self.vae, self.t5, self.clip = self._load_encoders()

    @staticmethod
    def _raise_encoder_auth_error(encoder_name: str, model_path: str, original_error: Exception):
        """Raise a clear error when encoder download fails due to authentication."""
        err_str = str(original_error).lower()
        is_auth = any(
            kw in err_str
            for kw in [
                "token",
                "permission",
                "private repository",
                "gated",
                "401",
                "403",
                "authentication",
                "login",
            ]
        )
        if is_auth:
            raise RuntimeError(
                f"Failed to download {encoder_name} encoder from '{model_path}'.\n"
                f"This model likely requires HuggingFace authentication.\n\n"
                f"To fix, provide a token using one of:\n"
                f"  1. --hf-token-file /path/to/.hf_token\n"
                f"  2. export HF_TOKEN=hf_xxx\n"
                f"  3. huggingface-cli login\n"
            ) from original_error
        raise RuntimeError(
            f"Failed to load {encoder_name} encoder from '{model_path}': {original_error}"
        ) from original_error

    def _load_encoders(self):
        """Load VAE, T5, and CLIP encoders."""
        logger.info("Loading encoders...")

        # Determine if we're using FLUX model (needs subfolders)
        vae_model_path = self.vae_path or self.model_path
        t5_model_path = self.t5_path or self.model_path
        clip_model_path = self.clip_path or self.model_path

        is_flux = "FLUX" in self.model_path or "flux" in self.model_path.lower()

        # Create encoder configs with subfolders for FLUX models
        vae_config = VAEConfig(
            type="autoencoder_kl",
            model_path=vae_model_path,
            subfolder="vae" if (is_flux and not self.vae_path) else None,
            precision=self.precision,
            device=self.device,
        )

        t5_config = T5XXLConfig(
            type="t5_xxl",
            model_path=t5_model_path,
            subfolder="text_encoder_2" if (is_flux and not self.t5_path) else None,
            tokenizer_subfolder="tokenizer_2" if (is_flux and not self.t5_path) else None,
            max_length=self.t5_max_length,
            precision=self.precision,
            device=self.device,
        )

        clip_config = CLIPLConfig(
            type="clip_l",
            model_path=clip_model_path,
            subfolder="text_encoder" if (is_flux and not self.clip_path) else None,
            tokenizer_subfolder="tokenizer" if (is_flux and not self.clip_path) else None,
            precision=self.precision,
            device=self.device,
        )

        # Load encoders with clear error messages on failure
        try:
            vae = get_encoder(vae_config)
        except Exception as e:
            self._raise_encoder_auth_error("VAE", vae_config.model_path, e)
        logger.info(f"Loaded VAE from {vae_config.model_path}")

        try:
            t5 = get_encoder(t5_config)
        except Exception as e:
            self._raise_encoder_auth_error("T5-XXL", t5_config.model_path, e)
        logger.info(f"Loaded T5-XXL from {t5_config.model_path}")

        try:
            clip = get_encoder(clip_config)
        except Exception as e:
            self._raise_encoder_auth_error("CLIP-L", clip_config.model_path, e)
        logger.info(f"Loaded CLIP-L from {clip_config.model_path}")

        # Set to eval mode
        vae.eval()
        t5.eval()
        clip.eval()

        return vae, t5, clip

    def load_data_distributed(self, **source_kwargs):
        """
        Load only the data needed for this rank (distributed-aware loading).

        This method loads data more efficiently by having each rank load only
        its assigned portion of the dataset, rather than loading everything
        and then splitting.

        Args:
            **source_kwargs: Source-specific arguments
                - For directory: input_dir
                - For huggingface: hf_dataset, hf_split
                - For webdataset: input_path

        Returns:
            List of samples assigned to this rank
        """
        if self.source_type == "directory":
            # Get file list (cheap operation, all ranks do this)
            from pathlib import Path

            from PIL import Image

            input_path = Path(source_kwargs["input_dir"])
            images_dir = input_path / "images"
            captions_dir = input_path / "captions"

            if not images_dir.exists():
                raise ValueError(f"Images directory not found: {images_dir}")
            if not captions_dir.exists():
                raise ValueError(f"Captions directory not found: {captions_dir}")

            # Get all image files
            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            all_image_files = []
            for ext in image_extensions:
                all_image_files.extend(sorted(images_dir.glob(f"*{ext}")))

            total_files = len(all_image_files)
            logger.info(f"Found {total_files} total images")

            # Apply max_samples limit BEFORE splitting across ranks so that
            # max_samples refers to the total number of samples, not per-rank.
            if self.max_samples and total_files > self.max_samples:
                logger.info(f"Limiting to {self.max_samples} total samples (before rank split)")
                all_image_files = all_image_files[: self.max_samples]
                total_files = self.max_samples

            # Split file list across ranks (before loading!)
            start_idx, end_idx = split_work_for_rank(total_files, self.rank, self.world_size)
            my_files = all_image_files[start_idx:end_idx]

            logger.info(f"Rank {self.rank}: Loading {len(my_files)} images (indices {start_idx}-{end_idx})")

            # Now load only this rank's files
            items = []
            for img_path in my_files:
                caption_path = captions_dir / f"{img_path.stem}.txt"
                if not caption_path.exists():
                    logger.warning(f"Caption not found for {img_path.name}, skipping")
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                    with open(caption_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                    items.append({"image": image, "caption": caption})
                except Exception as e:
                    logger.warning(f"Failed to load {img_path.name}: {e}")

            return items
        else:
            # For non-directory sources, fall back to loading all data
            logger.warning(
                f"Distributed loading not yet implemented for {self.source_type}, loading all data"
            )
            data_iter = self.load_data(**source_kwargs)
            all_items = list(data_iter)
            total_items = len(all_items)

            # Apply max_samples limit BEFORE splitting across ranks so that
            # max_samples refers to the total number of samples, not per-rank.
            if self.max_samples and total_items > self.max_samples:
                logger.info(f"Limiting to {self.max_samples} total samples (before rank split)")
                all_items = all_items[: self.max_samples]
                total_items = self.max_samples

            # Split work across ranks
            start_idx, end_idx = split_work_for_rank(total_items, self.rank, self.world_size)
            return all_items[start_idx:end_idx]

    def _preprocess_images_batch(self, images):
        """
        Preprocess batch of PIL images for VAE encoding.

        Args:
            images: List of PIL Images

        Returns:
            Tensor [B, C, H, W] in range [-1, 1]
        """
        tensors = []
        for image in images:
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array and normalize to [-1, 1]
            img_array = np.array(image).astype(np.float32) / 255.0
            img_array = img_array * 2.0 - 1.0

            # Convert to tensor (HWC -> CHW)
            img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1)))
            tensors.append(img_tensor)

        return torch.stack(tensors)

    def _encode_batch(self, images, captions):
        """
        Encode a batch of images and captions.

        Position IDs are NOT generated during preprocessing - they will be
        computed at runtime based on actual tensor shapes. This provides
        flexibility for variable-resolution training.

        Args:
            images: List of PIL Images
            captions: List of caption strings

        Returns:
            List of sample dicts with encoded tensors
        """
        with torch.no_grad():
            # Preprocess images
            images_tensor = self._preprocess_images_batch(images)
            images_tensor = images_tensor.to(self.device)

            # Encode images with VAE
            if self.vae_latent_mode == "resample":
                latents, mean, logvar = self.vae.encode_for_resample(images_tensor)
            else:
                latents = self.vae.encode(images_tensor)

            # Encode text with T5
            prompt_embeds = self.t5.encode(captions)

            # Encode text with CLIP
            _, pooled_prompt_embeds = self.clip.encode(captions)

        # Move to CPU and create samples (NO position IDs)
        samples = []
        for i in range(len(images)):
            sample = {
                "latents.pth": latents[i].cpu(),
                "prompt_embeds.pth": prompt_embeds[i].cpu(),
                "pooled_prompt_embeds.pth": pooled_prompt_embeds[i].cpu(),
            }
            if self.vae_latent_mode == "resample":
                sample["mean.pth"] = mean[i].cpu()
                sample["logvar.pth"] = logvar[i].cpu()
            samples.append(sample)

        return samples

    def run(self, **source_kwargs):
        """
        Execute the encoded dataset preparation pipeline.

        Args:
            **source_kwargs: Source-specific arguments (passed to load_data)

        Returns:
            Dictionary with processing statistics:
                - samples_processed: Number of samples successfully processed
                - samples_skipped: Number of samples that failed
                - shards_written: Number of output shards created
        """
        import torch.distributed as dist

        # Reproducible encoding: set seeds/cuDNN flags here (not at import time).
        set_reproducibility()

        start_time = time.time()
        rank_prefix = f"[RANK {self.rank}]"

        tqdm.write(f"{rank_prefix} Stage 1/4: Initialization")
        tqdm.write(f"{rank_prefix} Source: {self.source_type}")
        tqdm.write(
            f"{rank_prefix} Image preprocessing: size={self.image_size}, crop={self.center_crop}, variable_size={self.variable_size}"
        )

        # Synchronization point: ensure all ranks start together
        if dist.is_initialized():
            dist.barrier()

        # Load data using distributed-aware loading (each rank loads only its portion)
        tqdm.write(f"{rank_prefix} Stage 2/4: Loading data")
        items_to_process = self.load_data_distributed(**source_kwargs)
        tqdm.write(f"{rank_prefix} Loaded {len(items_to_process)} items")

        # Synchronization point: ensure all ranks finished loading
        if dist.is_initialized():
            dist.barrier()

        # Sort items by dimensions AFTER splitting to ensure batches have uniform sizes
        # This prevents tensor stacking errors while maintaining balanced workload across ranks
        items_to_process = sorted(
            items_to_process, key=lambda item: item["image"].size  # Returns (width, height)
        )

        total_to_process = len(items_to_process)

        tqdm.write(f"{rank_prefix} Stage 3/4: Encoding samples")

        # Process in batches
        batch_images = []
        batch_captions = []
        encoded_samples = []
        samples_processed = 0
        samples_skipped = 0
        shards_written = 0

        pbar = tqdm(
            total=total_to_process,
            desc=f"{rank_prefix} Encoding",
            unit="sample",
            disable=(self.rank != 0 and self.world_size > 4),
        )

        def _flush_batch():
            """Encode and accumulate the current batch, returning count."""
            nonlocal batch_images, batch_captions, encoded_samples, samples_processed
            nonlocal shards_written
            batch_samples = self._encode_batch(batch_images, batch_captions)
            encoded_samples.extend(batch_samples)
            count = len(batch_samples)
            samples_processed += count
            pbar.update(count)
            batch_images = []
            batch_captions = []

            if len(encoded_samples) >= self.shard_size:
                shard_offset = shards_written * self.world_size + self.rank
                num_shards = save_to_webdataset(
                    encoded_samples,
                    str(self.output_dir),
                    self.shard_size,
                    shard_offset=shard_offset,
                    compress=self.compress,
                )
                shards_written += num_shards
                encoded_samples = []

        for idx, item in enumerate(items_to_process):
            try:
                image = item["image"]
                if self.image_size or self.center_crop or self.variable_size:
                    image = preprocess_image(
                        image,
                        variable_size=self.variable_size,
                        size=self.image_size,
                        center_crop=self.center_crop,
                        max_size=self.max_size,
                    )

                current_size = image.size

                if batch_images and batch_images[0].size != current_size:
                    _flush_batch()

                batch_images.append(image)
                batch_captions.append(item["caption"])

                if len(batch_images) >= self.batch_size:
                    _flush_batch()

            except Exception as e:
                tqdm.write(f"{rank_prefix} Failed to encode sample {idx}: {e}")
                logger.debug(f"Sample {idx} encoding error", exc_info=True)
                samples_skipped += 1
                batch_images = []
                batch_captions = []
                continue

        # Process remaining batch
        if batch_images:
            try:
                _flush_batch()
            except Exception as e:
                tqdm.write(f"{rank_prefix} Failed to encode final batch: {e}")

        # Save remaining samples
        if encoded_samples:
            shard_offset = shards_written * self.world_size + self.rank
            num_shards = save_to_webdataset(
                encoded_samples,
                str(self.output_dir),
                self.shard_size,
                shard_offset=shard_offset,
                compress=self.compress,
            )
            shards_written += num_shards

        pbar.close()

        elapsed_time = time.time() - start_time
        tqdm.write(f"{rank_prefix} Stage 4/4: Complete")
        tqdm.write(
            f"{rank_prefix} Finished in {elapsed_time:.1f}s — "
            f"processed: {samples_processed}, skipped: {samples_skipped}, shards: {shards_written}"
        )

        # Generate empty encodings for CFG dropout (rank 0 only).
        # Uses the same T5/CLIP models already loaded with the same t5_max_length,
        # guaranteeing sequence length consistency with the encoded dataset.
        if self.rank == 0:
            self._generate_empty_encodings()

        # Critical synchronization point: ensure all ranks finished encoding before returning
        if dist.is_initialized():
            dist.barrier()

        return {
            "samples_processed": samples_processed,
            "samples_skipped": samples_skipped,
            "shards_written": shards_written,
        }

    def _generate_empty_encodings(self):
        """Generate and save T5/CLIP encodings for the empty string (rank 0 only)."""
        empty_dir = self.output_dir / "empty_encodings"
        empty_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            t5_empty = self.t5.encode([""])
            _, clip_empty = self.clip.encode([""])

        t5_np = t5_empty.cpu().float().numpy()
        clip_np = clip_empty.cpu().float().numpy()

        np.save(str(empty_dir / "t5_empty.npy"), t5_np)
        np.save(str(empty_dir / "clip_empty.npy"), clip_np)

        tqdm.write(
            f"[RANK 0] Saved empty encodings to {empty_dir} " f"(t5={t5_np.shape}, clip={clip_np.shape})"
        )
