# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Diffusion TaskEncoder for image-text pairs using Crude Data pattern.

This module provides TaskEncoder implementations for diffusion models:
- EncodedDiffusionTaskEncoder: Loads pre-encoded data (latents, embeddings)
- RawDiffusionTaskEncoder: Loads raw images and text (no encoding)

Encoding happens in the model, not in the TaskEncoder (following best practices).
Position IDs are generated in the model code, not here.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from megatron.energon import (
    Cooker,
    DefaultTaskEncoder,
    Sample,
    SampleDecoder,
    WorkerConfig,
    basic_sample_keys,
    stateless,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Sample Definition (with proper Sample inheritance)
# ============================================================================


@dataclass
class DiffusionSample(Sample):
    """
    Diffusion training sample with framework-standard field names.

    Inherits from megatron.energon.Sample to ensure __key__, __restore_key__,
    and __subflavors__ are properly tracked for deterministic training resumption.

    Position IDs (img_ids, text_ids) are generated at runtime based on actual
    tensor shapes, not stored in the dataset. This provides flexibility for
    variable-resolution training.

    Attributes:
        prompt_embeds: T5 text embeddings (seq_len, hidden_dim)
        pooled_prompt_embeds: CLIP pooled embeddings (hidden_dim,)
        latents: Image latents from VAE (C, H, W) — optional for resample-only datasets
        mean: VAE posterior mean (C, H, W) — present in 'resample' mode datasets
        logvar: VAE posterior log-variance (C, H, W) — present in 'resample' mode datasets
        caption: Original text caption (optional, for debugging)
    """

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    latents: Optional[torch.Tensor] = None
    mean: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    caption: str = ""
    timestep: Optional[torch.Tensor] = None


# ============================================================================
# Cooker Functions
# ============================================================================


@stateless
def cook_preencoded_diffusion(sample: dict) -> DiffusionSample:
    """
    Cooker for preencoded diffusion features with framework-standard keys.

    Loads precalculated VAE latents and text embeddings from disk.
    Position IDs are NOT loaded - they are generated at runtime based on
    actual tensor shapes for flexibility.

    Required standard keys:
        - 'latents.pth': VAE-encoded image latents
        - 'prompt_embeds.pth': T5 text embeddings
        - 'pooled_prompt_embeds.pth': CLIP pooled embeddings

    Args:
        sample: Raw sample dict from WebDataset

    Returns:
        DiffusionSample with all metadata properly forwarded
    """

    def load_tensor(data):
        """Helper to load tensor from bytes."""
        if data is None:
            return None
        if isinstance(data, (str, Path)):
            return torch.load(data, map_location="cpu")
        elif isinstance(data, bytes):
            return torch.load(io.BytesIO(data), map_location="cpu")
        elif isinstance(data, torch.Tensor):
            return data
        else:
            return data

    # Load required fields (position IDs not loaded)
    latents = load_tensor(sample.get("latents.pth"))
    prompt_embeds = load_tensor(sample.get("prompt_embeds.pth"))
    pooled_prompt_embeds = load_tensor(sample.get("pooled_prompt_embeds.pth"))

    # Load optional resample-mode fields
    mean = load_tensor(sample.get("mean.pth"))
    logvar = load_tensor(sample.get("logvar.pth"))

    if prompt_embeds is None or pooled_prompt_embeds is None:
        raise ValueError(
            f"Sample missing required keys. Expected: 'prompt_embeds.pth', "
            f"'pooled_prompt_embeds.pth'. Got: {list(sample.keys())}"
        )
    if latents is None and mean is None:
        raise ValueError(
            f"Sample must have 'latents.pth' or 'mean.pth'/'logvar.pth'. " f"Got: {list(sample.keys())}"
        )

    # Load caption if available
    caption = sample.get("caption.txt", b"")
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8")
    elif isinstance(caption, (str, Path)):
        try:
            if Path(caption).exists():
                with open(caption, "r") as f:
                    caption = f.read().strip()
            else:
                caption = str(caption)
        except (OSError, ValueError):
            caption = str(caption)
    else:
        caption = str(caption) if caption else ""

    # Extract timestep from JSON sidecar (MLPerf validation datasets)
    timestep = None
    if "json" in sample:
        import json as json_mod

        raw = sample["json"]
        if isinstance(raw, dict):
            decoded = raw
        elif isinstance(raw, bytes):
            decoded = json_mod.loads(raw.decode("utf-8"))
        else:
            decoded = json_mod.loads(raw)
        if "timestep" in decoded:
            timestep = torch.tensor(decoded["timestep"])

    return DiffusionSample(
        **basic_sample_keys(sample),
        latents=latents,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        mean=mean,
        logvar=logvar,
        caption=caption,
        timestep=timestep,
    )


@stateless
def cook_raw_images(sample: dict) -> Dict[str, Any]:
    """
    Cooker for raw images - just loads data, NO ENCODING.

    Encoding happens in the model's forward_step, not here.
    This follows the pattern where TaskEncoders only load data.

    Standard data keys:
        - 'images': Raw image data (or format-specific key like 'jpg', 'png', 'webp')
        - 'txt': Text caption

    Args:
        sample: Raw sample dict from WebDataset

    Returns:
        Dict with raw data ready for model encoding
    """
    return {
        **basic_sample_keys(sample),
        "images": sample.get("images"),
        "txt": sample.get("txt"),
    }


def load_numpy_tensor(data: Optional[bytes]) -> Optional[torch.Tensor]:
    """Load a bfloat16 tensor from numpy uint16 bytes (MLPerf format).

    The MLPerf Flux dataset stores bfloat16 tensors by reinterpreting them
    as uint16 numpy arrays serialized as .npy files. This function loads
    the .npy buffer (preserving shape), then reinterprets uint16 as
    bfloat16 via torch.

    Falls back to raw ``np.frombuffer`` for headerless byte buffers.
    """
    if data is None:
        return None
    if data[:6] == b"\x93NUMPY":
        arr = np.load(io.BytesIO(data))
    else:
        arr = np.frombuffer(data, dtype=np.uint16)
    return torch.from_numpy(arr.copy()).view(torch.bfloat16)


@stateless
def cook_preencoded_numpy_diffusion(sample: dict) -> DiffusionSample:
    """Cooker for MLPerf pre-encoded numpy data (bfloat16 as uint16 bytes).

    Expected keys in the WebDataset tar shard:
        - 't5.bytes':    T5 text embeddings
        - 'clip.bytes':  CLIP pooled embeddings
        - 'mean.bytes':  VAE posterior mean
        - 'logvar.bytes': VAE posterior log-variance

    No 'latents' are stored; the training loop resamples from mean/logvar.
    """
    prompt_embeds = load_numpy_tensor(sample.get("t5.bytes"))
    pooled_prompt_embeds = load_numpy_tensor(sample.get("clip.bytes"))
    mean = load_numpy_tensor(sample.get("mean.bytes"))
    logvar = load_numpy_tensor(sample.get("logvar.bytes"))

    if prompt_embeds is None or pooled_prompt_embeds is None:
        raise ValueError(
            f"Sample missing required keys. Expected: 't5.bytes', "
            f"'clip.bytes'. Got: {list(sample.keys())}"
        )
    if mean is None:
        raise ValueError(f"Sample missing 'mean.bytes'. Got: {list(sample.keys())}")

    timestep = None
    if "json" in sample:
        import json as json_mod

        raw = sample["json"]
        if isinstance(raw, dict):
            decoded = raw
        elif isinstance(raw, bytes):
            decoded = json_mod.loads(raw.decode("utf-8"))
        else:
            decoded = json_mod.loads(raw)
        if "timestep" in decoded:
            timestep = torch.tensor(decoded["timestep"])

    return DiffusionSample(
        **basic_sample_keys(sample),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        mean=mean,
        logvar=logvar,
        timestep=timestep,
    )


# ============================================================================
# TaskEncoders
# ============================================================================


class EncodedDiffusionTaskEncoder(DefaultTaskEncoder[DiffusionSample, DiffusionSample, dict, dict]):
    """
    TaskEncoder for PRE-ENCODED diffusion data.

    Use this when your dataset contains pre-encoded features:
    - latents.pth (VAE-encoded image latents)
    - prompt_embeds.pth (T5 text embeddings)
    - pooled_prompt_embeds.pth (CLIP pooled embeddings)

    Does NOT do any encoding - just loads from disk.
    For raw data, use RawDiffusionTaskEncoder instead.

    Outputs batch with standard keys:
    - 'latents'
    - 'prompt_embeds'
    - 'pooled_prompt_embeds'

    Use with dataset.yaml:
        ```yaml
        subflavors:
          encoding: preencoded
        ```
    """

    decoder = SampleDecoder(image_decode="pil")

    cookers = [
        Cooker(cook_preencoded_diffusion, has_subflavors={"encoding": "preencoded"}),
        Cooker(cook_preencoded_numpy_diffusion, has_subflavors={"encoding": "preencoded_numpy"}),
    ]

    def __init__(self, worker_config: Optional[WorkerConfig] = None):
        """Initialize pre-encoded TaskEncoder."""
        super().__init__()
        self.worker_config = worker_config
        logger.info("Initialized EncodedDiffusionTaskEncoder (preencoded / preencoded_numpy modes)")

    def batch(self, samples: List[DiffusionSample]) -> Dict[str, torch.Tensor]:
        """
        Batch pre-encoded samples.

        Position IDs are NOT included - they are generated at runtime
        in the forward step based on actual tensor shapes.

        Returns:
            Dict with keys: prompt_embeds, pooled_prompt_embeds,
            and conditionally latents and/or mean, logvar depending on
            which fields are present in the samples.
        """
        batch: Dict[str, torch.Tensor] = {
            "prompt_embeds": torch.stack([s.prompt_embeds for s in samples]),
            "pooled_prompt_embeds": torch.stack([s.pooled_prompt_embeds for s in samples]),
        }

        if samples[0].latents is not None:
            batch["latents"] = torch.stack([s.latents for s in samples])

        if samples[0].mean is not None:
            batch["mean"] = torch.stack([s.mean for s in samples])
            batch["logvar"] = torch.stack([s.logvar for s in samples])

        if samples[0].timestep is not None:
            batch["timestep"] = torch.stack([s.timestep for s in samples])

        return batch


class RawDiffusionTaskEncoder(DefaultTaskEncoder):
    """
    TaskEncoder for RAW diffusion data (images and text).

    Use this when your dataset contains raw files:
    - images (raw image files)
    - txt (text captions)

    This TaskEncoder:
    - Loads raw images and captions from disk
    - Does NOT do any encoding (no VAE, no T5, no CLIP)
    - Encoding happens on-the-fly in model's forward_step

    Outputs batch with standard keys:
    - 'images': List of PIL Images or image bytes
    - 'txt': List of caption strings

    Use with dataset.yaml:
        ```yaml
        subflavors:
          encoding: raw
        ```
    """

    decoder = SampleDecoder(image_decode="pil")

    cookers = [
        Cooker(cook_raw_images, has_subflavors={"encoding": "raw"}),
    ]

    def __init__(self, worker_config: Optional[WorkerConfig] = None):
        """Initialize raw diffusion TaskEncoder."""
        super().__init__()
        self.worker_config = worker_config
        logger.info("Initialized RawDiffusionTaskEncoder (no encoding, passes raw data)")

    def batch(self, samples: List[Dict]) -> Dict[str, Any]:
        """
        Batch raw samples.

        Returns:
            Dict with standard keys:
            - 'images': List of PIL Images or image bytes
            - 'txt': List of caption strings
        """
        return {
            "images": [s["images"] for s in samples],
            "txt": [s["txt"] for s in samples],
        }


__all__ = [
    "DiffusionSample",
    "EncodedDiffusionTaskEncoder",
    "RawDiffusionTaskEncoder",
    "cook_preencoded_diffusion",
    "cook_preencoded_numpy_diffusion",
    "cook_raw_images",
    "load_numpy_tensor",
]
