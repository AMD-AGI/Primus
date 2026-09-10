# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Video TaskEncoders for Wan diffusion models using Crude Data pattern.

This module provides TaskEncoder implementations for Wan video models:
- EncodedWanTaskEncoder: Loads pre-encoded data (VAE latents, UMT5 features)
- RawWanTaskEncoder: Loads raw frames and text (no encoding)

The split mirrors EncodedDiffusionTaskEncoder / RawDiffusionTaskEncoder in
image.py. Encoding happens in the model, not in the TaskEncoder.

Use the dataset.yaml subflavors field to choose:
    ```yaml
    subflavors:
      encoding: wan_preencoded  # or "wan_raw"
    ```
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
class WanSample(Sample):
    """
    Wan video training sample with framework-standard field names.

    Inherits from megatron.energon.Sample to ensure __key__, __restore_key__,
    and __subflavors__ are properly tracked for deterministic training resumption.

    Attributes:
        latents: Wan VAE video latents (C, T, H, W) — pre-encoded path
        encoder_hidden_states: UMT5 text features (S_txt, D_txt) — pre-encoded path
        frames: Raw pixel video (C, T, H, W) in range [-1, 1] — raw path
        txt: Raw caption string — raw path
        caption: Original text caption (optional, for debugging)
    """

    latents: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    frames: Optional[torch.Tensor] = None
    txt: str = ""
    caption: str = ""


# ============================================================================
# Cooker Helpers
# ============================================================================


def load_wan_tensor(data: Any) -> Optional[torch.Tensor]:
    """Load a tensor from bytes, a path, or an already-decoded tensor."""
    if data is None:
        return None
    if isinstance(data, (str, Path)):
        return torch.load(data, map_location="cpu")
    if isinstance(data, bytes):
        return torch.load(io.BytesIO(data), map_location="cpu")
    if isinstance(data, torch.Tensor):
        return data
    return data


def decode_wan_caption(raw: Any) -> str:
    """Decode a caption stored as bytes, a path, or a plain string."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    if isinstance(raw, (str, Path)):
        path = Path(str(raw))
        try:
            if path.exists():
                return path.read_text().strip()
        except (OSError, ValueError) as e:
            # Shards carry captions either inline or as a sidecar path, so an
            # unreadable path is a caption that happens to look like one.
            logger.debug(f"Could not read caption from path {path}: {e}")
        return str(raw)
    return str(raw)


# ============================================================================
# Cooker Functions
# ============================================================================


@stateless
def cook_wan_preencoded(sample: dict) -> WanSample:
    """
    Cooker for pre-encoded Wan features with framework-standard keys.

    Loads precalculated Wan VAE latents and UMT5 text features from disk.

    Required standard keys:
        - 'latents.pth': Wan VAE video latents (C, T, H, W)
        - 'encoder_hidden_states.pth': UMT5 text features (S_txt, D_txt)

    Optional keys:
        - 'caption.txt': Original caption (for debugging)

    Args:
        sample: Raw sample dict from WebDataset

    Returns:
        WanSample with all metadata properly forwarded
    """
    latents = load_wan_tensor(sample.get("latents.pth"))
    encoder_hidden_states = load_wan_tensor(sample.get("encoder_hidden_states.pth"))

    if latents is None:
        raise ValueError(f"Wan pre-encoded sample missing 'latents.pth'. Got: {list(sample.keys())}")
    if encoder_hidden_states is None:
        raise ValueError(
            f"Wan pre-encoded sample missing 'encoder_hidden_states.pth'. Got: {list(sample.keys())}"
        )

    caption = decode_wan_caption(sample.get("caption.txt"))

    return WanSample(
        **basic_sample_keys(sample),
        latents=latents,
        encoder_hidden_states=encoder_hidden_states,
        caption=caption,
    )


@stateless
def cook_wan_raw(sample: dict) -> WanSample:
    """
    Cooker for raw Wan videos - just loads data, NO ENCODING.

    Encoding happens in the model's forward_step, not here.

    Standard data keys:
        - 'frames': Raw video data (tensor or serialized bytes)
        - 'txt': Text caption

    Args:
        sample: Raw sample dict from WebDataset

    Returns:
        WanSample with raw data ready for model encoding
    """
    frames = sample.get("frames")
    if isinstance(frames, bytes):
        frames = load_wan_tensor(frames)

    txt = decode_wan_caption(sample.get("txt", ""))

    return WanSample(
        **basic_sample_keys(sample),
        frames=frames,
        txt=txt,
        caption=txt,
    )


# ============================================================================
# TaskEncoders
# ============================================================================


class EncodedWanTaskEncoder(DefaultTaskEncoder[WanSample, WanSample, dict, dict]):
    """
    TaskEncoder for PRE-ENCODED Wan video data.

    Use this when your dataset contains pre-encoded features:
    - latents.pth (Wan VAE video latents)
    - encoder_hidden_states.pth (UMT5 text features)

    Does NOT do any encoding - just loads from disk.
    For raw data, use RawWanTaskEncoder instead.

    Outputs batch with standard keys:
    - 'latents'
    - 'encoder_hidden_states'

    Use with dataset.yaml:
        ```yaml
        subflavors:
          encoding: wan_preencoded
        ```
    """

    decoder = SampleDecoder(image_decode="pil")

    cookers = [
        Cooker(cook_wan_preencoded, has_subflavors={"encoding": "wan_preencoded"}),
    ]

    def __init__(self, worker_config: Optional[WorkerConfig] = None):
        """Initialize pre-encoded Wan TaskEncoder."""
        super().__init__()
        self.worker_config = worker_config
        logger.info("Initialized EncodedWanTaskEncoder (wan_preencoded mode)")

    def batch(self, samples: List[WanSample]) -> Dict[str, torch.Tensor]:
        """
        Batch pre-encoded Wan samples.

        Returns:
            Dict with keys: latents, encoder_hidden_states
        """
        return {
            "latents": torch.stack([s.latents for s in samples]),
            "encoder_hidden_states": torch.stack([s.encoder_hidden_states for s in samples]),
        }


class RawWanTaskEncoder(DefaultTaskEncoder):
    """
    TaskEncoder for RAW Wan video data (frames and text).

    Use this when your dataset contains raw files:
    - frames (raw video frames)
    - txt (text captions)

    This TaskEncoder:
    - Loads raw frames and captions from disk
    - Does NOT do any encoding (no Wan VAE, no UMT5)
    - Encoding happens on-the-fly in model's forward_step

    Outputs batch with standard keys:
    - 'frames': List of video tensors
    - 'txt': List of caption strings

    Use with dataset.yaml:
        ```yaml
        subflavors:
          encoding: wan_raw
        ```
    """

    decoder = SampleDecoder(image_decode="pil")

    cookers = [
        Cooker(cook_wan_raw, has_subflavors={"encoding": "wan_raw"}),
    ]

    def __init__(self, worker_config: Optional[WorkerConfig] = None):
        """Initialize raw Wan TaskEncoder."""
        super().__init__()
        self.worker_config = worker_config
        logger.info("Initialized RawWanTaskEncoder (no encoding, passes raw data)")

    def batch(self, samples: List[WanSample]) -> Dict[str, Any]:
        """
        Batch raw Wan samples.

        Returns:
            Dict with standard keys:
            - 'frames': List of video tensors
            - 'txt': List of caption strings
        """
        return {
            "frames": [s.frames for s in samples],
            "txt": [s.txt for s in samples],
        }


__all__ = [
    "WanSample",
    "EncodedWanTaskEncoder",
    "RawWanTaskEncoder",
    "cook_wan_preencoded",
    "cook_wan_raw",
    "decode_wan_caption",
    "load_wan_tensor",
]
