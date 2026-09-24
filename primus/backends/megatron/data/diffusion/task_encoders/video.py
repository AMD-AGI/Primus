# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Video TaskEncoder for Wan diffusion models using Crude Data pattern.

EncodedWanTaskEncoder loads pre-encoded data (Wan VAE latents, UMT5 features),
mirroring EncodedDiffusionTaskEncoder in image.py. Wan has no raw-video path
yet; latents and text features are produced offline.

Select it with the dataset.yaml subflavors field:
    ```yaml
    subflavors:
      encoding: wan_preencoded
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
        latents: Wan VAE video latents (C, T, H, W)
        encoder_hidden_states: UMT5 text features (S_txt, D_txt)
        caption: Original text caption (optional, for debugging)
    """

    latents: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
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


__all__ = [
    "WanSample",
    "EncodedWanTaskEncoder",
    "cook_wan_preencoded",
    "decode_wan_caption",
    "load_wan_tensor",
]
