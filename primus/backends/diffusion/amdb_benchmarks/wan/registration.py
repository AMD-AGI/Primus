###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Build Wan2.1 I2V on the native Wan modules."""

from __future__ import annotations

from typing import Any

from primus.backends.diffusion.amdb_benchmarks.download import materialize_pretrained
from primus.backends.diffusion.models.registrations.wan import (
    build_wan_model as build_native_wan_model,
)


def build_wan_model(model_config: dict[str, Any]):
    """Native Wan DiT with Wan2.1 I2V cross-attention and the AMDB noise schedule."""
    config = dict(model_config or {})
    nested = dict(config.get("config") or {})
    nested.setdefault("model_type", "i2v")
    nested.setdefault("dit_hidden_size", 5120)
    nested.setdefault("dit_num_layers", 40)
    nested.setdefault("dit_num_heads", 40)
    nested.setdefault("dit_intermediate_size", 13824)
    nested.setdefault("dit_in_channels", 36)
    nested.setdefault("dit_out_channels", 16)
    nested.setdefault("dit_patch_size", (1, 2, 2))
    nested.setdefault("dit_freq_dim", 256)
    nested.setdefault("dit_text_dim", 4096)
    nested.setdefault("text_len", 512)
    nested.setdefault("vae_type", "wan_video_vae")
    nested.setdefault("separated_timestep", False)
    nested.setdefault("fuse_vae_embedding_in_latents", False)
    nested.setdefault("trainable_modules", "dit")
    nested.setdefault("height", 544)
    nested.setdefault("width", 960)
    nested.setdefault("num_frames", 9)
    nested.setdefault("amdb_sampling", True)
    nested.setdefault("accept_diffusers", True)
    config["config"] = nested

    pretrained = materialize_pretrained("wan2.1-i2v", config.get("load_from_pretrained_path") or "")
    if pretrained:
        config["load_from_pretrained_path"] = pretrained
    return build_native_wan_model(config)
