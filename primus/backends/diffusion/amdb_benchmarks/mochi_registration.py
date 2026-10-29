###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Build the official Mochi-1 DiT and VAE encoder for Primus post-training."""

from __future__ import annotations

import os

import torch

from primus.backends.diffusion.amdb_benchmarks.checkpoint_io import load_matching
from primus.backends.diffusion.amdb_benchmarks.download import materialize_pretrained
from primus.backends.diffusion.amdb_benchmarks.mochi.adapter import MochiForTraining
from primus.backends.diffusion.amdb_benchmarks.mochi.asymm_models_joint import (
    AsymmDiTJoint,
)
from primus.backends.diffusion.amdb_benchmarks.mochi.text_encoder import (
    MochiTextEncoder,
)
from primus.backends.diffusion.amdb_benchmarks.mochi.train_pipeline import (
    MochiTrainPipeline,
)
from primus.backends.diffusion.amdb_benchmarks.mochi.vae_models import Encoder
from primus.backends.diffusion.amdb_benchmarks.sampling import amdb_flow_sigma_grid
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import count_parameters


def _parse_resolution(value) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value)
    if "," in text:
        width, height = text.split(",", 1)
        return int(width), int(height)
    size = int(text)
    return size, size


def build_mochi_model(model_config: dict):
    cfg = dict(model_config.get("config") or {})
    encoder_cfg = dict(model_config.get("encoder") or {})
    width, height = _parse_resolution(model_config.get("resolution") or cfg.get("resolution") or "960,544")
    dit = AsymmDiTJoint(
        depth=48,
        patch_size=2,
        num_heads=24,
        hidden_size_x=3072,
        hidden_size_y=1536,
        mlp_ratio_x=4.0,
        mlp_ratio_y=4.0,
        in_channels=12,
        qk_norm=True,
        qkv_bias=False,
        out_bias=True,
        patch_embed_bias=True,
        timestep_mlp_bias=True,
        timestep_scale=1000.0,
        t5_feat_dim=4096,
        t5_token_length=256,
        rope_theta=10000.0,
        attention_mode=str(cfg.get("attention_mode") or "sdpa"),
    )
    if os.environ.get("BENCHMARK_DENOISER_ONLY") == "1":
        encoder = torch.nn.Identity()
        text_encoder = torch.nn.Identity()
    else:
        encoder = Encoder(
            in_channels=15,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 6],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            temporal_reductions=[1, 2, 3],
            spatial_reductions=[2, 2, 2],
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate",
        )
        t5_path = encoder_cfg.get("t5") or encoder_cfg.get("text_encoder") or "google/t5-v1_1-xxl"
        text_encoder = MochiTextEncoder(t5_path)

    pretrained = materialize_pretrained("mochi-1", model_config.get("load_from_pretrained_path") or "")
    if pretrained:
        missing, unexpected = load_matching(dit, pretrained, prefixes=("module.", "model.", "dit."))
        logger.info(f"Loaded Mochi DiT. missing={missing} unexpected={unexpected}")
    vae_path = encoder_cfg.get("autoencoder") or ""
    if vae_path:
        missing, unexpected = load_matching(encoder, vae_path, prefixes=("encoder.", "module."))
        logger.info(f"Loaded Mochi VAE encoder. missing={missing} unexpected={unexpected}")

    model = MochiForTraining(
        dit=dit,
        encoder=encoder,
        text_encoder=text_encoder,
        train_pipeline=MochiTrainPipeline(
            width=width,
            height=height,
            num_frames=int(model_config.get("num_frames") or cfg.get("num_frames") or 9),
            sigma_grid=amdb_flow_sigma_grid(shift=float(cfg.get("shift", 3.0))),
        ),
    )
    model.freeze_except()
    total, trainable = count_parameters(model)
    logger.info(f"mochi-1 parameters: total={total/1e9:.3f}B trainable={trainable/1e9:.3f}B")
    return model
