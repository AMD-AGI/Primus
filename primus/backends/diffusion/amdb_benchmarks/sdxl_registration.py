###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Build the official SDXL UNet for Primus post-training."""

from __future__ import annotations

import os

import torch

from primus.backends.diffusion.amdb_benchmarks.checkpoint_io import load_matching
from primus.backends.diffusion.amdb_benchmarks.download import materialize_pretrained
from primus.backends.diffusion.amdb_benchmarks.sdxl.adapter import SDXLForTraining
from primus.backends.diffusion.amdb_benchmarks.sdxl.autoencoder_kl import (
    SDXLAutoencoderKL,
)
from primus.backends.diffusion.amdb_benchmarks.sdxl.text_encoder import SDXLTextEncoder
from primus.backends.diffusion.amdb_benchmarks.sdxl.train_pipeline import (
    SDXLTrainPipeline,
)
from primus.backends.diffusion.amdb_benchmarks.sdxl.unet import UNetModel
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import count_parameters


def build_sdxl_unet() -> UNetModel:
    return UNetModel(
        adm_in_channels=2816,
        num_classes="sequential",
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_resolutions=(4, 2),
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        num_head_channels=64,
        transformer_depth=[1, 2, 10],
        context_dim=2048,
        use_linear_in_transformer=True,
        spatial_transformer_attn_type="softmax",
        use_checkpoint=False,
    )


def build_sdxl_model(model_config: dict):
    cfg = dict(model_config.get("config") or {})
    encoder_cfg = dict(model_config.get("encoder") or {})
    resolution = int(str(model_config.get("resolution") or cfg.get("resolution") or 1024).split(",")[0])
    dit = build_sdxl_unet()
    autoencoder = SDXLAutoencoderKL(scale_factor=float(cfg.get("scale_factor", 0.13025)))
    clip_l = encoder_cfg.get("clip_l") or "openai/clip-vit-large-patch14"
    clip_g = encoder_cfg.get("clip_g") or "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    if os.environ.get("BENCHMARK_DENOISER_ONLY") == "1":
        text_encoder = torch.nn.Identity()
    else:
        text_encoder = SDXLTextEncoder(clip_l, clip_g)

    pretrained = materialize_pretrained("sdxl", model_config.get("load_from_pretrained_path") or "")
    if pretrained:
        missing, unexpected = load_matching(dit, pretrained, prefixes=("model.diffusion_model.", "module."))
        logger.info(f"Loaded SDXL UNet. missing={missing} unexpected={unexpected}")
    vae_path = encoder_cfg.get("autoencoder") or ""
    if vae_path:
        missing, unexpected = load_matching(
            autoencoder, vae_path, prefixes=("first_stage_model.", "vae.", "module.")
        )
        logger.info(f"Loaded SDXL VAE. missing={missing} unexpected={unexpected}")

    model = SDXLForTraining(
        dit=dit,
        autoencoder=autoencoder,
        text_encoder=text_encoder,
        train_pipeline=SDXLTrainPipeline(
            resolution=resolution,
            snr_gamma=float(cfg.get("snr_gamma", 5.0)),
            scale_factor=float(cfg.get("scale_factor", 0.13025)),
        ),
        trainable_modules=cfg.get("trainable_modules") or "dit",
    )
    if hasattr(model, "freeze_except"):
        model.freeze_except()
    total, trainable = count_parameters(model)
    logger.info(f"sdxl parameters: total={total/1e9:.3f}B trainable={trainable/1e9:.3f}B")
    return model
