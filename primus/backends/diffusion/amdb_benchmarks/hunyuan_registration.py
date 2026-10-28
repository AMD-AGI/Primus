###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Build the official HunyuanVideo transformer for Primus post-training."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from primus.backends.diffusion.amdb_benchmarks.checkpoint_convert import (
    hunyuan_diffusers_to_native,
    is_diffusers_hunyuan,
)
from primus.backends.diffusion.amdb_benchmarks.checkpoint_io import (
    load_state_dict,
    strip_prefixes,
)
from primus.backends.diffusion.amdb_benchmarks.download import materialize_pretrained
from primus.backends.diffusion.amdb_benchmarks.hunyuan.adapter import (
    HunyuanVideoForTraining,
)
from primus.backends.diffusion.amdb_benchmarks.hunyuan.models import (
    HUNYUAN_VIDEO_CONFIG,
    HYVideoDiffusionTransformer,
)
from primus.backends.diffusion.amdb_benchmarks.hunyuan.text_encoder import (
    HunyuanTextEncoder,
)
from primus.backends.diffusion.amdb_benchmarks.hunyuan.train_pipeline import (
    HunyuanTrainPipeline,
)
from primus.backends.diffusion.amdb_benchmarks.sampling import amdb_flow_sigma_grid
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import count_parameters


def build_hunyuan_model(model_config: dict):
    cfg = dict(model_config.get("config") or {})
    encoder_cfg = dict(model_config.get("encoder") or {})
    variant = str(cfg.get("variant") or "HYVideo-T/2-cfgdistill")
    preset = dict(HUNYUAN_VIDEO_CONFIG[variant])
    args = SimpleNamespace(text_states_dim=4096, text_states_dim_2=768)
    dit = HYVideoDiffusionTransformer(
        args,
        in_channels=int(cfg.get("in_channels", 16)),
        out_channels=int(cfg.get("out_channels", 16)),
        **preset,
    )
    if os.environ.get("BENCHMARK_DENOISER_ONLY") == "1":
        text_encoder = torch.nn.Identity()
    else:
        llm_path = encoder_cfg.get("llm") or encoder_cfg.get("text_encoder")
        clip_path = (
            encoder_cfg.get("clip") or encoder_cfg.get("clip_encoder") or "openai/clip-vit-large-patch14"
        )
        if not llm_path:
            raise ValueError("hunyuan-video requires model.encoder.llm (the official LLaMA text encoder).")
        text_encoder = HunyuanTextEncoder(llm_path, clip_path)

    vae_path = encoder_cfg.get("autoencoder") or os.environ.get("VAE_CHECKPOINT")
    vae = None
    if vae_path:
        from diffusers import AutoencoderKLHunyuanVideo

        vae = AutoencoderKLHunyuanVideo.from_pretrained(vae_path).eval()
        vae.requires_grad_(False)

    pretrained = materialize_pretrained("hunyuan-video", model_config.get("load_from_pretrained_path") or "")
    if pretrained:
        state = strip_prefixes(load_state_dict(pretrained), ("module.", "model."))
        if is_diffusers_hunyuan(state):
            state = hunyuan_diffusers_to_native(state)
        result = dit.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded HunyuanVideo transformer. missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"HunyuanVideo checkpoint mismatch: missing={result.missing_keys[:5]} "
                f"unexpected={result.unexpected_keys[:5]}"
            )

    resolution = str(model_config.get("resolution") or "960,544")
    width, height = (int(part) for part in resolution.split(","))
    model = HunyuanVideoForTraining(
        dit=dit,
        text_encoder=text_encoder,
        vae=vae,
        # HunyuanVideo's FlowMatchEulerDiscreteScheduler pre-shifts its grid (shift=7.0,
        # use_dynamic_shifting=False) before AMDiffusionBenchmark applies its own shift.
        train_pipeline=HunyuanTrainPipeline(
            guidance=float(cfg.get("guidance", 1000.0)),
            sigma_grid=amdb_flow_sigma_grid(
                shift=float(cfg.get("shift", 3.0)),
                scheduler_shift=float(cfg.get("scheduler_shift", 7.0)),
            ),
            width=width,
            height=height,
            num_frames=int(model_config.get("num_frames") or 9),
        ),
    )
    model.freeze_except()
    total, trainable = count_parameters(model)
    logger.info(f"hunyuan-video parameters: total={total/1e9:.3f}B trainable={trainable/1e9:.3f}B")
    return model
