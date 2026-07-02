###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Register the Wan model builder."""

from __future__ import annotations

import glob
import os
from typing import Any

import torch
from safetensors.torch import load_file as safe_load_file

from primus.backends.diffusion.models.wan.adapter import WanForTraining
from primus.backends.diffusion.models.wan.components import WanComponents
from primus.backends.diffusion.models.wan.configuration_wanvideo import WanVideoConfig
from primus.backends.diffusion.models.wan.t5 import umt5_xxl_encoder_from_checkpoint
from primus.backends.diffusion.models.wan.train_pipeline import (
    WanFlowMatchTrainPipeline,
)
from primus.backends.diffusion.models.wan.vae2_1 import Wan2_1_VAE
from primus.backends.diffusion.models.wan.vae2_2 import Wan2_2_VAE
from primus.backends.diffusion.models.wan.wan_dit import WanModel as WanDiT
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import count_parameters


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return dict(safe_load_file(path))
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def _load_dit_weights_into_module(dit: torch.nn.Module, pretrained_path: str):
    """
    Support two common cases:
    1) Wan training export: `dit_model.safetensors` with keys like `blocks.0...`
    2) Official-ish export: `*model*.safetensors|bin` possibly with keys like `dit.blocks.0...` or `blocks.0...`
    """
    # Direct file
    if os.path.isfile(pretrained_path):
        state = _strip_module_prefix(_load_state_dict(pretrained_path))
        # Accept either `dit.*` or plain keys.
        if any(k.startswith("dit.") for k in state):
            state = {k[len("dit.") :]: v for k, v in state.items() if k.startswith("dit.")}
        result = dit.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded DiT from file. missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
        )
        return

    # Directory: prefer explicit Wan trainer file name, else fall back to pattern search
    candidates: list[str] = []
    for fname in ("dit_model.safetensors", "diffusion_pytorch_model.safetensors", "model.safetensors"):
        p = os.path.join(pretrained_path, fname)
        if os.path.exists(p):
            candidates.append(p)
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(pretrained_path, "*model*.safetensors")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(pretrained_path, "*model*.bin")))
    if not candidates:
        raise FileNotFoundError(f"No DiT weights found under {pretrained_path}")

    merged: dict[str, torch.Tensor] = {}
    for ckpt in candidates:
        part = _strip_module_prefix(_load_state_dict(ckpt))
        merged.update(part)

    if any(k.startswith("dit.") for k in merged):
        merged = {k[len("dit.") :]: v for k, v in merged.items() if k.startswith("dit.")}

    result = dit.load_state_dict(merged, strict=False)
    logger.info(
        f"Loaded DiT from dir. files={len(candidates)} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
    )


def _load_vae(vae: torch.nn.Module, ckpt_path: str | None):
    if not ckpt_path:
        return
    logger.info(f"Loading VAE from {ckpt_path}")
    state = _strip_module_prefix(_load_state_dict(ckpt_path))
    # VAE wrapper may keep real module under `.model`
    target = vae.model if hasattr(vae, "model") else vae
    result = target.load_state_dict(state, strict=False)
    logger.info(f"VAE loaded. missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")


def build_wan_model(model_config: dict):
    """
    YAML compatibility:
      model_config:
        name: wan
        load_from_pretrained_path: /path/to/Wan2.2-TI2V-5B   (optional)
        config: {...}                                       (optional overrides)
        encoder:
          t5_encoder: ...
          autoencoder: ...
    """
    encoder_cfg = model_config.get("encoder", {}) if isinstance(model_config.get("encoder"), dict) else {}
    cfg_dict: dict[str, Any] = dict(model_config.get("config", {}) or {})
    if encoder_cfg:
        cfg_dict.setdefault("encoder", encoder_cfg)

    # 1. Try to load config.json from pretrained path if available
    pretrained_path = model_config.get("load_from_pretrained_path")
    if pretrained_path:
        import json

        cfg_path = os.path.join(pretrained_path, "config.json")
        if os.path.exists(cfg_path):
            logger.info(f"Loading config from {cfg_path}")
            with open(cfg_path) as f:
                loaded_cfg = json.load(f)

            # Map WanModel checkpoint config keys to WanVideoConfig keys.
            mapping = {
                "dim": "dit_hidden_size",
                "num_layers": "dit_num_layers",
                "num_heads": "dit_num_heads",
                "ffn_dim": "dit_intermediate_size",
                "in_dim": "dit_in_channels",
                "out_dim": "dit_out_channels",
                "freq_dim": "dit_freq_dim",
                "text_len": "text_len",
            }

            for k, v in loaded_cfg.items():
                target_key = mapping.get(k)
                # Only infer values the user did not set explicitly in YAML.
                if target_key is not None and target_key not in cfg_dict:
                    cfg_dict[target_key] = v

    # Build a WanVideoConfig-compatible object to maximize reuse of existing fields.
    model_cfg = WanVideoConfig(**cfg_dict)

    # Build DiT (close to official Wan2.2 modules/model.py signature).
    # model_config.config.model_type: one of {"t2v","i2v","ti2v","s2v"}
    dit_task_type = getattr(model_cfg, "model_type", None)
    if dit_task_type not in ("t2v", "i2v", "ti2v", "s2v"):
        raise ValueError(
            "wan requires explicit `model_config.config.model_type` in YAML "
            "(one of: t2v / i2v / ti2v / s2v). "
            f"Got: {dit_task_type!r}"
        )
    dit = WanDiT(
        model_type=dit_task_type,
        patch_size=tuple(model_cfg.dit_patch_size),
        text_len=int(getattr(model_cfg, "text_len", 512)),
        in_dim=int(model_cfg.dit_in_channels),
        dim=int(model_cfg.dit_hidden_size),
        ffn_dim=int(model_cfg.dit_intermediate_size),
        freq_dim=int(model_cfg.dit_freq_dim),
        text_dim=int(model_cfg.dit_text_dim),
        out_dim=int(model_cfg.dit_out_channels),
        num_heads=int(model_cfg.dit_num_heads),
        num_layers=int(model_cfg.dit_num_layers),
        window_size=tuple(getattr(model_cfg, "dit_window_size", (-1, -1))),
        qk_norm=bool(getattr(model_cfg, "dit_qk_norm", True)),
        cross_attn_norm=bool(getattr(model_cfg, "dit_cross_attn_norm", True)),
        eps=float(model_cfg.dit_eps),
    )

    # Build VAE
    vae_type = getattr(model_cfg, "vae_type", "wan_video_vae_38")
    vae_ckpt = encoder_cfg.get("vae_checkpoint") or encoder_cfg.get("autoencoder")
    if vae_type in ("wan2.2", "wan_video_vae_38"):
        if not vae_ckpt:
            raise ValueError("wan requires `model_config.encoder.autoencoder` (Wan2.2 VAE checkpoint path)")
        vae = Wan2_2_VAE(z_dim=48, vae_pth=vae_ckpt)
        vae.upsampling_factor = 16
    else:
        if not vae_ckpt:
            raise ValueError("wan requires `model_config.encoder.autoencoder` (Wan2.1 VAE checkpoint path)")
        vae = Wan2_1_VAE(z_dim=16, vae_pth=vae_ckpt)
        vae.upsampling_factor = 8

    # Build text encoder (UMT5-XXL, encoder-only). Tokenization is handled by the dataset processor.
    t5_ckpt = encoder_cfg.get("t5_encoder")
    if not t5_ckpt:
        raise ValueError("wan requires `model_config.encoder.t5_encoder` (UMT5 encoder checkpoint path)")
    text_encoder = umt5_xxl_encoder_from_checkpoint(t5_ckpt, dtype=torch.bfloat16, device="cpu")

    # Optionally load pretrained DiT
    pretrained_path = model_config.get("load_from_pretrained_path")
    if pretrained_path:
        logger.info(f"Loading DiT weights from {pretrained_path}")
        _load_dit_weights_into_module(dit, pretrained_path)

    # Load encoders (VAE weights loaded into `.model`)
    _load_vae(vae, vae_ckpt)

    components = WanComponents(dit=dit, vae=vae, text_encoder=text_encoder, image_encoder=None)
    pipeline = WanFlowMatchTrainPipeline()

    model = WanForTraining(
        components=components,
        train_pipeline=pipeline,
        model_config=model_cfg,
        raw_config={
            "model_config": model_config,
            "resolved_model_cfg": getattr(model_cfg, "__dict__", cfg_dict),
        },
        trainable_modules=getattr(model_cfg, "trainable_modules", None),
    )

    total_params, trainable_params = count_parameters(model)
    logger.info(f"wan parameters: total={total_params/1e9:.3f}B trainable={trainable_params/1e9:.3f}B")
    if hasattr(model, "freeze_except"):
        model.freeze_except()
        total_params, trainable_params = count_parameters(model)
        logger.info(f"wan after freeze: total={total_params/1e9:.3f}B trainable={trainable_params/1e9:.3f}B")

    return model
