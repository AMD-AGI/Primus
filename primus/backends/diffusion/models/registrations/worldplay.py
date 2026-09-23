###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Register the in-tree HY-WorldPlay model."""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from primus.backends.diffusion.models.worldplay import (
    ARHunyuanVideo_1_5_DiffusionTransformer,
    WorldPlayARTrainPipeline,
    WorldPlayForTraining,
)
from primus.backends.diffusion.utils.log import logger
from primus.backends.diffusion.utils.train_utils import count_parameters

def build_worldplay_model(model_config: dict):
    base_path = model_config.get("load_from_pretrained_path")
    action_path = model_config.get("action_checkpoint")
    for name, value in (
        ("load_from_pretrained_path", base_path),
        ("action_checkpoint", action_path),
    ):
        if not value:
            raise ValueError(f"worldplay requires model.config.{name}")

    logger.info(f"Loading HunyuanVideo-1.5 AR base from {base_path}")
    with torch.device("cpu"):
        dit = ARHunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            str(base_path),
            local_attn_size=int(model_config.get("local_attn_size", -1)),
            sink_size=int(model_config.get("sink_size", 0)),
        )
    dit.add_discrete_action_parameters()
    logger.info(f"Overlaying HY-WorldPlay action checkpoint from {action_path}")
    state_dict = load_file(str(action_path), device="cpu")
    incompatible = dit.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "WorldPlay checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    dit.to(dtype=torch.float32)

    recipe = model_config.get("recipe", {}) or {}
    model = WorldPlayForTraining(
        dit=dit,
        train_pipeline=WorldPlayARTrainPipeline(
            train_time_shift=float(recipe.get("train_time_shift", 3.0)),
            logit_mean=float(recipe.get("logit_mean", 0.0)),
            logit_std=float(recipe.get("logit_std", 1.0)),
        ),
        raw_config=model_config,
    )
    total, trainable = count_parameters(model)
    logger.info(
        f"worldplay parameters: total={total / 1e9:.3f}B "
        f"trainable={trainable / 1e9:.3f}B"
    )
    return model
