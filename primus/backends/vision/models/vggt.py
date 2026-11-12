###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""VGGT wrapped for the Primus vision trainer.

The module, heads, and multitask loss live in ``vggt_model``. This class
normalizes cameras the way VGGT's trainer does, then returns one scalar loss.
"""

from __future__ import annotations

import torch

from primus.backends.vision.models.interface import VisionModelForTraining

_PRETRAINED_SKIP = {"", "none", "null", "random"}


def _enabled(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class VGGTForTraining(VisionModelForTraining):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        from primus.backends.vision.attention import install_aiter_attention
        import primus.backends.vision.vggt_model.layers.attention as vggt_attention

        install_aiter_attention(vggt_attention)
        from primus.backends.vision.vggt_model.loss import MultitaskLoss
        from primus.backends.vision.vggt_model.models.vggt import VGGT

        pretrained = str(config.get("pretrained") or "").strip()
        enable_camera = _enabled(config, "enable_camera", True)
        enable_depth = _enabled(config, "enable_depth", True)
        enable_point = _enabled(config, "enable_point", False)
        enable_track = _enabled(config, "enable_track", False)
        if pretrained.lower() in _PRETRAINED_SKIP:
            self.vggt = VGGT(
                img_size=int(config.get("img_size", 518)),
                patch_size=int(config.get("patch_size", 14)),
                embed_dim=int(config.get("embed_dim", 1024)),
                enable_camera=enable_camera,
                enable_point=enable_point,
                enable_depth=enable_depth,
                enable_track=enable_track,
            )
        else:
            self.vggt = VGGT.from_pretrained(pretrained)
            if not enable_point:
                self.vggt.point_head = None
            if not enable_track:
                self.vggt.track_head = None
            if not enable_depth:
                self.vggt.depth_head = None
            if not enable_camera:
                self.vggt.camera_head = None

        self.loss_fn = MultitaskLoss(
            camera={"weight": 5.0, "loss_type": "l1"} if enable_camera else None,
            depth={
                "weight": 1.0,
                "gradient_loss_fn": "grad",
                "valid_range": 0.98,
            }
            if enable_depth
            else None,
            point={
                "weight": 1.0,
                "gradient_loss_fn": "normal",
                "valid_range": 0.98,
            }
            if enable_point
            else None,
            track=None,
        )
        frozen = [str(name) for name in (config.get("frozen_modules") or []) if str(name).strip()]
        self._frozen = frozen
        self.find_unused_parameters = False

    def block_classes(self) -> tuple[type, ...]:
        from primus.backends.vision.vggt_model.layers.block import Block

        return (Block,)

    def param_groups(self) -> list[dict]:
        for name, param in self.vggt.named_parameters():
            param.requires_grad = not any(token in name for token in self._frozen)
        params = [param for param in self.vggt.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("VGGT has no trainable parameters. Check frozen_modules.")
        return [{"params": params}]

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        return self.vggt.state_dict()

    def forward(self, batch) -> torch.Tensor:
        from primus.backends.vision.vggt_model.train_utils.normalization import (
            normalize_camera_extrinsics_and_points_batch,
        )

        images = batch["images"]
        extrinsics, cam_points, world_points, depths = normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )
        device = next(self.parameters()).device
        prepared = {
            "images": images.to(device, non_blocking=True),
            "extrinsics": extrinsics.to(device, non_blocking=True),
            "intrinsics": batch["intrinsics"].to(device, non_blocking=True),
            "cam_points": cam_points.to(device, non_blocking=True),
            "world_points": world_points.to(device, non_blocking=True),
            "depths": depths.to(device, non_blocking=True),
            "point_masks": batch["point_masks"].to(device, non_blocking=True),
        }
        predictions = self.vggt(images=prepared["images"])
        loss_dict = self.loss_fn(predictions, prepared)
        loss = loss_dict["objective"]
        metrics: dict[str, torch.Tensor | float] = {
            "num_images": float(prepared["images"].shape[0] * prepared["images"].shape[1])
        }
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                metrics[key] = value.detach()
        self.last_metrics = metrics
        return loss


def build_vggt_model(model_config: dict) -> VGGTForTraining:
    return VGGTForTraining(model_config)
