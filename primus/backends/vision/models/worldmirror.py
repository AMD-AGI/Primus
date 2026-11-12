###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""WorldMirror wrapped for the Primus vision trainer.

The network and the loss modules live in the in-tree ``worldmirror`` package.
This class samples the conditioning flags, runs the loss container, and exposes
the stage-1 and stage-2 parameter groups. It does not start Lightning.
"""

from __future__ import annotations

import gc
import os
import random

import torch

from primus.backends.vision.models.interface import VisionModelForTraining

_PRETRAINED_SKIP = {"", "none", "null", "random"}
_GROUP_DEFAULTS = {
    "pretrained": 2.0e-5,
    "backbone": 1.0e-4,
    "new": 2.0e-4,
}


def _enabled(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _load_checkpoint(model, path: str) -> None:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        incoming = load_file(path)
    elif path.endswith(".ckpt") or path.endswith(".pt") or path.endswith(".pth"):
        incoming = torch.load(path, map_location="cpu")
        if isinstance(incoming, dict) and "state_dict" in incoming:
            incoming = incoming["state_dict"]
            incoming = {key.replace("model.", "", 1): value for key, value in incoming.items()}
    else:
        raise ValueError(f"Unsupported WorldMirror checkpoint format: {path}")

    current = model.state_dict()
    for key, value in current.items():
        loaded = incoming.get(key)
        if loaded is not None and loaded.shape == value.shape:
            current[key] = loaded
    model.load_state_dict(current, strict=False)
    del incoming, current
    gc.collect()


class WorldMirrorForTraining(VisionModelForTraining):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        from primus.backends.vision.attention import install_aiter_attention
        import primus.backends.vision.worldmirror.src.models.layers.attention as wm_attention

        install_aiter_attention(wm_attention)
        from primus.backends.vision.worldmirror.src.models.models.worldmirror import WorldMirror
        from primus.backends.vision.worldmirror.training.losses.camera import CameraLoss
        from primus.backends.vision.worldmirror.training.losses.container import LossContainer
        from primus.backends.vision.worldmirror.training.losses.depth import DepthLoss
        from primus.backends.vision.worldmirror.training.losses.normal import NormalLoss
        from primus.backends.vision.worldmirror.training.losses.point import PointLoss

        enable_cam = _enabled(config, "enable_cam", True)
        enable_pts = _enabled(config, "enable_pts", True)
        enable_depth = _enabled(config, "enable_depth", True)
        enable_norm = _enabled(config, "enable_norm", True)
        enable_gs = _enabled(config, "enable_gs", False)
        enable_cond = _enabled(config, "enable_cond", True)
        self.enable_cond = enable_cond
        self._frozen = [str(name) for name in (config.get("freeze_params") or [])]
        self._group_lrs = {
            "pretrained": float(config.get("pretrained_lr", _GROUP_DEFAULTS["pretrained"])),
            "backbone": float(config.get("backbone_lr", _GROUP_DEFAULTS["backbone"])),
            "new": float(config.get("new_lr", _GROUP_DEFAULTS["new"])),
        }
        self._pretrained_names = list(
            config.get("pretrained_params") or ["visual_geometry_transformer.patch_embed"]
        )
        self._new_names = list(
            config.get("new_params")
            or ["normal_head", "gs_head", "gs_renderer", "depth_embed", "rays_embed", "camera_embed"]
        )
        flags = dict(
            img_size=int(config.get("img_size", 518)),
            patch_size=int(config.get("patch_size", 14)),
            embed_dim=int(config.get("embed_dim", 1024)),
            gs_dim=int(config.get("gs_dim", 256)),
            enable_cam=enable_cam,
            enable_pts=enable_pts,
            enable_depth=enable_depth,
            enable_norm=enable_norm,
            enable_gs=enable_gs,
            enable_cond=enable_cond,
        )
        pretrained = str(config.get("pretrained") or "").strip()
        if pretrained.lower() in _PRETRAINED_SKIP:
            self.wm = _construct_worldmirror(WorldMirror, flags)
        elif pretrained.endswith((".safetensors", ".ckpt", ".pt", ".pth")):
            self.wm = _construct_worldmirror(WorldMirror, flags)
            _load_checkpoint(self.wm, pretrained)
        else:
            self.wm = _construct_worldmirror(WorldMirror, flags)
            _load_hub_weights(self.wm, pretrained)

        losses = [
            CameraLoss(),
            PointLoss(gamma=1.0, alpha=0.2, gradient_loss_fn="normal", valid_range=0.98),
            DepthLoss(gamma=1.0, alpha=0.2, gradient_loss_fn="grad", valid_range=0.98),
            NormalLoss(loss_type="AL", real_weight=1.0, pseudo_weight=1.0, ignore_datasets=[]),
        ]
        weights = [5.0, 1.0, 1.0, 1.0]
        if enable_gs:
            from primus.backends.vision.worldmirror.training.losses.render import (
                GSDepthLoss,
                RenderDepthLoss,
                RenderMSELoss,
                RenderPerceptualLoss,
            )

            losses.extend(
                [
                    GSDepthLoss(gamma=1.0, alpha=0.2, gradient_loss_fn="grad", valid_range=0.98),
                    RenderMSELoss(mask_mode="target_mask"),
                    RenderPerceptualLoss(mask_mode="target_mask"),
                    RenderDepthLoss(
                        depth_head_sup=True,
                        depth_conf_mask=True,
                        gamma=1.0,
                        alpha=0.2,
                        gradient_loss_fn="grad",
                        valid_range=0.98,
                        mask_mode="context_mask + target_mask",
                    ),
                ]
            )
            weights.extend([0.1, 1.0, 0.05, 0.1])
        self.criterion = LossContainer(
            enable_cam=enable_cam,
            enable_pts=enable_pts,
            enable_depth=enable_depth,
            enable_norm=enable_norm,
            enable_gs=enable_gs,
            losses=losses,
            weights=weights,
        )
        self.find_unused_parameters = True

    def block_classes(self) -> tuple[type, ...]:
        from primus.backends.vision.worldmirror.src.models.layers.block import Block

        return (Block,)

    def param_groups(self) -> list[dict]:
        for name, param in self.wm.named_parameters():
            param.requires_grad = not any(token in name for token in self._frozen)
        pretrained, backbone, new = [], [], []
        for name, param in self.wm.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in self._new_names):
                new.append(param)
            elif any(token in name for token in self._pretrained_names):
                pretrained.append(param)
            else:
                backbone.append(param)
        groups = []
        if pretrained:
            groups.append(
                {"params": pretrained, "lr": self._group_lrs["pretrained"], "weight_decay": 0.1}
            )
        if backbone:
            groups.append({"params": backbone, "lr": self._group_lrs["backbone"], "weight_decay": 0.1})
        if new:
            groups.append({"params": new, "lr": self._group_lrs["new"], "weight_decay": 0.1})
        if not groups:
            raise RuntimeError("WorldMirror has no trainable parameters. Check freeze_params.")
        return groups

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        return self.wm.state_dict()

    def _condition_flags(self) -> list[int]:
        if not self.enable_cond:
            return [0, 0, 0]
        return [random.randint(0, 1) for _ in range(3)]

    def forward(self, batch) -> torch.Tensor:
        from primus.backends.vision.worldmirror.training.utils.misc import compose_batches_from_list

        device = next(self.parameters()).device
        if isinstance(batch, dict):
            views = batch
        else:
            views = compose_batches_from_list(batch, device=device)
        predictions = self.wm(views, cond_flags=self._condition_flags(), is_inference=False)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
            loss, loss_dict = self.criterion(views, predictions)
        images = views["img"]
        metrics: dict[str, torch.Tensor | float] = {"num_images": float(images.shape[0] * images.shape[1])}
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                metrics[key] = value.detach()
        metrics["loss"] = loss.detach()
        self.last_metrics = metrics
        return loss


def _construct_worldmirror(model_cls, flags: dict):
    try:
        return model_cls(**flags)
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") == "gsplat" or "gsplat" in str(exc):
            raise RuntimeError(
                "WorldMirror's Gaussian head needs the gsplat package. "
                "Set VISION_BUILD_GSPLAT=1 so the install hook builds it."
            ) from exc
        raise


def _load_hub_weights(model, repo_id: str) -> None:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(repo_id=repo_id))
    matches = list(folder.glob("*.safetensors")) or list(folder.glob("*.bin")) or list(folder.glob("*.pt"))
    if not matches:
        raise FileNotFoundError(f"No weight file in Hugging Face repo {repo_id}")
    _load_checkpoint(model, str(matches[0]))


def build_worldmirror_model(model_config: dict) -> WorldMirrorForTraining:
    return WorldMirrorForTraining(model_config)
