###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Visual tower of the official Wan2.1 XLM-RoBERTa CLIP. Image-to-video uses the
# features from the penultimate vision block, shape [B, 257, 1280].

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WanClipVision(nn.Module):
    def __init__(self, checkpoint_path: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from primus.backends.diffusion.models.wan.clip import clip_xlm_roberta_vit_h_14

        model, _transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=dtype,
            device="cpu",
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state)
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.dtype = dtype

    def visual(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] in [-1, 1]. Returns [B, 257, 1280]."""
        size = (self.model.image_size, self.model.image_size)
        images = F.interpolate(images, size=size, mode="bicubic", align_corners=False)
        images = images.mul(0.5).add(0.5).clamp(0, 1)
        mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = images.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        images = (images - mean) / std
        return self.model.visual(images.to(dtype=self.dtype), use_31_block=True)
