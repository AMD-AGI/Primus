###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# SDXL conditions on the penultimate hidden states of CLIP-L and OpenCLIP-G,
# concatenated to 2048 channels, plus the CLIP-G pooled vector.

from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class SDXLTextEncoder(nn.Module):
    def __init__(self, clip_l: str, clip_g: str, max_length: int = 77):
        super().__init__()
        self.max_length = max_length
        self.tokenizer_l = CLIPTokenizer.from_pretrained(clip_l, max_length=max_length)
        self.tokenizer_g = CLIPTokenizer.from_pretrained(clip_g, max_length=max_length)
        self.clip_l = CLIPTextModel.from_pretrained(clip_l).eval()
        # OpenCLIP-G publishes pytorch_model.bin shards and no model.safetensors.
        # Leaving safetensors detection on asks Hugging Face to convert the repo,
        # and that service responds without the event id Transformers expects.
        self.clip_g = CLIPTextModelWithProjection.from_pretrained(clip_g, use_safetensors=False).eval()
        for module in (self.clip_l, self.clip_g):
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.clip_l.parameters()).device
        tokens_l = self.tokenizer_l(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens_g = self.tokenizer_g(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens_l = {key: value.to(device) for key, value in tokens_l.items()}
        tokens_g = {key: value.to(device) for key, value in tokens_g.items()}
        hidden_l = self.clip_l(**tokens_l, output_hidden_states=True).hidden_states[-2]
        output_g = self.clip_g(**tokens_g, output_hidden_states=True)
        hidden_g = output_g.hidden_states[-2]
        pooled = output_g.text_embeds
        return torch.cat([hidden_l, hidden_g], dim=-1), pooled
