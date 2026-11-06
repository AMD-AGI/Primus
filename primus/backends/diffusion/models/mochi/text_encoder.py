###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Mochi conditions on T5-XXL token features, length 256, matching the official
# AsymmDiTJoint.

from __future__ import annotations

import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer

from primus.backends.diffusion.utils.hf_load import disable_safetensors_auto_conversion


class MochiTextEncoder(nn.Module):
    def __init__(self, t5_path: str, max_length: int = 256):
        super().__init__()
        disable_safetensors_auto_conversion()
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(t5_path, max_length=max_length)
        # google/t5-v1_1-xxl publishes pytorch shards and no model.safetensors.
        self.encoder = T5EncoderModel.from_pretrained(t5_path, use_safetensors=False).eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def forward(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.encoder.parameters()).device
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])[0]
        return hidden, tokens["attention_mask"].bool()
