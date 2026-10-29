###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# HunyuanVideo text conditioning from the official prompt template: LLaMA
# hidden states (third from the last, cropped) and CLIP-L pooled features.

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from primus.backends.diffusion.amdb_benchmarks.download import tokenizer_dir
from primus.backends.diffusion.utils.hf_load import disable_safetensors_auto_conversion

VIDEO_PROMPT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)


class HunyuanTextEncoder(nn.Module):
    def __init__(self, llm_path: str, clip_path: str, *, crop_start: int = 95, text_len: int = 256):
        super().__init__()
        disable_safetensors_auto_conversion()
        self.crop_start = crop_start
        self.text_len = text_len
        self.llm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir(llm_path))
        self.llm = AutoModel.from_pretrained(llm_path).eval()
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        self.clip = CLIPTextModel.from_pretrained(clip_path).eval()
        for module in (self.llm, self.clip):
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.llm.parameters()).device
        templated = [VIDEO_PROMPT_TEMPLATE.format(prompt) for prompt in prompts]
        llm_tokens = self.llm_tokenizer(
            templated,
            padding="max_length",
            truncation=True,
            max_length=self.text_len + self.crop_start,
            return_tensors="pt",
        )
        llm_tokens = {key: value.to(device) for key, value in llm_tokens.items()}
        hidden = self.llm(**llm_tokens, output_hidden_states=True).hidden_states[-3]
        hidden = hidden[:, self.crop_start :]
        mask = llm_tokens["attention_mask"][:, self.crop_start :]

        clip_tokens = self.clip_tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        clip_tokens = {key: value.to(device) for key, value in clip_tokens.items()}
        pooled = self.clip(**clip_tokens).pooler_output
        return hidden, mask, pooled
