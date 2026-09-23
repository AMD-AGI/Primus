"""Hunyuan VAE / Qwen / ByT5 / SigLIP latent extraction for WorldPlay SFT.

Modified from Tencent HY-WorldPlay preprocessing source by AMD in 2026:
imports are redirected to the in-tree, preprocessing-only source closure.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import torch

_VENDOR = str(Path(__file__).resolve().parent / "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import (  # noqa: E402
    AutoencoderKLConv3D,
)
from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder  # noqa: E402
from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2  # noqa: E402
from hyvideo.models.text_encoders.byT5.format_prompt import (  # noqa: E402
    MultilingualPromptFormat,
)
from hyvideo.models.vision_encoder import VisionEncoder  # noqa: E402


class LatentExtractor:
    def __init__(self, checkpoint_path, device, target_size=(480, 832)):
        self.device = device
        self.target_size = target_size
        self._load_models(checkpoint_path)

    def _load_models(self, checkpoint_path):
        print(f"Loading encoders from {checkpoint_path}...")
        siglip_root = os.path.join(checkpoint_path, "vision_encoder/siglip")
        glyph_ckpt = os.path.join(
            checkpoint_path, "text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt"
        )
        missing = []
        if not os.path.isdir(os.path.join(siglip_root, "image_encoder")):
            missing.append(
                f"{siglip_root}/image_encoder  (gated FLUX.1-Redux-dev SigLIP)"
            )
        if not os.path.isfile(glyph_ckpt):
            missing.append(glyph_ckpt + "  (ModelScope Glyph-SDXL-v2)")
        if missing:
            raise FileNotFoundError(
                "Hunyuan encode extras are missing. Hub caches for Qwen/ByT5 "
                "are not enough. Download them with:\n"
                "  export HF_TOKEN=... HF_HOME=/data/models "
                "HF_HUB_CACHE=/data/models/hub\n"
                "  python examples/diffusion/scripts/download_worldplay_models.py --hf_token $HF_TOKEN\n"
                "Missing:\n  - " + "\n  - ".join(missing)
            )
        self.vae = (
            AutoencoderKLConv3D.from_pretrained(
                os.path.join(checkpoint_path, "vae"), torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        self.vision_encoder = VisionEncoder(
            vision_encoder_type="siglip",
            vision_encoder_precision="fp16",
            vision_encoder_path=siglip_root,
            processor_type=None,
            processor_path=None,
            output_key=None,
            logger=None,
            device=self.device,
        )
        self.text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=os.path.join(checkpoint_path, "text_encoder/llm"),
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=self.device,
        )
        self.text_len = self.text_encoder.max_length

        load_from = os.path.join(checkpoint_path, "text_encoder")
        glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
        byt5_args = dict(
            byT5_google_path=os.path.join(load_from, "byt5-small"),
            byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
            multilingual_prompt_format_color_path=os.path.join(
                glyph_root, "assets/color_idx.json"
            ),
            multilingual_prompt_format_font_path=os.path.join(
                glyph_root, "assets/multilingual_10-lang_idx.json"
            ),
            byt5_max_length=256,
        )
        byt5_kwargs = load_glyph_byT5_v2(
            byt5_args,
            device=(
                f"cuda:{self.device}"
                if isinstance(self.device, int)
                else str(self.device)
            ),
        )
        self.prompt_format = MultilingualPromptFormat(
            font_path=byt5_args["multilingual_prompt_format_font_path"],
            color_path=byt5_args["multilingual_prompt_format_color_path"],
        )
        self.byt5_model = byt5_kwargs["byt5_model"]
        self.byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]
        self.byt5_max_length = byt5_kwargs["byt5_max_length"]
        print("Encoders loaded.")

    def _process_byt5_prompt(self, prompt_text):
        byt5_embeddings = torch.zeros(
            (1, self.byt5_max_length, 1472), device=self.device
        )
        byt5_mask = torch.zeros(
            (1, self.byt5_max_length), device=self.device, dtype=torch.int64
        )
        pattern = r'\"(.*?)\"|"(.*?)"'
        matches = re.findall(pattern, prompt_text)
        glyph_texts = [m[0] or m[1] for m in matches]
        glyph_texts = list(dict.fromkeys(glyph_texts)) if len(glyph_texts) > 1 else glyph_texts
        if glyph_texts:
            text_styles = [{"color": None, "font-family": None} for _ in glyph_texts]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            inputs = self.byt5_tokenizer(
                formatted_text,
                padding="max_length",
                max_length=self.byt5_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_ids = inputs.input_ids.to(self.device)
            text_mask = inputs.attention_mask.to(self.device)
            byt5_embeddings = self.byt5_model(text_ids, attention_mask=text_mask.float())[0]
            byt5_mask = text_mask
        return byt5_embeddings, byt5_mask

    @torch.no_grad()
    def encode_caption(self, caption):
        text_inputs = self.text_encoder.text2tokens(
            caption, data_type="video", max_length=self.text_len
        )
        prompt_outputs = self.text_encoder.encode(
            text_inputs, data_type="video", device=self.device
        )
        prompt_embeds = prompt_outputs.hidden_state.to(
            dtype=self.text_encoder.dtype, device=self.device
        )
        attention_mask = (
            prompt_outputs.attention_mask.to(self.device)
            if prompt_outputs.attention_mask is not None
            else None
        )
        byt5_embeddings, byt5_masks = self._process_byt5_prompt(caption)
        return prompt_embeds, attention_mask, byt5_embeddings, byt5_masks
