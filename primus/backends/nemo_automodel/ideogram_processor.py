###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ideogram-4 preprocessing (no-fork, Primus-side): Flux-2 VAE image encode +
Qwen3-VL 13-layer text tap -> the cache the :class:`Ideogram4Adapter` consumes.

WHY (no Automodel/diffusers fork):
  AutoModel's processors live in the submodule (``tools/diffusion/processors``) and
  register into a closed ``ProcessorRegistry``. Rather than edit the submodule, this
  is a standalone Primus-side processor that produces a flat per-sample cache read
  by :class:`Ideogram4CacheDataloaderConfig` (``ideogram_cache_data.py``).

WHAT it reproduces (ported verbatim from the diffusers Ideogram-4 pipeline so the
cached conditioning matches training==inference):
  - IMAGE: ``AutoencoderKLFlux2`` encode -> 32-ch latent -> 2x2 patch pack to
    ``[128, gh, gw]`` (128 = ae*patch*patch, channel order ``(pa, pb, ae)`` matching
    the pipeline's decode ``view(B, gh, gw, patch, patch, ae)``), then the VAE
    ``bn`` normalisation on the packed-128 channels. This is exactly the space the
    ``FlowMatchingPipeline`` adds noise in.
  - TEXT: chat-template tokenise -> run the Qwen3-VL ``language_model`` decoder
    stack -> tap hidden states at ``QWEN3_VL_ACTIVATION_LAYERS`` (0,3,...,33,35) ->
    concat to ``[n, 53248]`` (= 4096 x 13, layer fastest). Stored non-padded (only
    the ``n`` real tokens); the cache loader left-pads per batch to match the
    adapter's ``[left-pad][text][image]`` layout.

The nf4 Ideogram text encoder needs bitsandbytes (absent from the pinned ROCm
container), so we tap the full-precision base ``Qwen/Qwen3-VL-8B-Instruct`` (same
architecture). The VAE is loaded from the (non-quantized) Ideogram ``vae/`` subfolder
so the bn stats are exact.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Qwen3-VL decoder layers tapped for the packed text conditioning (from the diffusers
# Ideogram-4 pipeline: QWEN3_VL_ACTIVATION_LAYERS).
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

# Default HF sources (see scripts/b3_download_weights.py).
DEFAULT_VAE_SOURCE = "ideogram-ai/ideogram-4-nf4-diffusers"
DEFAULT_TEXT_ENCODER_SOURCE = "Qwen/Qwen3-VL-8B-Instruct"


def _load_autoencoder_kl_flux2():
    try:
        from diffusers import AutoencoderKLFlux2  # type: ignore

        return AutoencoderKLFlux2
    except Exception:  # pragma: no cover
        from diffusers.models.autoencoders import AutoencoderKLFlux2  # type: ignore

        return AutoencoderKLFlux2


class Ideogram4Processor:
    """Encode (image, caption) pairs into the Ideogram-4 training cache format."""

    def __init__(
        self,
        *,
        vae_source: str = DEFAULT_VAE_SOURCE,
        vae_subfolder: str = "vae",
        text_encoder_source: str = DEFAULT_TEXT_ENCODER_SOURCE,
        tokenizer_source: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.vae_source = vae_source
        self.vae_subfolder = vae_subfolder
        self.text_encoder_source = text_encoder_source
        self.tokenizer_source = tokenizer_source or text_encoder_source
        self.device = device
        self.dtype = dtype

        self.vae = None
        self.patch_size = 2
        self.bn_mean = None  # [1, 1, 1, 128] float32
        self.bn_std = None  # [1, 1, 1, 128] float32
        self.text_encoder = None
        self.language_model = None
        self.tokenizer = None

    # ------------------------------------------------------------------ load
    def load_models(self) -> None:
        from transformers import AutoTokenizer, Qwen3VLModel

        AutoencoderKLFlux2 = _load_autoencoder_kl_flux2()

        logger.info("[Ideogram4Processor] loading VAE (%s/%s)", self.vae_source, self.vae_subfolder)
        vae = AutoencoderKLFlux2.from_pretrained(
            self.vae_source, subfolder=self.vae_subfolder, torch_dtype=self.dtype
        )
        vae = vae.to(self.device).eval()
        self.vae = vae

        # BN running stats on the packed-128 channel space (see pipeline decode).
        eps = float(getattr(vae.config, "batch_norm_eps", 1e-5))
        self.bn_mean = vae.bn.running_mean.detach().view(1, 1, 1, -1).float().to(self.device)
        self.bn_std = torch.sqrt(vae.bn.running_var.detach() + eps).view(1, 1, 1, -1).float().to(self.device)
        packed_ch = int(self.bn_mean.shape[-1])
        ae = int(getattr(vae.config, "latent_channels", packed_ch // 4))
        self.patch_size = int(round((packed_ch / ae) ** 0.5))
        assert ae * self.patch_size * self.patch_size == packed_ch, (
            f"packed channels {packed_ch} != ae({ae}) * patch^2({self.patch_size}^2)"
        )
        logger.info(
            "[Ideogram4Processor] VAE ready: ae=%d patch=%d packed=%d", ae, self.patch_size, packed_ch
        )

        logger.info("[Ideogram4Processor] loading text encoder (%s)", self.text_encoder_source)
        te = Qwen3VLModel.from_pretrained(self.text_encoder_source, torch_dtype=self.dtype)
        # We only tap the language model; drop the vision tower to save memory if present.
        if hasattr(te, "visual"):
            try:
                te.visual = None
            except Exception:  # pragma: no cover
                pass
        te = te.to(self.device).eval()
        self.text_encoder = te
        self.language_model = te.language_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_source)
        logger.info("[Ideogram4Processor] models loaded.")

    # ------------------------------------------------------------------ image
    @staticmethod
    def preprocess_image(image: Image.Image, resolution: int) -> torch.Tensor:
        """PIL RGB image -> resize+center-crop to (resolution, resolution), [-1,1], [1,3,H,W]."""
        image = image.convert("RGB")
        w, h = image.size
        scale = max(resolution / w, resolution / h)
        rw, rh = int(round(w * scale)), int(round(h * scale))
        image = image.resize((rw, rh), Image.LANCZOS)
        left = (rw - resolution) // 2
        top = (rh - resolution) // 2
        image = image.crop((left, top, left + resolution, top + resolution))
        arr = torch.from_numpy(np.array(image)).float() / 255.0  # [H,W,3]
        arr = (arr - 0.5) / 0.5
        return arr.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Image [1,3,H,W] in [-1,1] -> packed+BN latent [128, gh, gw] (fp16).

        Packing matches the diffusers Ideogram-4 decode inverse exactly:
        raw[ae, 2i+pa, 2j+pb] -> channel ((pa*patch)+pb)*ae + ae_idx of token (i,j).
        """
        vae = self.vae
        patch = self.patch_size
        x = image_tensor.to(self.device, dtype=self.dtype)
        raw = vae.encode(x).latent_dist.mode()  # [1, ae, H8, W8]
        raw = raw.float()
        b, ae, H, W = raw.shape
        gh, gw = H // patch, W // patch
        x6 = raw.view(b, ae, gh, patch, gw, patch)
        # -> [b, gh, gw, pa, pb, ae] -> [b, gh, gw, 128] (channel order (pa, pb, ae))
        packed = x6.permute(0, 2, 4, 3, 5, 1).reshape(b, gh, gw, patch * patch * ae)
        packed = (packed - self.bn_mean) / self.bn_std
        latents = packed.permute(0, 3, 1, 2).squeeze(0)  # [128, gh, gw]
        return latents.detach().cpu().to(torch.float16)

    # ------------------------------------------------------------------ text
    @torch.no_grad()
    def _tap_hidden_states(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor, pos_1d: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run the Qwen3-VL decoder stack, returning hidden states at each tap layer.

        Ported from Ideogram4Pipeline._get_text_encoder_hidden_states.
        """
        from transformers.masking_utils import create_causal_mask

        lm = self.language_model
        inputs_embeds = lm.embed_tokens(token_ids)

        position_ids_4d = pos_1d[None, ...].expand(4, pos_1d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=lm.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = lm.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: Dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(lm.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states
        return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

    @torch.no_grad()
    def encode_text(self, prompt: str, max_text_tokens: int) -> Optional[Dict[str, Any]]:
        """Caption -> {llm_features [n, 53248] fp16, text_length n}. Returns None if too long.

        No padding: a single sequence of exactly ``n`` real tokens is tapped, so the
        cached features are the non-pad rows the adapter/cache-loader left-pad later.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        toks = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        n = int(toks.shape[0])
        if n == 0 or n > max_text_tokens:
            return None

        token_ids = toks.view(1, n).to(self.device)
        attention_mask = torch.ones(1, n, dtype=torch.long, device=self.device)
        pos_1d = torch.arange(n, device=self.device).view(1, n)

        selected = self._tap_hidden_states(token_ids, attention_mask, pos_1d)
        # [13, 1, n, 4096] -> [1, n, 4096, 13] -> [1, n, 53248]  (layer fastest)
        feats = torch.stack(selected, dim=0).permute(1, 2, 3, 0).reshape(1, n, -1)
        feats = feats.squeeze(0).to(torch.float16).detach().cpu()  # [n, 53248]
        return {"llm_features": feats, "text_length": n}

    # ------------------------------------------------------------------ cache
    @staticmethod
    def get_cache_data(
        image_latents: torch.Tensor,
        text_enc: Dict[str, Any],
        *,
        prompt: str,
        image_path: str,
    ) -> Dict[str, Any]:
        gh, gw = int(image_latents.shape[1]), int(image_latents.shape[2])
        return {
            "image_latents": image_latents,  # [128, gh, gw] fp16
            "llm_features": text_enc["llm_features"],  # [n, 53248] fp16
            "text_length": int(text_enc["text_length"]),
            "grid_h": gh,
            "grid_w": gw,
            "prompt": prompt,
            "image_path": image_path,
            "model_type": "ideogram4",
        }
