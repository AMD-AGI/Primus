###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ideogram-4 offline preprocessing: image encode plus the layered text tap.

Produces the per-sample cache the dataloader reads. AutoModel's own processors
live in the submodule and register into a closed registry, so rather than edit it
this is a standalone processor whose output is a flat directory of files -- which
also means the cache outlives any change to either side's interfaces.

EVERYTHING HERE IS AN INVERSE OF SOMETHING IN THE INFERENCE PIPELINE, and that is
the whole correctness story. The cached conditioning has to be what the model
would see at sampling time, or training optimizes for inputs it will never
receive. Two places this is load-bearing and easy to get subtly wrong:

  THE LATENT PACKING ORDER. The autoencoder produces a latent with some number of
  channels; the model works on 2x2 patches of it flattened into the channel
  dimension. Which of the several possible orderings is correct is fixed by the
  pipeline's decode, which reshapes to (grid, grid, patch, patch, channels) -- so
  the packing here must permute to that same order, with the autoencoder channel
  varying fastest. A different-but-plausible order gives latents that decode to
  scrambled images, and since training never decodes, nothing would reveal it.
  The patch size is derived from the channel counts rather than assumed, and
  asserted, so a model with different dimensions fails here rather than silently
  packing wrong.

  THE NORMALIZATION SPACE. The batch-norm statistics are applied on the PACKED
  channels, not the raw ones, because that is the space the flow-matching
  pipeline adds noise in. Normalizing before packing would be a different
  transform.

  THE TEXT TAP. The conditioning is not the text encoder's output but its hidden
  states at a fixed set of decoder layers, concatenated with the LAYER index
  varying fastest. Both the layer set and the interleaving order are part of the
  model's input contract.

Features are stored WITHOUT padding -- exactly the real tokens -- and the
dataloader left-pads them to a constant width later. Storing them padded would
mean the cache encoded a padding decision, and changing it would mean re-encoding
everything.

A note on the text encoder: the quantized release of it needs a quantization
library that is not always available, so the full-precision base model of the same
architecture is used instead. The autoencoder is loaded from the non-quantized
subfolder for the same reason, and additionally because its normalization
statistics need to be exact.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# The decoder layers tapped for the packed text conditioning. Part of the model's
# input contract, not a tuning knob.
ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

DEFAULT_VAE_SOURCE = "ideogram-ai/ideogram-4-nf4-diffusers"
DEFAULT_TEXT_ENCODER_SOURCE = "Qwen/Qwen3-VL-8B-Instruct"


def _autoencoder_class():
    """The autoencoder class, from wherever this diffusers version exports it."""
    try:
        from diffusers import AutoencoderKLFlux2

        return AutoencoderKLFlux2
    except ImportError:
        from diffusers.models.autoencoders import AutoencoderKLFlux2

        return AutoencoderKLFlux2


class Ideogram4Processor:
    """Encodes image and caption pairs into the Ideogram-4 cache format."""

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
        self.norm_mean = None
        self.norm_std = None
        self.language_model = None
        self.tokenizer = None

    def load_models(self) -> None:
        from transformers import AutoTokenizer, Qwen3VLModel

        autoencoder_cls = _autoencoder_class()

        logger.info(
            "[Ideogram4Processor] loading the autoencoder from %s/%s",
            self.vae_source,
            self.vae_subfolder,
        )
        vae = autoencoder_cls.from_pretrained(
            self.vae_source, subfolder=self.vae_subfolder, torch_dtype=self.dtype
        )
        self.vae = vae.to(self.device).eval()

        # The statistics live on the packed channel space; see the module docstring.
        eps = float(getattr(vae.config, "batch_norm_eps", 1e-5))
        self.norm_mean = vae.bn.running_mean.detach().view(1, 1, 1, -1).float().to(self.device)
        self.norm_std = (
            torch.sqrt(vae.bn.running_var.detach() + eps).view(1, 1, 1, -1).float().to(self.device)
        )

        # Derived from the model's own dimensions rather than assumed, and asserted:
        # an autoencoder with a different channel count would otherwise be packed
        # wrong, and nothing downstream decodes an image to notice.
        packed_channels = int(self.norm_mean.shape[-1])
        latent_channels = int(getattr(vae.config, "latent_channels", packed_channels // 4))
        self.patch_size = int(round((packed_channels / latent_channels) ** 0.5))
        if latent_channels * self.patch_size**2 != packed_channels:
            raise ValueError(
                f"cannot factor {packed_channels} packed channels into "
                f"{latent_channels} latent channels times a square patch; this "
                "autoencoder does not have the geometry the packing assumes"
            )
        logger.info(
            "[Ideogram4Processor] autoencoder ready: %d latent channels, patch %d, " "%d packed channels",
            latent_channels,
            self.patch_size,
            packed_channels,
        )

        logger.info("[Ideogram4Processor] loading the text encoder from %s", self.text_encoder_source)
        text_encoder = Qwen3VLModel.from_pretrained(self.text_encoder_source, torch_dtype=self.dtype)
        # Only the language model is tapped, so drop the vision tower rather than
        # move it to the device.
        if getattr(text_encoder, "visual", None) is not None:
            text_encoder.visual = None
        text_encoder = text_encoder.to(self.device).eval()
        self.language_model = text_encoder.language_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_source)
        logger.info("[Ideogram4Processor] models loaded.")

    # --------------------------------------------------------------------- image
    @staticmethod
    def preprocess_image(image, resolution: int) -> torch.Tensor:
        """Resize and center-crop to a square, scaled to [-1, 1].

        Scales by the LARGER ratio and then crops, so the result is filled rather
        than letterboxed: padding bars would be encoded as real image content and
        the model would learn to produce them.
        """
        import numpy as np
        from PIL import Image

        image = image.convert("RGB")
        width, height = image.size
        scale = max(resolution / width, resolution / height)
        resized = image.resize((int(round(width * scale)), int(round(height * scale))), Image.LANCZOS)
        left = (resized.width - resolution) // 2
        top = (resized.height - resolution) // 2
        cropped = resized.crop((left, top, left + resolution, top + resolution))

        array = torch.from_numpy(np.array(cropped)).float() / 255.0
        array = (array - 0.5) / 0.5
        return array.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode, pack into 2x2 patches, and normalize. Returns ``[C, gh, gw]``.

        The permutation is the inverse of the pipeline's decode reshape: raw
        channel ``c`` of sub-position ``(pa, pb)`` in patch ``(i, j)`` becomes
        packed channel ``((pa * patch) + pb) * channels + c`` of token ``(i, j)``.
        """
        patch = self.patch_size
        x = image_tensor.to(self.device, dtype=self.dtype)
        raw = self.vae.encode(x).latent_dist.mode().float()

        batch, channels, height, width = raw.shape
        if height % patch or width % patch:
            raise ValueError(
                f"the latent grid {height}x{width} is not divisible by the patch size "
                f"{patch}; the resolution must be a multiple of the autoencoder's "
                "downsampling factor times the patch size"
            )
        grid_h, grid_w = height // patch, width // patch

        split = raw.view(batch, channels, grid_h, patch, grid_w, patch)
        # -> (batch, grid_h, grid_w, pa, pb, channels), then flatten the last three.
        packed = split.permute(0, 2, 4, 3, 5, 1).reshape(batch, grid_h, grid_w, patch * patch * channels)
        packed = (packed - self.norm_mean) / self.norm_std

        latents = packed.permute(0, 3, 1, 2).squeeze(0)
        # Stored narrow: the cache is large, and the pipeline promotes on load.
        return latents.detach().cpu().to(torch.float16)

    # ---------------------------------------------------------------------- text
    @torch.no_grad()
    def _tap_hidden_states(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor, positions: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run the decoder stack, returning the hidden states at each tapped layer.

        The stack is stepped by hand rather than run with output_hidden_states,
        because the model's rotary embedding needs the multi-axis position ids
        constructed here and the mask needs to be built for this call.
        """
        from transformers.masking_utils import create_causal_mask

        language_model = self.language_model
        inputs_embeds = language_model.embed_tokens(token_ids)

        # The rotary embedding takes three position axes; text uses the same
        # sequence position on each.
        expanded = positions[None, ...].expand(4, positions.shape[0], -1)
        text_positions = expanded[0]
        rotary_positions = expanded[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_positions,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, rotary_positions)

        wanted = set(ACTIVATION_LAYERS)
        highest = max(wanted)
        if highest >= len(language_model.layers):
            raise ValueError(
                f"the text conditioning taps layer {highest} but this text encoder has "
                f"only {len(language_model.layers)} layers; it is not the architecture "
                "the model was trained against"
            )

        captured: Dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for index, layer in enumerate(language_model.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_positions,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if index in wanted:
                captured[index] = hidden_states
            if index == highest:
                # Nothing above the last tapped layer contributes, so stop.
                break

        return [captured[i] for i in ACTIVATION_LAYERS]

    @torch.no_grad()
    def encode_text(self, prompt: str, max_text_tokens: int) -> Optional[Dict[str, Any]]:
        """Encode one caption, or return None if it does not fit the budget.

        Returning None rather than truncating is deliberate: the caller drops the
        sample, which shows up in its skip counts, whereas a truncated caption
        would silently become the training signal for its image.

        No padding is applied. Exactly the real tokens are tapped and stored; the
        dataloader left-pads to a constant width later.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        templated = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokens = self.tokenizer(templated, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        length = int(tokens.shape[0])
        if length == 0 or length > max_text_tokens:
            return None

        token_ids = tokens.view(1, length).to(self.device)
        attention_mask = torch.ones(1, length, dtype=torch.long, device=self.device)
        positions = torch.arange(length, device=self.device).view(1, length)

        tapped = self._tap_hidden_states(token_ids, attention_mask, positions)
        # Stack the taps and interleave with the LAYER index varying fastest, which
        # is the order the model's input projection expects.
        features = torch.stack(tapped, dim=0).permute(1, 2, 3, 0).reshape(1, length, -1)
        features = features.squeeze(0).to(torch.float16).detach().cpu()
        return {"llm_features": features, "text_length": length}

    # --------------------------------------------------------------------- cache
    @staticmethod
    def get_cache_data(
        image_latents: torch.Tensor,
        text_encoding: Dict[str, Any],
        *,
        prompt: str,
        image_path: str,
    ) -> Dict[str, Any]:
        """Assemble one cache entry.

        The prompt and source path are stored alongside the tensors. They are not
        read during training, and exist so that a cache can be inspected later
        without the source directory it came from.
        """
        return {
            "image_latents": image_latents,
            "llm_features": text_encoding["llm_features"],
            "text_length": int(text_encoding["text_length"]),
            "grid_h": int(image_latents.shape[1]),
            "grid_w": int(image_latents.shape[2]),
            "prompt": prompt,
            "image_path": image_path,
            "model_type": "ideogram4",
        }
