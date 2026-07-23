###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ideogram-4 flow-matching adapter + no-fork registration for the AutoModel
diffusion recipe.

WHY (no Automodel/diffusers fork):
  AutoModel selects a flow-matching model adapter by name via
  ``nemo_automodel.components.flow_matching.pipeline.create_adapter`` (a closed
  dict of {"flux","flux2","wan","hunyuan","qwen_image","simple"}). Ideogram-4 is
  not in that dict. Rather than edit the submodule, ``install()`` wraps
  ``create_adapter`` at runtime so ``flow_matching.adapter_type: ideogram4``
  resolves to :class:`Ideogram4Adapter`. The recipe imports ``create_adapter`` by
  name (``from ...pipeline import create_adapter``), so we patch it in BOTH the
  pipeline module and the already-imported recipe module namespace.

WHAT the adapter does:
  Ideogram-4 is a single-stream flow-matching DiT: text-conditioning tokens and
  patchified image latents live in ONE packed sequence, distinguished by a
  per-token ``indicator`` and joined by a block-diagonal mask (``segment_ids``),
  with 3-axis MRoPE (``position_ids``). This adapter maps AutoModel's latent-space
  flow-matching convention (noise ``x_t=(1-sigma)x0+sigma eps``; target
  ``v=eps-x0`` in the 128-dim packed-latent space) onto that packed contract:

  - ``prepare_inputs`` packs image latents ``[B,128,H_p,W_p] -> [B,n_img,128]``,
    prepends a zeroed text region for ``hidden_states [B,S,128]``, builds
    ``encoder_hidden_states [B,S,53248]`` (Qwen3-VL features over text, zeros over
    image), and the ``position_ids/segment_ids/indicator`` for the
    ``[pad][text][image]`` layout. Ideogram model time is ``t = 1 - sigma``.
  - ``forward`` runs the DiT, slices the image-token velocity, unpacks to
    ``[B,128,H_p,W_p]`` and NEGATES it: the DiT predicts ``x0 - eps`` (inference
    feeds ``-v`` to the scheduler) while AutoModel's target is ``eps - x0``.

  Batch keys consumed (from the Ideogram-4 preprocessor / cache):
    - ``image_latents``: ``[B,128,H_p,W_p]`` patchified + BN-normalized latents.
    - ``llm_features``: ``[B,max_text,53248]`` left-padded Qwen3-VL 13-layer feats.
    - ``text_lengths``: ``[B]`` int, real (non-pad) text token count per sample.

  ``install()`` is additive and safe: it only adds the ``ideogram4`` route, so a
  FLUX/Wan run is unaffected. Idempotent.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Per-token role / layout constants live with the diffusers transformer definition.
try:
    from diffusers.models.transformers.transformer_ideogram4 import (
        IMAGE_POSITION_OFFSET,
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
        SEQUENCE_PADDING_INDICATOR,
    )
except Exception:  # pragma: no cover - keep import-safe if diffusers is older
    IMAGE_POSITION_OFFSET = 4096
    LLM_TOKEN_INDICATOR = 1
    OUTPUT_IMAGE_INDICATOR = 2
    SEQUENCE_PADDING_INDICATOR = 0


def _base_adapter_cls():
    """Import AutoModel's ModelAdapter base lazily (submodule must be importable)."""
    from nemo_automodel.components.flow_matching.adapters.base import ModelAdapter

    return ModelAdapter


# Defined as a factory so importing this module never requires nemo_automodel until
# install()/first use — keeps the module import-safe in any context.
def _build_ideogram4_adapter_class():
    ModelAdapter = _base_adapter_cls()

    class Ideogram4Adapter(ModelAdapter):
        """Model adapter for Ideogram-4 single-stream flow-matching T2I."""

        def __init__(self, in_channels: int = 128, predict_negative_velocity: bool = True):
            self.in_channels = in_channels
            self.predict_negative_velocity = predict_negative_velocity

        @staticmethod
        def _prepare_ids(
            text_lengths: List[int],
            grid_h: int,
            grid_w: int,
            max_text_tokens: int,
            device: torch.device,
        ):
            """Build ``[left-pad][text][image]`` position/segment/indicator tensors.

            Mirrors ``Ideogram4Pipeline._prepare_ids`` so training == inference layout.
            """
            batch_size = len(text_lengths)
            num_image_tokens = grid_h * grid_w
            total_seq_len = max_text_tokens + num_image_tokens

            h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
            w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
            t_idx = torch.zeros_like(h_idx)
            image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

            position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
            segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
            indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

            for b, num_text in enumerate(text_lengths):
                num_text = int(num_text)
                offset = max_text_tokens - num_text
                text_pos = torch.arange(num_text)
                text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
                position_ids[b, offset : offset + num_text] = text_pos_3d
                position_ids[b, offset + num_text :] = image_pos
                indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
                indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
                segment_ids[b, offset : offset + num_text + num_image_tokens] = 1

            return position_ids.to(device), segment_ids.to(device), indicator.to(device)

        def _pack_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
            b, c, h, w = latents.shape
            return latents.reshape(b, c, h * w).permute(0, 2, 1).contiguous()

        def _unpack_image_latents(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
            b, _, c = tokens.shape
            return tokens.permute(0, 2, 1).contiguous().reshape(b, c, h, w)

        def prepare_inputs(self, context) -> Dict[str, Any]:
            batch = context.batch
            device = context.device
            dtype = context.dtype

            noisy = context.noisy_latents
            if noisy.ndim != 4:
                raise ValueError(
                    f"Ideogram4Adapter expects 4D patchified latents [B, C, H_p, W_p], got {noisy.ndim}D"
                )
            B, C, H_p, W_p = noisy.shape
            if C != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} packed channels, got {C}")
            num_image_tokens = H_p * W_p

            img_tokens = self._pack_image_latents(noisy)  # [B, n_img, 128]

            llm_features = batch["llm_features"].to(device, dtype=dtype, non_blocking=True)
            if llm_features.ndim == 2:
                llm_features = llm_features.unsqueeze(0)
            max_text = llm_features.shape[1]

            if context.cfg_dropout_prob > 0.0:
                drop = torch.rand(B, 1, 1, device=device) < context.cfg_dropout_prob
                llm_features = llm_features.masked_fill(drop, 0.0)

            text_lengths = batch.get("text_lengths")
            if text_lengths is None:
                text_lengths = [max_text] * B
            elif torch.is_tensor(text_lengths):
                text_lengths = text_lengths.tolist()

            text_z = torch.zeros(B, max_text, C, device=device, dtype=dtype)
            hidden_states = torch.cat([text_z, img_tokens], dim=1)  # [B, S, 128]

            img_feat_pad = torch.zeros(B, num_image_tokens, llm_features.shape[-1], device=device, dtype=dtype)
            encoder_hidden_states = torch.cat([llm_features, img_feat_pad], dim=1)  # [B, S, 53248]

            position_ids, segment_ids, indicator = self._prepare_ids(text_lengths, H_p, W_p, max_text, device)

            # Ideogram model time: 0=noise, 1=data => t = 1 - sigma.
            timestep = (1.0 - context.sigma).to(dtype)

            return {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "position_ids": position_ids,
                "segment_ids": segment_ids,
                "indicator": indicator,
                "_max_text": max_text,
                "_h_p": H_p,
                "_w_p": W_p,
            }

        def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
            max_text = inputs.pop("_max_text")
            h_p = inputs.pop("_h_p")
            w_p = inputs.pop("_w_p")

            out = model(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
                attention_kwargs=None,
                return_dict=False,
            )
            pred = self.post_process_prediction(out)  # [B, S, 128]

            img_pred = pred[:, max_text:]  # [B, n_img, 128]
            unpacked = self._unpack_image_latents(img_pred, h_p, w_p)  # [B, 128, H_p, W_p]

            # DiT predicts x0 - eps; AutoModel target is eps - x0.
            return -unpacked if self.predict_negative_velocity else unpacked

    return Ideogram4Adapter


# Cache the built class so identity is stable across calls.
_IDEOGRAM4_ADAPTER_CLS = None


def get_ideogram4_adapter_class():
    global _IDEOGRAM4_ADAPTER_CLS
    if _IDEOGRAM4_ADAPTER_CLS is None:
        _IDEOGRAM4_ADAPTER_CLS = _build_ideogram4_adapter_class()
    return _IDEOGRAM4_ADAPTER_CLS


def install() -> bool:
    """Route ``adapter_type == "ideogram4"`` to the Ideogram-4 adapter via a no-fork
    wrapper around AutoModel's ``create_adapter``.

    Additive and idempotent; never changes existing adapter behavior. Returns True.
    """
    import nemo_automodel.components.flow_matching.pipeline as P

    orig = P.create_adapter
    if getattr(orig, "_ideogram4_patched", False):
        return True

    def create_adapter_patched(adapter_type: str, **kwargs):
        if adapter_type == "ideogram4":
            return get_ideogram4_adapter_class()(**kwargs)
        return orig(adapter_type, **kwargs)

    create_adapter_patched._ideogram4_patched = True
    P.create_adapter = create_adapter_patched

    # The recipe does ``from ...pipeline import create_adapter`` (bound by name), so
    # patch its module namespace too if it is already imported.
    try:
        import nemo_automodel.recipes.diffusion.train as T

        if getattr(T, "create_adapter", None) is orig:
            T.create_adapter = create_adapter_patched
    except Exception as exc:  # pragma: no cover
        logger.debug("[PrimusIdeogram] recipe namespace patch skipped: %s", exc)

    logger.info("[PrimusIdeogram] Registered 'ideogram4' flow-matching adapter (no-fork).")
    return True
