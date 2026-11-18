###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ideogram-4 flow-matching adapter, registered without forking AutoModel.

WHY A WRAPPER AND NOT AN EDIT:
  AutoModel picks a flow-matching adapter by name from a closed dictionary in
  ``flow_matching.pipeline.create_adapter``, and Ideogram-4 is not in it. Rather
  than edit the submodule, ``install`` wraps that function at runtime so
  ``flow_matching.adapter_type: ideogram4`` resolves here. The wrapper is purely
  additive, so a FLUX or Wan run is unaffected.

  It patches two namespaces, and the second is easy to miss: the recipe does
  ``from ...pipeline import create_adapter``, which binds the function by value at
  import time, so patching only the pipeline module would leave the recipe holding
  the original.

WHAT THE ADAPTER DOES:
  Ideogram-4 is a single-stream flow-matching diffusion transformer: text
  conditioning and patchified image latents occupy ONE packed sequence,
  distinguished by a per-token indicator and joined by a block-diagonal mask, with
  three-axis rotary position ids. This adapter maps AutoModel's latent-space
  flow-matching convention onto that packed contract.

  ``prepare_inputs`` packs the image latents, prepends a zeroed text region,
  builds the encoder features and the position, segment and indicator ids for the
  ``[pad][text][image]`` layout, and converts the noise level to the model's time
  convention. It also builds the variable-length packing on the host.

  ``forward`` publishes the packing, runs the transformer, slices out the
  image-token velocity, unpacks it, and NEGATES it -- the transformer predicts
  ``x0 - eps`` while AutoModel's target is ``eps - x0``. Getting that sign wrong
  produces a model that trains to the exact opposite of the objective, so it is
  stated rather than left to the reader.

Batch keys consumed, as produced by the Ideogram-4 preprocessor or its cache:
  image_latents  patchified and normalized latents, ``[B, C, H_p, W_p]``
  llm_features   left-padded language-model features, ``[B, max_text, F]``
  text_lengths   ``[B]`` real, non-pad text token count per sample
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

from primus.backends.automodel.models.ideogram4._varlen_common import (
    precompute_cu_seqlens_active,
)
from primus.backends.automodel.models.ideogram4.cu_seqlens import (
    build_cu_seqlens,
    static_max_seqlen,
)
from primus.backends.automodel.models.ideogram4.packing_buffer import publish_packing

logger = logging.getLogger(__name__)


# The values this code was written against, in the order _layout_constants returns.
FALLBACK_LAYOUT_CONSTANTS = (65536, 3, 2, -1)
_LAYOUT_CONSTANT_NAMES = (
    "IMAGE_POSITION_OFFSET",
    "LLM_TOKEN_INDICATOR",
    "OUTPUT_IMAGE_INDICATOR",
    "SEQUENCE_PADDING_INDICATOR",
)


def _layout_constants():
    """The per-token role constants, read from diffusers rather than hardcoded.

    These are the layout contract with the model. A stale literal here would
    mislabel every token with no further error, so they are read from the model's
    own module. The fallback applies only when diffusers has no Ideogram-4 module
    at all, which means no Ideogram-4 model can be built either, so the values can
    only reach tests of the layout arithmetic. A module that exists but lacks one
    of the names means they moved upstream, and that raises.
    """
    try:
        import diffusers.models.transformers.transformer_ideogram4 as module
    except ImportError:
        return FALLBACK_LAYOUT_CONSTANTS

    missing = [name for name in _LAYOUT_CONSTANT_NAMES if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"diffusers' Ideogram-4 module no longer defines {', '.join(missing)}. These "
            "constants are the token-layout contract with the model, so the adapter will "
            "not guess them; re-check this adapter against the diffusers pin."
        )
    return tuple(getattr(module, name) for name in _LAYOUT_CONSTANT_NAMES)


def _base_adapter_cls():
    from nemo_automodel.components.flow_matching.adapters.base import ModelAdapter

    return ModelAdapter


def _build_ideogram4_adapter_class():
    """Build the adapter class.

    A factory rather than a module-level class so that importing this module never
    requires AutoModel, which keeps it import-safe for tests and for the patch
    condition.
    """
    ModelAdapter = _base_adapter_cls()
    (
        IMAGE_POSITION_OFFSET,
        LLM_TOKEN_INDICATOR,
        OUTPUT_IMAGE_INDICATOR,
        SEQUENCE_PADDING_INDICATOR,
    ) = _layout_constants()

    class Ideogram4Adapter(ModelAdapter):
        """Flow-matching adapter for the Ideogram-4 single-stream transformer."""

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
            """Build the position, segment and indicator ids for one batch.

            Mirrors the inference pipeline's own construction, so that the training
            layout and the inference layout are the same. They have to be: a model
            trained on one token layout and sampled on another produces images that
            look subtly wrong for no visible reason.
            """
            batch_size = len(text_lengths)
            num_image_tokens = grid_h * grid_w
            total_seq_len = max_text_tokens + num_image_tokens

            h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
            w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
            t_idx = torch.zeros_like(h_idx)
            image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

            position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
            segment_ids = torch.full(
                (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
            )
            indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

            for row, num_text in enumerate(text_lengths):
                num_text = int(num_text)
                offset = max_text_tokens - num_text
                text_pos = torch.arange(num_text)
                position_ids[row, offset : offset + num_text] = torch.stack(
                    [text_pos, text_pos, text_pos], dim=1
                )
                position_ids[row, offset + num_text :] = image_pos
                indicator[row, offset : offset + num_text] = LLM_TOKEN_INDICATOR
                indicator[row, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
                # One segment covering text and image together: they attend jointly,
                # and the leading padding keeps the default padding id so it forms a
                # segment of its own.
                segment_ids[row, offset : offset + num_text + num_image_tokens] = 1

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
                    "Ideogram4Adapter expects 4-D patchified latents [B, C, H_p, W_p], " f"got {noisy.ndim}-D"
                )
            batch_size, channels, grid_h, grid_w = noisy.shape
            if channels != self.in_channels:
                raise ValueError(f"expected {self.in_channels} packed channels, got {channels}")
            num_image_tokens = grid_h * grid_w

            image_tokens = self._pack_image_latents(noisy)

            llm_features = batch["llm_features"].to(device, dtype=dtype, non_blocking=True)
            if llm_features.ndim != 3 or llm_features.shape[0] != batch_size:
                raise ValueError(
                    f"llm_features must be [B, max_text, F] with B={batch_size} to match the "
                    f"latents, got shape {tuple(llm_features.shape)}"
                )
            # The width the dataloader produced, before the reserved column below.
            text_capacity = llm_features.shape[1]

            # THE RESERVED PAD COLUMN. One always-pad position, so every row keeps at
            # least one padding token and therefore contributes exactly two segments.
            # Without it a caption filling the full width yields a single-segment row,
            # the segment count varies with the batch, and the compiled graph is rebuilt
            # for each distinct length pattern. It costs one token position; the
            # alternative -- capping captions -- would mean dropping a real token to
            # satisfy the compiler. See cu_seqlens.py for the full argument.
            precompute = precompute_cu_seqlens_active()
            if precompute:
                pad_column = torch.zeros(batch_size, 1, llm_features.shape[-1], device=device, dtype=dtype)
                llm_features = torch.cat([pad_column, llm_features], dim=1)
            max_text = llm_features.shape[1]

            if context.cfg_dropout_prob > 0.0:
                # Ideogram-4 uses asymmetric guidance: the inference pipeline never runs
                # this transformer without text. Its unconditional branch is a separate
                # unconditional transformer, fed zeroed features over an image-only
                # layout. Dropping captions here would train a case sampling never
                # produces, in a layout it never uses.
                raise ValueError(
                    "flow_matching.cfg_dropout_prob must be 0 for Ideogram-4. Its pipeline "
                    "guides with a separate unconditional transformer rather than by running "
                    "this one without text, so caption dropout would train an input it never "
                    f"sees at sampling time (got {context.cfg_dropout_prob})."
                )

            text_lengths = batch.get("text_lengths")
            if text_lengths is None:
                text_lengths = [text_capacity] * batch_size
            elif torch.is_tensor(text_lengths):
                text_lengths = text_lengths.tolist()
            text_lengths = [int(t) for t in text_lengths]
            if len(text_lengths) != batch_size:
                raise ValueError(f"text_lengths has {len(text_lengths)} entries for a batch of {batch_size}")
            if min(text_lengths) < 0:
                raise ValueError(f"text_lengths must be non-negative, got {text_lengths}")
            if max(text_lengths) > text_capacity:
                raise ValueError(
                    f"the longest text_length is {max(text_lengths)} but llm_features is "
                    f"only {text_capacity} wide; the dataloader must left-pad to at least "
                    "the longest caption in the batch"
                )

            text_region = torch.zeros(batch_size, max_text, channels, device=device, dtype=dtype)
            hidden_states = torch.cat([text_region, image_tokens], dim=1)

            image_feature_pad = torch.zeros(
                batch_size, num_image_tokens, llm_features.shape[-1], device=device, dtype=dtype
            )
            encoder_hidden_states = torch.cat([llm_features, image_feature_pad], dim=1)

            position_ids, segment_ids, indicator = self._prepare_ids(
                text_lengths, grid_h, grid_w, max_text, device
            )

            # Built once here and shared by every layer, so the packing is a graph input
            # rather than something each layer recovers from the mask. Left on the HOST:
            # ``forward`` publishes it with one host-to-device copy, and nothing reads
            # it back from the device.
            cu_seqlens = None
            max_seqlen = None
            if precompute:
                cu_seqlens = build_cu_seqlens(text_lengths, max_text, num_image_tokens)
                max_seqlen = static_max_seqlen(max_text, num_image_tokens)

            return {
                "hidden_states": hidden_states,
                # The model's time convention runs the other way from the noise level.
                "timestep": (1.0 - context.sigma).to(dtype),
                "encoder_hidden_states": encoder_hidden_states,
                "position_ids": position_ids,
                "segment_ids": segment_ids,
                "indicator": indicator,
                "_max_text": max_text,
                "_grid_h": grid_h,
                "_grid_w": grid_w,
                "_cu_seqlens": cu_seqlens,
                "_max_seqlen": max_seqlen,
            }

        def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
            max_text = inputs.pop("_max_text")
            grid_h = inputs.pop("_grid_h")
            grid_w = inputs.pop("_grid_w")
            cu_seqlens = inputs.pop("_cu_seqlens", None)
            max_seqlen = inputs.pop("_max_seqlen", None)

            if cu_seqlens is not None:
                # Publish onto the attention modules, which is where the processor reads
                # it. Two ordering rules apply and both are silent when broken: this must
                # happen BEFORE the model call, and nothing may republish between the
                # forward and its backward, because per-layer compilation lives inside
                # the checkpoint wrapper and each block re-reads this buffer during the
                # backward recompute. See packing_buffer.py.
                publish_packing(
                    model,
                    cu_seqlens,
                    max_seqlen,
                    device=inputs["hidden_states"].device,
                    # Having built a packing, a model that cannot read it is a
                    # misconfiguration rather than a fallback: every layer would derive
                    # its own from the mask, and if that happened on only some ranks it
                    # would quietly average two different attention paths into one
                    # gradient. Non-zero ranks log nothing after startup, so raising is
                    # the only way this becomes visible at all.
                    required=True,
                )

            out = model(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
                return_dict=False,
            )
            prediction = self.post_process_prediction(out)

            # Only the image tokens carry a velocity the loss reads.
            image_prediction = prediction[:, max_text:]
            unpacked = self._unpack_image_latents(image_prediction, grid_h, grid_w)

            # The transformer predicts x0 - eps; AutoModel's target is eps - x0.
            return -unpacked if self.predict_negative_velocity else unpacked

    return Ideogram4Adapter


# Cached so the class identity is stable across calls, which matters for isinstance
# checks and for anything that keys off the type.
_ADAPTER_CLS = None


def get_ideogram4_adapter_class():
    global _ADAPTER_CLS
    if _ADAPTER_CLS is None:
        _ADAPTER_CLS = _build_ideogram4_adapter_class()
    return _ADAPTER_CLS


def install() -> bool:
    """Make ``adapter_type: ideogram4`` resolve to this adapter.

    Additive and idempotent: it only adds a route, so every existing adapter type
    behaves exactly as before. The recipe's binding is re-checked on every call,
    including after the pipeline is already patched.
    """
    import nemo_automodel.components.flow_matching.pipeline as pipeline

    current = pipeline.create_adapter
    if getattr(current, "_ideogram4_patched", False):
        create_adapter = current
    else:
        original = current

        def create_adapter(adapter_type: str, **kwargs):
            if adapter_type == "ideogram4":
                return get_ideogram4_adapter_class()(**kwargs)
            return original(adapter_type, **kwargs)

        create_adapter._ideogram4_patched = True
        create_adapter._ideogram4_original = original
        pipeline.create_adapter = create_adapter
        logger.info("[PrimusIdeogram4] registered the 'ideogram4' flow-matching adapter.")

    _patch_recipe_binding(create_adapter)
    return True


def _patch_recipe_binding(create_adapter) -> None:
    """Point the recipe's own ``create_adapter`` name at the patched function.

    The recipe imports the function by name, binding it by value. A recipe imported
    after the pipeline was patched already holds the patched function; one imported
    before holds the original and is rebound here. Anything else bound there is
    someone else's wrapper, which this does not replace.
    """
    try:
        import nemo_automodel.recipes.diffusion.train as recipe
    except ImportError as exc:
        logger.warning(
            "[PrimusIdeogram4] could not import the diffusion recipe (%s), so its "
            "create_adapter was not checked; 'adapter_type: ideogram4' may be reported "
            "as unknown.",
            exc,
        )
        return

    bound = getattr(recipe, "create_adapter", None)
    if bound is create_adapter or getattr(bound, "_ideogram4_patched", False):
        return
    if bound is create_adapter._ideogram4_original:
        recipe.create_adapter = create_adapter
        return
    logger.warning(
        "[PrimusIdeogram4] the diffusion recipe's create_adapter is %r, neither AutoModel's "
        "nor the Ideogram-4 route, so it was left alone and 'adapter_type: ideogram4' "
        "will only resolve if that function forwards to the pipeline's.",
        bound,
    )
