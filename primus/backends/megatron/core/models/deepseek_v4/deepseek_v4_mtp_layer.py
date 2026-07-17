###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 multi-token-prediction (MTP) layer.

V4's MTP differs from the V3 / upstream :class:`MultiTokenPredictionLayer`
in two ways that the upstream layer cannot express on its own:

1. **Multi-stream (mHC) inner layer.** The MTP inner transformer layer is a
   :class:`DeepseekV4HybridLayer`, which — when ``hc_mult > 1`` — operates on
   the K-stream form ``[B, S, K, D]`` (see ``deepseek_v4_block``). The upstream
   MTP layer feeds its inner layer a single-stream ``[S, B, D]`` tensor, so we
   must lift to K streams before the inner layer and collapse back after it.

2. **Per-depth ``hc_head_fn``.** The released V4 checkpoint gives each MTP
   depth its *own* small :class:`HyperHead` (``hc_head_fn``) to collapse the K
   streams — it does **not** reuse the main trunk's HyperHead (techblog
   §8 / DeepSeek-V4 report). This is gated by
   ``config.mtp_use_separate_hc_head``.

The class subclasses upstream :class:`MultiTokenPredictionLayer` so it keeps
the shared embedding-roll / ``enorm`` / ``hnorm`` / ``eh_proj`` /
final-layernorm machinery and plugs straight into
:class:`MultiTokenPredictionBlock` (which builds each depth via
``build_module(layer_spec, ...)`` and therefore honours our ``module`` slot).
Only the inner-layer call site (``_proj_and_transformer_layer``) is overridden
to insert the lift / collapse, and ``forward`` is wrapped to capture
``position_ids`` (V4 attention derives its dual-RoPE from absolute positions,
and the upstream layer does not thread ``position_ids`` into
``_proj_and_transformer_layer``).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.spec_utils import build_module

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    _lift_streams_in,
    _lower_streams_out,
)
from primus.backends.megatron.core.transformer.hyper_connection import HyperHead

try:  # get_fp8_context lives in megatron.core.fp8_utils across recent versions
    from megatron.core.fp8_utils import get_fp8_context
except Exception:  # pragma: no cover - defensive: keep BF16 path working

    def get_fp8_context(config, *args, **kwargs):  # type: ignore[misc]
        return nullcontext()


class DeepseekV4MTPLayer(MultiTokenPredictionLayer):
    """One V4 MTP depth (mHC-aware, per-depth HyperHead)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Upstream MultiTokenPredictionLayer builds ``self.mtp_model_layer``
        # WITHOUT threading ``pg_collection`` (see
        # ``multi_token_prediction.py``: ``build_module(self.submodules.
        # mtp_model_layer, config=..., vp_stage=..., layer_number=...,
        # is_mtp_layer=True)``). For a V4 :class:`DeepseekV4HybridLayer` whose
        # MLP is a :class:`DeepseekV4MoE`, ``pg_collection=None`` selects the
        # *local-experts* (non-expert-parallel) path, which (a) instantiates
        # ALL routed experts on every rank and (b) breaks the DDP grad-bucket
        # invariant (``len(per_param_grad_ready_counts) == len(params)`` in
        # ``param_and_grad_buffer.reset``) because the local path's per-expert
        # modules do not match the EP dispatcher's grad-ready bookkeeping that
        # the main decoder relies on. Rebuild the inner layer WITH the
        # ``pg_collection`` the MTP block already holds so the MTP MoE uses the
        # exact same expert-parallel dispatcher path as the main decoder. Free
        # the throwaway first build before rebuilding to bound init memory.
        pg_collection = kwargs.get("pg_collection", None)
        vp_stage = kwargs.get("vp_stage", None)
        if pg_collection is not None and getattr(self.submodules, "mtp_model_layer", None) is not None:
            self.mtp_model_layer = None  # drop the pg_collection-less build
            self.mtp_model_layer = build_module(
                self.submodules.mtp_model_layer,
                config=self.config,
                vp_stage=vp_stage,
                layer_number=self.layer_number,
                is_mtp_layer=True,
                pg_collection=pg_collection,
            )

        self.hc_mult = int(getattr(self.config, "hc_mult", 1) or 1)
        use_separate_head = bool(getattr(self.config, "mtp_use_separate_hc_head", True))

        self.mtp_hyper_head: Optional[HyperHead] = None
        if self.hc_mult > 1:
            if not use_separate_head:
                # A single-stream MTP inner layer (no per-depth head) would
                # require the inner DeepseekV4HybridLayer to be built with
                # hc_mult=1, but it inherits the model's hc_mult and emits the
                # K-stream form. Fail loud instead of silently shape-crashing.
                raise NotImplementedError(
                    "DeepseekV4MTPLayer with hc_mult>1 requires "
                    "config.mtp_use_separate_hc_head=True (per-depth HyperHead); "
                    "single-stream MTP-with-mHC is not wired."
                )
            self.mtp_hyper_head = HyperHead(
                hidden_size=int(self.config.hidden_size),
                hc_mult=self.hc_mult,
                eps=float(getattr(self.config, "hc_eps", 1.0e-6)),
            )

        # Stash for ``_proj_and_transformer_layer`` (set in ``forward``).
        self._mtp_position_ids: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Capture ``position_ids`` then defer to the upstream forward.

        The model calls this with keyword args
        (``input_ids=``, ``position_ids=``, ``hidden_states=`` ...), so we
        read ``position_ids`` from ``kwargs``. V4 attention consumes absolute
        positions; the MTP token/label shift is handled by the upstream
        embedding roll, so we thread the *unrolled* ``position_ids`` into the
        inner hybrid layer.
        """
        if "position_ids" in kwargs:
            self._mtp_position_ids = kwargs["position_ids"]
        elif len(args) >= 2:
            self._mtp_position_ids = args[1]
        return super().forward(*args, **kwargs)

    # ------------------------------------------------------------------

    def _proj_and_transformer_layer(
        self,
        hidden_states: torch.Tensor,
        decoder_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """eh_proj -> (lift -> V4 hybrid layer -> per-depth HyperHead) -> norm.

        Mirrors the upstream method's fp8 / rng context handling but inserts
        the K-stream lift / collapse around the inner
        :class:`DeepseekV4HybridLayer` so the mHC math matches the main
        decoder block exactly (same ``_lift_streams_in`` / ``_lower_streams_out``
        helpers).
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        with rng_context:
            with fp8_context:
                # [S, B, D] single-stream after eh_proj.
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            with transformer_layer_fp8_context:
                # Lift to the K-stream form the V4 hybrid layer expects.
                x = _lift_streams_in(
                    hidden_states, pre_process=True, hc_mult=self.hc_mult
                )  # [B, S, K, D] (or [B, S, D] when hc_mult == 1)
                x, _ = self.mtp_model_layer(
                    x,
                    position_ids=self._mtp_position_ids,
                    token_ids=None,
                )
                # Per-depth collapse K streams -> single stream.
                if self.hc_mult > 1 and self.mtp_hyper_head is not None:
                    x = self.mtp_hyper_head(x)  # [B, S, D]
                hidden_states = _lower_streams_out(x, post_process=True, hc_mult=self.hc_mult)  # [S, B, D]

        hidden_states = self._postprocess(hidden_states)
        return hidden_states


__all__ = ["DeepseekV4MTPLayer"]
