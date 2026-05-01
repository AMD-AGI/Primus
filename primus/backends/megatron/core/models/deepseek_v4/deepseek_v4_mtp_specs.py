###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 MTP (Multi-Token Prediction) block spec.

Plan-2 P16 wires V4 onto Megatron's upstream
:class:`MultiTokenPredictionBlock` (multi-token prediction with the
classic eh_proj + per-depth transformer layer + final layernorm
recipe). The V4 specialization is:

* Each MTP-depth's inner layer is a :class:`DeepseekV4HybridLayer` (with
  HC, hash routing, and clamped-SwiGLU all wired through the same spec
  tree as the main decoder).
* The two pre-projection norms (``enorm`` over the embedding,
  ``hnorm`` over the prior hidden state) and the post-MTP final
  layernorm reuse the V4 RMSNorm provider.
* The eh_proj is a column-parallel linear (also from the provider).

What this file does *not* include:

* The MTP block forward path itself — that's owned by upstream
  :class:`MultiTokenPredictionBlock` once we hand it the spec.
* The ``HyperHead`` / loss-aware shifting / RouterReplay snapshot
  matching (those live inside :func:`process_mtp_loss` and the V4
  layer's ``forward``).

Reference: techblog §7 ("MTP V4 head") and
``DeepSeek-V4-Flash/inference/model.py:MTPBlock``.

The legacy primus-owned :class:`DeepseekV4MTPBlock` (in
``deepseek_v4_mtp.py``) is preserved behind the
``v4_use_custom_mtp_block`` config flag for back-compat with prior
research checkpoints; new training runs should leave that flag at
``False`` (the default) so the spec-based path here is used.
"""

from __future__ import annotations

from typing import Optional

from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
    DeepSeekV4SpecProvider,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)


def _v4_mtp_layer_spec(
    *,
    config: DeepSeekV4TransformerConfig,
    transformer_layer_spec: ModuleSpec,
    provider: DeepSeekV4SpecProvider,
) -> ModuleSpec:
    """One MTP depth's :class:`MultiTokenPredictionLayer` spec.

    Args:
        config: V4 transformer config (carries ``hidden_size``, etc.).
        transformer_layer_spec: the V4 hybrid-layer spec used as the
            inner layer for this depth. Plan-2 wires it through
            unchanged so the MTP layer reuses HC / hash routing /
            clamped-SwiGLU exactly the way the main decoder does.
        provider: V4 spec provider (resolves RMSNorm + column-parallel
            linear modules).
    """
    del config
    norm_module = provider.v4_norm_module()
    column_parallel = provider.column_parallel_linear()

    return ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=norm_module,
            hnorm=norm_module,
            eh_proj=column_parallel,
            mtp_model_layer=transformer_layer_spec,
            layer_norm=norm_module,
        ),
    )


def get_v4_mtp_block_spec(
    config: DeepSeekV4TransformerConfig,
    *,
    transformer_layer_spec: ModuleSpec,
    vp_stage: Optional[int] = None,
) -> ModuleSpec:
    """Return a :class:`ModuleSpec` for the V4 MTP block.

    The returned spec wraps :class:`MultiTokenPredictionBlock` with one
    :class:`MultiTokenPredictionLayer` spec per MTP depth (V4-Flash
    uses ``mtp_num_layers=1``; larger variants may use more depths).
    The block submodules carry the per-depth specs; the block's
    ``__init__`` walks them and instantiates :class:`MultiTokenPredictionLayer`
    instances that reuse the V4 hybrid layer for inner attention / MLP
    math.

    Args:
        config: V4 transformer config. Must have
            ``mtp_num_layers >= 1`` (caller checks before invoking
            this helper).
        transformer_layer_spec: the V4 hybrid-layer spec for the main
            decoder. The same spec is reused for each MTP depth so the
            inner attention + MoE math matches the main decoder
            exactly. Plan-2 §16 confirms V4 uses a single
            (non-repeated) shape for MTP layers.
        vp_stage: optional virtual-pipeline stage index. Passed through
            to upstream MTP code; ignored on non-VP runs.

    Returns:
        A ``ModuleSpec`` that builds a fully-wired
        :class:`MultiTokenPredictionBlock` when handed to
        :func:`build_module` with ``config`` + ``pg_collection``
        kwargs.

    Notes:
        * V4 has no ``mtp_use_repeated_layer`` carry-over; the spec
          replicates the inner layer ``mtp_num_layers`` times. (V4
          checkpoints store one set of MTP weights per depth.)
        * The ``HyperHead`` per-depth collapse lives inside
          :class:`DeepseekV4HybridLayer` itself (via the layer's HC
          mixers); the upstream MTP block does not need V4-specific
          HyperHead awareness.
    """
    del vp_stage  # currently only forwarded by callers; not needed here

    if int(config.mtp_num_layers or 0) < 1:
        raise ValueError(
            "get_v4_mtp_block_spec requires mtp_num_layers >= 1; "
            f"got mtp_num_layers={config.mtp_num_layers!r}."
        )

    provider = DeepSeekV4SpecProvider(config=config)
    mtp_layer_specs = [
        _v4_mtp_layer_spec(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            provider=provider,
        )
        for _ in range(int(config.mtp_num_layers))
    ]

    return ModuleSpec(
        module=MultiTokenPredictionBlock,
        submodules=MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs),
    )


__all__ = ["get_v4_mtp_block_spec"]
