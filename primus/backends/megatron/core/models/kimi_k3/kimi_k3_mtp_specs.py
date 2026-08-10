###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 Multi-Token Prediction (MTP) block spec.

Wires Kimi K3 onto Megatron's upstream :class:`MultiTokenPredictionBlock`
-- per-depth ``eh_proj`` + a K3 decoder layer + RMSNorm -- exactly as
DeepSeek-V4 did at its P16 (``6c5875d``,
``deepseek_v4_mtp_specs.py::get_v4_mtp_block_spec``). The loss half is
upstream's :func:`process_mtp_loss`, called from
:meth:`KimiK3Model.forward`.

What the tech report actually specifies
--------------------------------------

* **One MTP layer.** Table 1: ``MTP Layers | 1 layer``. The released
  ``config.json`` ships ``num_nextn_predict_layers: 0`` and the released
  modelling code has no MTP module, so the release was *stripped*, not
  trained without one.
* **Its input is the final hidden state.** §4.1.4 (Draft Model
  Fine-Tuning) is the only prose about MTP: the EAGLE-3 draft's fusion
  matrix is initialised ``[0 0 I]`` "so that the fused representation
  coincides at initialization with the high-level feature h -- **the input
  on which the MTP layer was pre-trained**". The high-level feature is the
  output of the final AttnRes block, and §2.2 says "the final output layer
  then aggregates all N block representations" -- which is
  :class:`AttentionResidualHead`. So the MTP layer consumes the tensor the
  LM head consumes, after ``attn_res_head`` and after
  ``final_layernorm``. That is also precisely what upstream hands it
  (``gpt_model.py:529-541`` then ``:596-610``), so no divergence is
  needed.
* **"an MTP layer that mirrors the structure of a backbone layer"**
  (§4.1.4). K3's backbone is heterogeneous, so *which* layer is a
  resolved ambiguity -- see :meth:`KimiK3TransformerConfig.mtp_layer_is_kda`.
  The default mirrors the **final** backbone layer, which §2.1 guarantees
  is a Gated MLA layer: "An additional Gated MLA layer is placed at the
  end of the backbone, ensuring that the final layer always performs
  global attention."

What the report does **not** specify, and what was chosen instead
-----------------------------------------------------------------

* **The loss weight.** ``mtp_loss_scaling_factor`` keeps upstream's /
  DeepSeek-V3's ``0.1``. A choice, not a reproduction.
* **Whether the MTP layer itself runs attention residuals.** It does
  **not**, and the argument is structural rather than aesthetic:

  1. AttnRes is defined (report Eq. 8-9) over the outputs of *preceding
     backbone layers* plus the token embedding. The MTP layer is not a
     member of that stack.
  2. The block representations have already been consumed --
     ``attn_res_head`` collapsed them into the very tensor the MTP layer
     receives. Re-attending over them would double-count.
  3. A single MTP layer has no cross-layer history of its own, so its
     mixers would softmax over one candidate. That is the identity, and
     it would leave ``2 * 2 * hidden`` parameters that can never receive a
     gradient -- permanently disarming
     ``test_every_parameter_receives_a_finite_gradient``, which is this
     project's cheapest detector of an unwired submodule.
  4. There is nothing to carry it in: ``_lower_res_out`` deliberately
     returns the bare ``[s, b, h]`` hidden state on the ``post_process``
     stage (``kimi_k3_block.py:249-250``), and neither
     :class:`MultiTokenPredictionBlock` nor
     :class:`MultiTokenPredictionLayer` has a slot for a second tensor.

  Mechanically this is the ``use_attn_residuals=False`` override on
  :class:`KimiK3Layer`. Without it the MTP layer would inherit
  ``config.attn_res_block_size``, compute ``num_blocks_in =
  ceil(layer_idx / block_size) > 0``, be handed no ``block_residual``, and
  trip the drift assert at ``kimi_k3_block.py:482-487`` on the first
  forward. That failure is loud, which is why this file's job is to make
  the *quiet* alternative -- appending a spurious ninth checkpoint --
  impossible.

Two upstream contracts this file exists to satisfy
--------------------------------------------------

* ``enorm`` / ``hnorm`` / ``layer_norm`` are invoked as **raw callables**,
  ``self.submodules.enorm(config=..., hidden_size=..., eps=...)``
  (``multi_token_prediction.py:782-792``, ``:843-847``), not through
  ``build_module``. So they must be the norm *class* from the provider and
  **not** the ``ModuleSpec(module=...)`` wrapper that
  ``kimi_k3_layer_specs._build_norm_spec`` returns for the decoder tree.
* The inner layer's ``self_attention`` spec must declare
  ``params['attn_mask_type']`` in
  ``{padding, causal, no_mask, padding_causal}``
  (``multi_token_prediction.py:773-780``), and ``ModuleSpec.params``
  defaults to ``{}``. The MLA spec already declared it; the KDA spec now
  does too, and :class:`KimiDeltaAttention` accepts and ignores it.
"""

from __future__ import annotations

from typing import Optional

import torch  # noqa: F401  # must precede any transformer_engine import
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.models.kimi_k3.build_context import (
    resolve_k3_provider,
)
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    KimiK3TransformerConfig,
)

__all__ = [
    "get_kimi_k3_mtp_block_spec",
    "build_kimi_k3_mtp_inner_layer_spec",
    "kimi_k3_mtp_layer_index",
]


def kimi_k3_mtp_layer_index(config: KimiK3TransformerConfig, depth: int = 0) -> int:
    """Global layer index handed to the MTP depth ``depth``'s inner layer.

    ``num_layers + depth``, i.e. the MTP layers continue the backbone's
    numbering. Two things read it and neither is cosmetic:

    * ``KimiK3Layer.layer_number`` defaults to ``layer_idx + 1``, which is
      the key Megatron's MoE aux-loss tracker indexes by
      (``router.py:464-479``). Reusing a decoder layer's number would make
      an MTP depth's aux loss overwrite a decoder layer's.

      Note :class:`MultiTokenPredictionLayer` *also* passes its own
      ``layer_number`` (1-based within the MTP block,
      ``multi_token_prediction.py:839``) which wins, because
      ``build_module`` kwargs override spec ``params``. Upstream comments
      that this is deliberate. The index below is therefore what
      ``layer_idx`` becomes, and the guard is that it is out of the
      decoder's range either way.
    * It is what would drive the attention-residual append schedule -- so
      it is deliberately a value for which ``appends_checkpoint`` would be
      **true** at the production geometry (93 % 12 != 0 ... but 96 % 12 ==
      0 for other shapes), and the ``use_attn_residuals=False`` override is
      what makes that unreachable rather than shape-dependent luck. The
      spec-construction test asserts the override, not the index.
    """
    return int(config.num_layers) + int(depth)


def build_kimi_k3_mtp_inner_layer_spec(
    config: KimiK3TransformerConfig,
    *,
    provider,
    depth: int = 0,
) -> ModuleSpec:
    """The :class:`KimiK3Layer` spec used as one MTP depth's inner layer.

    A full K3 decoder layer -- the same attention module, the same Stable
    Latent MoE with its latent bottleneck and ``situ`` activation, the same
    RMSNorms -- with attention residuals turned off for the reasons in the
    module docstring.

    Args:
        config: the runtime config. ``mtp_layer_type`` selects the attention
            variant; ``moe_layer_freq``'s last entry selects the FFN.
        provider: the K3 spec provider.
        depth: 0-indexed MTP depth.
    """
    # Imported here rather than at module scope: kimi_k3_layer_specs imports
    # the MLA module lazily to keep TransformerEngine off the KDA test path,
    # and a top-level import here would put this module in that same cycle.
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_layer_specs import (
        build_kimi_k3_layer_spec,
        get_kimi_k3_moe_layer_pattern,
    )

    # "Mirrors the structure of a backbone layer" (§4.1.4). The FFN half of
    # that is the *last* backbone layer's choice, never the first: K3's dense
    # layers are the leading ``first_k_dense_replace`` ones, so reading the
    # pattern's head would give the MTP layer a dense FFN on every real shape.
    moe_pattern = get_kimi_k3_moe_layer_pattern(config)
    is_moe = bool(moe_pattern[-1]) if moe_pattern else False

    # ``is_mtp_layer`` is deliberately *not* set here: upstream passes it as a
    # build_module keyword (``multi_token_prediction.py:835-841``) and a spec
    # param of the same name collides with it, because build_module unpacks
    # both into one call. ``KimiK3Layer`` records it and forwards it to the MoE
    # router, whose aux-loss tracker keys MTP depths separately from decoder
    # layers (``router.py:454``, ``:468``, ``:531``, ``:541``).
    return build_kimi_k3_layer_spec(
        config,
        provider=provider,
        layer_idx=kimi_k3_mtp_layer_index(config, depth),
        is_moe=is_moe,
        is_kda=config.mtp_layer_is_kda(),
        use_attn_residuals=False,
    )


def get_kimi_k3_mtp_block_spec(
    config: KimiK3TransformerConfig,
    *,
    provider=None,
    vp_stage: Optional[int] = None,
) -> ModuleSpec:
    """The Kimi K3 MTP block spec.

    One :class:`MultiTokenPredictionLayer` spec per depth, wrapped in a
    :class:`MultiTokenPredictionBlock`. Handed to that class's ``spec``
    argument by :class:`KimiK3Model`.

    Args:
        config: the runtime config. ``mtp_num_layers`` must be ``>= 1``;
            :meth:`KimiK3TransformerConfig._resolve_mtp_fields` has already
            normalised a YAML ``0`` to ``None``, so reaching here with 0 or
            ``None`` is a caller bug and raises.
        provider: the K3 spec provider. Resolved from ``config`` when
            omitted.
        vp_stage: accepted for signature parity with
            ``get_v4_mtp_block_spec`` and with the decoder spec builder.
            Unused: upstream derives the MTP layer offset from the config and
            the ``vp_stage`` it is given at build time
            (``multi_token_prediction.py:533-544``, ``:749``), and nothing
            here is stage-dependent.

    Returns:
        A ``ModuleSpec`` that builds a wired
        :class:`MultiTokenPredictionBlock` when handed ``config`` and
        ``pg_collection``.
    """
    del vp_stage  # forwarded by callers for parity; nothing here reads it

    depths = int(config.mtp_num_layers or 0)
    if depths < 1:
        raise ValueError(
            "get_kimi_k3_mtp_block_spec requires mtp_num_layers >= 1; got "
            f"mtp_num_layers={config.mtp_num_layers!r} "
            f"(num_nextn_predict_layers={config.num_nextn_predict_layers!r}). "
            "Enable MTP by setting num_nextn_predict_layers in the yaml."
        )

    if provider is None:
        provider = resolve_k3_provider(config)

    # Raw class, not ModuleSpec: MultiTokenPredictionLayer calls these three
    # slots directly (see the module docstring).
    norm_module = provider.k3_norm_module()
    assert norm_module is not None, "Kimi K3 norm module must be provided by KimiK3SpecProvider."
    column_parallel = provider.column_parallel_linear()

    layer_specs = [
        ModuleSpec(
            module=MultiTokenPredictionLayer,
            submodules=MultiTokenPredictionLayerSubmodules(
                enorm=norm_module,
                hnorm=norm_module,
                eh_proj=column_parallel,
                mtp_model_layer=build_kimi_k3_mtp_inner_layer_spec(config, provider=provider, depth=depth),
                layer_norm=norm_module,
            ),
        )
        # One spec per depth even when they are structurally identical:
        # MultiTokenPredictionBlock._build_layers walks the list and builds one
        # module per entry unless config.mtp_use_repeated_layer is set
        # (multi_token_prediction.py:1336-1343), so a shared spec object would
        # still give independent weights but a shorter list would silently
        # build fewer depths than mtp_num_layers.
        for depth in range(depths)
    ]

    return ModuleSpec(
        module=MultiTokenPredictionBlock,
        submodules=MultiTokenPredictionBlockSubmodules(layer_specs=layer_specs),
    )
