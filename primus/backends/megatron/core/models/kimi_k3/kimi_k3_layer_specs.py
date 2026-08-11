###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 spec entry points.

:func:`get_kimi_k3_runtime_decoder_spec` is the one symbol
``kimi_k3_builders`` needs; everything else here builds a piece of the
tree it returns. Mirrors ``deepseek_v4_layer_specs.py`` in shape: a
per-layer builder that dispatches on the layer index, a stage slicer that
respects PP/VP partitioning, and a block-level assembler.

Two per-layer patterns drive the dispatch, and they are independent:

* ``config.linear_attention_freq`` -- ``1`` selects Kimi Delta Attention,
  ``0`` selects NoPE MLA. Already normalised to a length-``num_layers``
  ``list`` by ``KimiK3TransformerConfig.__post_init__``, so this file only
  indexes it.
* ``config.moe_layer_freq`` -- ``1`` selects the Stable Latent MoE, ``0``
  the dense ``situ`` MLP. Parsed here with upstream's own semantics
  (``gpt_layer_specs.py``).

Every MLP spec in this file fills the ``MLPSubmodules.activation_func``
module slot. That is load-bearing, not tidiness: the K3 yamls set
``use_te_activation_func: true``, and with the slot empty ``MLP.__init__``
falls back to ``config.activation_func`` -- ``F.silu`` -- applied to the
**fused** ``[gate | up]`` tensor (``mlp.py``), which is both the wrong
activation and double the width ``linear_fc2`` expects.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List, Optional

import torch  # noqa: F401  # must precede any transformer_engine import
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from primus.backends.megatron.core.models.kimi_k3.build_context import (
    resolve_k3_provider,
)
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_block import (
    KimiK3Layer,
    KimiK3LayerSubmodules,
    KimiK3TransformerBlock,
    KimiK3TransformerBlockSubmodules,
    attn_res_num_blocks_before,
)
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    KimiK3TransformerConfig,
)
from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
    AttentionResidualHead,
    AttentionResidualMixer,
)
from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
    KimiDeltaAttention,
    KimiDeltaAttentionSubmodules,
)
from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_moe_specs import (
    build_stable_latent_moe_spec,
)

if TYPE_CHECKING:
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        KimiK3SpecProvider,
    )

logger = logging.getLogger(__name__)

__all__ = [
    "get_kimi_k3_runtime_decoder_spec",
    "get_kimi_k3_moe_layer_pattern",
    "build_kimi_k3_layer_spec",
]


# ---------------------------------------------------------------------------
# Per-layer patterns
# ---------------------------------------------------------------------------


def get_kimi_k3_moe_layer_pattern(config: KimiK3TransformerConfig) -> List[int]:
    """Per-layer MoE pattern, ``1`` = MoE and ``0`` = dense.

    Upstream's semantics verbatim (``gpt_layer_specs.py``): an int
    ``N`` means ``i % N == 0``, a list is used as-is and length-checked.

    The string form (``"([0]*1+[1]*7)"``, which is what the K3 yamls write)
    is resolved here as well. It normally arrives already evaluated, because
    Primus's ``megatron.args.moe_layer_freq`` patch runs at the args layer --
    but only the *launcher* path goes through that patch. A config built
    in-process from a parsed yaml, which is what the config tests and the
    projection code do (``primus/core/projection/training_config.py``
    evaluates it for the same reason), still carries the raw string. The
    character whitelist is the one Megatron's own pattern evaluator uses
    (``arguments.py``).

    With no experts configured the whole stack is dense, which keeps this
    file usable for a no-MoE debug shape.
    """
    if not int(config.num_moe_experts or 0):
        return [0] * int(config.num_layers)

    freq = config.moe_layer_freq
    num_layers = int(config.num_layers)
    if isinstance(freq, str):
        if re.search(r"[^,\d\[\]\(\)\+\*\s]", freq):
            raise ValueError(f"moe_layer_freq contains unsupported characters: {freq!r}")
        freq = eval(freq, {"__builtins__": {}}, {})  # noqa: S307
    if isinstance(freq, bool):
        raise ValueError(f"Invalid moe_layer_freq: bool, {freq}")
    if isinstance(freq, int):
        return [1 if (i % freq == 0) else 0 for i in range(num_layers)]
    if isinstance(freq, (list, tuple)):
        if len(freq) != num_layers:
            raise ValueError(
                f"moe_layer_freq has length {len(freq)}, expected num_layers={num_layers}: {freq}"
            )
        return [int(x) for x in freq]
    raise ValueError(f"Invalid moe_layer_freq: {type(freq).__name__}, {freq}")


# ---------------------------------------------------------------------------
# Leaf specs
# ---------------------------------------------------------------------------


def _build_norm_spec(*, config: KimiK3TransformerConfig, provider: "KimiK3SpecProvider") -> ModuleSpec:
    del config
    norm_module = provider.k3_norm_module()
    assert norm_module is not None, "Kimi K3 norm module must be provided by KimiK3SpecProvider."
    return ModuleSpec(module=norm_module)


def _build_kda_spec(*, config: KimiK3TransformerConfig, provider: "KimiK3SpecProvider") -> ModuleSpec:
    """One Kimi Delta Attention layer.

    :class:`KimiDeltaAttention` passes every constructor argument to its
    projections itself (``kimi_delta_attention.py``), so these
    slots carry bare classes rather than parameterised ``ModuleSpec``\\ s.

    ``f_a_proj`` is the exception and is deliberately **not** the same
    class as the rest. It is the replicated low-rank down-projection of
    the decay gate, and ``_duplicated_linear_kwargs``
    (``kimi_delta_attention.py``) replicates a non-``TELinear``
    class by asking for ``gather_output=True`` -- which TE's
    column-parallel wrapper rejects outright
    (``transformer_engine.py``). The Megatron-native
    :class:`ColumnParallelLinear` accepts it, and at TP=1 the gather is an
    identity. Its     ``TELinear`` sibling is not usable here either:
    ``TELinear.__init__`` requires ``skip_weight_param_allocation``
    (``transformer_engine.py``) and KDA's ``build_module`` call does
    not pass it.

    ``attn_mask_type`` is declared even though KDA has no mask tensor and
    no softmax: ``MultiTokenPredictionLayer.__init__`` reads
    ``self_attention.params['attn_mask_type']`` off the inner layer's spec
    and asserts it is one of ``SUPPORTED_ATTN_MASK``
    (``multi_token_prediction.py``), and ``ModuleSpec.params``
    defaults to ``{}`` (``spec_utils.py``) so the missing key would make
    the MTP block refuse to construct with a confusing message about mask
    types. ``causal`` is the truthful value -- KDA's causality comes from
    the recurrence and the causal short convolution -- and
    :class:`KimiDeltaAttention` records it without acting on it.
    """
    del config
    column = provider.column_parallel_linear()
    return ModuleSpec(
        module=KimiDeltaAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=KimiDeltaAttentionSubmodules(
            q_proj=column,
            k_proj=column,
            v_proj=column,
            f_a_proj=provider.column_parallel_linear_with_gather_output(),
            f_b_proj=column,
            b_proj=column,
            g_proj=column,
            # out_norm stays at the KimiGatedRMSNorm default: the head-wise
            # norm and the sigmoid output gate have to share one fp32 region
            # to match fla's fused kernel, which no Megatron norm does.
            o_proj=provider.row_parallel_linear(),
        ),
    )


def _build_full_attention_spec(
    *, config: KimiK3TransformerConfig, provider: "KimiK3SpecProvider"
) -> ModuleSpec:
    """One NoPE MLA layer with the sigmoid output gate.

    Imported from its concrete module path rather than through the
    ``kimi_k3`` package ``__init__``: that package deliberately does not
    export the MLA module, because re-exporting it would make every
    ``import ...kimi_k3`` pull in TransformerEngine, which the KDA tests
    do not need.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kimi_k3_mla_attention import (
        get_kimi_k3_mla_attention_spec,
    )

    return get_kimi_k3_mla_attention_spec(config=config, backend=provider, attn_mask_type=AttnMaskType.causal)


def _build_dense_mlp_spec(*, config: KimiK3TransformerConfig, provider: "KimiK3SpecProvider") -> ModuleSpec:
    """The dense ``situ`` SwiGLU FFN used by the first ``first_k_dense_replace`` layers.

    Upstream :class:`MLP`, not a bespoke class: with ``gated_linear_unit``
    on, ``linear_fc1`` already emits the fused ``[gate | up]`` tensor K3
    wants, and the ``activation_func`` slot is where ``situ`` goes.
    """
    del config
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=provider.column_parallel_linear(),
            linear_fc2=provider.row_parallel_linear(),
            activation_func=provider.k3_mlp_activation_func(),
        ),
    )


# ---------------------------------------------------------------------------
# Layer + block
# ---------------------------------------------------------------------------


def build_kimi_k3_layer_spec(
    config: KimiK3TransformerConfig,
    *,
    provider: "KimiK3SpecProvider",
    layer_idx: int,
    is_moe: bool,
    is_kda: Optional[bool] = None,
    use_attn_residuals: Optional[bool] = None,
) -> ModuleSpec:
    """One :class:`KimiK3Layer` spec.

    Args:
        config: the runtime config.
        provider: the K3 spec provider.
        layer_idx: 0-indexed **global** layer index. Selects the attention
            variant (unless ``is_kda`` overrides it) and drives the
            attention-residual checkpoint schedule.
        is_moe: routed Stable Latent MoE rather than the dense ``situ`` MLP.
        is_kda: override the interleave. ``None`` reads
            ``config.is_kda_layer(layer_idx)``. The MTP spec sets it, because
            an MTP layer has no position in the interleave to read.
        use_attn_residuals: override the attention-residual mechanism for this
            layer alone. ``None`` derives it from
            ``config.attn_res_block_size`` as every decoder layer does.
            ``False`` builds neither mixer and makes the layer's forward take
            its plain-residual branch -- which is what the MTP layer needs,
            because it sits *after* ``attn_res_head`` has already collapsed the
            checkpoint set and there is no ``block_residual`` left to mix.

    There is deliberately no ``is_mtp_layer`` argument. Upstream passes it as a
    ``build_module`` keyword when it constructs an MTP depth's inner layer
    (``multi_token_prediction.py``), and a spec ``param`` of the same
    name would collide -- ``build_module`` unpacks ``params`` and ``kwargs``
    into one call, so the layer would be constructed with two values for it and
    raise ``TypeError``.
    """
    if is_kda is None:
        is_kda = bool(config.is_kda_layer(layer_idx))
    is_kda = bool(is_kda)
    attention = (
        _build_kda_spec(config=config, provider=provider)
        if is_kda
        else _build_full_attention_spec(config=config, provider=provider)
    )
    mlp = (
        build_stable_latent_moe_spec(config=config, provider=provider)
        if is_moe
        else _build_dense_mlp_spec(config=config, provider=provider)
    )

    use_res = bool(config.attn_res_block_size) if use_attn_residuals is None else bool(use_attn_residuals)
    # Layer 0 enters with no checkpoints, so its pre-attention mix is skipped
    # (``modeling_kimi_linear.py``) and the mixer would be dead weight.
    # See the matching comment in ``KimiK3Layer.__init__``.
    runs_pre_attn_mix = use_res and attn_res_num_blocks_before(layer_idx, config.attn_res_block_size) > 0
    submodules = KimiK3LayerSubmodules(
        input_layernorm=_build_norm_spec(config=config, provider=provider),
        self_attention=attention,
        pre_mlp_layernorm=_build_norm_spec(config=config, provider=provider),
        mlp=mlp,
        attn_res_mixer=ModuleSpec(module=AttentionResidualMixer) if runs_pre_attn_mix else None,
        mlp_res_mixer=ModuleSpec(module=AttentionResidualMixer) if use_res else None,
    )
    params = {"layer_idx": layer_idx, "is_kda_layer": is_kda}
    if use_attn_residuals is not None:
        params["use_attn_residuals"] = bool(use_attn_residuals)
    return ModuleSpec(module=KimiK3Layer, params=params, submodules=submodules)


# Kept as the historical private name so nothing outside this module has to
# change; ``build_kimi_k3_layer_spec`` is what the MTP spec module imports.
_build_layer_spec = build_kimi_k3_layer_spec


def _build_stage_layer_specs(
    config: KimiK3TransformerConfig,
    *,
    provider: "KimiK3SpecProvider",
    vp_stage: Optional[int],
    pp_rank: Optional[int],
) -> List[ModuleSpec]:
    """This stage's layer specs, indexed by **global** layer id.

    The global index is what selects the attention variant and drives the
    attention-residual checkpoint schedule, so the slicing has to keep it
    -- which is why the specs carry ``layer_idx`` explicitly rather than
    relying on their position in the list.
    """
    num_layers = int(config.num_layers)
    moe_pattern = get_kimi_k3_moe_layer_pattern(config)

    try:
        local_count = int(get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank))
        layer_offset = int(get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank))
    except Exception:
        # Mirrors deepseek_v4_layer_specs.py: without an
        # initialised parallel_state (CPU spec-tree tests) build the whole
        # stack on one stage.
        local_count = num_layers
        layer_offset = 0

    local_start = max(0, layer_offset)
    local_end = min(num_layers, local_start + max(0, local_count))

    return [
        _build_layer_spec(
            config,
            provider=provider,
            layer_idx=layer_idx,
            is_moe=bool(moe_pattern[layer_idx]),
        )
        for layer_idx in range(local_start, local_end)
    ]


def get_kimi_k3_runtime_decoder_spec(
    config: KimiK3TransformerConfig,
    *,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> ModuleSpec:
    """The Kimi K3 runtime decoder spec tree.

    Mirrors ``get_deepseek_v4_runtime_decoder_spec``
    (``deepseek_v4_layer_specs.py``). The returned block
    submodules always carry a non-empty ``layer_specs``.
    """
    provider = resolve_k3_provider(config)

    layer_specs = _build_stage_layer_specs(config, provider=provider, vp_stage=vp_stage, pp_rank=pp_rank)
    assert layer_specs, "Kimi K3 requires non-empty stage layer specs."

    pattern = "".join("K" if s.params["is_kda_layer"] else "F" for s in layer_specs)
    logger.info(
        "[Primus:Kimi-K3] spec provider=%s, %d layer specs, attention pattern=%s",
        type(provider).__name__,
        len(layer_specs),
        pattern,
    )

    block_submodules = KimiK3TransformerBlockSubmodules(
        layer_specs=layer_specs,
        # Built on the post_process stage only; see KimiK3TransformerBlock.
        attn_res_head=(ModuleSpec(module=AttentionResidualHead) if config.attn_res_block_size else None),
        final_layernorm=_build_norm_spec(config=config, provider=provider),
    )
    return ModuleSpec(module=KimiK3TransformerBlock, submodules=block_submodules)
