###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3 model patches.

Companion to ``minimax_m3_config_patches``, which owns the config-class
selection. This module owns the patches that change what the model *builds*.
Today that is one patch: the per-layer attention swap below.

M3's other two deviations from a stock GQA+MoE decoder need no patch, because
Megatron expresses them exactly:

  - ``swigluoai`` (clamp, quick-gelu gate, ``+1`` on the linear half) is
    ``quick_geglu`` + ``glu_linear_offset`` + ``activation_func_clamp_value``;
  - ``use_gemma_norm`` (RMSNorm weighted by ``1 + w``) is
    ``layernorm_zero_centered_gamma``.

Both are derived and validated in ``MSATransformerConfig.__post_init__``.

--- Attention swap ---

M3 reuses upstream ``GPTModel``, so the whole change is one substitution in the
decoder layer specs: on the layers where ``sparse_attention_freq`` is 1, the
``self_attention`` slot's ``module`` becomes
:class:`~primus.backends.megatron.core.transformer.minimax_m3.MinimaxSparseAttention`
instead of upstream ``SelfAttention``. Everything else -- the qkv/proj
submodules, the MoE layer spec, the MTP spec -- is upstream's.

This *wraps* ``get_gpt_decoder_layer_specs`` rather than replacing it: Primus
already ships its own fork of that function for LFM2 short-conv layers
(``gpt_decoder_layer_specs_patches``, priority 42), and a second full
replacement would fight over the same binding. Running at priority 45 means we
post-process whatever that fork (or upstream) returns, so the two compose.

``get_gpt_decoder_block_spec`` -- the branch ``gpt_builder`` takes for MoE
models like M3 -- calls ``get_gpt_decoder_layer_specs`` through the module
global, so patching the module attribute reaches it. ``gpt_builders.py``
imports the symbol by name instead, so it needs its own rebind; see
``_rebind.rebind_everywhere``.
"""

import dataclasses

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.minimax_m3.sparse_attention"


def _swap_self_attention(layer_spec, sparse_attention_cls, cache):
    """Return ``layer_spec`` with its ``self_attention`` module swapped for MSA.

    The specs come from a small set of shared template objects (one dense, one
    MoE), so rewritten copies are cached by identity instead of being rebuilt
    once per layer.
    """
    cached = cache.get(id(layer_spec))
    if cached is not None:
        return cached

    from megatron.core.transformer.attention import SelfAttention

    submodules = getattr(layer_spec, "submodules", None)
    self_attention = getattr(submodules, "self_attention", None)
    module = getattr(self_attention, "module", None)

    if module is None or not (isinstance(module, type) and issubclass(module, SelfAttention)):
        # Another Primus patch already put something else in this slot (e.g. an
        # LFM2 short conv). Leave it alone rather than silently dropping it.
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}]   Layer spec has self_attention module {module!r}, "
            "not a SelfAttention subclass; leaving it unchanged."
        )
        cache[id(layer_spec)] = layer_spec
        return layer_spec

    patched_spec = dataclasses.replace(
        layer_spec,
        submodules=dataclasses.replace(
            submodules,
            self_attention=dataclasses.replace(
                self_attention,
                module=sparse_attention_cls,
                submodules=_widen_submodules(self_attention.submodules),
            ),
        ),
    )
    cache[id(layer_spec)] = patched_spec
    return patched_spec


def _widen_submodules(self_attention_submodules):
    """Carry the upstream attention submodules into the MSA dataclass.

    MSA needs four more modules than ``SelfAttentionSubmodules`` has fields for
    (the indexer's two projections and two norms), so the spec's submodules
    object has to be rebuilt as the wider class -- ``dataclasses.replace`` keeps
    the original class and cannot add fields.

    The four are taken from the same spec provider upstream uses, so a Primus
    patch that swaps the provider (turbo) is picked up here too. This mirrors
    how DSA's indexer submodules are supplied in
    ``megatron/core/models/gpt/experimental_attention_variant_module_specs.py``.
    """
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    from primus.backends.megatron.core.transformer.minimax_m3 import (
        MinimaxSparseAttentionSubmodules,
    )

    # Resolved lazily: Primus's turbo patch replaces TESpecProvider in that
    # module, and this runs after it (priority 45).
    backend = TESpecProvider()
    return MinimaxSparseAttentionSubmodules(
        **dataclasses.asdict(self_attention_submodules),
        linear_index_q=backend.linear(),
        linear_index_k=backend.linear(),
        index_q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
        index_k_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
    )


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Swap SelfAttention for MinimaxSparseAttention on the decoder layers selected by "
        "sparse_attention_freq, keeping upstream GPTModel."
    ),
    condition=lambda ctx: bool(getattr(get_args(ctx), "minimax_sparse_attention", False)),
    priority=45,  # must wrap after te_spec_provider_patches and gpt_decoder_layer_specs_patches
)
def patch_minimax_sparse_attention(ctx: PatchContext):
    from megatron.core.models.gpt import gpt_layer_specs as megatron_gpt_layer_specs

    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )
    from primus.backends.megatron.core.transformer.minimax_m3 import (
        MinimaxSparseAttention,
    )

    if is_patched(megatron_gpt_layer_specs, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}]   Already patched; skipping.")
        return

    original = megatron_gpt_layer_specs.get_gpt_decoder_layer_specs

    def get_gpt_decoder_layer_specs(config, *args, **kwargs):
        layer_specs = original(config, *args, **kwargs)

        if not isinstance(config, MSATransformerConfig) or not config.minimax_sparse_attention:
            return layer_specs

        pattern = config.sparse_layer_pattern
        cache = {}
        return [
            _swap_self_attention(layer_spec, MinimaxSparseAttention, cache) if pattern[i] else layer_spec
            for i, layer_spec in enumerate(layer_specs)
        ]

    rebound = rebind_everywhere(
        megatron_gpt_layer_specs, "get_gpt_decoder_layer_specs", get_gpt_decoder_layer_specs
    )
    mark_patched(megatron_gpt_layer_specs, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}]   Wrapped get_gpt_decoder_layer_specs: MSA layers now build "
        f"MinimaxSparseAttention; rebound in: {', '.join(rebound)}"
    )
