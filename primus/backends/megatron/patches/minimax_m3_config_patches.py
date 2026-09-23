###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3 config-class selection.

``core_transformer_config_from_args`` picks the config dataclass a run builds:
``TransformerConfig`` by default, ``MLATransformerConfig`` when
``args.multi_latent_attention``. Primus's own model families (Kimi-K3,
DeepSeek-V4) get theirs by passing ``config_class=`` from their builders, but
MiniMax-M3 reuses upstream ``GPTModel`` and therefore upstream ``gpt_builder``,
which calls ``core_transformer_config_from_args(args)`` with no class.

So this patch adds the missing branch: ``args.minimax_sparse_attention`` ->
``MSATransformerConfig``, exactly mirroring the ``multi_latent_attention``
branch one line above it upstream. An explicit ``config_class=`` from a caller
still wins.

It also closes a conflict upstream leaves silent. That ``multi_latent_attention``
branch has no ``config_class is None`` guard, so it overwrites whatever the
caller asked for -- the same trap Kimi-K3 documents in ``kimi_k3_builders.py``.
A preset with both flags set would therefore build a plain
``MLATransformerConfig``: every ``sparse_*`` field would vanish,
``MSATransformerConfig`` would never be constructed, and the mutual-exclusion
check in its ``__post_init__`` would never run. The wrapper raises first.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "megatron.minimax_m3.config"


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Make core_transformer_config_from_args select MSATransformerConfig when "
        "minimax_sparse_attention is set, the way it selects MLATransformerConfig for "
        "multi_latent_attention."
    ),
    condition=lambda ctx: bool(getattr(get_args(ctx), "minimax_sparse_attention", False)),
)
def patch_minimax_m3_config(ctx: PatchContext):
    import megatron.training.arguments as arguments_module

    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    if is_patched(arguments_module, _PATCH_KEY):
        log_rank_0(f"[Patch:{_PATCH_KEY}]   Already patched; skipping.")
        return

    original = arguments_module.core_transformer_config_from_args

    def core_transformer_config_from_args(args, config_class=None):
        wants_msa = bool(getattr(args, "minimax_sparse_attention", False))

        if wants_msa and getattr(args, "multi_latent_attention", False):
            raise ValueError(
                "minimax_sparse_attention and multi_latent_attention cannot both be set: "
                "MSA runs on GQA, and core_transformer_config_from_args would silently "
                "replace the config class with MLATransformerConfig, dropping every "
                "sparse_* field. Turn one of them off in the model preset."
            )

        if config_class is None and wants_msa:
            config_class = MSATransformerConfig

        return original(args, config_class=config_class)

    rebound = rebind_everywhere(
        arguments_module, "core_transformer_config_from_args", core_transformer_config_from_args
    )
    mark_patched(arguments_module, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}]   Patched core_transformer_config_from_args -> MSATransformerConfig "
        f"when minimax_sparse_attention is set; rebound in: {', '.join(rebound)}"
    )
