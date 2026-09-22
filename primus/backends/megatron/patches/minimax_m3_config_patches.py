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
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
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
        if config_class is None and getattr(args, "minimax_sparse_attention", False):
            config_class = MSATransformerConfig
        return original(args, config_class=config_class)

    arguments_module.core_transformer_config_from_args = core_transformer_config_from_args
    mark_patched(arguments_module, _PATCH_KEY)
    log_rank_0(
        f"[Patch:{_PATCH_KEY}]   Patched "
        "megatron.training.arguments.core_transformer_config_from_args -> "
        "MSATransformerConfig when minimax_sparse_attention is set"
    )

    # gpt_builders imports the symbol directly, so the module attribute above
    # does not reach it; rebind its local name too.
    try:
        import gpt_builders as gpt_builders_module  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        log_rank_0(
            f"[Patch:{_PATCH_KEY}]   Failed to import gpt_builders; cannot patch its local "
            "core_transformer_config_from_args binding."
        )
        raise RuntimeError("Failed to import required module gpt_builders") from exc

    gpt_builders_module.core_transformer_config_from_args = core_transformer_config_from_args
    log_rank_0(
        f"[Patch:{_PATCH_KEY}]   Patched gpt_builders.core_transformer_config_from_args -> same wrapper"
    )
